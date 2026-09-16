import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import timm
from copy import deepcopy
from timm.models.layers import to_2tuple

"""
# Swin Transformer 特征提取步骤输入输出总结
输入: (4, 1, 128, 1024) -> 尺寸调整 -> (4, 1, 224, 224)    (batch_size, channels, freq, time) 调整到标准输入尺寸
     ↓
Patch嵌入: (4, 1, 224, 224) -> 4x4卷积 -> (4, 3136, 96)    (batch_size, num_patches, embed_dim) 4x4 patch分割
     ↓
Stage 1: (4, 3136, 96) -> SW-MSA + FFN -> (4, 3136, 96)    (batch_size, H*W, C) 窗口注意力+前馈网络
     ↓
Patch合并1: (4, 3136, 96) -> 降采样 -> (4, 784, 192)    (batch_size, H/2*W/2, 2C) 2x2邻域合并，通道翻倍
     ↓
Stage 2: (4, 784, 192) -> W-MSA + SW-MSA -> (4, 784, 192)    (batch_size, H*W, C) 交替窗口注意力
     ↓
Patch合并2: (4, 784, 192) -> 降采样 -> (4, 196, 384)    (batch_size, H/4*W/4, 4C) 继续降采样
     ↓
Stage 3: (4, 196, 384) -> SW-MSA + W-MSA -> (4, 196, 384)    (batch_size, H*W, C) 深层特征提取
     ↓
Patch合并3: (4, 196, 384) -> 降采样 -> (4, 49, 768)    (batch_size, H/8*W/8, 8C) 最终降采样
     ↓
Stage 4: (4, 49, 768) -> W-MSA + SW-MSA -> (4, 49, 768)    (batch_size, H*W, C) 最高层特征
     ↓
全局池化: (4, 49, 768) -> 平均池化 -> (4, 768)    (batch_size, feature_dim) 全局特征表示
     ↓
分类预测: (4, 768) -> MLP -> (4, 4)    (batch_size, num_classes) ICBHI 4类输出

# Swin Transformer核心创新详解
Swin核心机制:
├── 窗口注意力 (Window-based Multi-head Self-Attention):
│   ├── 局部窗口: 将特征图分割为固定大小窗口(7×7)
│   ├── 窗口内注意力: 只在窗口内计算自注意力
│   ├── 线性复杂度: O(M²) vs 全局注意力的O(n²)
│   └── 空间局部性: 保持CNN的归纳偏置
├── 移位窗口 (Shifted Window Multi-head Self-Attention):
│   ├── 窗口移位: 相邻层采用不同的窗口分割
│   ├── 跨窗口连接: 建立不同窗口间的信息交流
│   ├── 循环移位: 高效实现窗口移位操作
│   └── 全局建模: 通过移位实现全局感受野
├── 分层结构 (Hierarchical Feature Maps):
│   ├── 多尺度特征: 类似CNN的金字塔结构
│   ├── 逐层降采样: 每个stage减半分辨率
│   ├── 通道加倍: 补偿分辨率降低的信息损失
│   └── 密集预测: 支持检测和分割任务
└── Patch合并 (Patch Merging):
    ├── 2×2邻域: 合并相邻的2×2 patch
    ├── 线性投影: 降维操作减少计算量
    ├── 分辨率减半: 构建层次特征表示
    └── 信息聚合: 保留重要的空间信息

# Swin vs ViT 关键差异
Swin优势:
├── 计算效率: 线性复杂度 vs ViT的二次复杂度
├── 多尺度特征: 分层结构适合密集预测任务
├── 归纳偏置: 保持CNN的空间局部性
├── 可扩展性: 支持各种下游视觉任务
└── 实用性: 在检测、分割等任务上表现优异

ViT优势:
├── 全局建模: 每层都有全局感受野
├── 架构纯净: 纯Transformer设计更简洁
├── 理论清晰: 更直接的序列到序列建模
└── 大规模数据: 在大规模数据下表现更佳
"""


class SwinModel(nn.Module):
    """
    Swin Transformer model for audio classification.
    :param label_dim: number of classes
    :param input_fdim: frequency dimension of input spectrogram
    :param input_tdim: time dimension of input spectrogram
    :param imagenet_pretrain: whether to use ImageNet pre-trained weights
    :param model_size: which Swin architecture to use ('tiny', 'small', 'base', 'large')
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
                 imagenet_pretrain=True, audioset_pretrain=False, model_size='tiny', verbose=True, mix_beta=None,
                 freeze_base=False, freeze_layers=0):
        super(SwinModel, self).__init__()
        
        self.mix_beta = mix_beta
        
        # 调试信息 - 添加此行来验证传入的参数
        if verbose:
            print(f"Debug - 收到的 imagenet_pretrain 参数: {imagenet_pretrain}")
        
        if verbose:
            print('---------------Swin Transformer Model Summary---------------')
            print(f'Using Swin Transformer-{model_size} architecture')
            print('ImageNet pretraining: {:s}'.format(str(imagenet_pretrain)))
        
        # 明确将 imagenet_pretrain 赋值给 pretrained 变量，避免混淆
        pretrained = imagenet_pretrain
        if verbose:
            print(f"Debug - pretrained 参数设置为: {pretrained}")
        
        # 根据模型大小选择对应的 Swin 模型名称和特征维度
        if model_size == 'tiny':
            model_name = "swin_tiny_patch4_window7_224"
            self.final_feat_dim = 768
            self.embed_dim = 96
        elif model_size == 'small':
            model_name = "swin_small_patch4_window7_224"
            self.final_feat_dim = 768
            self.embed_dim = 96
        elif model_size == 'base':
            model_name = "swin_base_patch4_window7_224"
            self.final_feat_dim = 1024
            self.embed_dim = 128
        elif model_size == 'large':
            model_name = "swin_large_patch4_window7_224"
            self.final_feat_dim = 1536
            self.embed_dim = 192
        else:
            print(f"不支持的 Swin 模型大小: {model_size}，使用 'tiny' 作为备选")
            model_name = "swin_tiny_patch4_window7_224"
            self.final_feat_dim = 768
            self.embed_dim = 96
        
        # 计算输入尺寸，确保能被patch_size整除
        self.patch_size = 4  # Swin 使用 4x4 patch
        input_fdim_padded = ((input_fdim + self.patch_size - 1) // self.patch_size) * self.patch_size
        input_tdim_padded = ((input_tdim + self.patch_size - 1) // self.patch_size) * self.patch_size

        if verbose:
            if input_fdim != input_fdim_padded or input_tdim != input_tdim_padded:
                print(f"调整输入尺寸从 {input_fdim}x{input_tdim} 到 {input_fdim_padded}x{input_tdim_padded} 以适应 patch_size={self.patch_size}")
        
        # 尝试加载模型
        try:
            # 使用 timm 创建模型
            self.swin = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=1,  # 单声道输入
                num_classes=0,  # 移除分类头
                img_size=(224, 224)  # 使用标准尺寸
            )
            if verbose:
                print(f"模型创建成功: {model_name}, pretrained={pretrained}")
                if pretrained:
                    print(f"成功加载带有预训练权重的 {model_name} 模型")
        except Exception as e:
            print(f"加载 {model_name} 预训练权重失败: {e}，尝试使用随机初始化")
            try:
                self.swin = timm.create_model(
                    model_name,
                    pretrained=False,
                    in_chans=1,
                    num_classes=0,
                    img_size=(224, 224)
                )
                if verbose:
                    print(f"使用随机初始化的 {model_name} 模型")
            except Exception as e2:
                print(f"创建模型失败: {e2}，使用备用模型")
                self._create_fallback_model()
        
        # 添加分类头
        self.mlp_head = nn.Sequential(
            nn.Linear(self.final_feat_dim, self.final_feat_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(self.final_feat_dim // 2),
            nn.Linear(self.final_feat_dim // 2, label_dim)
        )
        
        # 计算patch数量
        f_dim, t_dim = self.get_shape(input_fdim_padded, input_tdim_padded)
        self.num_patches = f_dim * t_dim
        
        # 为了与AST接口兼容
        self.v = type('', (), {})()
        self.v.patch_embed = type('', (), {})()
        self.v.patch_embed.num_patches = self.num_patches
        
        if verbose:
            print(f'最终特征维度: {self.final_feat_dim}')
            print(f'Patch大小: {self.patch_size}x{self.patch_size}')
            print(f'Patch数量: {self.num_patches} ({f_dim}x{t_dim})')
            print(f"使用{'预训练' if pretrained else '随机初始化'}的Swin-{model_size}模型")
        
        # 冻结层功能
        if freeze_base and hasattr(self.swin, 'layers'):
            self._freeze_layers(freeze_layers)
            if verbose:
                print(f'冻结前 {freeze_layers} 个Swin层')
    
    def _create_fallback_model(self):
        """创建一个备用模型，当Swin模型加载失败时使用"""
        print("创建备用ViT模型")
        try:
            # 尝试加载vit_small
            self.swin = timm.create_model(
                'vit_small_patch16_224',
                pretrained=True,
                in_chans=1,
                num_classes=0,
                img_size=(224, 224)
            )
            # 调整特征维度
            self.final_feat_dim = 384  # vit_small的特征维度
            print("成功加载备用ViT模型")
        except Exception as e:
            print(f"加载备用ViT模型失败: {e}，使用CNN备用")
            # 创建一个简单的CNN作为备用
            layers = []
            in_channels = 1
            out_channels = 16
            
            # 添加卷积层
            for i in range(4):
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU())
                in_channels = out_channels
                out_channels *= 2
            
            # 全局池化
            layers.append(nn.AdaptiveAvgPool2d(1))
            
            # 创建CNN模型
            self.swin = nn.Sequential(*layers)
            
            # 确保特征维度一致性
            self.final_feat_dim = in_channels
    
    def _freeze_layers(self, freeze_layers):
        """冻结指定层的参数"""
        if freeze_layers <= 0:
            return
        
        # 冻结patch_embed
        if hasattr(self.swin, 'patch_embed'):
            for param in self.swin.patch_embed.parameters():
                param.requires_grad = False
            print("冻结 Swin patch_embed")
        
        # 冻结前几个层
        if hasattr(self.swin, 'layers'):
            n_layers = len(self.swin.layers)
            layers_to_freeze = min(freeze_layers, n_layers)
            
            for i in range(layers_to_freeze):
                for param in self.swin.layers[i].parameters():
                    param.requires_grad = False
                print(f"冻结 Swin 层 {i}")
    
    def get_shape(self, input_fdim, input_tdim):
        """计算Swin Transformer的输出特征图尺寸"""
        # 计算patch的数量
        f_dim = input_fdim // self.patch_size
        t_dim = input_tdim // self.patch_size
        
        # Swin在每个stage会减半分辨率
        for _ in range(3):  # Swin有4个stage，分辨率缩小3次
            f_dim = (f_dim + 1) // 2
            t_dim = (t_dim + 1) // 2
        
        return f_dim, t_dim
    
    def load_audio_pretrained(self, pretrained_path):
        """从音频预训练模型加载权重"""
        if os.path.exists(pretrained_path):
            print(f"加载音频预训练权重: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # 获取模型权重
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 过滤并加载权重
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() 
                              if k in model_dict and model_dict[k].shape == v.shape}
            print(f"加载 {len(pretrained_dict)}/{len(model_dict)} 个参数")
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            return True
        else:
            print(f"未找到预训练音频模型: {pretrained_path}")
            return False
    
    def square_patch(self, patch, hw_num_patch):
        """将特征向量重组为二维图形，用于PatchMix"""
        h, w = hw_num_patch
        B, C = patch.size(0), patch.size(1)
        square = patch.reshape(B, C, h, w)
        return square

    def flatten_patch(self, square):
        """将二维特征图展平为向量，用于PatchMix"""
        B, C, h, w = square.shape
        patch = square.reshape(B, C, h * w).transpose(1, 2)
        return patch

    def patch_mix(self, features, target, time_domain=False, hw_num_patch=None):
        """实现类似PatchMix的混合增强功能"""
        if self.mix_beta > 0:
            lam = np.random.beta(self.mix_beta, self.mix_beta)
        else:
            lam = 1

        batch_size = features.size(0)
        device = features.device

        # 创建随机索引用于混合
        index = torch.randperm(batch_size).to(device)
        
        # 对特征图进行混合
        if not time_domain:  # 空间维度混合
            B, C, H, W = features.shape
            num_total = H * W
            num_mask = int(num_total * (1. - lam))
            
            # 将特征图转为二维表示
            flat_features = features.reshape(B, C, -1)
            
            # 选择随机位置进行混合
            mask_idx = torch.randperm(num_total)[:num_mask].to(device)
            for i in range(batch_size):
                flat_features[i, :, mask_idx] = flat_features[index[i], :, mask_idx]
                
            # 恢复原始形状
            mixed_features = flat_features.reshape(B, C, H, W)
            lam = 1 - (num_mask / num_total)
        else:  # 时间维度混合
            B, C, H, W = features.shape
            num_mask = int(W * (1. - lam))
            
            # 选择随机时间帧进行混合
            mask_idx = torch.randperm(W)[:num_mask].to(device)
            for i in range(batch_size):
                features[i, :, :, mask_idx] = features[index[i], :, :, mask_idx]
                
            mixed_features = features
            lam = 1 - (num_mask / W)
        
        y_a, y_b = target, target[index]
        return mixed_features, y_a, y_b, lam, index
    
    @autocast()
    def forward(self, x, y=None, patch_mix=False, time_domain=False):
        """
        前向传播
        :param x: 输入频谱图，预期形状: (batch_size, channels, frequency_bins, time_frames)
        :param y: 标签，用于PatchMix
        :param patch_mix: 是否启用PatchMix增强
        :param time_domain: 是否在时间域上执行PatchMix
        :return: 特征表示
        """
        # 确保输入格式正确
        if x.dim() == 3:
            x = x.unsqueeze(1)  # 添加通道维度
        
        # 如果输入的形状是 [B, C, Time, Freq]，则交换最后两个维度
        if x.size(2) > x.size(3):
            x = x.transpose(2, 3)
        
        # 检查并调整输入尺寸，以匹配模型期望的尺寸
        if hasattr(self.swin, 'patch_embed') and hasattr(self.swin.patch_embed, 'img_size'):
            expected_size = self.swin.patch_embed.img_size
            if isinstance(expected_size, int):
                expected_size = (expected_size, expected_size)
                
            if x.shape[2] != expected_size[0] or x.shape[3] != expected_size[1]:
                # 仅在首次调整时打印
                if not hasattr(self, '_size_adjusted'):
                    print(f"调整输入尺寸从 {x.shape[2:]} 到 {expected_size}")
                    self._size_adjusted = True
                x = nn.functional.interpolate(
                    x, size=expected_size, mode='bilinear', align_corners=False
                )
        
        # 如果需要PatchMix，并且有标签，在模型前向传播过程中应用
        if patch_mix and y is not None and self.mix_beta is not None and hasattr(self.swin, 'patch_embed'):
            # 提取特征，但在最后一层之前
            if hasattr(self.swin, 'forward_features') and callable(self.swin.forward_features):
                features = self.swin.forward_features(x)
                
                # 假设特征形状为 [B, L, C]，需要转换为 [B, C, H, W] 用于混合
                # 首先估计特征图的高度和宽度
                h_patch = int(np.sqrt(features.size(1)))
                w_patch = features.size(1) // h_patch
                
                # 重塑为特征图格式 [B, C, H, W]
                features_2d = features.transpose(1, 2).reshape(features.size(0), features.size(2), h_patch, w_patch)
                
                # 应用PatchMix
                mixed_features, y_a, y_b, lam, index = self.patch_mix(
                    features_2d, y, time_domain=time_domain, hw_num_patch=(h_patch, w_patch)
                )
                
                # 转回原始格式
                mixed_features = mixed_features.reshape(features.size(0), features.size(2), -1).transpose(1, 2)
                
                # 计算混合特征
                return mixed_features, y_a, y_b, lam, index
        
        # 提取特征 - 使用timm的Swin或备用模型
        if hasattr(self.swin, 'forward_features') and callable(self.swin.forward_features):
            # 使用timm的forward_features方法直接获取特征
            features = self.swin.forward_features(x)
        else:
            # 使用备用模型或手动提取特征
            features = self.swin(x)
            if features.dim() > 2:
                features = torch.flatten(features, 1)
        
        # 返回特征向量，不经过分类头
        if not patch_mix:
            return features
        else:
            # PatchMix功能暂不实现，直接返回特征
            return features


# 保留原始PatchEmbed类以保持兼容性
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x