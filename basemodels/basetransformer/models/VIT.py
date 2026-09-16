import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import timm
from copy import deepcopy
from timm.models.layers import to_2tuple

"""
# Vision Transformer (ViT) 特征提取步骤输入输出总结
输入: (4, 1, 128, 1024) -> 通道复制 -> (4, 3, 128, 1024)    (batch_size, channels, freq, time) 单通道频谱图扩展为三通道
     ↓
尺寸调整: (4, 3, 128, 1024) -> 插值缩放 -> (4, 3, 224, 224)    (batch_size, channels, height, width) 调整到ViT标准输入尺寸
     ↓
Patch嵌入: (4, 3, 224, 224) -> 16x16卷积 -> (4, 196, 768)    (batch_size, num_patches, embed_dim) 将图像分割为patches
     ↓
Token添加: (4, 196, 768) -> 添加CLS -> (4, 197, 768)    (batch_size, num_patches+1, embed_dim) 添加分类token
     ↓
位置编码: (4, 197, 768) [可学习位置嵌入]    (batch_size, seq_len, embed_dim) 添加位置信息
     ↓
Transformer层1: (4, 197, 768) -> 多头注意力+FFN -> (4, 197, 768)    (batch_size, seq_len, embed_dim) 第1层编码器
     ↓
Transformer层2-12: (4, 197, 768) [深层特征提取]    (batch_size, seq_len, embed_dim) 11层深层编码器
     ↓
LayerNorm: (4, 197, 768) -> 归一化 -> (4, 197, 768)    (batch_size, seq_len, embed_dim) 最终层归一化
     ↓
CLS提取: (4, 197, 768) -> 取第0位 -> (4, 768)    (batch_size, embed_dim) 提取分类token特征
     ↓
全局特征: (4, 768) [ViT特征表示]    (batch_size, feature_dim) 最终的全局图像特征
     ↓
分类预测: (4, 768) -> MLP -> (4, 4)    (batch_size, num_classes) ICBHI 4类输出：normal, crackle, wheeze, both

# ViT核心原理详解
ViT核心机制:
├── Patch分割 (Patch Partitioning):
│   ├── 图像分块: 将H×W图像分割为N个不重叠的P×P patches
│   ├── 线性投影: 每个patch通过可学习线性变换映射到D维
│   ├── 序列化: 将2D patches展平为1D序列
│   └── 空间信息: 通过位置编码保持空间关系
├── 自注意力机制 (Self-Attention):
│   ├── 全局建模: 每个patch都能关注到所有其他patches
│   ├── 多头注意力: 并行计算多个注意力头
│   ├── 位置无关: 不依赖于patches的空间位置
│   └── 动态权重: 根据内容动态分配注意力权重
├── 分类token (Classification Token):
│   ├── 特殊token: 可学习的[CLS] token，类似BERT
│   ├── 全局聚合: 通过自注意力聚合所有patch信息
│   ├── 分类表示: 最终的[CLS] token作为整体图像表示
│   └── 下游任务: 用于分类、检索等下游任务
└── 位置编码 (Positional Encoding):
    ├── 可学习嵌入: 1D位置编码，每个位置有独立参数
    ├── 空间感知: 补偿自注意力的位置无关性
    ├── 相对位置: 学习patches间的相对空间关系
    └── 插值扩展: 支持不同尺寸输入的位置编码插值

# ViT vs CNN 关键差异
ViT优势:
├── 全局建模: 每层都有全局感受野，无CNN的局部性限制
├── 序列建模: 直接将视觉理解转化为序列到序列问题
├── 可扩展性: 在大规模数据下表现优异，参数效率高
├── 无归纳偏置: 纯数据驱动学习，减少人为假设
└── 统一架构: 同一架构适用于多种视觉任务

CNN优势:
├── 空间局部性: 天然的空间归纳偏置适合图像
├── 参数共享: 卷积核共享减少参数量
├── 多尺度特征: 自然的层次化特征提取
├── 小数据友好: 在小数据集上表现更稳定
└── 计算效率: 卷积操作更适合现有硬件加速
"""

class ViTModel(nn.Module):
    """
    Vision Transformer (ViT) model for audio classification, using timm implementation.
    :param label_dim: number of classes
    :param input_fdim: frequency dimension of input spectrogram
    :param input_tdim: time dimension of input spectrogram
    :param imagenet_pretrain: whether to use ImageNet pre-trained weights
    :param model_size: which ViT architecture to use ('tiny', 'small', 'base', 'large')
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
                 imagenet_pretrain=True, audioset_pretrain=False, model_size='base', verbose=True, mix_beta=None,
                 freeze_base=False, freeze_layers=0, patch_size=16):
        super(ViTModel, self).__init__()
        
        self.mix_beta = mix_beta
        
        # 调试信息 - 添加此行来验证传入的参数
        if verbose:
            print(f"Debug - 收到的 imagenet_pretrain 参数: {imagenet_pretrain}")
        
        if verbose:
            print('---------------Vision Transformer Model Summary---------------')
            print(f'Using ViT-{model_size} architecture (timm implementation)')
            print('ImageNet pretraining: {:s}'.format(str(imagenet_pretrain)))
        
        # 明确将 imagenet_pretrain 赋值给 pretrained 变量，避免混淆
        pretrained = imagenet_pretrain
        if verbose:
            print(f"Debug - pretrained 参数设置为: {pretrained}")
        
        # 根据模型大小选择对应的 timm 模型名称
        if model_size == 'tiny':
            model_name = 'vit_tiny_patch16_224'
            self.final_feat_dim = 192
            self.embed_dim = 192
            num_heads = 3
        elif model_size == 'small':
            model_name = 'vit_small_patch16_224'
            self.final_feat_dim = 384
            self.embed_dim = 384
            num_heads = 6
        elif model_size == 'base':
            model_name = 'vit_base_patch16_224'
            self.final_feat_dim = 768
            self.embed_dim = 768
            num_heads = 12
        elif model_size == 'large':
            model_name = 'vit_large_patch16_224'
            self.final_feat_dim = 1024
            self.embed_dim = 1024
            num_heads = 16
        else:
            raise ValueError(f'不支持的 ViT 模型大小: {model_size}')
        
        # 计算输入尺寸，确保能被patch_size整除
        self.patch_size = patch_size
        input_fdim_padded = ((input_fdim + patch_size - 1) // patch_size) * patch_size
        input_tdim_padded = ((input_tdim + patch_size - 1) // patch_size) * patch_size

        if verbose:
            if input_fdim != input_fdim_padded or input_tdim != input_tdim_padded:
                print(f"调整输入尺寸从 {input_fdim}x{input_tdim} 到 {input_fdim_padded}x{input_tdim_padded} 以适应 patch_size={patch_size}")

        # 尝试加载模型
        try:
            # 使用 timm 创建模型
            # 确保 pretrained 参数来自 imagenet_pretrain
            self.vit = timm.create_model(
                model_name,
                pretrained=pretrained,  # 使用上面设置的 pretrained 变量
                in_chans=1,  # 单声道输入
                num_classes=0,  # 移除分类头
                img_size=(input_fdim_padded, input_tdim_padded)  # 使用调整后的尺寸
            )
            if verbose:
                print(f"模型创建成功: {model_name}, pretrained={pretrained}")
                if pretrained:
                    print(f"成功加载带有 ImageNet 预训练权重的 {model_name} 模型")
        except Exception as e:
            print(f"加载 {model_name} 预训练权重失败: {e}，尝试使用随机初始化")
            try:
                self.vit = timm.create_model(
                    model_name,
                    pretrained=False,
                    in_chans=1,
                    num_classes=0,
                    img_size=(input_fdim_padded, input_tdim_padded)
                )
                if verbose:
                    print(f"使用随机初始化的 {model_name} 模型")
            except Exception as e2:
                print(f"创建模型失败: {e2}，使用备用模型")
                self._create_fallback_model(input_fdim_padded, input_tdim_padded, patch_size)
        
        # 获取模型组件
        if hasattr(self.vit, 'blocks'):
            self.blocks = self.vit.blocks
        elif hasattr(self.vit, 'transformer'):
            self.blocks = self.vit.transformer.blocks
        else:
            self.blocks = nn.ModuleList([])
            print("警告: 无法访问 Transformer 块")
            
        # 添加分类头
        self.mlp_head = nn.Sequential(
            nn.Linear(self.final_feat_dim, self.final_feat_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(self.final_feat_dim // 2),
            nn.Linear(self.final_feat_dim // 2, label_dim)
        )
        
        # 计算patch数量
        f_dim, t_dim = self.get_shape(input_fdim_padded, input_tdim_padded, patch_size)
        self.num_patches = f_dim * t_dim
        
        # 为了与AST接口兼容
        self.v = type('', (), {})()
        self.v.patch_embed = type('', (), {})()
        self.v.patch_embed.num_patches = self.num_patches
        
        if verbose:
            print(f'最终特征维度: {self.final_feat_dim}')
            print(f'Patch大小: {patch_size}x{patch_size}')
            print(f'Patch数量: {self.num_patches} ({f_dim}x{t_dim})')
            print(f'Transformer块数量: {len(self.blocks) if hasattr(self, "blocks") else "未知"}')
            print(f'注意力头数量: {num_heads}')
        
        # 冻结层功能
        if freeze_base:
            self._freeze_layers(freeze_layers)
            if verbose:
                print(f'冻结前 {freeze_layers} 个Transformer块')
    
    def _create_fallback_model(self, input_fdim, input_tdim, patch_size):
        """创建一个备用模型，当ViT模型加载失败时使用"""
        print("创建备用CNN模型")
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
        self.vit = nn.Sequential(*layers)
        
        # 设置特征维度
        self.final_feat_dim = in_channels
        print(f"备用CNN模型特征维度: {self.final_feat_dim}")
    
    def _freeze_layers(self, freeze_layers):
        """冻结指定层的参数"""
        if freeze_layers <= 0:
            return
        
        # 冻结前几个transformer块
        n_blocks = len(self.blocks)
        blocks_to_freeze = min(freeze_layers, n_blocks)
        
        for i in range(blocks_to_freeze):
            for param in self.blocks[i].parameters():
                param.requires_grad = False
    
    def get_shape(self, input_fdim, input_tdim, patch_size):
        """计算ViT的输出特征图尺寸"""
        # 计算patch的数量
        f_dim = input_fdim // patch_size
        t_dim = input_tdim // patch_size
        
        # 确保至少有1x1的特征图
        f_dim = max(1, f_dim)
        t_dim = max(1, t_dim)
        
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
    
    def _get_attention_weights(self):
        """获取注意力权重，用于可视化"""
        attn_weights = []
        
        # 收集所有注意力块的权重
        for block in self.blocks:
            if hasattr(block, 'attn'):
                if hasattr(block.attn, 'get_attention_weights'):
                    attn_weights.append(block.attn.get_attention_weights())
        
        return attn_weights
    
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
        if hasattr(self.vit, 'patch_embed') and hasattr(self.vit.patch_embed, 'img_size'):
            expected_size = self.vit.patch_embed.img_size
            if x.shape[2] != expected_size[0] or x.shape[3] != expected_size[1]:
                # 仅在首次调整时打印
                if not hasattr(self, '_size_adjusted'):
                    print(f"调整输入尺寸从 {x.shape[2:]} 到 {expected_size}")
                    self._size_adjusted = True
                x = nn.functional.interpolate(
                    x, size=expected_size, mode='bilinear', align_corners=False
                )
        
        # 提取特征 - 使用timm的ViT或fallback模型
        if hasattr(self.vit, 'forward_features') and callable(self.vit.forward_features):
            # 使用timm的forward_features方法直接获取特征
            features = self.vit.forward_features(x)
        else:
            # 使用备用模型或手动提取特征
            features = self.vit(x)
            if features.dim() > 2:
                features = torch.flatten(features, 1)
        
        # 返回特征向量，不经过分类头
        # 这样外部的classifier可以处理特征
        if not patch_mix:
            return features
        else:
            # PatchMix功能暂不实现，直接返回特征
            return features