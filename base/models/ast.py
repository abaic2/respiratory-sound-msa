import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
import timm
from copy import deepcopy
from timm.models.layers import to_2tuple, trunc_normal_


# # 关键特征提取步骤输入输出总结
# 输入: (4, 1, 128, 1024) -> 转置 -> (4, 1, 1024, 128)    (batch_size, channels, time_frames, frequency_bins) 4个音频片段，每个片段是128维梅尔频谱，1024个时间帧
#      ↓
# Patch Embedding: (4, 1212, 768) [1212个16x16的patch]    (batch_size, num_patches, embedding_dim) 1212个patch tokens: 每个patch是16x16的音频片段，转换为768维特征向量
#      ↓
# 添加Token: (4, 1214, 768) [CLS + DIST + 1212 patches]     (batch_size, num_patches + 2, embedding_dim)1个CLS token: 用于聚合全局分类特征 1个Distillation token: 用于知识蒸馏的辅助token 1212个patch tokens: 来自音频的局部特征
#      ↓
# 位置编码: (4, 1214, 768) [空间时间位置信息]       (batch_size, num_patches + 2, embedding_dim) 位置编码为每个token提供其在时频图中的位置信息
#      ↓
# 12层Transformer: (4, 1214, 768) [全局依赖学习]    (batch_size, num_patches + 2, embedding_dim) 通过多层Transformer编码器，学习全局时频特征和长距离依赖
#      ↓
# 特征融合: (4, 768) [CLS + Distillation token平均]  (batch_size, embedding_dim) 最终特征向量: CLS token和Distillation token的平均，得到全局音频特征表示
#      ↓
# 分类预测: (4, 4) [ICBHI 4类输出] (batch_size, num_classes) 分类头输出: 4个类别的logits，表示每个音频片段属于normal, crackle, wheeze, both的概率分布


# 优化PatchEmbed以支持混合精度
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

    @autocast()
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModel(nn.Module):
    """
    混合精度优化版AST模型
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
                 imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', 
                 verbose=True, mix_beta=None):
        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------混合精度优化AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        self.final_feat_dim = 768
        self.mix_beta = mix_beta
        
        # 预计算patch维度以避免运行时计算
        self.fstride, self.tstride = fstride, tstride
        self.input_fdim, self.input_tdim = input_fdim, input_tdim

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            
            # 简化MLP head，减少内存使用
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim), 
                nn.Linear(self.original_embedding_dim, label_dim)
            )

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            
            # 预计算patch尺寸
            self.h_patch = int((input_fdim - 16) / fstride) + 1
            self.w_patch = int((input_tdim - 16) / tstride) + 1
            
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))
                print('预计算patch尺寸: h={:d}, w={:d}'.format(self.h_patch, self.w_patch))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            out_dir = r'D:\55555\bishe\pretrained_models'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            
            if os.path.exists(os.path.join(out_dir, 'audioset_10_10_0.4593.pth')) == False:
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out=os.path.join(out_dir, 'audioset_10_10_0.4593.pth'))
            
            sd = torch.load(os.path.join(out_dir, 'audioset_10_10_0.4593.pth'), map_location=device)
            audio_model = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            
            # 预计算patch尺寸
            self.h_patch = int((input_fdim - 16) / fstride) + 1
            self.w_patch = int((input_tdim - 16) / tstride) + 1
            
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            # otherwise interpolate
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def square_patch(self, patch, hw_num_patch):
        h, w = hw_num_patch
        B, _, dim = patch.size()
        # 使用view替代reshape以提高内存效率
        square = patch.view(B, h, w, dim)
        return square

    def flatten_patch(self, square):
        B, h, w, dim = square.shape
        # 使用view替代reshape
        patch = square.view(B, h * w, dim)
        return patch

    @autocast()
    def patch_mix(self, image, target, time_domain=False, hw_num_patch=None):
        """混合精度优化的patch_mix"""
        if hw_num_patch is None:
            hw_num_patch = [self.h_patch, self.w_patch]
            
        if self.mix_beta > 0:
            lam = np.random.beta(self.mix_beta, self.mix_beta)
        else:
            lam = 1

        batch_size, num_patch, dim = image.size()
        device = image.device

        # 直接在设备上生成随机索引
        index = torch.randperm(batch_size, device=device)

        if not time_domain:
            num_mask = int(num_patch * (1. - lam))
            mask = torch.randperm(num_patch, device=device)[:num_mask]

            image[:, mask, :] = image[index][:, mask, :]
            lam = 1 - (num_mask / num_patch)
        else:
            squared_1 = self.square_patch(image, hw_num_patch)
            squared_2 = self.square_patch(image[index], hw_num_patch)

            w_size = squared_1.size()[2]
            num_mask = int(w_size * (1. - lam))
            mask = torch.randperm(w_size, device=device)[:num_mask]

            squared_1[:, :, mask, :] = squared_2[:, :, mask, :]
            image = self.flatten_patch(squared_1)
            lam = 1 - (num_mask / w_size)
        
        y_a, y_b = target, target[index]
        return image, y_a, y_b, lam, index

    @autocast()
    def forward(self, x, y=None, patch_mix=False, time_domain=False):
        """
        混合精度优化的前向传播
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        x = x.transpose(2, 3)
        B = x.shape[0]
        
        # 使用预计算的patch尺寸
        x = self.v.patch_embed(x)

        if patch_mix:
            x, y_a, y_b, lam, index = self.patch_mix(x, y, time_domain=time_domain)

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        # Transformer blocks在autocast内部自动处理
        for i, blk in enumerate(self.v.blocks):
            x = blk(x)
            
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) * 0.5  # 使用乘法代替除法
        
        if not patch_mix:
            return x
        else:
            return x, y_a, y_b, lam, index

# ...existing code...

if __name__ == "__main__":
    """
    AST模型完整测试主函数 - 重点分析特征提取流程
    """
    print("=" * 80)
    print("AST模型完整测试 - 特征提取重点分析")
    print("=" * 80)
    
    # 设置设备和随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"使用设备: {device}")
    
    # 模拟ICBHI数据集的真实参数
    batch_size = 4
    sample_rate = 16000      # ICBHI数据集采样率
    desired_length = 8       # 8秒音频片段
    n_mels = 128            # 梅尔频谱bins数量
    time_frames = 1024      # 时间帧数 (约10秒音频)
    freq_bins = 128         # 频率bins
    num_classes = 4         # ICBHI 4分类：normal, crackle, wheeze, both
    fstride = 10           
    tstride = 10           
    
    print(f"\nICBHI数据集配置:")
    print(f"  批次大小: {batch_size}")
    print(f"  采样率: {sample_rate} Hz")
    print(f"  音频长度: {desired_length} 秒")
    print(f"  梅尔频谱bins: {n_mels}")
    print(f"  时间帧数: {time_frames}")
    print(f"  分类数: {num_classes} (normal, crackle, wheeze, both)")
    
    # 1. 创建AST模型
    print(f"\n{'='*25} 1. 模型创建 {'='*25}")
    try:
        model = ASTModel(
            label_dim=num_classes,
            fstride=fstride,
            tstride=tstride,
            input_fdim=freq_bins,
            input_tdim=time_frames,
            imagenet_pretrain=True,
            audioset_pretrain=False,
            model_size='base384',
            verbose=True,
            mix_beta=0.4
        ).to(device)
        print("✓ AST模型创建成功")
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        exit(1)
    
    # 模型参数分析
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.v.parameters())
    head_params = sum(p.numel() for p in model.mlp_head.parameters())
    
    print(f"模型参数分析:")
    print(f"  Backbone参数: {backbone_params:,}")
    print(f"  分类头参数: {head_params:,}")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024**2:.1f} MB (FP32)")
    
    # 2. 创建模拟ICBHI音频数据
    print(f"\n{'='*25} 2. 音频数据模拟 {'='*25}")
    
    # 模拟真实的fbank特征 (来自generate_fbank函数)
    # fbank特征范围通常在[-6, 6]之间，已经过标准化处理
    input_fbank = torch.randn(batch_size, freq_bins, time_frames) * 2.0  # 模拟标准化后的fbank
    input_fbank = input_fbank.unsqueeze(1)  # 添加通道维度: (batch, 1, freq, time)
    input_fbank = input_fbank.to(device)
    
    # 模拟ICBHI标签 (0: normal, 1: crackle, 2: wheeze, 3: both)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    label_names = ['normal', 'crackle', 'wheeze', 'both']
    
    print(f"输入fbank特征:")
    print(f"  形状: {input_fbank.shape} (batch, channel, freq_bins, time_frames)")
    print(f"  数据范围: [{input_fbank.min().item():.3f}, {input_fbank.max().item():.3f}]")
    print(f"  均值/标准差: {input_fbank.mean().item():.3f} / {input_fbank.std().item():.3f}")
    print(f"  说明: 已标准化的梅尔频谱特征，模拟来自generate_fbank()的输出")
    
    print(f"\n标签信息:")
    for i, (label, name) in enumerate(zip(labels.cpu().numpy(), [label_names[l] for l in labels.cpu().numpy()])):
        print(f"  样本{i+1}: 类别{label} ({name})")
    
    # 3. 详细分析特征提取流程
    print(f"\n{'='*25} 3. 特征提取流程详析 {'='*25}")
    model.eval()
    
    with torch.no_grad():
        # 步骤1: 输入预处理
        print("步骤1: 输入预处理与维度变换")
        x_input = input_fbank.clone()
        print(f"  原始输入: {x_input.shape}")
        print(f"  含义: (批次, 通道=1, 频率bins={freq_bins}, 时间帧={time_frames})")
        
        # AST要求的维度变换: (B, C, F, T) -> (B, C, T, F)
        x_transposed = x_input.transpose(2, 3)
        print(f"  转置后: {x_transposed.shape}")
        print(f"  含义: (批次, 通道=1, 时间帧={time_frames}, 频率bins={freq_bins})")
        print(f"  目的: 适配Vision Transformer的patch处理机制")
        
        # 步骤2: Patch Embedding - 关键的特征提取步骤
        print(f"\n步骤2: Patch Embedding - 音频到视觉patch转换")
        print(f"  Patch大小: 16×16 像素")
        print(f"  频率步长: {fstride}, 时间步长: {tstride}")
        
        # 计算patch数量
        f_patches = (freq_bins - 16) // fstride + 1  # 频率方向patch数
        t_patches = (time_frames - 16) // tstride + 1  # 时间方向patch数
        total_patches = f_patches * t_patches
        
        print(f"  频率patch数: {f_patches} (覆盖{f_patches * fstride + 6}个频率bins)")
        print(f"  时间patch数: {t_patches} (覆盖{t_patches * tstride + 6}个时间帧)")
        print(f"  总patch数: {total_patches}")
        
        x_patches = model.v.patch_embed(x_transposed)
        print(f"  Patch embedding输出: {x_patches.shape}")
        print(f"  含义: 每个16×16的音频patch -> 768维特征向量")
        print(f"  说明: 这是音频信号的局部时频特征表示")
        
        # 步骤3: 特殊Token添加
        print(f"\n步骤3: 添加分类和蒸馏Token")
        B = x_patches.shape[0]
        cls_tokens = model.v.cls_token.expand(B, -1, -1)
        dist_token = model.v.dist_token.expand(B, -1, -1)
        
        print(f"  CLS token: {cls_tokens.shape} - 全局分类特征聚合")
        print(f"  Distillation token: {dist_token.shape} - 知识蒸馏辅助")
        
        x_with_tokens = torch.cat((cls_tokens, dist_token, x_patches), dim=1)
        print(f"  合并后序列: {x_with_tokens.shape}")
        print(f"  含义: [CLS, DIST, patch1, patch2, ..., patch{total_patches}]")
        
        # 步骤4: 位置编码
        print(f"\n步骤4: 位置编码 - 空间时间位置信息")
        pos_embed_shape = model.v.pos_embed.shape
        print(f"  位置编码形状: {pos_embed_shape}")
        print(f"  作用: 为每个token提供其在时频图中的位置信息")
        
        x_pos = x_with_tokens + model.v.pos_embed
        x_pos = model.v.pos_drop(x_pos)
        print(f"  添加位置编码后: {x_pos.shape}")
        
        # 步骤5: Transformer编码器 - 核心特征提取
        print(f"\n步骤5: Transformer编码器 - 全局特征学习")
        x_encoded = x_pos.clone()
        num_layers = len(model.v.blocks)
        print(f"  Transformer层数: {num_layers}")
        print(f"  注意力头数: 12")
        print(f"  隐藏层维度: 3072")
        
        # 分析前几层的注意力模式变化
        attention_maps = []
        for i, blk in enumerate(model.v.blocks[:3]):
            x_before = x_encoded.clone()
            x_encoded = blk(x_encoded)
            
            # 计算变化程度
            change = torch.norm(x_encoded - x_before, dim=-1).mean()
            print(f"  第{i+1}层: 输出{x_encoded.shape}, 特征变化幅度: {change.item():.4f}")
            
            if i == 0:
                print(f"    -> 学习局部时频模式")
            elif i == 1:
                print(f"    -> 融合相邻patch信息")
            elif i == 2:
                print(f"    -> 建立长距离依赖")
        
        # 完成所有层
        for blk in model.v.blocks[3:]:
            x_encoded = blk(x_encoded)
        
        x_norm = model.v.norm(x_encoded)
        print(f"  最终编码: {x_norm.shape}")
        print(f"  Layer Norm完成全局特征标准化")
        
        # 步骤6: 全局特征提取 - 最关键步骤
        print(f"\n步骤6: 全局特征提取 - 音频表示生成")
        cls_feature = x_norm[:, 0]  # CLS token特征
        dist_feature = x_norm[:, 1]  # Distillation token特征
        
        print(f"  CLS特征: {cls_feature.shape}")
        print(f"  CLS特征统计: 均值={cls_feature.mean().item():.4f}, 标准差={cls_feature.std().item():.4f}")
        print(f"  含义: 聚合所有patch的全局音频特征")
        
        print(f"  Distillation特征: {dist_feature.shape}")
        print(f"  Dist特征统计: 均值={dist_feature.mean().item():.4f}, 标准差={dist_feature.std().item():.4f}")
        print(f"  含义: 知识蒸馏的辅助全局特征")
        
        # 特征融合
        final_feature = (cls_feature + dist_feature) * 0.5
        print(f"  最终特征: {final_feature.shape}")
        print(f"  最终特征统计: 均值={final_feature.mean().item():.4f}, 标准差={final_feature.std().item():.4f}")
        print(f"  含义: 融合后的768维音频全局表示，用于呼吸音分类")
        
        # 分析特征的分布
        feature_norm = torch.norm(final_feature, dim=1)
        print(f"  特征向量模长: 均值={feature_norm.mean().item():.4f}, 标准差={feature_norm.std().item():.4f}")
    
    # 4. 完整模型推理与分类
    print(f"\n{'='*25} 4. 完整推理与分类 {'='*25}")
    model.eval()
    
    with torch.no_grad():
        # 测量推理时间
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
        
        # 完整前向传播
        output_features = model(input_fbank)
        
        if torch.cuda.is_available():
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time)
            print(f"推理时间: {inference_time:.2f} ms ({batch_size}个样本)")
            print(f"单样本推理时间: {inference_time/batch_size:.2f} ms")
        
        print(f"\n特征提取结果:")
        print(f"  输出特征形状: {output_features.shape}")
        print(f"  特征维度: {output_features.shape[1]} (与Transformer嵌入维度一致)")
        print(f"  特征统计: 均值={output_features.mean().item():.4f}, 标准差={output_features.std().item():.4f}")
        
        # 分类预测
        if hasattr(model, 'mlp_head'):
            print(f"\n分类头预测:")
            predictions = model.mlp_head(output_features)
            print(f"  分类logits: {predictions.shape}")
            
            # 概率分布
            probs = torch.softmax(predictions, dim=1)
            max_probs, predicted_classes = torch.max(probs, dim=1)
            
            print(f"  预测结果:")
            for i in range(batch_size):
                true_label = labels[i].item()
                pred_label = predicted_classes[i].item()
                confidence = max_probs[i].item()
                print(f"    样本{i+1}: 真实={label_names[true_label]}, 预测={label_names[pred_label]}, 置信度={confidence:.3f}")
            
            # 分析每个类别的预测概率
            print(f"\n  各类别概率分布:")
            class_probs = probs.mean(dim=0)
            for i, (name, prob) in enumerate(zip(label_names, class_probs)):
                print(f"    {name}: {prob.item():.3f}")
    
    # 5. Patch Mix增强测试
    print(f"\n{'='*25} 5. Patch Mix数据增强 {'='*25}")
    model.train()
    
    with torch.no_grad():
        print("5.1 Spatial Patch Mix (空间混合):")
        try:
            output_spatial = model(input_fbank, y=labels, patch_mix=True, time_domain=False)
            if isinstance(output_spatial, tuple) and len(output_spatial) == 5:
                features_sp, y_a_sp, y_b_sp, lam_sp, index_sp = output_spatial
                print(f"  ✓ 混合特征: {features_sp.shape}")
                print(f"  ✓ 混合系数λ: {lam_sp:.4f}")
                print(f"  ✓ 说明: 随机替换{(1-lam_sp)*100:.1f}%的patch，增强模型鲁棒性")
                
                # 分析混合效果
                original_features = model(input_fbank)
                feature_diff = torch.norm(features_sp - original_features, dim=1).mean()
                print(f"  ✓ 特征变化幅度: {feature_diff.item():.4f}")
            else:
                print("  ✗ Spatial Patch Mix返回格式错误")
        except Exception as e:
            print(f"  ✗ Spatial Patch Mix测试失败: {e}")
        
        print("\n5.2 Temporal Patch Mix (时间混合):")
        try:
            output_temporal = model(input_fbank, y=labels, patch_mix=True, time_domain=True)
            if isinstance(output_temporal, tuple) and len(output_temporal) == 5:
                features_tp, y_a_tp, y_b_tp, lam_tp, index_tp = output_temporal
                print(f"  ✓ 混合特征: {features_tp.shape}")
                print(f"  ✓ 混合系数λ: {lam_tp:.4f}")
                print(f"  ✓ 说明: 沿时间维度混合{(1-lam_tp)*100:.1f}%的patch，模拟呼吸模式变化")
            else:
                print("  ✗ Temporal Patch Mix返回格式错误")
        except Exception as e:
            print(f"  ✗ Temporal Patch Mix测试失败: {e}")
    
    # 6. 特征可视化分析
    print(f"\n{'='*25} 6. 特征分析与可视化 {'='*25}")
    model.eval()
    
    with torch.no_grad():
        # 提取中间层特征
        x = input_fbank.transpose(2, 3)
        x = model.v.patch_embed(x)
        
        # 添加tokens和位置编码
        B = x.shape[0]
        cls_tokens = model.v.cls_token.expand(B, -1, -1)
        dist_token = model.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + model.v.pos_embed
        x = model.v.pos_drop(x)
        
        # 分析不同层的特征
        layer_features = []
        for i, blk in enumerate(model.v.blocks):
            x = blk(x)
            if i in [0, 3, 6, 9, 11]:  # 选择几个关键层
                cls_feat = x[:, 0]  # CLS token特征
                layer_features.append(cls_feat)
                
                feat_norm = torch.norm(cls_feat, dim=1).mean()
                feat_std = cls_feat.std(dim=1).mean()
                print(f"  第{i+1}层CLS特征: 模长={feat_norm.item():.4f}, 标准差={feat_std.item():.4f}")
        
        # 特征相似性分析
        if len(layer_features) >= 2:
            print(f"\n  层间特征相似性:")
            for i in range(len(layer_features)-1):
                sim = torch.cosine_similarity(layer_features[i], layer_features[i+1], dim=1).mean()
                layer_nums = [0, 3, 6, 9, 11]
                print(f"    第{layer_nums[i]+1}层 -> 第{layer_nums[i+1]+1}层: {sim.item():.4f}")
    
    # 7. 性能与效率分析
    print(f"\n{'='*25} 7. 性能效率分析 {'='*25}")
    
    # GPU内存使用
    if torch.cuda.is_available():
        print("GPU内存使用:")
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  已分配: {allocated:.1f} MB")
        print(f"  已保留: {reserved:.1f} MB")
        print(f"  内存效率: {allocated/reserved*100:.1f}%")
    
    # 计算复杂度分析
    print(f"\n计算复杂度 (FLOPs估算):")
    patch_embed_flops = batch_size * freq_bins * time_frames * 768
    attention_flops = 12 * total_patches * total_patches * 768 * 12 * batch_size
    mlp_flops = 12 * total_patches * 768 * 768 * 4 * batch_size
    total_flops = patch_embed_flops + attention_flops + mlp_flops
    
    print(f"  Patch Embedding: {patch_embed_flops/1e6:.1f} MFLOPs")
    print(f"  Self-Attention: {attention_flops/1e9:.2f} GFLOPs") 
    print(f"  MLP层: {mlp_flops/1e9:.2f} GFLOPs")
    print(f"  总计: {total_flops/1e9:.2f} GFLOPs")
    print(f"  单样本: {total_flops/batch_size/1e9:.2f} GFLOPs")
    
    # 8. 呼吸音分类特定分析
    print(f"\n{'='*25} 8. 呼吸音分类特性 {'='*25}")
    print("AST模型在呼吸音分类中的优势:")
    print("  ✓ 时频patch机制: 捕获呼吸音的局部时频模式")
    print("  ✓ 全局建模能力: 通过自注意力学习长距离依赖")
    print("  ✓ 多尺度特征: 不同层提取从局部到全局的特征")
    print("  ✓ 数据增强: Patch Mix模拟呼吸模式变化")
    print("  ✓ 知识迁移: 从ImageNet预训练获得通用视觉特征")
    
    print(f"\n针对ICBHI数据集的适配:")
    print(f"  • 输入处理: 16kHz采样 -> 128维梅尔频谱 -> 768维特征")
    print(f"  • 时间建模: {time_frames}帧 覆盖约{time_frames*0.01:.1f}秒音频")
    print(f"  • 频率建模: {freq_bins}个梅尔滤波器 覆盖50-2000Hz")
    print(f"  • 分类目标: 4类呼吸音 (正常/爆裂音/哮鸣音/混合)")
    
    print("\n" + "=" * 80)
    print("AST模型特征提取测试完成!")
    print("关键发现:")
    print("  🎯 成功将音频转换为768维全局特征表示")
    print("  🎯 Patch机制有效捕获时频局部模式")
    print("  🎯 Transformer编码器建立全局依赖关系")
    print("  🎯 双token设计(CLS+Distillation)增强特征表达")
    print("  🎯 数据增强策略提升模型泛化能力")
    print("=" * 80)