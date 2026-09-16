import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import os
import wget
import timm
from copy import deepcopy
from timm.models.layers import to_2tuple,trunc_normal_

# 🆕 改进的Pre-Patch MSA模块 - 采用点积注意力融合
class PrePatchMSA(nn.Module):
    """
    Pre-Patch多尺度注意力模块
    在Patch Embedding之前直接对原始频谱图(128×1024)进行多尺度注意力增强
    🔥 采用点积注意力融合策略
    """
    def __init__(self, input_channels=1, r=4, use_skip_connection=False, skip_ratio=0.3, 
                 skip_type='standard', fusion_type='dot_product_attention'):
        super(PrePatchMSA, self).__init__()
        self.input_channels = input_channels
        self.use_skip_connection = use_skip_connection
        self.skip_ratio = skip_ratio
        self.skip_type = skip_type
        self.fusion_type = fusion_type
        
        # 🔍 局部注意力 - 使用1×1卷积（参考MSA_small的local_att结构）
        inter_channels = max(1, input_channels // r)  # 确保至少有1个通道
        
        self.local_att = nn.Sequential(
            nn.Conv2d(input_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_channels),
        )

        # 🔍 多尺度上下文模块（参考model_utils.py中的context结构）
        self.context1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((32, 256)),  # 小尺度：局部时频模式
            nn.Conv2d(input_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_channels)
        )

        self.context2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((64, 512)),  # 中尺度：中等时频模式
            nn.Conv2d(input_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_channels)
        )

        # 🔍 全局注意力（参考MSA_small的global_att结构）
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_channels),
        )

        self.sigmoid = nn.Sigmoid()
        
        # 🔥 点积注意力融合模块
        if fusion_type == 'dot_product_attention':
            # 🔥 点积注意力：query, key, value投影
            self.query_proj = nn.Conv2d(input_channels, input_channels, kernel_size=1)
            self.key_proj = nn.Conv2d(input_channels * 4, input_channels, kernel_size=1)
            self.value_proj = nn.Conv2d(input_channels * 4, input_channels, kernel_size=1)
            
            # 缩放因子
            self.scale = (input_channels) ** -0.5
            
            # 输出投影
            self.out_proj = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, kernel_size=1),
                nn.BatchNorm2d(input_channels)
            )
            print("🔥 使用点积注意力融合 (scaled dot-product attention)")
            
        elif fusion_type == 'multi_head_dot_product':
            # 🔥 多头点积注意力
            self.num_heads = 4
            self.head_dim = input_channels // self.num_heads
            
            self.query_proj = nn.Conv2d(input_channels, input_channels, kernel_size=1)
            self.key_proj = nn.Conv2d(input_channels * 4, input_channels, kernel_size=1)
            self.value_proj = nn.Conv2d(input_channels * 4, input_channels, kernel_size=1)
            
            self.scale = (self.head_dim) ** -0.5
            
            self.out_proj = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, kernel_size=1),
                nn.BatchNorm2d(input_channels)
            )
            print("🔥 使用多头点积注意力融合 (multi-head dot-product attention)")
            
        elif fusion_type == 'self_attention_fusion':
            # 🔥 自注意力融合
            self.self_attention = SelfAttentionFusion(input_channels)
            print("🔥 使用自注意力融合 (self-attention fusion)")
            
        elif fusion_type == 'cross_dot_product':
            # 🔥 交叉点积注意力
            self.cross_attention = CrossDotProductAttention(input_channels)
            print("🔥 使用交叉点积注意力融合 (cross dot-product attention)")
            
        elif fusion_type == 'attention_fusion':
            # 🔥 基于特征的自适应注意力融合（原版）
            self.attention_fusion = nn.Sequential(
                nn.Conv2d(input_channels * 4, inter_channels, kernel_size=1),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, 4, kernel_size=1),
                nn.Softmax(dim=1)  # 在通道维度上应用softmax，保证权重和为1
            )
            print("🔥 使用自适应注意力融合 (基于特征的动态权重)")
            
        elif fusion_type == 'weighted_sum':
            # 可学习的4个分支权重：[局部, 全局, 小尺度, 中尺度]
            self.fusion_weights = nn.Parameter(torch.ones(4))  # 初始化为等权重
            print("🔥 使用可学习加权求和融合 (4个分支权重)")
            
        else:
            # 默认：简单加法
            print("📍 使用简单加法融合 (baseline)")
        
        # 🆕 跳跃连接相关参数
        if use_skip_connection:
            if skip_type == 'learnable':
                self.skip_weight = nn.Parameter(torch.tensor(skip_ratio))
            elif skip_type == 'adaptive':
                self.skip_adapter = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Conv2d(input_channels, input_channels//r, 1),
                    nn.ReLU(),
                    nn.Conv2d(input_channels//r, input_channels, 1),
                    nn.Sigmoid()
                )
        
    def forward(self, x, return_intermediates=False):
        """
        前向传播 - 🔥 采用点积注意力融合
        Args:
            x: [B, C, F, T] = [B, 1, 128, 1024] (原始频谱图)
            return_intermediates: 是否返回中间结果用于可视化
        Returns:
            增强后的频谱图 [B, 1, 128, 1024]
        """
        h, w = x.shape[2], x.shape[3]  # 获取输入 x 的高度和宽度 (128, 1024)

        # 🔍 步骤1：局部注意力
        xl = self.local_att(x)  # [B, 1, 128, 1024]
        
        # 🔍 步骤2：小尺度上下文
        c1 = self.context1(x)  # [B, 1, 32, 256]
        c1_upsampled = F.interpolate(c1, size=[h, w], mode='bilinear', align_corners=False)
        
        # 🔍 步骤3：中尺度上下文
        c2 = self.context2(x)  # [B, 1, 64, 512]
        c2_upsampled = F.interpolate(c2, size=[h, w], mode='bilinear', align_corners=False)
        
        # 🔍 步骤4：全局上下文
        xg = self.global_att(x)  # [B, 1, 1, 1]
        xg_broadcasted = xg.expand_as(x)  # [B, 1, 128, 1024]
        
        # 🔥 步骤5：点积注意力融合
        xlg = self._dot_product_fusion(xl, xg_broadcasted, c1_upsampled, c2_upsampled, x)
        
        # 🎯 步骤6：生成注意力权重
        wei = self.sigmoid(xlg)  # [B, 1, 128, 1024] 权重范围[0,1]
        
        # 🚀 步骤7：应用注意力增强
        if self.use_skip_connection:
            if self.skip_type == 'standard':
                xo = x + self.skip_ratio * (2 * x * wei)
            elif self.skip_type == 'learnable':
                xo = x + self.skip_weight * (2 * x * wei)
            elif self.skip_type == 'adaptive':
                adaptive_weight = self.skip_adapter(x)
                xo = x + adaptive_weight * (2 * x * wei)
            else:
                xo = x + self.skip_ratio * (2 * x * wei)
        else:
            xo = 2 * x * wei  # [B, 1, 128, 1024] 注意力加权增强
        
        if return_intermediates:
            return xo, {
                'input': x,
                'local_attention': xl,
                'small_scale': c1,
                'medium_scale': c2,
                'global_context': xg,
                'fused_features': xlg,
                'attention_weights': wei,
                'enhanced_spectrogram': xo,
                'fusion_weights': self._get_current_weights(xl, xg_broadcasted, c1_upsampled, c2_upsampled, x)
            }
        
        return xo
    
    def _dot_product_fusion(self, xl, xg, c1, c2, x_orig):
        """🔥 点积注意力融合方法"""
        if self.fusion_type == 'dot_product_attention':
            # 🔥 标准点积注意力融合
            # 使用局部特征作为query
            Q = self.query_proj(xl)  # [B, C, H, W]
            
            # 拼接所有特征作为key和value
            concat_features = torch.cat([xl, xg, c1, c2], dim=1)  # [B, 4C, H, W]
            K = self.key_proj(concat_features)   # [B, C, H, W]
            V = self.value_proj(concat_features) # [B, C, H, W]
            
            # 计算注意力分数
            B, C, H, W = Q.shape
            Q_flat = Q.view(B, C, -1)  # [B, C, HW]
            K_flat = K.view(B, C, -1)  # [B, C, HW] 
            V_flat = V.view(B, C, -1)  # [B, C, HW]
            
            # 点积注意力：Q·K^T
            attention_scores = torch.bmm(Q_flat.transpose(1, 2), K_flat) * self.scale  # [B, HW, HW]
            attention_weights = F.softmax(attention_scores, dim=-1)  # [B, HW, HW]
            
            # 应用注意力到值
            attended_values = torch.bmm(V_flat, attention_weights.transpose(1, 2))  # [B, C, HW]
            attended_values = attended_values.view(B, C, H, W)  # [B, C, H, W]
            
            # 输出投影和残差连接
            output = self.out_proj(attended_values)
            return output + xl  # 残差连接
            
        elif self.fusion_type == 'multi_head_dot_product':
            # 🔥 多头点积注意力
            Q = self.query_proj(xl)  # [B, C, H, W]
            
            concat_features = torch.cat([xl, xg, c1, c2], dim=1)  # [B, 4C, H, W]
            K = self.key_proj(concat_features)   # [B, C, H, W]
            V = self.value_proj(concat_features) # [B, C, H, W]
            
            B, C, H, W = Q.shape
            
            # 重塑为多头格式
            Q = Q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # [B, heads, HW, head_dim]
            K = K.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # [B, heads, HW, head_dim]
            V = V.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # [B, heads, HW, head_dim]
            
            # 多头注意力计算
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, heads, HW, HW]
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            attended_values = torch.matmul(attention_weights, V)  # [B, heads, HW, head_dim]
            
            # 合并多头
            attended_values = attended_values.transpose(2, 3).contiguous().view(B, C, H, W)
            
            # 输出投影和残差连接
            output = self.out_proj(attended_values)
            return output + xl
            
        elif self.fusion_type == 'self_attention_fusion':
            # 🔥 自注意力融合
            return self.self_attention(xl, xg, c1, c2)
            
        elif self.fusion_type == 'cross_dot_product':
            # 🔥 交叉点积注意力
            return self.cross_attention(xl, xg, c1, c2)
            
        elif self.fusion_type == 'attention_fusion':
            # 🔥 基于特征的自适应注意力融合（原版）
            concat_features = torch.cat([xl, xg, c1, c2], dim=1)  # [B, 4C, H, W]
            attention_weights = self.attention_fusion(concat_features)  # [B, 4, H, W]
            
            fused = (attention_weights[:, 0:1] * xl + 
                    attention_weights[:, 1:2] * xg + 
                    attention_weights[:, 2:3] * c1 + 
                    attention_weights[:, 3:4] * c2)
            return fused
            
        elif self.fusion_type == 'weighted_sum':
            # 可学习加权求和
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = weights[0] * xl + weights[1] * xg + weights[2] * c1 + weights[3] * c2
            return fused
            
        else:
            # 默认：简单加法融合
            return xl + xg + c1 + c2
    
    def _get_current_weights(self, xl, xg, c1, c2, x_orig):
        """获取当前融合权重（用于可视化）"""
        if self.fusion_type in ['dot_product_attention', 'multi_head_dot_product']:
            return {
                'type': self.fusion_type,
                'description': f'Dynamic {self.fusion_type} with learned attention weights',
                'note': 'Attention weights are computed dynamically via dot-product'
            }
        elif self.fusion_type in ['self_attention_fusion', 'cross_dot_product']:
            return {
                'type': self.fusion_type,
                'description': f'Advanced {self.fusion_type} mechanism'
            }
        elif self.fusion_type == 'attention_fusion':
            concat_features = torch.cat([xl, xg, c1, c2], dim=1)
            attention_weights = self.attention_fusion(concat_features)
            return {
                'local': attention_weights[:, 0].mean().item(),
                'global': attention_weights[:, 1].mean().item(),
                'small_scale': attention_weights[:, 2].mean().item(), 
                'medium_scale': attention_weights[:, 3].mean().item(),
                'attention_map': attention_weights.detach()
            }
        elif self.fusion_type == 'weighted_sum':
            weights = F.softmax(self.fusion_weights, dim=0)
            return {
                'local': weights[0].item(),
                'global': weights[1].item(), 
                'small_scale': weights[2].item(),
                'medium_scale': weights[3].item()
            }
        else:
            return {'local': 0.25, 'global': 0.25, 'small_scale': 0.25, 'medium_scale': 0.25}

    def get_fusion_weights(self):
        """🔍 获取当前融合权重（用于分析）"""
        if self.fusion_type in ['dot_product_attention', 'multi_head_dot_product', 'self_attention_fusion', 'cross_dot_product']:
            return {
                'type': self.fusion_type,
                'description': f'Dynamic attention-based fusion with {self.fusion_type} strategy',
                'note': 'Weights are computed dynamically based on input content'
            }
        elif self.fusion_type == 'attention_fusion':
            return {
                'type': 'feature_based_attention',
                'description': 'Dynamic attention-based fusion with feature-based weights'
            }
        elif self.fusion_type == 'weighted_sum':
            weights = F.softmax(self.fusion_weights, dim=0)
            return {
                'type': 'learnable_weighted',
                'weights': {
                    'local_attention': weights[0].item(),
                    'global_context': weights[1].item(),
                    'small_scale_context': weights[2].item(),
                    'medium_scale_context': weights[3].item()
                },
                'raw_params': self.fusion_weights.detach().cpu().numpy()
            }
        else:
            return {
                'type': 'simple_addition',
                'weights': {'all_branches': 0.25}
            }


# override the timm package to relax the input shape constraint.
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


class ASTModel(nn.Module):
    """
    The AST model.
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True, mix_beta=None):
        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        self.final_feat_dim = 768
        self.mix_beta = mix_beta

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            # ... [ImageNet预训练部分保持不变] ...
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
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

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
                # TODO can use sinusoidal positional embedding instead
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
            
            # 🔧 修改：灵活的预训练模型路径查找
            # 优先级：用户指定路径 > 环境变量 > 默认路径
            audioset_model_paths = [
                '/home/u202420085410012/55555/bishe/pretrained_models/audioset_10_10_0.4593.pth',  # 您的路径
                os.path.join(os.path.expanduser('~'), 'bishe', 'pretrained_models', 'audioset_10_10_0.4593.pth'),  # 用户目录
                os.path.join(os.path.dirname(__file__), '..', '..', 'pretrained_models', 'audioset_10_10_0.4593.pth'),  # 相对路径
                os.path.join('.', 'pretrained_models', 'audioset_10_10_0.4593.pth'),  # 当前目录
            ]
            
            # 查找现有的预训练模型文件
            audioset_model_path = None
            for path in audioset_model_paths:
                if os.path.exists(path):
                    audioset_model_path = path
                    if verbose:
                        print(f"✅ 找到AudioSet预训练模型: {path}")
                    break
            
            # 如果没有找到，则下载到默认位置
            if audioset_model_path is None:
                out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'pretrained_models')
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                audioset_model_path = os.path.join(out_dir, 'audioset_10_10_0.4593.pth')
                
                if verbose:
                    print("📥 下载AudioSet预训练模型...")
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, audioset_model_path)
                if verbose:
                    print("✅ AudioSet预训练模型下载完成")
            
            # 加载预训练权重
            try:
                sd = torch.load(audioset_model_path, map_location=device)
                audio_model = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
                audio_model = torch.nn.DataParallel(audio_model)
                audio_model.load_state_dict(sd, strict=False)
                self.v = audio_model.module.v
                self.original_embedding_dim = self.v.pos_embed.shape[2]
                self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))
                
                if verbose:
                    print("✅ AudioSet权重加载成功")
            except Exception as e:
                if verbose:
                    print(f"❌ AudioSet权重加载失败: {e}")
                    print("🔄 回退到ImageNet预训练...")
                # 回退到ImageNet预训练
                raise e

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
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
        square = patch.reshape(B, h, w, dim)
        return square

    def flatten_patch(self, square):
        B, h, w, dim = square.shape
        patch = square.reshape(B, h * w, dim)
        return patch

    def patch_mix(self, image, target, time_domain=False, hw_num_patch=None):
        if self.mix_beta > 0:
            lam = np.random.beta(self.mix_beta, self.mix_beta)
        else:
            lam = 1

        batch_size, num_patch, dim = image.size()
        device = image.device

        index = torch.randperm(batch_size).to(device)

        if not time_domain:
            num_mask = int(num_patch * (1. - lam))
            mask = torch.randperm(num_patch)[:num_mask].to(device)

            image[:, mask, :] = image[index][:, mask, :]
            lam = 1 - (num_mask / num_patch)
        else:
            squared_1 = self.square_patch(image, hw_num_patch)
            squared_2 = self.square_patch(image[index], hw_num_patch)

            w_size = squared_1.size()[2]
            num_mask = int(w_size * (1. - lam))
            mask = torch.randperm(w_size)[:num_mask].to(device)

            squared_1[:, :, mask, :] = squared_2[:, :, mask, :]
            image = self.flatten_patch(squared_1)
            lam = 1 - (num_mask / w_size)
        
        y_a, y_b = target, target[index]
        return image, y_a, y_b, lam, index

    @autocast()
    def forward(self, x, y=None, patch_mix=False, time_domain=False, return_features=False):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :param return_features: 如果为True，返回特征向量；如果为False，返回分类结果
        :return: prediction
        """
        # x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        h_patch, w_patch = int((x.size()[2] - 16) / 10) + 1, int((x.size()[3] - 16) / 10) + 1

        B = x.shape[0]
        x = self.v.patch_embed(x)

        if patch_mix:
            x, y_a, y_b, lam, index = self.patch_mix(x, y, time_domain=time_domain, hw_num_patch=[h_patch, w_patch])

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for i, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        
        # 🔧 根据参数决定是否通过分类头
        if return_features:
            # 返回特征向量，不通过分类头
            if not patch_mix:
                return x
            else:
                return x, y_a, y_b, lam, index
        else:
            # 原始行为：通过分类头返回分类结果
            x = self.mlp_head(x)
            if not patch_mix:
                return x
            else:
                return x, y_a, y_b, lam, index


# 🔥 多尺度交叉注意力融合模块
class MultiScaleCrossAttention(nn.Module):
    """多尺度交叉注意力融合"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 查询、键、值投影
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        
        # 注意力权重计算
        self.attention_conv = nn.Sequential(
            nn.Conv2d(channels * 4, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 4, 1),
            nn.Softmax(dim=1)
        )
        
        # 输出投影
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)
        
    def forward(self, xl, xg, c1, c2):
        # 使用局部特征作为查询
        Q = self.q_proj(xl)  # [B, C, H, W]
        
        # 拼接其他特征作为键和值的基础
        features_stack = torch.stack([xl, xg, c1, c2], dim=2)  # [B, C, 4, H, W]
        B, C, N, H, W = features_stack.shape
        
        # 计算注意力权重
        concat_features = torch.cat([xl, xg, c1, c2], dim=1)  # [B, 4C, H, W]
        attention_weights = self.attention_conv(concat_features)  # [B, 4, H, W]
        
        # 应用注意力权重
        weighted_features = (attention_weights.unsqueeze(2) * features_stack).sum(dim=2)  # [B, C, H, W]
        
        # 残差连接和归一化
        output = self.out_proj(weighted_features)
        output = self.norm(output + xl)
        
        return output


# 🔥 改进的Transformer特征融合 - 注意力版本
class AdvancedTransformerFusion(nn.Module):
    """高级Transformer特征融合 - 支持注意力融合"""
    def __init__(self, embed_dim=768, fusion_type='attention_weighted'):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention_weighted':
            # 🔥 基于注意力的权重计算
            self.attention_weights = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim // 4, 2),
                nn.Softmax(dim=-1)
            )
            print("🔥 使用注意力加权融合CLS和DIST tokens")
            
        elif fusion_type == 'multi_head_attention':
            # 🔥 多头注意力融合
            self.multi_head_attention = nn.MultiheadAttention(
                embed_dim=embed_dim, 
                num_heads=8, 
                batch_first=True,
                dropout=0.1
            )
            self.norm = nn.LayerNorm(embed_dim)
            print("🔥 使用多头注意力融合CLS和DIST tokens")
            
        elif fusion_type == 'cross_attention_fusion':
            # 🔥 交叉注意力融合
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=embed_dim, 
                num_heads=8, 
                batch_first=True
            )
            self.feed_forward = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 4, embed_dim)
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            print("🔥 使用交叉注意力融合CLS和DIST tokens")
            
        elif fusion_type == 'gated_attention':
            # 🔥 门控注意力融合
            self.gate_attention = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Tanh(),
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid()
            )
            self.content_transform = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, embed_dim)
            )
            print("🔥 使用门控注意力融合CLS和DIST tokens")
            
        elif fusion_type == 'weighted_average':
            # 可学习的CLS和DIST token权重
            self.token_weights = nn.Parameter(torch.ones(2))
            print("🔥 使用可学习加权平均融合CLS和DIST tokens")
            
        else:
            print("📍 使用简单平均融合CLS和DIST tokens")
    
    def forward(self, cls_token, dist_token):
        """🔥 注意力融合CLS和Distillation tokens"""
        if self.fusion_type == 'attention_weighted':
            # 🔥 基于注意力的加权
            concat_tokens = torch.cat([cls_token, dist_token], dim=-1)  # [B, 2*embed_dim]
            attention_weights = self.attention_weights(concat_tokens)  # [B, 2]
            
            weighted_sum = (attention_weights[:, 0:1] * cls_token + 
                           attention_weights[:, 1:2] * dist_token)
            return weighted_sum
            
        elif self.fusion_type == 'multi_head_attention':
            # 🔥 多头注意力融合
            # 将tokens组织为序列
            token_seq = torch.stack([cls_token, dist_token], dim=1)  # [B, 2, embed_dim]
            
            # 自注意力
            attended, _ = self.multi_head_attention(token_seq, token_seq, token_seq)
            
            # 残差连接和归一化
            attended = self.norm(attended + token_seq)
            
            # 平均池化得到最终特征
            return attended.mean(dim=1)
            
        elif self.fusion_type == 'cross_attention_fusion':
            # 🔥 交叉注意力融合
            # 使用CLS作为查询，DIST作为键值
            cls_expanded = cls_token.unsqueeze(1)  # [B, 1, embed_dim]
            dist_expanded = dist_token.unsqueeze(1)  # [B, 1, embed_dim]
            
            # 交叉注意力：CLS attend to DIST
            attended_cls, _ = self.cross_attention(cls_expanded, dist_expanded, dist_expanded)
            attended_cls = self.norm1(attended_cls + cls_expanded)
            
            # 前馈网络
            ff_output = self.feed_forward(attended_cls)
            final_output = self.norm2(ff_output + attended_cls)
            
            return final_output.squeeze(1)
            
        elif self.fusion_type == 'gated_attention':
            # 🔥 门控注意力融合
            concat_tokens = torch.cat([cls_token, dist_token], dim=-1)
            
            # 计算门控权重
            gate = self.gate_attention(concat_tokens)
            
            # 内容变换
            content = self.content_transform(concat_tokens)
            
            # 门控融合
            return gate * content + (1 - gate) * cls_token
            
        elif self.fusion_type == 'weighted_average':
            # 可学习加权平均
            weights = F.softmax(self.token_weights, dim=0)
            return weights[0] * cls_token + weights[1] * dist_token
            
        else:
            # 默认：简单平均
            return (cls_token + dist_token) / 2
    
    def get_fusion_weights(self, cls_token=None, dist_token=None):
        """获取当前融合权重"""
        if self.fusion_type == 'attention_weighted' and cls_token is not None:
            concat_tokens = torch.cat([cls_token, dist_token], dim=-1)
            attention_weights = self.attention_weights(concat_tokens)
            return {
                'cls_weight': attention_weights[:, 0].mean().item(),
                'dist_weight': attention_weights[:, 1].mean().item(),
                'fusion_type': 'attention_weighted'
            }
        elif self.fusion_type == 'weighted_average':
            weights = F.softmax(self.token_weights, dim=0)
            return {
                'cls_weight': weights[0].item(),
                'dist_weight': weights[1].item(),
                'fusion_type': 'weighted_average'
            }
        elif self.fusion_type in ['multi_head_attention', 'cross_attention_fusion', 'gated_attention']:
            return {
                'fusion_type': self.fusion_type,
                'description': f'Dynamic {self.fusion_type} fusion'
            }
        else:
            return {'cls_weight': 0.5, 'dist_weight': 0.5, 'fusion_type': 'simple_average'}


class AST_Early_MSA(nn.Module):
    """
    Pre-Patch MSA AST：在原始频谱图级别应用MSA，然后进行Patch Embedding
    🔥 采用注意力融合策略
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024,
                 imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True,
                 mix_beta=None, use_msa=True, use_skip_connection=False, skip_ratio=0.3, 
                 skip_type='standard', fusion_type='attention_fusion', token_fusion_type='attention_weighted'):
        super(AST_Early_MSA, self).__init__()
        
        # 基础配置
        self.returns_features = True
        self.use_msa = use_msa
        self.use_skip_connection = use_skip_connection
        self.skip_ratio = skip_ratio
        self.skip_type = skip_type
        self.fusion_type = fusion_type
        self.token_fusion_type = token_fusion_type
        self.final_feat_dim = 768
        self.fstride = fstride
        self.tstride = tstride
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.mix_beta = mix_beta
        
        # 🆕 Pre-Patch MSA模块（用于原始频谱图）
        if use_msa:
            if verbose:
                print(f"✅ 使用Pre-Patch MSA (跳跃连接: {use_skip_connection})")
                print(f"🔥 多尺度融合方式: {fusion_type}")
                if use_skip_connection:
                    print(f"   跳跃连接参数: ratio={skip_ratio}, type={skip_type}")
            
            self.pre_patch_msa = PrePatchMSA(
                input_channels=1, 
                r=4, 
                use_skip_connection=use_skip_connection,
                skip_ratio=skip_ratio,
                skip_type=skip_type,
                fusion_type=fusion_type  # 🔥 传入注意力融合类型
            )
        else:
            if verbose:
                print("ℹ️ 不使用MSA模块（纯AST baseline）")
            self.pre_patch_msa = None
        
        # 🔧 基础AST组件
        self._build_ast_components(
            model_size=model_size,
            imagenet_pretrain=imagenet_pretrain,
            audioset_pretrain=audioset_pretrain,
            verbose=verbose
        )
        
        # 🔥 Transformer特征融合模块
        self.transformer_fusion = AdvancedTransformerFusion(
            embed_dim=768,
            fusion_type=token_fusion_type
        )
        
        # 🆕 计算patch维度
        self.f_dim, self.t_dim = self._get_patch_dimensions()
        if verbose:
            print(f'🔍 Patch dimensions: freq={self.f_dim}, time={self.t_dim}')
            print(f'🔥 Token融合方式: {token_fusion_type}')
            print(f'📊 处理流程: 原始频谱图({input_fdim}×{input_tdim}) -> Pre-MSA(注意力融合) -> Patch({self.f_dim}×{self.t_dim}) -> Transformer -> 注意力Token融合')
        
        # 🔧 删除内部分类器
        self.mlp_head = nn.Identity()
        
    def _build_ast_components(self, model_size, imagenet_pretrain, audioset_pretrain, verbose):
        """构建AST的基础组件"""
        import timm
        from timm.models.layers import trunc_normal_
        
        # 使用自定义的PatchEmbed
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        
        # 创建ViT模型
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
            
            # 🔧 修改patch embedding为音频特定的
            self._setup_audio_patch_embed(imagenet_pretrain)
            
            # 🔧 修改位置编码
            self._setup_position_embedding(imagenet_pretrain)
            
        else:
            # 🆕 AudioSet预训练模型的处理
            self._setup_audioset_pretrained(model_size, verbose)
    
    def _get_patch_dimensions(self):
        """计算patch维度"""
        test_input = torch.randn(1, 1, self.input_fdim, self.input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(self.fstride, self.tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
    
    def _setup_audio_patch_embed(self, imagenet_pretrain):
        """设置音频patch embedding"""
        f_dim, t_dim = self._get_patch_dimensions()
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        
        # 修改投影层
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        if imagenet_pretrain == True:
            new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj
    
    def _setup_position_embedding(self, imagenet_pretrain):
        """设置位置编码"""
        f_dim, t_dim = self._get_patch_dimensions()
        num_patches = f_dim * t_dim
        
        if imagenet_pretrain == True:
            # 从预训练模型获取位置编码
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
            
            # 调整时间维度
            if t_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
            
            # 调整频率维度
            if f_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            
            # 展平位置编码
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
            # 与cls token和distillation token拼接
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detained(), new_pos_embed], dim=1))
        else:
            # 随机初始化位置编码
            from timm.models.layers import trunc_normal_
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)
    
    def _setup_audioset_pretrained(self, model_size, verbose):
        """🆕 设置AudioSet预训练模型（完整实现）"""
        if model_size != 'base384':
            raise ValueError('🚫 AudioSet预训练目前只支持base384模型')
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 🔧 多路径查找AudioSet预训练模型
        audioset_model_paths = [
            '/home/u202420085410012/55555/bishe/pretrained_models/audioset_10_10_0.4593.pth',  # 您的路径
            os.path.join(os.path.expanduser('~'), 'bishe', 'pretrained_models', 'audioset_10_10_0.4593.pth'),  # 用户目录
            os.path.join(os.path.dirname(__file__), '..', '..', 'pretrained_models', 'audioset_10_10_0.4593.pth'),  # 相对路径
            os.path.join('.', 'pretrained_models', 'audioset_10_10_0.4593.pth'),  # 当前目录
            # 可以根据需要添加更多路径
        ]
        
        # 查找现有的预训练模型文件
        audioset_model_path = None
        for path in audioset_model_paths:
            if os.path.exists(path):
                audioset_model_path = path
                if verbose:
                    print(f"✅ 找到AudioSet预训练模型: {path}")
                break
        
        # 如果没有找到，则下载到默认位置
        if audioset_model_path is None:
            out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'pretrained_models')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            audioset_model_path = os.path.join(out_dir, 'audioset_10_10_0.4593.pth')
            
            if verbose:
                print("📥 AudioSet模型未找到，正在下载...")
                print(f"   下载路径: {audioset_model_path}")
            try:
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, audioset_model_path)
                if verbose:
                    print("✅ AudioSet预训练模型下载完成")
            except Exception as download_error:
                if verbose:
                    print(f"❌ 下载失败: {download_error}")
                    print("🔄 回退到ImageNet预训练...")
                self._build_imagenet_fallback(model_size, verbose)
                return
        
        # 🔧 加载AudioSet预训练权重
        if verbose:
            print("🔄 加载AudioSet预训练权重...")
        
        try:
            # 创建一个临时的基础AST模型来加载预训练权重
            temp_ast = ASTModel(
                label_dim=527, 
                fstride=10, 
                tstride=10, 
                input_fdim=128, 
                input_tdim=1024, 
                imagenet_pretrain=False,  # 注意：这里设为False
                audioset_pretrain=False,  # 注意：这里设为False
                model_size='base384', 
                verbose=False
            )
            
            # 包装为DataParallel（因为原始保存时使用了DataParallel）
            temp_ast = torch.nn.DataParallel(temp_ast)
            
            # 加载权重
            sd = torch.load(audioset_model_path, map_location=device)
            temp_ast.load_state_dict(sd, strict=False)
            
            if verbose:
                print("✅ AudioSet权重加载成功")
                
        except Exception as e:
            if verbose:
                print(f"❌ AudioSet权重加载失败: {e}")
                print("🔄 回退到ImageNet预训练...")
            # 回退到ImageNet预训练
            self._build_imagenet_fallback(model_size, verbose)
            return
        
        # 🔧 提取ViT组件
        self.v = temp_ast.module.v
        self.original_embedding_dim = self.v.pos_embed.shape[2]  # 应该是768
        
        # 🔧 计算当前配置的patch维度
        f_dim, t_dim = self._get_patch_dimensions()
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        
        if verbose:
            print(f'🔍 AudioSet模型patch维度: freq={f_dim}, time={t_dim}')
            print(f'🔍 总patch数: {num_patches}')
        
        # 🔧 调整位置编码以适配当前输入尺寸
        # AudioSet原始模型的位置编码是为12x101=1212个patches设计的
        try:
            original_pos_embed = self.v.pos_embed[:, 2:, :].detach()  # 跳过CLS和DIST tokens
            original_pos_embed = original_pos_embed.reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            
            # 根据当前输入尺寸调整位置编码
            if t_dim <= 101:
                # 如果时间维度较小，从中间裁剪
                start_t = max(0, 50 - int(t_dim/2))
                end_t = min(101, start_t + t_dim)
                new_pos_embed = original_pos_embed[:, :, :, start_t:end_t]
            else:
                # 如果时间维度较大，插值放大
                new_pos_embed = torch.nn.functional.interpolate(
                    original_pos_embed, size=(12, t_dim), mode='bilinear', align_corners=False
                )
            
            if f_dim < 12:
                # 如果频率维度较小，从中间裁剪
                start_f = max(0, 6 - int(f_dim/2))
                end_f = min(12, start_f + f_dim)
                new_pos_embed = new_pos_embed[:, :, start_f:end_f, :]
            elif f_dim > 12:
                # 如果频率维度较大，插值放大
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(f_dim, t_dim), mode='bilinear', align_corners=False
                )
            
            # 🔧 重塑并拼接位置编码
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(
                torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1)
            )
            
            if verbose:
                print("✅ AudioSet位置编码调整完成")
                print(f"   位置编码形状: {self.v.pos_embed.shape}")
                
        except Exception as pos_error:
            if verbose:
                print(f"⚠️ 位置编码调整失败: {pos_error}")
                print("   使用默认位置编码...")
            # 如果位置编码调整失败，使用默认方式
            pass
        
        # 🔧 存储原始patch数量等信息
        self.original_num_patches = 1212  # AudioSet原始patch数量
        self.oringal_hw = int(np.sqrt(576))  # 原始ImageNet hw (24x24=576, 但AudioSet是12x101)
        
        if verbose:
            print("🎉 AudioSet预训练模型设置完成")
    
    def _build_imagenet_fallback(self, model_size, verbose):
        """ImageNet预训练回退方案"""
        if verbose:
            print("🔄 使用ImageNet预训练作为回退方案...")
        
        import timm
        
        # 创建ImageNet预训练模型
        self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=True)
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        
        # 设置音频相关组件
        self._setup_audio_patch_embed(imagenet_pretrain=True)
        self._setup_position_embedding(imagenet_pretrain=True)
        
        if verbose:
            print("✅ ImageNet预训练回退完成")
    
    def square_patch(self, patch, hw_num_patch):
        """将patch序列重塑为2D"""
        h, w = hw_num_patch
        B, _, dim = patch.size()
        square = patch.reshape(B, h, w, dim)
        return square

    def flatten_patch(self, square):
        """将2D patch展平为序列"""
        B, h, w, dim = square.shape
        patch = square.reshape(B, h * w, dim)
        return patch

    def patch_mix(self, image, target, time_domain=False, hw_num_patch=None):
        """Patch混合数据增强"""
        import numpy as np
        
        if self.mix_beta > 0:
            lam = np.random.beta(self.mix_beta, self.mix_beta)
        else:
            lam = 1

        batch_size, num_patch, dim = image.size()
        device = image.device

        index = torch.randperm(batch_size).to(device)

        if not time_domain:
            num_mask = int(num_patch * (1. - lam))
            mask = torch.randperm(num_patch)[:num_mask].to(device)
            image[:, mask, :] = image[index][:, mask, :]
            lam = 1 - (num_mask / num_patch)
        else:
            squared_1 = self.square_patch(image, hw_num_patch)
            squared_2 = self.square_patch(image[index], hw_num_patch)
            w_size = squared_1.size()[2]
            num_mask = int(w_size * (1. - lam))
            mask = torch.randperm(w_size)[:num_mask].to(device)
            squared_1[:, :, mask, :] = squared_2[:, :, mask, :]
            image = self.flatten_patch(squared_1)
            lam = 1 - (num_mask / w_size)
        
        y_a, y_b = target, target[index]
        return image, y_a, y_b, lam, index

    def forward(self, x, y=None, patch_mix=False, time_domain=False, return_intermediates=False):
        """
        前向传播
        Args:
            x: 输入频谱图 [B, 1, 128, 1024] 或 [B, 128, 1024]
            y: 标签（用于patch_mix）
            patch_mix: 是否使用patch混合数据增强
            time_domain: patch_mix的类型
            return_intermediates: 是否返回中间结果（用于可视化）
        Returns:
            特征向量 [B, 768] 或 (特征向量, 中间结果)
        """
        # 🔧 预处理：确保输入维度正确
        if x.dim() == 3:  # [B, 128, 1024]
            x = x.unsqueeze(1)  # [B, 1, 128, 1024]
        
        B = x.shape[0]
        intermediates = {}
        
        # 🆕 步骤1：Pre-Patch MSA增强（🔥 使用注意力融合！）
        if self.pre_patch_msa is not None:
            if return_intermediates:
                x_enhanced, msa_intermediates = self.pre_patch_msa(x, return_intermediates=True)
                intermediates['pre_msa'] = msa_intermediates
                # 🔥 保存融合权重信息
                intermediates['fusion_weights'] = self.pre_patch_msa.get_fusion_weights()
            else:
                x_enhanced = self.pre_patch_msa(x)
            
            # 保存MSA前后对比
            if return_intermediates:
                intermediates['original_spectrogram'] = x
                intermediates['enhanced_spectrogram'] = x_enhanced
        else:
            x_enhanced = x
            if return_intermediates:
                intermediates['original_spectrogram'] = x
                intermediates['enhanced_spectrogram'] = x
        
        # 🔧 步骤2：转置操作（AST要求的格式）
        x_transposed = x_enhanced.transpose(2, 3)  # [B, 1, 128, 1024] -> [B, 1, 1024, 128]
        if return_intermediates:
            intermediates['transposed'] = x_transposed
        
        # 🎯 步骤3：Patch Embedding
        x_patches = self.v.patch_embed(x_transposed)  # [B, num_patches, 768]
        if return_intermediates:
            intermediates['patch_embedded'] = x_patches
        
        # 🔧 步骤4：Patch Mix数据增强（如果需要）
        if patch_mix:
            x_patches, y_a, y_b, lam, index = self.patch_mix(
                x_patches, y, time_domain=time_domain, hw_num_patch=[self.f_dim, self.t_dim]
            )

        # 🎯 步骤5：添加tokens
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x_with_tokens = torch.cat((cls_tokens, dist_token, x_patches), dim=1)  # [B, num_patches+2, 768]
        if return_intermediates:
            intermediates['with_tokens'] = x_with_tokens
        
        # 🎯 步骤6：位置编码
        x_with_pos = x_with_tokens + self.v.pos_embed
        x_dropped = self.v.pos_drop(x_with_pos)
        if return_intermediates:
            intermediates['with_position'] = x_with_pos
        
        # 🎯 步骤7：Transformer blocks
        transformer_outputs = []
        x_current = x_dropped
        for i, blk in enumerate(self.v.blocks):
            x_current = blk(x_current)
            if return_intermediates and i % 3 == 0:  # 每3层保存一次
                transformer_outputs.append(x_current.clone())
        
        if return_intermediates:
            intermediates['transformer_outputs'] = transformer_outputs
        
        # 🎯 步骤8：🔥 注意力Token融合
        x_normed = self.v.norm(x_current)
        cls_token = x_normed[:, 0]  # CLS token
        dist_token = x_normed[:, 1]  # Distillation token
        
        # 🔥 使用高级融合方法
        final_features = self.transformer_fusion(cls_token, dist_token)
        
        if return_intermediates:
            intermediates['final_features'] = final_features
            intermediates['cls_token'] = cls_token
            intermediates['dist_token'] = dist_token
            # 🔥 保存Token融合权重
            intermediates['token_fusion_weights'] = self.transformer_fusion.get_fusion_weights(cls_token, dist_token)
        
        # 🔧 返回结果
        if return_intermediates:
            if not patch_mix:
                return final_features, intermediates  # [B, 768], dict
            else:
                return final_features, y_a, y_b, lam, index, intermediates
        else:
            if not patch_mix:
                return final_features  # [B, 768]
            else:
                return final_features, y_a, y_b, lam, index
    
    def get_all_fusion_weights(self):
        """🔍 获取所有融合权重（用于分析和可视化）"""
        weights_info = {}
        
        # Pre-patch MSA融合权重
        if self.pre_patch_msa is not None:
            weights_info['pre_patch_msa'] = self.pre_patch_msa.get_fusion_weights()
        
        # Token融合权重
        weights_info['token_fusion'] = self.transformer_fusion.get_fusion_weights()
        
        return weights_info


# 🆕 为兼容性添加其他模型类
class AST_Lightweight_MSAF(nn.Module):
    """
    轻量级AST-MSA集成，支持跳跃连接
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
                 imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True, 
                 mix_beta=None, use_msa=True, use_skip_connection=False, skip_ratio=0.3, skip_type='standard'):
        super(AST_Lightweight_MSAF, self).__init__()
        
        # 🆕 添加标识属性和参数
        self.returns_features = True  # 🔧 修改：这个模型返回特征向量，让外部分类器处理
        self.use_msa = use_msa
        self.use_skip_connection = use_skip_connection
        self.skip_ratio = skip_ratio
        self.skip_type = skip_type
        self.final_feat_dim = 768  # 兼容性
        
        # 基础AST模型
        self.ast = ASTModel(
            label_dim=label_dim,  # 这里的label_dim实际上不会被使用，因为我们使用特征提取
            fstride=fstride,
            tstride=tstride,
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain,
            audioset_pretrain=audioset_pretrain,
            model_size=model_size,
            verbose=verbose,
            mix_beta=mix_beta
        )
        
        # 🆕 根据参数选择是否使用MSA
        if use_msa:
            if use_skip_connection:
                print(f"✅ 使用带跳跃连接的MSA (skip_ratio: {skip_ratio}, skip_type: {skip_type})")
                try:
                    from .model_utils import MSA_small_with_skip
                    self.msa = MSA_small_with_skip(channels=768, r=4, skip_ratio=skip_ratio, skip_type=skip_type)
                except ImportError as e:
                    print(f"❌ 无法导入跳跃连接MSA: {e}")
                    print("   回退到原始MSA")
                    try:
                        from .model_utils import MSA_small
                        self.msa = MSA_small(channels=768, r=4)
                    except ImportError:
                        print("❌ MSA_small 也未找到，将不使用MSA模块")
                        self.msa = None
                    self.use_skip_connection = False
            else:
                print("✅ 使用原始MSA（无跳跃连接）")
                try:
                    from .model_utils import MSA_small
                    self.msa = MSA_small(channels=768, r=4)
                except ImportError:
                    print("❌ MSA_small 未找到，将不使用MSA模块")
                    self.msa = None
        else:
            print("ℹ️ 不使用MSA模块（纯AST baseline）")
            self.msa = None
        
        # 特征投影层
        self.feat_proj = nn.Linear(768, 768)
        
        # 🔧 删除内部分类器，让外部分类器处理
        # 🆕 为兼容性添加一个虚拟的mlp_head
        self.mlp_head = nn.Identity()  # 不实际使用，只是为了兼容性
        
    def forward(self, x, y=None, patch_mix=False, time_domain=False):
        # 🔧 使用return_features=True从AST获取特征
        ast_output = self.ast(x, y, patch_mix, time_domain, return_features=True)
        
        # 处理patch_mix情况
        if patch_mix:
            features, y_a, y_b, lam, index = ast_output
        else:
            features = ast_output
            
        # 🆕 只有在使用MSA时才应用MSA
        if self.msa is not None:
            # 特征投影
            projected_feat = self.feat_proj(features)  # [B, 768]
            
            # reshape为4D用于MSA
            feat_4d = projected_feat.unsqueeze(-1).unsqueeze(-1)  # [B, 768, 1, 1]
            
            # 应用多尺度注意力（自动适配是否有跳跃连接）
            enhanced_feat = self.msa(feat_4d)  # [B, 768, 1, 1]
            
            # 回到2D
            final_feat = enhanced_feat.squeeze(-1).squeeze(-1)  # [B, 768]
            
            # 残差连接
            final_feat = final_feat + features
        else:
            # 不使用MSA，直接使用AST特征
            final_feat = features
        
        # 🔧 返回特征向量而不是分类结果，让外部分类器处理
        if not patch_mix:
            return final_feat  # [B, 768] - 特征向量
        else:
            return final_feat, y_a, y_b, lam, index  # 特征向量 + patch_mix参数