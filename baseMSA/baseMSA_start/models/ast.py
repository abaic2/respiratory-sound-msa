import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
import timm
from copy import deepcopy
from timm.models.layers import to_2tuple,trunc_normal_

"""
原始输入: (4, 1, 128, 1024)
    (batch_size, channels, freq_bins, time_frames)
    单通道音频频谱图，128个频率bin，1024个时间帧
     ↓
转置操作: (4, 1, 128, 1024) -> transpose(2,3) -> (4, 1, 1024, 128)
    (batch_size, channels, time_frames, freq_bins)
    转换为AST期望的格式：时间在前，频率在后

Patch投影: (4, 1, 1024, 128) -> Conv2d(16x16, stride=10x10) -> (4, 1224, 768)
    (batch_size, num_patches, embed_dim)
    
    计算过程:
    - 时间维度patches: (1024-16)/10 + 1 = 102个
    - 频率维度patches: (128-16)/10 + 1 = 12个  
    - 总patches数: 102 × 12 = 1224个
    - 每个patch映射为768维特征向量

Patch序列转2D: (4, 1224, 768) -> square_patch -> (4, 12, 102, 768)
    (batch_size, freq_patches, time_patches, embed_dim)
    将1D patch序列重塑为2D空间结构
     ↓
维度重排: (4, 12, 102, 768) -> permute(0,3,1,2) -> (4, 768, 12, 102)
    (batch_size, channels, height, width)
    转换为MSA期望的通道优先格式
     ↓
🔍 局部注意力: (4, 768, 12, 102) -> local_att -> (4, 768, 12, 102)
    (batch_size, channels, height, width)
    1×1卷积捕获patch间的局部相关性，医学意义：检测相邻时频区域的细微异常
     ↓
🔍 小尺度上下文: (4, 768, 12, 102) -> AdaptiveAvgPool2d(4×4) -> (4, 768, 4, 4) -> 插值 -> (4, 768, 12, 102)
    (batch_size, channels, height, width)
    4×4池化后插值恢复，捕获小区域时频模式，医学意义：识别局部呼吸异常
     ↓
🔍 中尺度上下文: (4, 768, 12, 102) -> AdaptiveAvgPool2d(8×8) -> (4, 768, 8, 8) -> 插值 -> (4, 768, 12, 102)
    (batch_size, channels, height, width)
    8×8池化后插值恢复，捕获中等区域模式，医学意义：理解呼吸周期特征
     ↓
🔍 全局注意力: (4, 768, 12, 102) -> AdaptiveAvgPool2d(1×1) -> (4, 768, 1, 1) -> 广播 -> (4, 768, 12, 102)
    (batch_size, channels, height, width)
    全局池化后广播，捕获整体信息，医学意义：理解整体肺功能状态
     ↓
多尺度融合: xl + c1 + c2 + xg -> (4, 768, 12, 102)
    (batch_size, channels, height, width)
    四种尺度特征简单相加融合
     ↓
注意力权重: (4, 768, 12, 102) -> Sigmoid -> (4, 768, 12, 102)
    (batch_size, channels, height, width)
    生成[0,1]范围的像素级注意力权重
     ↓
特征增强: 2 × input × attention_weights -> (4, 768, 12, 102)
    (batch_size, channels, height, width)
    原始patch特征的注意力加权增强
     ↓
维度恢复: (4, 768, 12, 102) -> permute(0,2,3,1) -> (4, 12, 102, 768)
    (batch_size, freq_patches, time_patches, embed_dim)
    转回空间优先格式
     ↓
序列展平: (4, 12, 102, 768) -> flatten_patch -> (4, 1224, 768)
    (batch_size, num_patches, embed_dim)
    重新展平为patch序列，保留MSA增强效果


Token添加: (4, 1224, 768) -> 添加CLS+DIST -> (4, 1226, 768)
    (batch_size, seq_len, embed_dim)
    在序列开头添加分类token和蒸馏token
     ↓
位置编码: (4, 1226, 768) + pos_embed -> (4, 1226, 768)
    (batch_size, seq_len, embed_dim)
    加入学习到的位置编码，为每个位置提供位置信息
     ↓
Dropout: (4, 1226, 768) -> 随机失活 -> (4, 1226, 768)
    (batch_size, seq_len, embed_dim)
    位置编码后的dropout正则化

Transformer层1: (4, 1226, 768) -> 自注意力+FFN -> (4, 1226, 768)
    (batch_size, seq_len, embed_dim)
    第1层：学习patch间的基础交互关系
     ↓
Transformer层2: (4, 1226, 768) -> 自注意力+FFN -> (4, 1226, 768)
    (batch_size, seq_len, embed_dim)
    第2层：构建更复杂的特征关系
     ↓
Transformer层3-11: (4, 1226, 768) [深层特征学习]
    (batch_size, seq_len, embed_dim)
    中间层：逐步抽象和精化特征表示
     ↓
Transformer层12: (4, 1226, 768) -> 自注意力+FFN -> (4, 1226, 768)
    (batch_size, seq_len, embed_dim)
    最后一层：形成最终的高级语义表示

LayerNorm: (4, 1226, 768) -> 最终归一化 -> (4, 1226, 768)
    (batch_size, seq_len, embed_dim)
    最终层归一化，稳定特征分布
     ↓
Token提取: (4, 1226, 768) -> 提取前2个token -> (4, 2, 768)
    (batch_size, num_tokens, embed_dim)
    提取CLS token和蒸馏token
     ↓
Token融合: (token[0] + token[1]) / 2 -> (4, 768)
    (batch_size, feature_dim)
    融合两个token得到最终的音频全局特征向量
"""
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
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
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
            
            out_dir = '/home/yujieyang/bishe/pretrained_models'
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


import torch
import torch.nn as nn
from .ast import ASTModel
from .model_utils import MSA_small

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


import torch
import torch.nn as nn
from .ast import ASTModel
from .model_utils import MSA_small

class AST_Early_MSA(nn.Module):
    """
    早期融合AST-MSA：在Patch Embedding后立即应用MSA
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024,
                 imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True,
                 mix_beta=None, use_msa=True, use_skip_connection=False, skip_ratio=0.3, skip_type='standard'):
        super(AST_Early_MSA, self).__init__()
        
        # 基础配置
        self.returns_features = True
        self.use_msa = use_msa
        self.use_skip_connection = use_skip_connection
        self.skip_ratio = skip_ratio
        self.skip_type = skip_type
        self.final_feat_dim = 768
        self.fstride = fstride
        self.tstride = tstride
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.mix_beta = mix_beta
        
        # 🔧 基础AST组件（不使用完整的ASTModel）
        self._build_ast_components(
            model_size=model_size,
            imagenet_pretrain=imagenet_pretrain,
            audioset_pretrain=audioset_pretrain,
            verbose=verbose
        )
        
        # 🆕 计算patch维度用于MSA
        self.f_dim, self.t_dim = self._get_patch_dimensions()
        if verbose:
            print(f'🔍 Patch dimensions: freq={self.f_dim}, time={self.t_dim}')
        
        # 🆕 MSA模块（用于patch特征）
        if use_msa:
            if use_skip_connection:
                print(f"✅ 使用早期跳跃连接MSA (skip_ratio: {skip_ratio}, skip_type: {skip_type})")
                try:
                    from .model_utils import MSA_small_with_skip
                    self.patch_msa = MSA_small_with_skip(channels=768, r=4, skip_ratio=skip_ratio, skip_type=skip_type)
                except ImportError:
                    print("❌ 跳跃连接MSA未找到，使用原始MSA")
                    from .model_utils import MSA_small
                    self.patch_msa = MSA_small(channels=768, r=4)
                    self.use_skip_connection = False
            else:
                print("✅ 使用早期原始MSA")
                from .model_utils import MSA_small
                self.patch_msa = MSA_small(channels=768, r=4)
        else:
            print("ℹ️ 不使用MSA模块（纯AST baseline）")
            self.patch_msa = None
        
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
            # AudioSet预训练模型的处理
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
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(self.fstride, self.tstride))
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
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
        else:
            # 随机初始化位置编码
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)
    
    def _setup_audioset_pretrained(self, model_size, verbose):
        """设置AudioSet预训练模型"""
        # 这里可以根据需要实现AudioSet预训练模型的加载
        # 为简化，暂时抛出异常
        raise NotImplementedError("AudioSet预训练模型的早期MSA融合暂未实现")
    
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

    def forward(self, x, y=None, patch_mix=False, time_domain=False):
        """前向传播"""
        # 🔧 预处理：转置和获取patch维度
        # x = x.unsqueeze(1)  # 如果需要的话
        x = x.transpose(2, 3)  # [B, 1, time, freq] -> [B, 1, freq, time]
        
        B = x.shape[0]
        
        # 🎯 第1步：Patch Embedding
        x = self.v.patch_embed(x)  # [B, num_patches, 768]
        
        # 🆕 第2步：早期MSA增强（关键创新！）
        if self.patch_msa is not None:
            # 将patch序列重塑为2D用于MSA
            x_2d = self.square_patch(x, [self.f_dim, self.t_dim])  # [B, f_dim, t_dim, 768]
            x_2d = x_2d.permute(0, 3, 1, 2)  # [B, 768, f_dim, t_dim]
            
            # 🚀 应用MSA进行多尺度特征增强
            x_enhanced = self.patch_msa(x_2d)  # [B, 768, f_dim, t_dim]
            
            # 转回patch序列格式
            x_enhanced = x_enhanced.permute(0, 2, 3, 1)  # [B, f_dim, t_dim, 768]
            x = self.flatten_patch(x_enhanced)  # [B, num_patches, 768]
        
        # 🔧 第3步：Patch Mix数据增强（如果需要）
        if patch_mix:
            x, y_a, y_b, lam, index = self.patch_mix(x, y, time_domain=time_domain, hw_num_patch=[self.f_dim, self.t_dim])

        # 🎯 第4步：添加tokens
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)  # [B, num_patches+2, 768]
        
        # 🎯 第5步：位置编码
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        # 🎯 第6步：Transformer blocks
        for i, blk in enumerate(self.v.blocks):
            x = blk(x)
        
        # 🎯 第7步：最终处理
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2  # 融合CLS和DIST tokens
        
        # 🔧 返回特征向量
        if not patch_mix:
            return x  # [B, 768]
        else:
            return x, y_a, y_b, lam, index