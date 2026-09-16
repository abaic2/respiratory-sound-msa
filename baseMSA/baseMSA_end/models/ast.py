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
输入频谱图: (4, 1, 128, 1024) 
    (batch_size, channels, freq_bins, time_frames) 
    单通道音频频谱图
     ↓
转置操作: (4, 1, 128, 1024) -> transpose(2,3) -> (4, 1, 1024, 128)
    (batch_size, channels, time_frames, freq_bins) 
    时频维度交换，符合AST输入格式
     ↓
Patch嵌入: (4, 1, 1024, 128) -> Conv2d(16x16, stride=10x10) -> (4, 102*12, 768)
    (batch_size, num_patches, embed_dim) 
    音频patch嵌入，102个时间patch × 12个频率patch = 1224个patches
     ↓
Token添加: (4, 1224, 768) -> 添加CLS+DIST -> (4, 1226, 768)
    (batch_size, num_patches+2, embed_dim) 
    添加分类token和蒸馏token
     ↓
位置编码: (4, 1226, 768) + pos_embed -> (4, 1226, 768)
    (batch_size, seq_len, embed_dim) 
    添加音频特定的位置编码
     ↓
Transformer块1: (4, 1226, 768) -> 自注意力+FFN -> (4, 1226, 768)
    (batch_size, seq_len, embed_dim) 
    第1层音频Transformer编码
     ↓
Transformer块2-12: (4, 1226, 768) [深层音频特征学习]
    (batch_size, seq_len, embed_dim) 
    11层深层音频编码器处理
     ↓
LayerNorm: (4, 1226, 768) -> 最终归一化 -> (4, 1226, 768)
    (batch_size, seq_len, embed_dim) 
    最终层归一化
     ↓
Token融合: (4, 1226, 768) -> (token[0] + token[1])/2 -> (4, 768)
    (batch_size, embed_dim) 
    融合CLS token和蒸馏token得到音频全局特征
"""

"""
AST特征: (4, 768) [高级音频语义特征]
    (batch_size, feature_dim) 
    AST提取的768维音频特征向量
     ↓
特征投影: (4, 768) -> Linear(768->768) -> (4, 768)
    (batch_size, feature_dim) 
    可学习的特征投影变换
     ↓
维度扩展: (4, 768) -> unsqueeze(-1,-1) -> (4, 768, 1, 1)
    (batch_size, channels, height, width) 
    为MSA处理添加空间维度
     ↓
局部注意力: (4, 768, 1, 1) -> Conv2d(1x1) -> (4, 768, 1, 1)
    (batch_size, channels, height, width) 
    局部特征注意力，捕获通道间局部相关性
     ↓
小尺度上下文: (4, 768, 1, 1) -> AdaptiveAvgPool2d(4x4) -> (4, 768, 1, 1)
    (batch_size, channels, height, width) 
    4×4上下文特征（实际为1×1，语义上代表小尺度）
     ↓
中尺度上下文: (4, 768, 1, 1) -> AdaptiveAvgPool2d(8x8) -> (4, 768, 1, 1)
    (batch_size, channels, height, width) 
    8×8上下文特征（实际为1×1，语义上代表中尺度）
     ↓
全局注意力: (4, 768, 1, 1) -> AdaptiveAvgPool2d(1x1) -> (4, 768, 1, 1)
    (batch_size, channels, height, width) 
    全局特征注意力，捕获整体语义信息
     ↓
多尺度融合: xl + xg + c1 + c2 -> (4, 768, 1, 1)
    (batch_size, channels, height, width) 
    多尺度特征简单相加融合
     ↓
注意力权重: (4, 768, 1, 1) -> Sigmoid -> (4, 768, 1, 1)
    (batch_size, channels, height, width) 
    生成[0,1]范围的注意力权重
     ↓
特征增强: 2 * input * attention_weights -> (4, 768, 1, 1)
    (batch_size, channels, height, width) 
    原特征的注意力加权增强
     ↓
维度恢复: (4, 768, 1, 1) -> squeeze(-1,-1) -> (4, 768)
    (batch_size, feature_dim) 
    移除添加的空间维度
     ↓
残差连接: enhanced_features + original_features -> (4, 768)
    (batch_size, feature_dim) 
    MSA增强特征与原始AST特征残差融合
"""

"""
融合特征: (4, 768) [AST+MSA增强特征]
    (batch_size, feature_dim) 
    结合AST语义理解和MSA多尺度注意力的特征
     ↓
LayerNorm: (4, 768) -> 归一化 -> (4, 768)
    (batch_size, feature_dim) 
    特征归一化，稳定训练
     ↓
Dropout: (4, 768) -> 随机失活(0.1) -> (4, 768)
    (batch_size, feature_dim) 
    防止过拟合的正则化
     ↓
分类预测: (4, 768) -> Linear(768->4) -> (4, 4)
    (batch_size, num_classes) 
    ICBHI肺音4类预测：normal, crackle, wheeze, both
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