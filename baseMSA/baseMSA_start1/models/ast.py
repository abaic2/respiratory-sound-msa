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

# 🆕 Pre-Patch MSA模块 - 基于model_utils.py中的MSA结构设计
class PrePatchMSA(nn.Module):
    """
    Pre-Patch多尺度注意力模块
    在Patch Embedding之前直接对原始频谱图(128×1024)进行多尺度注意力增强
    基于model_utils.py中的MSA_small架构设计
    """
    def __init__(self, input_channels=1, r=4, use_skip_connection=False, skip_ratio=0.3, skip_type='standard'):
        super(PrePatchMSA, self).__init__()
        self.input_channels = input_channels
        self.use_skip_connection = use_skip_connection
        self.skip_ratio = skip_ratio
        self.skip_type = skip_type
        
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
        前向传播
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
        
        # 🔀 步骤5：多尺度特征融合（参考model_utils.py的融合方式）
        xlg = xl + xg_broadcasted + c1_upsampled + c2_upsampled  # [B, 1, 128, 1024]
        
        # 🎯 步骤6：生成注意力权重
        wei = self.sigmoid(xlg)  # [B, 1, 128, 1024] 权重范围[0,1]
        
        # 🚀 步骤7：应用注意力增强
        if self.use_skip_connection:
            if self.skip_type == 'standard':
                # 标准跳跃连接：原始 + 注意力增强
                xo = x + self.skip_ratio * (2 * x * wei)
            elif self.skip_type == 'learnable':
                # 可学习跳跃连接
                xo = x + self.skip_weight * (2 * x * wei)
            elif self.skip_type == 'adaptive':
                # 自适应跳跃连接
                adaptive_weight = self.skip_adapter(x)
                xo = x + adaptive_weight * (2 * x * wei)
            else:
                # 默认标准跳跃连接
                xo = x + self.skip_ratio * (2 * x * wei)
        else:
            # 原始版本：直接应用注意力增强（参考MSA的增强方式）
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
                'enhanced_spectrogram': xo
            }
        
        return xo


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


class AST_Early_MSA(nn.Module):
    """
    Pre-Patch MSA AST：在原始频谱图级别应用MSA，然后进行Patch Embedding
    支持AudioSet预训练模型
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
        
        # 🆕 Pre-Patch MSA模块（用于原始频谱图）
        if use_msa:
            if verbose:
                print(f"✅ 使用Pre-Patch MSA (跳跃连接: {use_skip_connection})")
                if use_skip_connection:
                    print(f"   跳跃连接参数: ratio={skip_ratio}, type={skip_type}")
            
            self.pre_patch_msa = PrePatchMSA(
                input_channels=1, 
                r=4, 
                use_skip_connection=use_skip_connection,
                skip_ratio=skip_ratio,
                skip_type=skip_type
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
        
        # 🆕 计算patch维度
        self.f_dim, self.t_dim = self._get_patch_dimensions()
        if verbose:
            print(f'🔍 Patch dimensions: freq={self.f_dim}, time={self.t_dim}')
            print(f'📊 处理流程: 原始频谱图({input_fdim}×{input_tdim}) -> Pre-MSA -> Patch({self.f_dim}×{self.t_dim}) -> Transformer')
        
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
        
        # 🆕 步骤1：Pre-Patch MSA增强（关键创新！）
        if self.pre_patch_msa is not None:
            if return_intermediates:
                x_enhanced, msa_intermediates = self.pre_patch_msa(x, return_intermediates=True)
                intermediates['pre_msa'] = msa_intermediates
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
        
        # 🎯 步骤8：最终处理
        x_normed = self.v.norm(x_current)
        final_features = (x_normed[:, 0] + x_normed[:, 1]) / 2  # 融合CLS和DIST tokens
        
        if return_intermediates:
            intermediates['final_features'] = final_features
        
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