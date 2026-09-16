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

# 导入MSAF相关模块
from .model_utils import MSAF_small, MSA_small

# 保持原有的PatchEmbed和ASTModel类不变
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

# 保持原有ASTModel完全不变
class ASTModel(nn.Module):
    """
    原有的AST模型保持不变
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
            
            # 修复预训练模型下载路径
            out_dir = '/home/u202420085410012/55555/bishe/pretrained_models'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            
            pretrained_model_path = os.path.join(out_dir, 'audioset_10_10_0.4593.pth')
            
            # 检查文件是否存在，如果不存在则尝试下载
            if not os.path.exists(pretrained_model_path):
                print(f"🔄 AudioSet预训练模型不存在，正在下载到: {pretrained_model_path}")
                try:
                    # this model performs 0.4593 mAP on the audioset eval set
                    audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                    wget.download(audioset_mdl_url, out=pretrained_model_path)
                    print(f"✅ 下载完成: {pretrained_model_path}")
                except Exception as e:
                    print(f"❌ 下载失败: {str(e)}")
                    print("请手动下载AudioSet预训练权重并放置到正确路径")
                    raise FileNotFoundError(f"无法下载AudioSet预训练权重: {pretrained_model_path}")
            else:
                print(f"✅ 找到AudioSet预训练权重: {pretrained_model_path}")
            
            # 加载预训练权重
            try:
                sd = torch.load(pretrained_model_path, map_location=device)
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
                    print('AudioSet预训练加载成功!')
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
                
                print(f"✅ AudioSet预训练权重加载成功，路径: {pretrained_model_path}")
                
            except Exception as e:
                print(f"❌ 加载AudioSet预训练权重失败: {str(e)}")
                raise RuntimeError(f"无法加载AudioSet预训练权重: {pretrained_model_path}")

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
    def forward(self, x, y=None, patch_mix=False, time_domain=False):
        # 单特征AST的原始forward逻辑
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

        x = self.mlp_head(x)

        if not patch_mix:
            return x
        else:
            return x, y_a, y_b, lam, index

class ASTModelBranch(nn.Module):
    def forward(self, x, y=None, patch_mix=False, time_domain=False):
        # 输入应该是 (batch_size, 1, height, width)
        B = x.shape[0]
        
        # 确保输入维度正确
        if len(x.shape) == 3:  # (B, H, W)
            x = x.unsqueeze(1)  # -> (B, 1, H, W)
        
        # 调试信息
        if self.verbose:
            print(f"Input shape: {x.shape}")
        
        # 将输入转换为patches
        x = x.transpose(2, 3)  # 现在应该是 (B, 1, W, H)
        
        # 继续原有的forward逻辑...
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
        
        if not patch_mix:
            return x
        else:
            return x, y_a, y_b, lam, index

# ===================== 新增双特征MSAF模型 =====================

class AST_Dual_Feature_MSAF(nn.Module):
    """
    双特征AST-MSAF模型
    支持两种不同音频特征的融合（如Mel频谱图 + CQT图）
    """
    def __init__(self, label_dim=4, model_size='base384', spatial_size=8, 
                 feature_combo='mel+cqt', shared_weights=False):
        super(AST_Dual_Feature_MSAF, self).__init__()
        
        self.spatial_size = spatial_size
        self.shared_weights = shared_weights
        
        # 解析特征组合
        if '+' in feature_combo:
            self.feature_types = feature_combo.split('+')
        else:
            self.feature_types = ['mel', 'cqt']  # 默认组合
            
        self.feature_combo = feature_combo
        
        if shared_weights:
            # 共享权重模式：使用同一个AST处理不同特征
            self.ast_shared = ASTModel(
                label_dim=label_dim,
                model_size=model_size,
                imagenet_pretrain=True,
                audioset_pretrain=False,
                verbose=True
            )
            self.feat_proj_shared = nn.Linear(768, 768)
        else:
            # 独立权重模式：为每种特征创建独立的AST
            self.ast_branch1 = ASTModel(
                label_dim=label_dim,
                model_size=model_size,
                imagenet_pretrain=True,
                audioset_pretrain=False,
                verbose=True
            )
            
            self.ast_branch2 = ASTModel(
                label_dim=label_dim,
                model_size=model_size,
                imagenet_pretrain=True,
                audioset_pretrain=False,
                verbose=False  # 避免重复打印
            )
            
            # 独立的特征投影层
            self.feat_proj1 = nn.Linear(768, 768)
            self.feat_proj2 = nn.Linear(768, 768)
        
        # 多尺度注意力融合模块
        self.msaf = MSAF_small(channels=768, r=4)
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            nn.Linear(768, label_dim)
        )
        
    def forward(self, feature_input, y=None, patch_mix=False, time_domain=False):
        """
        修复前向传播方法，正确处理输入
        """
        # 处理输入格式
        if isinstance(feature_input, dict):
            # 字典输入模式 - 按照feature_types顺序提取
            feature_list = [feature_input[ft] for ft in self.feature_types]
            feature1, feature2 = feature_list[0], feature_list[1]
        elif isinstance(feature_input, (list, tuple)):
            # 列表/元组输入模式
            feature1, feature2 = feature_input[0], feature_input[1]
        else:
            # 如果传入单一tensor，复制为两个相同特征（用于调试）
            feature1 = feature2 = feature_input
        
        if self.shared_weights:
            # 共享权重处理
            ast_output1 = self.ast_shared(feature1, y, patch_mix, time_domain)
            ast_output2 = self.ast_shared(feature2, y, patch_mix, time_domain)
            
            if patch_mix:
                features1, y_a, y_b, lam, index = ast_output1
                features2, _, _, _, _ = ast_output2
            else:
                features1 = ast_output1
                features2 = ast_output2
            
            projected_feat1 = self.feat_proj_shared(features1)
            projected_feat2 = self.feat_proj_shared(features2)
        else:
            # 独立权重处理
            ast_output1 = self.ast_branch1(feature1, y, patch_mix, time_domain)
            ast_output2 = self.ast_branch2(feature2, y, patch_mix, time_domain)
            
            if patch_mix:
                features1, y_a, y_b, lam, index = ast_output1
                features2, _, _, _, _ = ast_output2
            else:
                features1 = ast_output1
                features2 = ast_output2
            
            projected_feat1 = self.feat_proj1(features1)
            projected_feat2 = self.feat_proj2(features2)
        
        # 为MSAF创建4D输入
        feat1_4d = projected_feat1.unsqueeze(-1).unsqueeze(-1)  # [B, 768, 1, 1]
        feat1_expanded = feat1_4d.expand(-1, -1, self.spatial_size, self.spatial_size)
        
        feat2_4d = projected_feat2.unsqueeze(-1).unsqueeze(-1)  # [B, 768, 1, 1]
        feat2_expanded = feat2_4d.expand(-1, -1, self.spatial_size, self.spatial_size)
        
        # MSAF融合两种特征
        enhanced_feat = self.msaf(feat1_expanded, feat2_expanded)  # [B, 768, 8, 8]
        
        # 全局平均池化
        final_feat = F.adaptive_avg_pool2d(enhanced_feat, 1).squeeze(-1).squeeze(-1)
        
        # 残差连接
        final_feat = final_feat + (features1 + features2) / 2
        
        # 分类
        output = self.classifier(final_feat)
        
        if not patch_mix:
            return output
        else:
            return output, y_a, y_b, lam, index

class AST_Configurable_MSAF(nn.Module):
    """
    可配置的多特征AST-MSAF模型
    支持任意数量的特征融合（通过级联MSAF实现）
    """
    def __init__(self, label_dim=4, model_size='base384', spatial_size=8, 
                 feature_types=['mel', 'cqt'], shared_weights=False):
        super(AST_Configurable_MSAF, self).__init__()
        
        self.spatial_size = spatial_size
        self.feature_types = feature_types
        self.num_features = len(feature_types)
        self.shared_weights = shared_weights
        
        if self.num_features < 2:
            raise ValueError("MSAF需要至少2种特征进行融合")
        
        if shared_weights:
            # 共享权重模式
            self.ast_shared = ASTModel(
                label_dim=label_dim,
                model_size=model_size,
                imagenet_pretrain=True,
                audioset_pretrain=False,
                verbose=True
            )
            self.feat_proj_shared = nn.Linear(768, 768)
        else:
            # 独立权重模式
            self.ast_branches = nn.ModuleDict()
            self.feat_projs = nn.ModuleDict()
            
            for i, feat_type in enumerate(feature_types):
                self.ast_branches[feat_type] = ASTModel(
                    label_dim=label_dim,
                    model_size=model_size,
                    imagenet_pretrain=True,
                    audioset_pretrain=False,
                    verbose=(i == 0)  # 只有第一个打印信息
                )
                self.feat_projs[feat_type] = nn.Linear(768, 768)
        
        # MSAF融合模块
        self.msaf = MSAF_small(channels=768, r=4)
        
        # 如果有超过2个特征，需要额外的融合层
        if self.num_features > 2:
            self.additional_fusion = nn.ModuleList([
                MSAF_small(channels=768, r=4) for _ in range(self.num_features - 2)
            ])
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            nn.Linear(768, label_dim)
        )
        
    def forward(self, feature_dict, y=None, patch_mix=False, time_domain=False):
        """
        前向传播
        """
        if isinstance(feature_dict, dict):
            # 字典输入模式
            features_list = [feature_dict[ft] for ft in self.feature_types]
        elif isinstance(feature_dict, (list, tuple)):
            # 列表/元组输入模式
            features_list = list(feature_dict)
        else:
            raise ValueError("输入应为字典、列表或元组格式")
        
        # 提取各个特征
        extracted_features = []
        
        if self.shared_weights:
            # 共享权重处理
            for i, feature in enumerate(features_list):
                ast_output = self.ast_shared(feature, y, patch_mix, time_domain)
                if patch_mix and i == 0:
                    features, y_a, y_b, lam, index = ast_output
                elif patch_mix:
                    features, _, _, _, _ = ast_output
                else:
                    features = ast_output
                
                projected_feat = self.feat_proj_shared(features)
                extracted_features.append(projected_feat)
        else:
            # 独立权重处理
            for i, (feat_type, feature) in enumerate(zip(self.feature_types, features_list)):
                ast_output = self.ast_branches[feat_type](feature, y, patch_mix, time_domain)
                if patch_mix and i == 0:
                    features, y_a, y_b, lam, index = ast_output
                elif patch_mix:
                    features, _, _, _, _ = ast_output
                else:
                    features = ast_output
                
                projected_feat = self.feat_projs[feat_type](features)
                extracted_features.append(projected_feat)
        
        # 将特征转换为4D
        features_4d = []
        for feat in extracted_features:
            feat_4d = feat.unsqueeze(-1).unsqueeze(-1)
            feat_expanded = feat_4d.expand(-1, -1, self.spatial_size, self.spatial_size)
            features_4d.append(feat_expanded)
        
        # 逐步融合特征
        if self.num_features == 2:
            # 双特征直接融合
            fused_feat = self.msaf(features_4d[0], features_4d[1])
        else:
            # 多特征逐步融合
            fused_feat = self.msaf(features_4d[0], features_4d[1])
            
            for i in range(2, self.num_features):
                fused_feat = self.additional_fusion[i-2](fused_feat, features_4d[i])
        
        # 全局平均池化
        final_feat = F.adaptive_avg_pool2d(fused_feat, 1).squeeze(-1).squeeze(-1)
        
        # 残差连接
        avg_original_feat = sum(extracted_features) / len(extracted_features)
        final_feat = final_feat + avg_original_feat
        
        # 分类
        output = self.classifier(final_feat)
        
        if not patch_mix:
            return output
        else:
            return output, y_a, y_b, lam, index

# 单特征增强模型（使用MSA而非MSAF）
class AST_Single_Feature_MSA(nn.Module):
    """
    单特征AST-MSA模型，用于对比实验
    """
    def __init__(self, label_dim=4, model_size='base384', spatial_size=8):
        super(AST_Single_Feature_MSA, self).__init__()
        
        self.spatial_size = spatial_size
        
        # 基础AST模型
        self.ast = ASTModel(
            label_dim=label_dim,
            model_size=model_size,
            imagenet_pretrain=True,
            audioset_pretrain=False,
            verbose=True
        )
        
        # 多尺度注意力模块（单特征）
        self.msa = MSA_small(channels=768, r=4)
        
        # 特征投影层
        self.feat_proj = nn.Linear(768, 768)
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            nn.Linear(768, label_dim)
        )
        
    def forward(self, x, y=None, patch_mix=False, time_domain=False):
        # AST特征提取
        ast_output = self.ast(x, y, patch_mix, time_domain)
        
        # 处理patch_mix情况
        if patch_mix:
            features, y_a, y_b, lam, index = ast_output
        else:
            features = ast_output
            
        # 特征投影
        projected_feat = self.feat_proj(features)  # [B, 768]
        
        # 扩展为4D用于MSA
        feat_4d = projected_feat.unsqueeze(-1).unsqueeze(-1)  # [B, 768, 1, 1]
        feat_expanded = feat_4d.expand(-1, -1, self.spatial_size, self.spatial_size)  # [B, 768, 8, 8]
        
        # 应用多尺度注意力
        enhanced_feat = self.msa(feat_expanded)  # [B, 768, 8, 8]
        
        # 全局平均池化回到2D
        final_feat = F.adaptive_avg_pool2d(enhanced_feat, 1).squeeze(-1).squeeze(-1)  # [B, 768]
        
        # 残差连接
        final_feat = final_feat + features
        
        # 分类
        output = self.classifier(final_feat)
        
        if not patch_mix:
            return output
        else:
            return output, y_a, y_b, lam, index