import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
from copy import deepcopy
from timm.models.layers import to_2tuple, trunc_normal_
import math
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("✅ 使用官方 mamba_ssm 库")
except ImportError:
    print("警告: mamba_ssm 未安装，将使用自定义Mamba实现")
    MAMBA_AVAILABLE = False


class CustomMamba(nn.Module):
    """
    自定义Mamba实现，基于状态空间模型
    当mamba_ssm包不可用时的fallback实现
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        d_inner = int(self.expand * d_model)
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        
        # 卷积层
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=d_inner,
        )
        
        # SSM参数
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        
        # 状态空间参数
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        
        # 激活函数
        self.act = nn.SiLU()
        
    def forward(self, x):
        """
        x: (B, L, D) 其中 B=batch_size, L=sequence_length, D=d_model
        """
        B, L, D = x.shape
        
        # 输入投影
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # 每个都是 (B, L, d_inner)
        
        # 卷积
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]  # 截断到原始长度
        x = x.transpose(1, 2)  # (B, L, d_inner)
        
        # 激活
        x = self.act(x)
        
        # SSM
        x = self.ssm(x)
        
        # 门控
        x = x * self.act(z)
        
        # 输出投影
        x = self.out_proj(x)
        
        return x
    
    def ssm(self, x):
        """简化的状态空间模型实现"""
        B, L, D = x.shape
        
        # 计算时间步长和状态
        dt = self.dt_proj(x)  # (B, L, d_inner)
        dt = torch.softplus(dt + self.dt_proj.bias)
        
        # 状态矩阵
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # 简化的状态空间计算
        # 这里使用简化版本，实际Mamba有更复杂的选择性机制
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        
        # 输入投影
        x_dbl = self.x_proj(x)  # (B, L, 2*d_state)
        B_part, C_part = x_dbl.chunk(2, dim=-1)
        
        # 简化的扫描操作
        y = []
        h = torch.zeros(B, self.A_log.shape[1], self.A_log.shape[0], device=x.device)
        
        for i in range(L):
            h = dA[:, i] * h + (dt[:, i].unsqueeze(-1) * B_part[:, i].unsqueeze(-1)) * x[:, i].unsqueeze(1)
            y_i = torch.sum(h * C_part[:, i].unsqueeze(-1), dim=1)
            y.append(y_i)
        
        y = torch.stack(y, dim=1)
        
        # 添加跳跃连接
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y


class MambaBlock(nn.Module):
    """Mamba块，包含Layer Norm和残差连接"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, norm_eps=1e-5):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model, eps=norm_eps)
        
        # 检查CUDA是否可用来决定使用哪个实现
        use_official_mamba = MAMBA_AVAILABLE and torch.cuda.is_available()
        
        if use_official_mamba:
            try:
                self.mamba = Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                self.use_custom = False
            except Exception as e:
                print(f"⚠️ 官方Mamba初始化失败: {e}，切换到自定义实现")
                self.mamba = CustomMamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                self.use_custom = True
        else:
            self.mamba = CustomMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.use_custom = True
    
    def forward(self, x):
        """
        x: (B, L, D)
        """
        # 确保数据在正确的设备上
        if MAMBA_AVAILABLE and hasattr(self, 'use_custom') and not self.use_custom:
            # 使用官方mamba时，确保数据在CUDA上
            if not x.is_cuda and torch.cuda.is_available():
                x = x.cuda()
        
        return x + self.mamba(self.norm(x))


class PatchEmbed(nn.Module):
    """将频谱图分割成补丁并嵌入"""
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


class MambaModel(nn.Module):
    """
    基于Mamba的音频频谱图分类模型
    
    Mamba相比Transformer的优势：
    1. 线性复杂度 O(L) vs O(L²)
    2. 更好的长序列建模能力
    3. 选择性状态空间机制
    4. 更高的推理效率
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
                 embed_dim=768, depth=12, d_state=16, d_conv=4, expand=2,
                 imagenet_pretrain=False, audioset_pretrain=False, verbose=True, mix_beta=None):
        super(MambaModel, self).__init__()
        
        if verbose:
            print('---------------Mamba Model Summary---------------')
            print(f'Embed dimension: {embed_dim}')
            print(f'Depth: {depth}')
            print(f'State dimension: {d_state}')
            print(f'Expand ratio: {expand}')
            print(f'Input shape: {input_fdim} x {input_tdim}')
            print(f'CUDA available: {torch.cuda.is_available()}')
            print(f'mamba_ssm available: {MAMBA_AVAILABLE}')
        
        self.final_feat_dim = embed_dim
        self.mix_beta = mix_beta
        self.embed_dim = embed_dim
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.fstride = fstride
        self.tstride = tstride
        
        # 计算补丁数量
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, embed_dim)
        num_patches = f_dim * t_dim
        self.num_patches = num_patches
        
        if verbose:
            print(f'Frequency stride: {fstride}, Time stride: {tstride}')
            print(f'Number of patches: {num_patches}')
        
        # 补丁嵌入
        self.patch_embed = PatchEmbed(
            img_size=(input_fdim, input_tdim),
            patch_size=(16, 16),
            in_chans=1,
            embed_dim=embed_dim
        )
        
        # 覆盖投影层以匹配stride
        self.patch_embed.proj = nn.Conv2d(
            1, embed_dim, 
            kernel_size=(16, 16), 
            stride=(fstride, tstride)
        )
        self.patch_embed.num_patches = num_patches
        
        # 自适应位置嵌入 - 修复关键问题！
        # 不使用固定尺寸的位置嵌入，而是使用自适应的
        max_patches = 3000  # 设置一个足够大的最大patch数量
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches + 1, embed_dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Dropout
        self.pos_drop = nn.Dropout(p=0.1)
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(depth)
        ])
        
        # 最终层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 分类头 - 为了兼容性，同时提供head和mlp_head
        self.head = nn.Linear(embed_dim, label_dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim), 
            nn.Linear(embed_dim, label_dim)
        )
        
        # 初始化权重
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        
        if verbose:
            total_params = sum(p.numel() for p in self.parameters())
            print(f'Total parameters: {total_params:,}')
            print('------------------------------------------------')
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024, embed_dim=768):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, embed_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
    
    def interpolate_pos_encoding(self, x, h, w):
        """
        自适应插值位置编码以匹配输入尺寸
        """
        npatch = x.shape[1] - 1  # 减去CLS token
        N = self.pos_embed.shape[1] - 1  # 减去CLS token的位置编码
        
        # 如果patch数量匹配，直接返回对应长度的位置编码
        if npatch <= N:
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:npatch+1]
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:N+1]  # 使用所有可用的patch位置编码
        
        dim = x.shape[-1]
        
        # 计算原始位置编码的网格尺寸
        # 使用实际的patch数量来推断网格尺寸
        h0 = int(math.sqrt(N))
        w0 = N // h0
        
        # 确保h0 * w0 = N
        while h0 * w0 != N and h0 > 1:
            h0 -= 1
            w0 = N // h0
        
        if h0 * w0 != N:
            # 如果无法完美分解，使用近似方形
            h0 = w0 = int(math.sqrt(N))
            # 截断到正确的大小
            patch_pos_embed = patch_pos_embed[:, :h0*w0]
            N = h0 * w0
        
        # 重塑为2D网格
        try:
            patch_pos_embed = patch_pos_embed.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)
        except RuntimeError:
            # 如果重塑失败，使用简单的线性插值
            print(f"⚠️ 位置编码重塑失败，使用线性插值。N={N}, h0={h0}, w0={w0}, 实际大小={patch_pos_embed.shape}")
            # 简单的重复或截断来匹配所需长度
            if npatch > N:
                # 需要扩展
                repeat_factor = (npatch + N - 1) // N  # 向上取整
                patch_pos_embed = patch_pos_embed.repeat(1, repeat_factor)[:, :npatch]
            else:
                # 需要截断
                patch_pos_embed = patch_pos_embed[:, :npatch]
            
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        
        # 双线性插值到目标尺寸
        if h != h0 or w != w0:
            patch_pos_embed = F.interpolate(
                patch_pos_embed,
                size=(h, w),
                mode='bicubic',
                align_corners=False,
            )
        
        # 重塑回序列格式
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    def load_sl_official_weights(self):
        """
        加载官方SL权重的兼容方法
        对于Mamba模型，我们暂时跳过这个步骤，因为没有对应的预训练权重
        """
        print("⚠️ Mamba模型暂不支持SL官方权重加载，跳过此步骤")
        print("💡 Mamba模型将使用随机初始化的权重进行训练")
        pass
    
    def square_patch(self, patch, hw_num_patch):
        """兼容AST的patch reshaping方法"""
        h, w = hw_num_patch
        B, _, dim = patch.size()
        square = patch.reshape(B, h, w, dim)
        return square

    def flatten_patch(self, square):
        """兼容AST的patch flattening方法"""
        B, h, w, dim = square.shape
        patch = square.reshape(B, h * w, dim)
        return patch
    
    def patch_mix(self, image, target, time_domain=False, hw_num_patch=None):
        """PatchMix数据增强 - 兼容AST的接口"""
        if self.mix_beta is None:
            self.mix_beta = 0
            
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
        """
        前向传播 - 兼容AST的接口
        :param x: 输入频谱图 (batch_size, time_frame_num, frequency_bins)
        :param y: 标签（用于patch_mix）
        :param patch_mix: 是否应用patch mixing
        :param time_domain: patch mix的时域选项
        :return: 特征向量或分类结果
        """
        # 修复输入维度处理 - 兼容AST的输入格式
        # 输入 x: (B, T, F) -> 需要转换为 (B, 1, F, T)
        if x.dim() == 3:  # (B, T, F)
            x = x.transpose(2, 3)  # (B, F, T) 
            x = x.unsqueeze(1)  # (B, 1, F, T)
        elif x.dim() == 4 and x.shape[1] == 1:  # (B, 1, F, T)
            pass  # 已经是正确格式
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}. Expected (B, T, F) or (B, 1, F, T)")
        
        # 计算patch数量（兼容AST）
        h_patch = int((x.size()[2] - 16) / self.fstride) + 1
        w_patch = int((x.size()[3] - 16) / self.tstride) + 1
        
        B = x.shape[0]
        
        # 补丁嵌入
        x = self.patch_embed(x)  # (B, N, D)
        
        # PatchMix增强
        if patch_mix and y is not None:
            x, y_a, y_b, lam, index = self.patch_mix(x, y, time_domain=time_domain, hw_num_patch=[h_patch, w_patch])
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, D)
        
        # 自适应位置嵌入 - 修复关键问题！
        pos_embed = self.interpolate_pos_encoding(x, h_patch, w_patch)
        
        # 确保位置嵌入的长度匹配
        if pos_embed.shape[1] != x.shape[1]:
            # 如果还是不匹配，截断或填充
            if pos_embed.shape[1] > x.shape[1]:
                pos_embed = pos_embed[:, :x.shape[1], :]
            else:
                # 扩展位置嵌入
                pad_length = x.shape[1] - pos_embed.shape[1]
                pad_embed = torch.zeros(1, pad_length, self.embed_dim, device=x.device)
                pos_embed = torch.cat([pos_embed, pad_embed], dim=1)
        
        x = x + pos_embed
        x = self.pos_drop(x)
        
        # 通过Mamba blocks
        for i, blk in enumerate(self.blocks):
            try:
                x = blk(x)
            except RuntimeError as e:
                if "is_cuda" in str(e):
                    print(f"⚠️ CUDA错误在第{i}层，切换到CPU模式")
                    # 将模型移到CPU
                    self.cpu()
                    x = x.cpu()
                    x = blk(x)
                else:
                    raise e
        
        # 最终归一化
        x = self.norm(x)
        
        # 提取CLS token特征
        cls_features = x[:, 0]  # (B, D)
        
        # 兼容AST的返回格式
        if not patch_mix:
            return cls_features
        else:
            return cls_features, y_a, y_b, lam, index


# 为了兼容性，提供AST接口
def ASTModel(**kwargs):
    """兼容原有AST模型接口的Mamba模型"""
    
    # 转换参数
    mamba_kwargs = {
        'label_dim': kwargs.get('label_dim', 527),
        'fstride': kwargs.get('fstride', 10),
        'tstride': kwargs.get('tstride', 10),
        'input_fdim': kwargs.get('input_fdim', 128),
        'input_tdim': kwargs.get('input_tdim', 1024),
        'embed_dim': 768,  # 固定为768以匹配AST
        'depth': 12,       # 默认12层
        'verbose': kwargs.get('verbose', True),
        'mix_beta': kwargs.get('mix_beta', None),
        'imagenet_pretrain': kwargs.get('imagenet_pretrain', False),
        'audioset_pretrain': kwargs.get('audioset_pretrain', False),
    }
    
    # 如果原来使用AudioSet预训练，我们在这里可以加载对应的权重
    if kwargs.get('audioset_pretrain', False):
        print("注意: Mamba模型暂不支持AudioSet预训练权重，使用随机初始化")
    
    model = MambaModel(**mamba_kwargs)
    
    return model


def create_mamba_audio_model(
    label_dim=4,
    input_fdim=128,
    input_tdim=1024,
    embed_dim=768,
    depth=12,
    d_state=16,
    d_conv=4,
    expand=2,
    **kwargs
):
    """创建音频Mamba模型的便捷函数"""
    
    print("🐍 创建基于Mamba的音频分类模型")
    
    model = MambaModel(
        label_dim=label_dim,
        input_fdim=input_fdim,
        input_tdim=input_tdim,
        embed_dim=embed_dim,
        depth=depth,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        verbose=True,
        **kwargs
    )
    
    return model


# 测试代码
if __name__ == "__main__":
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 测试模型
    model = create_mamba_audio_model(
        label_dim=4,
        input_fdim=128,
        input_tdim=1024,
        embed_dim=768,
        depth=4,  # 减少层数以加快测试
    )
    
    # 测试load_sl_official_weights方法
    print("\n🧪 测试load_sl_official_weights方法:")
    model.load_sl_official_weights()
    
    # 将模型移到相应设备
    model = model.to(device)
    
    # 测试不同尺寸的输入
    test_sizes = [
        (2, 1024, 128),   # 标准尺寸
        (2, 2048, 128),   # 更长的时间序列
        (2, 512, 128),    # 更短的时间序列
    ]
    
    for size in test_sizes:
        print(f"\n🧪 测试输入尺寸: {size}")
        x = torch.randn(*size).to(device)
        
        with torch.no_grad():
            try:
                output = model(x)
                print(f"✅ 输出形状: {output.shape}")
            except Exception as e:
                print(f"❌ 错误: {e}")
    
    print("✅ Mamba模型测试完成！")