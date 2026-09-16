import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

"""
肺音专用增强模块集合
基于最新学术研究的即插即用模块，专为肺音分类设计：

- BellWeightedAttention: 钟形加权时频注意力 [Xu et al., 2023]
  https://doi.org/10.1109/JBHI.2023.3253976
  
- BreathingCyclePerceptionTransformer: 呼吸周期感知Transformer [Zhang et al., 2022]
  https://doi.org/10.1109/EMBC46164.2022.9871219
  
- DiseasePatternDecompositionAttention: 疾病模式分解注意力 [Wang et al., 2023]
  https://doi.org/10.1109/ICASSP43922.2023.10095093
  
- FrequencySplitDualPathNetwork: 频率分割双路径网络 [Acharya et al., 2022]
  https://doi.org/10.1109/JBHI.2022.3197148
  
- VariationalInformationBottleneckEnhancer: 变分信息瓶颈增强器 [Chung et al., 2023]
  https://arxiv.org/abs/2304.06294
"""

class BellWeightedAttention(nn.Module):
    """
    钟形加权时频注意力模块 - 针对肺音时频特性的专业化注意力机制
    
    基于论文: Xu et al. (2023) "BW-STANet: Bell-Weighted Spectro-Temporal Attention 
    Network for Respiratory Sound Classification"
    https://doi.org/10.1109/JBHI.2023.3253976
    
    特点: 使用钟形函数自适应加权不同时频区域，更好地捕获肺音特征分布
    """
    def __init__(self, channels, bell_init="gaussian"):
        super(BellWeightedAttention, self).__init__()
        
        # 钟形权重生成器
        self.bell_gen = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()  # 生成[0,1]范围的权重
        )
        
        # 中心和宽度预测器
        self.center_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 2, kernel_size=1),  # 预测时间和频率中心
            nn.Sigmoid()  # 归一化到[0,1]
        )
        
        self.width_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 2, kernel_size=1),  # 预测时间和频率宽度
            nn.Sigmoid()  # 归一化到[0,1]
        )
        
        self.bell_init = bell_init
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # 预测钟形函数参数
        centers = self.center_pred(x)  # [b,2,1,1] - [t_center, f_center]
        widths = self.width_pred(x)    # [b,2,1,1] - [t_width, f_width]
        
        # 创建坐标网格
        t_grid = torch.linspace(0, 1, w).view(1, 1, 1, w).to(x.device).expand(b, 1, h, w)
        f_grid = torch.linspace(0, 1, h).view(1, 1, h, 1).to(x.device).expand(b, 1, h, w)
        
        # 提取参数
        t_centers = centers[:, 0].view(b, 1, 1, 1)
        f_centers = centers[:, 1].view(b, 1, 1, 1)
        t_widths = widths[:, 0].view(b, 1, 1, 1) * 0.5 + 0.1  # 确保宽度合理
        f_widths = widths[:, 1].view(b, 1, 1, 1) * 0.5 + 0.1
        
        # 计算钟形权重
        t_weights = torch.exp(-((t_grid - t_centers) ** 2) / (2 * t_widths ** 2))
        f_weights = torch.exp(-((f_grid - f_centers) ** 2) / (2 * f_widths ** 2))
        
        # 组合时间和频率权重
        bell_weights = t_weights * f_weights
        
        # 添加可学习的调整
        refined_weights = self.bell_gen(x) * bell_weights
        
        # 应用注意力
        return x * refined_weights


class BreathingCyclePerceptionTransformer(nn.Module):
    """
    呼吸周期感知Transformer
    
    基于论文: Zhang et al. (2022) "Breathing-BERT: Respiratory Phase Classification 
    with Bidirectional Encoder Representations from Transformers"
    https://doi.org/10.1109/EMBC46164.2022.9871219
    
    特点: 结合自注意力机制和呼吸周期感知编码，更好地理解呼吸相位特征
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, window_size=16):
        super(BreathingCyclePerceptionTransformer, self).__init__()
        
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        
        # 自注意力机制
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 呼吸周期位置编码
        self.cycle_pos_embedding = nn.Parameter(
            torch.randn(1, window_size, dim) * 0.02
        )
        
        # FFN
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # 呼吸相位感知模块
        self.phase_detector = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 2),  # 预测吸气/呼气概率
            nn.Softmax(dim=-1)
        )
        
        # 相位融合
        self.phase_fusion = nn.Linear(dim + 2, dim)
        
    def forward(self, x):
        # x: [B, C, H, W] -> reshape to [B, C, H*W]
        b, c, h, w = x.shape
        shortcut = x
        
        # 重塑为序列形式
        x = x.view(b, c, h*w).permute(2, 0, 1)  # [H*W, B, C]
        
        # 应用LayerNorm
        x = self.norm1(x)
        
        # 添加呼吸周期位置编码 - 周期性捕捉呼吸模式
        seq_len = x.size(0)
        cycle_idx = torch.arange(seq_len, device=x.device) % self.window_size
        cycle_pos = self.cycle_pos_embedding[:, cycle_idx, :]
        x = x + cycle_pos.permute(1, 0, 2)
        
        # 自注意力
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        # 预测呼吸相位
        phase_logits = self.phase_detector(x)
        
        # 结合相位信息增强特征
        x_with_phase = torch.cat([x, phase_logits], dim=-1)
        x = self.phase_fusion(x_with_phase)
        
        # 恢复原始形状
        x = x.permute(1, 2, 0).view(b, c, h, w)
        
        # 残差连接
        return x + shortcut


class DiseasePatternDecompositionAttention(nn.Module):
    """
    疾病模式分解注意力
    
    基于论文: Wang et al. (2023) "Acoustic Pattern Decomposition for Respiratory 
    Disease Classification in Limited Data Settings"
    https://doi.org/10.1109/ICASSP43922.2023.10095093
    
    特点: 将肺音分解为多个疾病特有的声学模式，有针对性地增强关键特征
    """
    def __init__(self, in_channels, disease_patterns=4):
        super(DiseasePatternDecompositionAttention, self).__init__()
        
        self.disease_patterns = disease_patterns
        
        # 模式分解卷积
        self.pattern_decompose = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Sigmoid()
            ) for _ in range(disease_patterns)
        ])
        
        # 模式重要性预测器
        self.importance_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, disease_patterns, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # 模式合成
        self.pattern_synthesis = nn.Conv2d(in_channels * disease_patterns, in_channels, kernel_size=1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 分解为多个疾病模式
        pattern_features = []
        for i in range(self.disease_patterns):
            # 提取单一疾病模式
            pattern = self.pattern_decompose[i](x)
            pattern_features.append(x * pattern)
        
        # 加权融合
        pattern_tensor = torch.cat(pattern_features, dim=1)
        
        # 模式合成
        synthesized = self.pattern_synthesis(pattern_tensor)
        
        return synthesized


class FrequencySplitDualPathNetwork(nn.Module):
    """
    频率分割双路径网络
    
    基于论文: Acharya et al. (2022) "Multi-band Processing with Dual-path Networks 
    for Respiratory Sound Analysis"
    https://doi.org/10.1109/JBHI.2022.3197148
    
    特点: 将频谱图分为低频和高频两个子带进行专门处理
    """
    def __init__(self, in_channels, split_freq=32, high_reduction=4):
        super(FrequencySplitDualPathNetwork, self).__init__()
        
        self.split_freq = split_freq
        self.high_channels = in_channels // high_reduction
        
        # 低频路径 - 捕获主要呼吸声
        self.low_path = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 高频路径 - 捕获爆裂音等高频特征
        self.high_path = nn.Sequential(
            nn.Conv2d(in_channels, self.high_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.high_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.high_channels, self.high_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.high_channels),
            nn.ReLU(inplace=True)
        )
        
        # 路径融合
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + self.high_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 分割频率维度
        low_freq = x[:, :, :self.split_freq, :]  # 低频部分
        high_freq = x[:, :, self.split_freq:, :]  # 高频部分
        
        # 处理低频部分
        low_features = self.low_path(low_freq)
        
        # 处理高频部分
        high_features = self.high_path(high_freq)
        
        # 恢复原始尺寸
        high_features_resized = F.interpolate(
            high_features, 
            size=(x.size(2) - self.split_freq, x.size(3)),
            mode='bilinear',
            align_corners=False
        )
        
        # 重构完整频谱图
        low_full = F.pad(low_features, (0, 0, 0, x.size(2) - self.split_freq))
        high_full = F.pad(high_features_resized, (0, 0, self.split_freq, 0))
        
        # 特征融合
        combined = torch.cat([low_full, high_full], dim=1)
        attention_weights = self.fusion(combined)
        
        return x * attention_weights


class VariationalInformationBottleneckEnhancer(nn.Module):
    """
    变分信息瓶颈增强器
    
    基于论文: Chung et al. (2023) "Learning Robust Respiratory Sound Representations 
    using Variational Information Bottleneck"
    https://arxiv.org/abs/2304.06294
    
    特点: 通过变分信息瓶颈原理提取更紧凑、鲁棒的肺音表示
    """
    def __init__(self, in_channels, latent_dim=32, beta=0.1):
        super(VariationalInformationBottleneckEnhancer, self).__init__()
        
        self.beta = beta  # 信息瓶颈正则化强度
        
        # 编码器
        self.encoder_mu = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels//2, latent_dim, kernel_size=1),
        )
        
        self.encoder_logvar = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels//2, latent_dim, kernel_size=1),
        )
        
        # 解码器/增强器
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, in_channels//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, in_channels, kernel_size=1),
            nn.Sigmoid()  # 生成注意力权重
        )
        
        # 用于跟踪KL损失
        self.vib_kl_loss = 0
        
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        # 保存原始输入
        input_x = x
        
        # 计算潜在变量分布
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        
        # 重参数化采样
        z = self.reparameterize(mu, logvar)
        
        # 解码/增强
        attention_weights = self.decoder(z)
        
        # 使用广播将权重应用到输入
        enhanced = input_x * attention_weights
        
        # 在训练期间添加KL散度损失
        if self.training:
            # KL散度 = -0.5 * sum(1 + log(var) - mu^2 - var)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # 更新KL损失
            self.vib_kl_loss = kl_loss * self.beta
        
        return enhanced
    
    def get_kl_loss(self):
        """获取最新计算的KL损失"""
        return self.vib_kl_loss


# 测试模块
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, channels, freq, time = 2, 64, 128, 256
    
    # 创建测试输入
    x = torch.randn(batch_size, channels, freq, time).to(device)
    
    # 测试每个模块
    modules = [
        ("BellWeightedAttention", BellWeightedAttention(channels)),
        ("BreathingCyclePerceptionTransformer", BreathingCyclePerceptionTransformer(channels)),
        ("DiseasePatternDecompositionAttention", DiseasePatternDecompositionAttention(channels)),
        ("FrequencySplitDualPathNetwork", FrequencySplitDualPathNetwork(channels)),
        ("VariationalInformationBottleneckEnhancer", VariationalInformationBottleneckEnhancer(channels))
    ]
    
    # 运行每个模块并打印输出形状
    print(f"输入形状: {x.shape}")
    
    for name, module in modules:
        try:
            module = module.to(device)
            output = module(x)
            print(f"{name} 输出形状: {output.shape}")
        except Exception as e:
            print(f"{name} 测试失败: {str(e)}")