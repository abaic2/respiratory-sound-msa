import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# 导入模型和工具函数
from ast1 import AST_Early_MSA
from icbhi_util import get_annotations, get_individual_cycles_torchaudio, generate_fbank

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid'] = False

# Pre-Patch MSA模块 - 在原始频谱图上直接应用MSA
class PrePatchMSA(nn.Module):
    """在Patch Embedding之前应用的多尺度注意力模块 - 修复版"""
    def __init__(self, input_channels=1, r=4):
        super(PrePatchMSA, self).__init__()
        self.input_channels = input_channels
        
        # 🔍 局部注意力 - 在原始频谱图上检测局部时频模式
        self.local_att = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        
        # 🔍 多尺度池化 - 直接在时频图上捕获不同尺度模式
        self.pool1 = nn.AdaptiveAvgPool2d((32, 256))   # 小尺度
        self.pool2 = nn.AdaptiveAvgPool2d((64, 512))   # 中尺度
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局
        
        # 🆕 卷积处理层（对应图中的Conv）
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.global_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        
        # 注意力生成网络
        self.att_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, return_intermediates=False):
        """
        x: [B, C, F, T] = [B, 1, 128, 1024] (原始频谱图)
        """
        B, C, F, T = x.shape
        
        # 🔍 局部注意力：3x3卷积捕获邻域时频关系
        xl = self.local_att(x)  # [B, 1, 128, 1024]
        
        # 🔍 小尺度上下文处理：Pool → Conv → UnPool
        c1_pooled = self.pool1(x)  # [B, 1, 32, 256]
        c1_conv = self.conv1(c1_pooled)  # [B, 1, 32, 256]
        c1_upsampled = torch.nn.functional.interpolate(
            c1_conv, size=(F, T), mode='bilinear', align_corners=False
        )  # [B, 1, 128, 1024]
        
        # 🔍 中尺度上下文处理：Pool → Conv → UnPool
        c2_pooled = self.pool2(x)  # [B, 1, 64, 512]
        c2_conv = self.conv2(c2_pooled)  # [B, 1, 64, 512]
        c2_upsampled = torch.nn.functional.interpolate(
            c2_conv, size=(F, T), mode='bilinear', align_corners=False
        )  # [B, 1, 128, 1024]
        
        # 🔍 全局上下文处理：Pool → Conv → Broadcast
        xg_pooled = self.global_pool(x)  # [B, 1, 1, 1]
        xg_conv = self.global_conv(xg_pooled)  # [B, 1, 1, 1]
        xg_broadcasted = xg_conv.expand_as(x)  # [B, 1, 128, 1024]
        
        # 🔀 多尺度特征融合
        xlg = xl + c1_upsampled + c2_upsampled + xg_broadcasted  # [B, 1, 128, 1024]
        
        # 🎯 生成像素级注意力权重
        wei = self.att_conv(xlg)  # [B, 1, 128, 1024]
        wei = self.sigmoid(wei)   # [B, 1, 128, 1024] 权重范围[0,1]
        
        # 🚀 应用注意力增强原始频谱图
        xo = x + x * wei * 2  # [B, 1, 128, 1024] 注意力加权增强
        
        if return_intermediates:
            return xo, {
                'input': x,
                'local_attention': xl,
                'small_scale': c1_upsampled,  # 返回上采样后的结果
                'medium_scale': c2_upsampled,  # 返回上采样后的结果
                'global_context': xg_broadcasted,  # 返回广播后的结果
                'fused_features': xlg,
                'attention_weights': wei,
                'enhanced_spectrogram': xo,
                # 🆕 额外返回池化阶段的中间结果用于详细可视化
                'small_scale_pooled': c1_pooled,
                'medium_scale_pooled': c2_pooled,
                'global_pooled': xg_pooled
            }
        
        return xo

# 创建模拟的args对象
class FeatureArgs:
    def __init__(self):
        self.sample_rate = 16000
        self.desired_length = 8
        self.pad_types = 'repeat'
        self.class_split = 'lungsound'

class PrePatchMSAVisualizer:
    """Pre-Patch MSA 可视化器"""
    
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.args = FeatureArgs()
        self.output_dir = r'D:\bishe\MVST-main\basevisualization\baseastfeature\save'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化组件
        self._initialize_components()
        
        print('🎯 Pre-Patch MSA 可视化器初始化完成')
        print(f"   数据文件夹: {data_folder}")
        print(f"   输出目录: {self.output_dir}")

    def _initialize_components(self):
        """初始化组件"""
        # Pre-Patch MSA模块
        self.pre_patch_msa = PrePatchMSA(input_channels=1, r=4)
        
        # Patch Embedding组件
        self.patch_embed = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(10, 10))
        
        # 计算patch维度
        self.f_dim, self.t_dim = self._get_patch_dimensions()
        print(f"   🔍 Patch维度: 频率={self.f_dim}, 时间={self.t_dim}")

    def _get_patch_dimensions(self):
        """计算patch维度"""
        test_input = torch.randn(1, 1, 128, 1024)
        test_out = self.patch_embed(test_input.transpose(2, 3))
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def load_sample_data(self):
        """加载一个代表性样本"""
        print("\n🔍 加载代表性ICBHI样本...")
        
        if not os.path.exists(self.data_folder):
            return self._create_mock_sample()
        
        # 获取音频文件列表
        audio_files = [f for f in os.listdir(self.data_folder) if f.endswith('.wav')]
        
        if len(audio_files) == 0:
            return self._create_mock_sample()
        
        # 尝试加载真实数据
        for audio_file in audio_files[:5]:  # 尝试前5个文件
            filename = audio_file.replace('.wav', '')
            annotation_path = os.path.join(self.data_folder, filename + '.txt')
            
            if not os.path.exists(annotation_path):
                continue
                
            try:
                # 读取标注
                annotations = pd.read_csv(
                    annotation_path, 
                    names=['Start', 'End', 'Crackles', 'Wheezes'], 
                    delimiter='\t'
                )
                
                # 提取周期
                cycles_data = get_individual_cycles_torchaudio(
                    self.args, annotations, self.data_folder, filename, 
                    16000, n_cls=4
                )
                
                if cycles_data:
                    audio_data, label = cycles_data[0]  # 取第一个周期
                    
                    # 使用icbhi_util生成fbank特征
                    fbank_feature = generate_fbank(audio_data, 16000, n_mels=128)
                    
                    if fbank_feature is not None:
                        # 转换格式: (T, F, 1) -> (1, F, T)
                        fbank_tensor = torch.tensor(fbank_feature).squeeze(-1)  # (T, F)
                        fbank_tensor = fbank_tensor.transpose(0, 1)  # (F, T)
                        fbank_tensor = fbank_tensor.unsqueeze(0)  # (1, F, T)
                        
                        # 调整时间维度到1024
                        if fbank_tensor.shape[2] != 1024:
                            fbank_tensor = fbank_tensor.unsqueeze(0)  # (1, 1, F, T)
                            fbank_tensor = torch.nn.functional.interpolate(
                                fbank_tensor, size=(128, 1024), mode='bilinear'
                            )
                            fbank_tensor = fbank_tensor.squeeze(0)  # (1, F, T)
                        
                        class_names = {0: '正常', 1: '爆裂音', 2: '喘鸣音', 3: '混合音'}
                        print(f"   ✅ 加载真实样本: {filename}, 标签: {class_names[label]}")
                        
                        return fbank_tensor.unsqueeze(0), label, filename  # (1, 1, F, T)
                        
            except Exception as e:
                print(f"   ⚠️ 文件 {filename} 处理失败: {e}")
                continue
        
        # 如果都失败了，创建模拟数据
        return self._create_mock_sample()

    def _create_mock_sample(self):
        """创建模拟样本"""
        print("   🔄 创建模拟ICBHI样本...")
        
        # 创建模拟fbank特征 (1, 1, 128, 1024)
        fbank = np.random.randn(128, 1024) * 0.5
        
        # 添加爆裂音特征模式
        for _ in range(8):
            t_pos = np.random.randint(50, 950)
            f_pos = np.random.randint(60, 120)
            fbank[f_pos:f_pos+10, t_pos:t_pos+5] += 3.0
        
        # 添加持续的低频能量
        fbank[10:30, 200:800] += 1.5
        
        # 标准化
        mean, std = -4.2677393, 4.5689974
        fbank = (fbank - mean) / (std * 2)
        
        # 转换为torch张量
        fbank_tensor = torch.tensor(fbank, dtype=torch.float32)
        fbank_tensor = fbank_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)
        
        print("   📊 创建了模拟爆裂音样本")
        
        return fbank_tensor, 1, "mock_crackle"

    def visualize_pre_patch_msa_pipeline(self):
        """可视化Pre-Patch MSA完整流程"""
        print("\n" + "="*80)
        print('🔍 Pre-Patch MSA 完整流程可视化')
        print("="*80)
        
        # 加载样本
        input_fbank, label, filename = self.load_sample_data()
        class_names = {0: '正常', 1: '爆裂音', 2: '喘鸣音', 3: '混合音'}
        
        print(f"\n📊 处理样本: {filename} ({class_names[label]})")
        print(f"   原始输入形状: {input_fbank.shape}")
        
        with torch.no_grad():
            # 🎯 步骤1：原始频谱图 (1, 1, 128, 1024)
            original_spectrogram = input_fbank.clone()
            print(f"\n🎯 步骤1 - 原始频谱图: {original_spectrogram.shape}")
            
            # 🎯 步骤2：Pre-Patch MSA处理
            enhanced_spectrogram, msa_intermediates = self.pre_patch_msa(
                original_spectrogram, return_intermediates=True
            )
            print(f"🎯 步骤2 - Pre-Patch MSA增强: {enhanced_spectrogram.shape}")
            
            # 🎯 步骤3：转置准备Patch Embedding
            transposed_original = original_spectrogram.transpose(2, 3)  # (1, 1, 1024, 128)
            transposed_enhanced = enhanced_spectrogram.transpose(2, 3)  # (1, 1, 1024, 128)
            print(f"🎯 步骤3 - 转置操作: {transposed_enhanced.shape}")
            
            # 🎯 步骤4：Patch Embedding对比
            patch_original = self.patch_embed(transposed_original).flatten(2).transpose(1, 2)
            patch_enhanced = self.patch_embed(transposed_enhanced).flatten(2).transpose(1, 2)
            print(f"🎯 步骤4 - Patch Embedding: {patch_enhanced.shape}")
        
        # 🎨 创建综合可视化
        self._create_pre_patch_visualization(
            original_spectrogram, enhanced_spectrogram, msa_intermediates,
            patch_original, patch_enhanced, label, filename
        )

    def _create_pre_patch_visualization(self, original_spec, enhanced_spec, msa_intermediates,
                                      patch_original, patch_enhanced, label, filename):
        """创建Pre-Patch MSA可视化"""
        
        class_names = {0: '正常', 1: '爆裂音', 2: '喘鸣音', 3: '混合音'}
        
        # 创建大图 - 分为多个部分
        fig = plt.figure(figsize=(24, 20))
        
        # 第一部分：原始频谱图处理 (2行4列)
        gs1 = GridSpec(2, 4, figure=fig, top=0.95, bottom=0.7, hspace=0.3, wspace=0.3)
        
        # 第二部分：Pre-Patch MSA多尺度分析 (2行4列)
        gs2 = GridSpec(2, 4, figure=fig, top=0.65, bottom=0.4, hspace=0.3, wspace=0.3)
        
        # 第三部分：增强效果对比 (2行3列)
        gs3 = GridSpec(2, 3, figure=fig, top=0.35, bottom=0.05, hspace=0.3, wspace=0.3)
        
        # 🎨 第一部分：原始频谱图和增强对比
        self._plot_spectrogram_comparison(fig, gs1, original_spec, enhanced_spec, 
                                        msa_intermediates, label, filename)
        
        # 🎨 第二部分：Pre-Patch MSA多尺度分析
        self._plot_pre_patch_msa_analysis(fig, gs2, msa_intermediates)
        
        # 🎨 第三部分：Patch Embedding对比
        self._plot_patch_embedding_comparison(fig, gs3, patch_original, patch_enhanced,
                                            original_spec, enhanced_spec)
        
        # 设置总标题
        fig.suptitle(f'Pre-Patch MSA 完整流程可视化\n样本: {filename} ({class_names[label]})', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 保存图片
        save_path = os.path.join(self.output_dir, f'pre_patch_msa_pipeline_{filename}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"\n✅ Pre-Patch MSA可视化已保存: {save_path}")

    def _plot_spectrogram_comparison(self, fig, gs, original_spec, enhanced_spec, 
                                   msa_intermediates, label, filename):
        """绘制频谱图对比"""
        
        # 1. 原始频谱图
        ax1 = fig.add_subplot(gs[0, 0])
        original_2d = original_spec[0, 0].numpy()
        im1 = ax1.imshow(original_2d, aspect='auto', origin='lower', cmap='viridis')
        ax1.set_title('原始FilterBank频谱图\n(128×1024)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('时间帧')
        ax1.set_ylabel('频率bin')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # 2. Pre-Patch MSA增强后
        ax2 = fig.add_subplot(gs[0, 1])
        enhanced_2d = enhanced_spec[0, 0].numpy()
        im2 = ax2.imshow(enhanced_2d, aspect='auto', origin='lower', cmap='viridis')
        ax2.set_title('Pre-Patch MSA增强后\n(128×1024)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('时间帧')
        ax2.set_ylabel('频率bin')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # 3. 增强差异图
        ax3 = fig.add_subplot(gs[0, 2])
        diff = enhanced_2d - original_2d
        im3 = ax3.imshow(diff, aspect='auto', origin='lower', cmap='RdBu_r')
        ax3.set_title('增强差异\n(增强后-原始)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('时间帧')
        ax3.set_ylabel('频率bin')
        plt.colorbar(im3, ax=ax3, fraction=0.046, label='差异强度')
        
        # 4. 注意力权重图
        ax4 = fig.add_subplot(gs[0, 3])
        attention_weights = msa_intermediates['attention_weights'][0, 0].numpy()
        im4 = ax4.imshow(attention_weights, aspect='auto', origin='lower', cmap='hot')
        ax4.set_title('Pre-Patch 注意力权重\n(像素级)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('时间帧')
        ax4.set_ylabel('频率bin')
        plt.colorbar(im4, ax=ax4, fraction=0.046, label='权重[0,1]')
        
        # 5-8. 统计分析
        # 5. 原始频谱统计
        ax5 = fig.add_subplot(gs[1, 0])
        original_flat = original_2d.flatten()
        ax5.hist(original_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax5.set_title('原始频谱值分布', fontsize=12, fontweight='bold')
        ax5.set_xlabel('频谱值')
        ax5.set_ylabel('频次')
        ax5.grid(True, alpha=0.3)
        
        # 6. 增强频谱统计
        ax6 = fig.add_subplot(gs[1, 1])
        enhanced_flat = enhanced_2d.flatten()
        ax6.hist(enhanced_flat, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax6.set_title('增强频谱值分布', fontsize=12, fontweight='bold')
        ax6.set_xlabel('频谱值')
        ax6.set_ylabel('频次')
        ax6.grid(True, alpha=0.3)
        
        # 7. 注意力权重分布
        ax7 = fig.add_subplot(gs[1, 2])
        attention_flat = attention_weights.flatten()
        ax7.hist(attention_flat, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax7.set_title('注意力权重分布', fontsize=12, fontweight='bold')
        ax7.set_xlabel('权重值')
        ax7.set_ylabel('频次')
        ax7.grid(True, alpha=0.3)
        
        # 8. 增强效果统计对比
        ax8 = fig.add_subplot(gs[1, 3])
        
        stats_data = {
            '原始': [original_2d.mean(), original_2d.std(), original_2d.max(), original_2d.min()],
            '增强': [enhanced_2d.mean(), enhanced_2d.std(), enhanced_2d.max(), enhanced_2d.min()]
        }
        
        metrics = ['均值', '标准差', '最大值', '最小值']
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        ax8.bar(x_pos - width/2, stats_data['原始'], width, label='原始', alpha=0.7, color='blue')
        ax8.bar(x_pos + width/2, stats_data['增强'], width, label='增强', alpha=0.7, color='red')
        
        ax8.set_title('统计指标对比', fontsize=12, fontweight='bold')
        ax8.set_xlabel('统计指标')
        ax8.set_ylabel('数值')
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(metrics)
        ax8.legend()
        ax8.grid(True, alpha=0.3)

    def _plot_pre_patch_msa_analysis(self, fig, gs, msa_intermediates):
        """绘制Pre-Patch MSA多尺度分析 - 基于MSA架构图设计"""
        
        # 获取MSA中间结果
        input_feat = msa_intermediates['input']  # F_FUSE
        local_att = msa_intermediates['local_attention']  # f_s
        small_scale = msa_intermediates['small_scale']  # f_c1
        medium_scale = msa_intermediates['medium_scale']  # f_c2
        global_context = msa_intermediates['global_context']  # 类似f_c4
        fused_features = msa_intermediates['fused_features']
        attention_weights = msa_intermediates['attention_weights']  # α
        
        # 🔧 修复：正确转换为2D用于可视化
        input_2d = input_feat.squeeze().cpu().numpy()
        local_2d = local_att.squeeze().cpu().numpy()
        
        # 🔧 修复：处理池化后的特征
        # 小尺度和中尺度需要上采样回原始尺寸进行可视化
        H, W = input_2d.shape
        
        # 处理小尺度特征 (原本是32×256，需要上采样)
        if small_scale.dim() == 4:  # [B, C, H, W]
            small_upsampled = torch.nn.functional.interpolate(
                small_scale, size=(H, W), mode='bilinear', align_corners=False
            )
            small_2d = small_upsampled.squeeze().cpu().numpy()
        else:
            small_2d = small_scale.squeeze().cpu().numpy()
        
        # 处理中尺度特征 (原本是64×512，需要上采样)
        if medium_scale.dim() == 4:  # [B, C, H, W]
            medium_upsampled = torch.nn.functional.interpolate(
                medium_scale, size=(H, W), mode='bilinear', align_corners=False
            )
            medium_2d = medium_upsampled.squeeze().cpu().numpy()
        else:
            medium_2d = medium_scale.squeeze().cpu().numpy()
        
        # 🔧 修复：处理全局上下文 (1×1的标量值)
        if global_context.dim() == 4:  # [B, C, H, W]
            global_value = global_context.squeeze().cpu().numpy()
            if global_value.ndim == 0:  # 标量
                # 创建与原始尺寸相同的广播图
                global_2d = np.full((H, W), global_value)
            elif global_value.shape == (1, 1):  # 1×1矩阵
                global_2d = np.full((H, W), global_value[0, 0])
            else:
                # 如果已经是正确尺寸，直接使用
                global_2d = global_value
        else:
            # 处理其他情况
            global_value = global_context.squeeze().cpu().numpy()
            if global_value.ndim == 0:
                global_2d = np.full((H, W), global_value)
            else:
                global_2d = global_value
        
        fused_2d = fused_features.squeeze().cpu().numpy()
        weights_2d = attention_weights.squeeze().cpu().numpy()
        
        # === 第一行：多尺度特征提取分支 ===
        
        # 1. 输入特征 F_FUSE
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(input_2d, cmap='viridis', aspect='auto', origin='lower')
        ax1.set_title('🎯 输入特征 F_FUSE\n(C×H×W)', fontsize=10, fontweight='bold')
        ax1.set_xlabel('时间帧')
        ax1.set_ylabel('频率bin')
        plt.colorbar(im1, ax=ax1, shrink=0.6)
        
        # 2. 局部注意力分支 f_s
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(local_2d, cmap='plasma', aspect='auto', origin='lower')
        ax2.set_title('🔍 局部注意力 f_s\n3×3 Conv → 局部模式', fontsize=10, fontweight='bold')
        ax2.set_xlabel('时间帧')
        ax2.set_ylabel('频率bin')
        plt.colorbar(im2, ax=ax2, shrink=0.6)
        
        # 3. 小尺度上下文 f_c1 
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(small_2d, cmap='coolwarm', aspect='auto', origin='lower')
        ax3.set_title('🔍 小尺度上下文 f_c1\nPool(32×256) → UnPool', fontsize=10, fontweight='bold')
        ax3.set_xlabel('时间帧')
        ax3.set_ylabel('频率bin')
        plt.colorbar(im3, ax=ax3, shrink=0.6)
        
        # 4. 中尺度上下文 f_c2
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(medium_2d, cmap='RdYlBu', aspect='auto', origin='lower')
        ax4.set_title('🔍 中尺度上下文 f_c2\nPool(64×512) → UnPool', fontsize=10, fontweight='bold')
        ax4.set_xlabel('时间帧')
        ax4.set_ylabel('频率bin')
        plt.colorbar(im4, ax=ax4, shrink=0.6)
        
        # === 第二行：融合与注意力权重 ===
        
        # 5. 全局上下文 (处理后的广播图)
        ax5 = fig.add_subplot(gs[1, 0])
        im5 = ax5.imshow(global_2d, cmap='inferno', aspect='auto', origin='lower')
        ax5.set_title('🌐 全局上下文\nPool(1×1) → Broadcast', fontsize=10, fontweight='bold')
        ax5.set_xlabel('时间帧')
        ax5.set_ylabel('频率bin')
        plt.colorbar(im5, ax=ax5, shrink=0.6)
        
        # 6. 多尺度融合后特征
        ax6 = fig.add_subplot(gs[1, 1])
        im6 = ax6.imshow(fused_2d, cmap='magma', aspect='auto', origin='lower')
        ax6.set_title('🔀 多尺度融合\nf_s + f̄_c1 + f̄_c2 + f̄_global', fontsize=10, fontweight='bold')
        ax6.set_xlabel('时间帧')
        ax6.set_ylabel('频率bin')
        plt.colorbar(im6, ax=ax6, shrink=0.6)
        
        # 7. 注意力权重 α (Sigmoid激活)
        ax7 = fig.add_subplot(gs[1, 2])
        im7 = ax7.imshow(weights_2d, cmap='hot', aspect='auto', vmin=0, vmax=1, origin='lower')
        ax7.set_title('⊕ 注意力权重 α\nSigmoid(融合特征)', fontsize=10, fontweight='bold')
        ax7.set_xlabel('时间帧')
        ax7.set_ylabel('频率bin')
        plt.colorbar(im7, ax=ax7, shrink=0.6)
        
        # 8. MSA架构流程图解和统计分析
        ax8 = fig.add_subplot(gs[1, 3])
        
        # 🆕 计算各分支的特征统计
        stats_data = {
            '局部f_s': [local_2d.mean(), local_2d.std(), local_2d.max(), local_2d.min()],
            '小尺度f_c1': [small_2d.mean(), small_2d.std(), small_2d.max(), small_2d.min()],
            '中尺度f_c2': [medium_2d.mean(), medium_2d.std(), medium_2d.max(), medium_2d.min()],
            '全局': [global_2d.mean(), global_2d.std(), global_2d.max(), global_2d.min()]
        }
        
        # 计算各分支的平均强度用于可视化
        branch_strengths = [
            abs(local_2d).mean(),
            abs(small_2d).mean(), 
            abs(medium_2d).mean(),
            abs(global_2d).mean()
        ]
        
        branch_names = ['局部', '小尺度', '中尺度', '全局']
        colors = ['purple', 'blue', 'green', 'orange']
        
        # 绘制分支强度对比
        bars = ax8.bar(branch_names, branch_strengths, color=colors, alpha=0.7)
        ax8.set_title('🔍 各分支特征强度', fontsize=10, fontweight='bold')
        ax8.set_ylabel('平均特征强度')
        ax8.tick_params(axis='x', rotation=45)
        ax8.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, strength in zip(bars, branch_strengths):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{strength:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 🆕 添加MSA处理信息文本
        info_text = f"""
MSA处理统计:
• 输入: {H}×{W}
• 局部: 3×3卷积
• 小尺度: 32×256池化
• 中尺度: 64×512池化  
• 全局: 1×1池化
• 权重范围: [{weights_2d.min():.3f}, {weights_2d.max():.3f}]
"""
        
        ax8.text(0.02, 0.02, info_text, transform=ax8.transAxes, fontsize=7,
                 verticalalignment='bottom', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))

    def _plot_patch_embedding_comparison(self, fig, gs, patch_original, patch_enhanced,
                                       original_spec, enhanced_spec):
        """绘制Patch Embedding对比"""
        
        # 1. 原始Patch特征范数
        ax1 = fig.add_subplot(gs[0, 0])
        original_norms = torch.norm(patch_original[0], dim=1).numpy()
        patch_map_orig = original_norms.reshape(self.f_dim, self.t_dim)
        im1 = ax1.imshow(patch_map_orig, aspect='auto', origin='lower', cmap='plasma')
        ax1.set_title('原始频谱的Patch特征\n(12×102 patches)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('时间patches')
        ax1.set_ylabel('频率patches')
        plt.colorbar(im1, ax=ax1, fraction=0.046, label='特征范数')
        
        # 2. 增强Patch特征范数
        ax2 = fig.add_subplot(gs[0, 1])
        enhanced_norms = torch.norm(patch_enhanced[0], dim=1).numpy()
        patch_map_enh = enhanced_norms.reshape(self.f_dim, self.t_dim)
        im2 = ax2.imshow(patch_map_enh, aspect='auto', origin='lower', cmap='plasma')
        ax2.set_title('MSA增强后的Patch特征\n(12×102 patches)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('时间patches')
        ax2.set_ylabel('频率patches')
        plt.colorbar(im2, ax=ax2, fraction=0.046, label='特征范数')
        
        # 3. Patch特征改进热图
        ax3 = fig.add_subplot(gs[0, 2])
        improvement_ratio = enhanced_norms / (original_norms + 1e-8)
        improvement_map = improvement_ratio.reshape(self.f_dim, self.t_dim)
        im3 = ax3.imshow(improvement_map, aspect='auto', origin='lower', cmap='YlOrRd')
        ax3.set_title('🚀 Patch特征改进比例\n(增强/原始)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('时间patches')
        ax3.set_ylabel('频率patches')
        plt.colorbar(im3, ax=ax3, fraction=0.046, label='改进比例')
        
        # 4. Patch特征统计对比
        ax4 = fig.add_subplot(gs[1, 0])
        
        # 显示前50个patch的对比
        num_patches_show = 50
        x_positions = np.arange(num_patches_show)
        width = 0.35
        
        ax4.bar(x_positions - width/2, original_norms[:num_patches_show], width, 
                label='原始', alpha=0.7, color='lightblue')
        ax4.bar(x_positions + width/2, enhanced_norms[:num_patches_show], width, 
                label='MSA增强', alpha=0.7, color='lightcoral')
        
        ax4.set_title(f'Patch特征范数对比\n(前{num_patches_show}个patches)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Patch索引')
        ax4.set_ylabel('特征范数')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 整体改进统计
        ax5 = fig.add_subplot(gs[1, 1])
        
        improvement_stats = {
            '均值改进': (enhanced_norms.mean() - original_norms.mean()) / original_norms.mean() * 100,
            '标准差改进': (enhanced_norms.std() - original_norms.std()) / original_norms.std() * 100,
            '最大值改进': (enhanced_norms.max() - original_norms.max()) / original_norms.max() * 100,
            '总能量改进': (enhanced_norms.sum() - original_norms.sum()) / original_norms.sum() * 100
        }
        
        metrics = list(improvement_stats.keys())
        improvements = list(improvement_stats.values())
        colors = ['green' if x > 0 else 'red' for x in improvements]
        
        bars = ax5.bar(range(len(metrics)), improvements, color=colors, alpha=0.7)
        ax5.set_title('Pre-Patch MSA 改进效果\n(百分比)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('改进指标')
        ax5.set_ylabel('改进百分比 (%)')
        ax5.set_xticks(range(len(metrics)))
        ax5.set_xticklabels(metrics, rotation=45)
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 添加数值标签
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                    f'{improvement:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 6. 频谱图能量分布对比
        ax6 = fig.add_subplot(gs[1, 2])
        
        # 计算频率维度的能量分布
        original_freq_energy = original_spec[0, 0].numpy().mean(axis=1)  # 沿时间轴平均
        enhanced_freq_energy = enhanced_spec[0, 0].numpy().mean(axis=1)  # 沿时间轴平均
        
        freq_bins = range(len(original_freq_energy))
        ax6.plot(freq_bins, original_freq_energy, 'b-', linewidth=2, label='原始', alpha=0.7)
        ax6.plot(freq_bins, enhanced_freq_energy, 'r-', linewidth=2, label='MSA增强', alpha=0.7)
        
        ax6.set_title('频率维度能量分布对比', fontsize=12, fontweight='bold')
        ax6.set_xlabel('频率bin')
        ax6.set_ylabel('平均能量')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    def run_visualization(self):
        """运行可视化"""
        print("🚀 开始Pre-Patch MSA可视化")
        print("="*60)
        
        try:
            self.visualize_pre_patch_msa_pipeline()
            
            print("\n" + "="*60)
            print(f"🎉 Pre-Patch MSA可视化完成！")
            print(f"   所有结果已保存到: {self.output_dir}")
            
            # 打印Pre-Patch MSA的优势
            print(f"\n💡 Pre-Patch MSA的优势:")
            print(f"   ✅ 直接在原始频谱图上进行注意力增强")
            print(f"   ✅ 保留完整的时频分辨率信息")
            print(f"   ✅ 增强病理声音的频谱特征")
            print(f"   ✅ 为后续Patch Embedding提供更好的输入")
            print(f"   ✅ 可以更精细地处理像素级特征")
            print("="*60)
            
        except Exception as e:
            print(f"❌ 可视化过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    # 数据路径配置
    data_folder = r"D:\bishe\data\ICBHI_final_database\icbhi_dataset"
    
    print("🎯 Pre-Patch MSA 可视化")
    print(f"   数据文件夹: {data_folder}")
    print("="*60)
    
    print("💡 Pre-Patch MSA概念:")
    print("   • 在Patch Embedding之前直接对原始频谱图应用MSA")
    print("   • 像素级注意力增强，保留完整时频分辨率")
    print("   • 增强病理声音特征，为Patch提取提供更好的输入")
    print("="*60)
    
    # 创建可视化器
    visualizer = PrePatchMSAVisualizer(data_folder=data_folder)
    
    # 运行可视化
    visualizer.run_visualization()

if __name__ == "__main__":
    main()