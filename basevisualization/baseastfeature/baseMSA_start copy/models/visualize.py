import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import torchaudio
from torchaudio import transforms as T
import os
from collections import namedtuple

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入MSA模块 (直接从ast1.py复制)
class MSA(nn.Module):
    '''
    多特征融合 AFF, 一个像素级尺度，多个语义级尺度
    '''
    def __init__(self, channels=64, r=4):
        super(MSA, self).__init__()
        # 修复：确保inter_channels至少为1
        inter_channels = max(1, int(channels // r))
        
        # 修复：对于单通道输入，使用特殊的网络结构
        if channels == 1:
            # 对于单通道音频输入，使用更适合的结构
            inter_channels = 4  # 固定使用4个中间通道
            
            self.local_att = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channels),
            )

            self.context1 = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels)
            )

            self.context2 = nn.Sequential(
                nn.AdaptiveAvgPool2d((8, 8)),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels)
            )

            self.context3 = nn.Sequential(
                nn.AdaptiveAvgPool2d((16, 16)),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels)
            )

            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )

        self.sigmoid = nn.Sigmoid()

    def forward_with_intermediates(self, x):
        """返回中间结果用于可视化"""
        h, w = x.shape[2], x.shape[3]
        
        xa = x 
        xl = self.local_att(xa)
        c1 = self.context1(xa)
        c2 = self.context2(xa)
        c3 = self.context3(xa)
        xg = self.global_att(xa)

        # 将 c1, c2, c3 还原到原本的大小
        c1_resized = F.interpolate(c1, size=[h, w], mode='nearest')
        c2_resized = F.interpolate(c2, size=[h, w], mode='nearest')
        c3_resized = F.interpolate(c3, size=[h, w], mode='nearest')

        xlg = xl + xg + c1_resized + c2_resized + c3_resized
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei

        return {
            'input': x,
            'local': xl,
            'context1': c1,
            'context2': c2, 
            'context3': c3,
            'global': xg,
            'context1_resized': c1_resized,
            'context2_resized': c2_resized,
            'context3_resized': c3_resized,
            'fused': xlg,
            'attention_weights': wei,
            'output': xo
        }

# 复制icbhi_util.py中的函数
def generate_fbank(audio, sample_rate, n_mels=128): 
    """
    use torchaudio library to convert mel fbank for AST model
    """    
    assert sample_rate == 16000, 'input audio sampling rate must be 16kHz'
    fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True, sample_frequency=sample_rate, use_energy=False, window_type='hanning', num_mel_bins=n_mels, dither=0.0, frame_shift=10)
    
    mean, std =  -4.2677393, 4.5689974
    fbank = (fbank - mean) / (std * 2) # mean / std
    fbank = fbank.unsqueeze(-1).numpy()
    return fbank

def load_real_icbhi_data(specific_file=None):
    """加载真实的ICBHI数据"""
    
    # 如果指定了具体文件，直接加载
    if specific_file and os.path.exists(specific_file):
        fpath = specific_file
        filename = os.path.basename(fpath)
        print(f"📁 加载指定ICBHI数据: {filename}")
        
        # 使用torchaudio加载音频
        try:
            data, original_sr = torchaudio.load(fpath)
            print(f"✅ 成功加载音频文件")
            print(f"📊 原始采样率: {original_sr}Hz")
            print(f"📊 音频形状: {data.shape}")
            
            # 重采样到16kHz（如果需要）
            target_sr = 16000
            if original_sr != target_sr:
                print(f"🔄 重采样从 {original_sr}Hz 到 {target_sr}Hz")
                resample = T.Resample(original_sr, target_sr)
                data = resample(data)
            
            # 如果是立体声，取第一个通道
            if data.shape[0] > 1:
                print("🔊 检测到多通道音频，使用第一个通道")
                data = data[0:1, :]
            
            # 限制音频长度（用于可视化）
            max_duration = 20  # 秒
            max_samples = target_sr * max_duration
            if data.shape[1] > max_samples:
                print(f"✂️ 截取音频到 {max_duration} 秒")
                data = data[:, :max_samples]
            
            return data, target_sr, filename
            
        except Exception as e:
            print(f"❌ 加载音频文件失败: {e}")
            return generate_simulated_icbhi_data()
    
    # 如果没有指定文件，尝试查找数据文件夹
    icbhi_paths = [
        r"D:\bishe\data\ICBHI_final_database\icbhi_dataset",
        r"./data/ICBHI_final_database/icbhi_dataset",
        r"./",
        r"../data"
    ]
    
    for data_folder in icbhi_paths:
        if os.path.exists(data_folder):
            print(f"📁 搜索ICBHI数据文件夹: {data_folder}")
            wav_files = [f for f in os.listdir(data_folder) if f.endswith('.wav')]
            
            if wav_files:
                # 选择第一个文件
                sample_file = wav_files[0]
                fpath = os.path.join(data_folder, sample_file)
                return load_real_icbhi_data(fpath)
    
    print("⚠️ 未找到ICBHI数据文件，将生成模拟数据")
    return generate_simulated_icbhi_data()

def generate_simulated_icbhi_data():
    """生成模拟的ICBHI肺音数据"""
    print("📊 生成模拟ICBHI肺音数据...")
    
    sample_rate = 16000
    duration = 20  # 秒
    t = np.linspace(0, duration, sample_rate * duration)
    
    # 模拟肺音信号：包含正常呼吸和异常音
    # 基础呼吸音 (100-600 Hz)
    breath_sound = 0.3 * np.sin(2 * np.pi * 200 * t) * np.exp(-0.1 * (t % 4))
    
    # 添加crackles (高频爆裂音 600-2000 Hz)
    from scipy.signal import butter, lfilter
    crackles_signal = np.random.randn(len(t)) * (t % 8 < 0.1)
    b, a = butter(5, [600, 2000], 'band', fs=sample_rate)
    crackles = lfilter(b, a, crackles_signal) * 0.2
    
    # 添加wheezes (单频哨鸣音 100-1000 Hz)
    wheezes = 0.15 * np.sin(2 * np.pi * 400 * t) * (np.sin(2 * np.pi * 0.2 * t) > 0.5)
    
    # 合成音频
    audio = breath_sound + crackles + wheezes
    audio = audio + 0.05 * np.random.randn(len(audio))  # 添加噪声
    
    # 转换为torch tensor
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    
    return audio_tensor, sample_rate, "simulated_lung_sound.wav"

def create_icbhi_real_msa_visualization(specific_file=None):
    """创建真实ICBHI数据通过MSA的可视化"""
    
    # 1. 加载真实ICBHI数据
    if specific_file is None:
        # 使用您提供的具体文件路径
        specific_file = r"D:\bishe\data\ICBHI_final_database\icbhi_dataset\101_1b1_Al_sc_Meditron.wav"
    
    try:
        audio, sr, filename = load_real_icbhi_data(specific_file)
    except Exception as e:
        print(f"❌ 加载指定文件失败: {e}")
        audio, sr, filename = generate_simulated_icbhi_data()
    
    print(f"🎵 音频文件: {filename}")
    print(f"📊 音频形状: {audio.shape}, 采样率: {sr}Hz")
    
    # 2. 使用icbhi_util.py中的generate_fbank函数生成频谱图
    print("📊 使用icbhi_util.py生成fbank特征...")
    fbank = generate_fbank(audio, sr, n_mels=128)
    
    # 移除最后一个维度 (从 [T, 128, 1] 到 [T, 128])
    fbank_2d = fbank.squeeze(-1)
    
    print(f"📊 Fbank特征形状: {fbank_2d.shape}")
    
    # 🔧 修复：转置fbank以匹配预期的格式 [mel_bins, time_frames]
    if fbank_2d.shape[1] == 128:  # 如果第二个维度是128（mel bins）
        fbank_2d = fbank_2d.T  # 转置为 [128, T]
        print(f"🔄 转置fbank特征为: {fbank_2d.shape}")
    
    # 调整尺寸以适应可视化 (限制时间维度)
    max_time_frames = 1024
    if fbank_2d.shape[1] > max_time_frames:
        print(f"✂️ 截取频谱图时间维度到 {max_time_frames} 帧")
        fbank_2d = fbank_2d[:, :max_time_frames]
    elif fbank_2d.shape[1] < max_time_frames:
        # 重复填充
        repeat_times = max_time_frames // fbank_2d.shape[1] + 1
        fbank_2d = np.tile(fbank_2d, (1, repeat_times))[:, :max_time_frames]
        print(f"🔄 填充频谱图时间维度到 {max_time_frames} 帧")
    
    # 3. 转换为torch tensor用于MSA处理
    fbank_tensor = torch.from_numpy(fbank_2d).float().unsqueeze(0).unsqueeze(0)  # [1, 1, 128, 1024]
    
    print(f"📊 MSA输入张量形状: {fbank_tensor.shape}")
    
    # 4. 通过MSA处理
    print("🔄 通过MSA模块处理...")
    msa = MSA(channels=1, r=4)
    msa.eval()
    
    with torch.no_grad():
        intermediates = msa.forward_with_intermediates(fbank_tensor)
    
    print("✅ MSA处理完成")
    
    # 5. 创建可视化
    fig = plt.figure(figsize=(24, 18))
    
    # 定义子图布局 (5行5列)
    gs = fig.add_gridspec(5, 5, hspace=0.4, wspace=0.3)
    
    def plot_feature_map(ax, data, title, cmap='viridis', add_colorbar=True):
        """绘制特征图的辅助函数"""
        if isinstance(data, torch.Tensor):
            data = data.squeeze().numpy()
        
        # 🔧 修复：确保数据是2D且非空
        if data.ndim == 0:  # 标量数据
            print(f"⚠️ 警告: {title} 包含标量数据，跳过可视化")
            ax.text(0.5, 0.5, 'No Data\n(Scalar)', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            return None
        elif data.ndim == 1:  # 1D数据
            print(f"⚠️ 警告: {title} 是1D数据，转换为2D")
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            # 多维数据，取第一个通道或平均
            if data.shape[0] == 1:
                data = data[0]
            else:
                data = data.mean(axis=0)
            
            # 如果还是多维，继续处理
            while data.ndim > 2:
                data = data.mean(axis=0)
        
        # 确保数据不为空
        if data.size == 0:
            print(f"⚠️ 警告: {title} 包含空数据，跳过可视化")
            ax.text(0.5, 0.5, 'Empty Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            return None
        
        try:
            im = ax.imshow(data, aspect='auto', origin='lower', cmap=cmap, 
                          interpolation='bilinear')
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('Time Frames')
            ax.set_ylabel('Mel Bins')
            
            if add_colorbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # 添加数值统计
            mean_val = data.mean()
            std_val = data.std()
            min_val = data.min()
            max_val = data.max()
            
            stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nmin={min_val:.3f}\nmax={max_val:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            return im
            
        except Exception as e:
            print(f"❌ 绘制 {title} 时出错: {e}")
            print(f"📊 数据形状: {data.shape}")
            ax.text(0.5, 0.5, f'Error\n{str(e)}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            return None
    
    # === 第一行：原始数据和输入 ===
    # 原始音频波形
    ax1 = fig.add_subplot(gs[0, :2])
    if audio.dim() > 1:
        audio_plot = audio.squeeze().numpy()
    else:
        audio_plot = audio.numpy()
    
    time_axis = np.linspace(0, len(audio_plot)/sr, len(audio_plot))
    ax1.plot(time_axis, audio_plot, 'b-', linewidth=0.5)
    ax1.set_title(f'ICBHI肺音信号: {filename}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('时间 (秒)')
    ax1.set_ylabel('幅度')
    ax1.grid(True, alpha=0.3)
    
    # 原始fbank特征
    ax2 = fig.add_subplot(gs[0, 2:4])
    plot_feature_map(ax2, fbank_2d, 'ICBHI Fbank特征\n(icbhi_util.py生成)')
    
    # MSA输入
    ax3 = fig.add_subplot(gs[0, 4])
    plot_feature_map(ax3, intermediates['input'], 'MSA输入\n[1,1,128,1024]')
    
    # === 第二行：MSA五个分支 ===
    branch_data = [
        ('local', '局部注意力\n(3×3卷积)', 'Reds'),
        ('context1', '上下文1\n(4×4池化)', 'Blues'),
        ('context2', '上下文2\n(8×8池化)', 'Blues'),
        ('context3', '上下文3\n(16×16池化)', 'Blues'),
        ('global', '全局注意力\n(1×1池化)', 'Oranges')
    ]
    
    for i, (key, title, cmap) in enumerate(branch_data):
        ax = fig.add_subplot(gs[1, i])
        plot_feature_map(ax, intermediates[key], title, cmap)
    
    # === 第三行：上采样后的上下文分支 ===
    ax6 = fig.add_subplot(gs[2, 0])
    plot_feature_map(ax6, intermediates['local'], '局部分支\n(原尺寸)', 'Reds')
    
    ax7 = fig.add_subplot(gs[2, 1])
    plot_feature_map(ax7, intermediates['context1_resized'], '上下文1\n(上采样到128×1024)', 'Blues')
    
    ax8 = fig.add_subplot(gs[2, 2])
    plot_feature_map(ax8, intermediates['context2_resized'], '上下文2\n(上采样到128×1024)', 'Blues')
    
    ax9 = fig.add_subplot(gs[2, 3])
    plot_feature_map(ax9, intermediates['context3_resized'], '上下文3\n(上采样到128×1024)', 'Blues')
    
    ax10 = fig.add_subplot(gs[2, 4])
    plot_feature_map(ax10, intermediates['global'], '全局分支\n(广播到128×1024)', 'Oranges')
    
    # === 第四行：融合和注意力 ===
    ax11 = fig.add_subplot(gs[3, :2])
    plot_feature_map(ax11, intermediates['fused'], '特征融合\nxlg = xl + xg + c1 + c2 + c3', 'plasma')
    
    ax12 = fig.add_subplot(gs[3, 2:4])
    plot_feature_map(ax12, intermediates['attention_weights'], 'Sigmoid注意力权重\nwei = σ(xlg)', 'RdYlBu')
    
    ax13 = fig.add_subplot(gs[3, 4])
    plot_feature_map(ax13, intermediates['output'], '最终输出\nxo = 2×x×wei', 'viridis')
    
    # === 第五行：详细分析 ===
    # 输入vs输出对比
    ax14 = fig.add_subplot(gs[4, :2])
    input_data = intermediates['input'].squeeze().numpy()
    output_data = intermediates['output'].squeeze().numpy()
    
    # 确保数据不为空且维度正确
    if input_data.size > 0 and output_data.size > 0 and input_data.shape == output_data.shape:
        enhancement = output_data - input_data
        
        try:
            im14 = ax14.imshow(enhancement, aspect='auto', origin='lower', 
                              cmap='RdBu_r', interpolation='bilinear')
            ax14.set_title('增强效果 (输出 - 输入)', fontweight='bold')
            ax14.set_xlabel('Time Frames')
            ax14.set_ylabel('Mel Bins')
            plt.colorbar(im14, ax=ax14, fraction=0.046, pad=0.04)
        except Exception as e:
            ax14.text(0.5, 0.5, f'Enhancement Error\n{str(e)}', ha='center', va='center', 
                     transform=ax14.transAxes, fontsize=10)
            ax14.set_title('增强效果 (输出 - 输入)', fontweight='bold')
    else:
        ax14.text(0.5, 0.5, 'Data Mismatch', ha='center', va='center', 
                 transform=ax14.transAxes, fontsize=14)
        ax14.set_title('增强效果 (输出 - 输入)', fontweight='bold')
    
    # 注意力权重分布
    ax15 = fig.add_subplot(gs[4, 2])
    try:
        weights_data = intermediates['attention_weights'].squeeze().numpy()
        if weights_data.size > 0:
            weights_flat = weights_data.flatten()
            ax15.hist(weights_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax15.set_title('注意力权重分布', fontweight='bold')
            ax15.set_xlabel('权重值')
            ax15.set_ylabel('频次')
            ax15.grid(True, alpha=0.3)
        else:
            ax15.text(0.5, 0.5, 'No Weight Data', ha='center', va='center', 
                     transform=ax15.transAxes, fontsize=14)
            ax15.set_title('注意力权重分布', fontweight='bold')
    except Exception as e:
        ax15.text(0.5, 0.5, f'Weight Error\n{str(e)}', ha='center', va='center', 
                 transform=ax15.transAxes, fontsize=10)
        ax15.set_title('注意力权重分布', fontweight='bold')
    
    # 频率维度分析
    ax16 = fig.add_subplot(gs[4, 3])
    try:
        if input_data.size > 0 and output_data.size > 0 and len(input_data.shape) == 2:
            freq_profile_input = input_data.mean(axis=1)
            freq_profile_output = output_data.mean(axis=1)
            
            mel_bins = np.arange(len(freq_profile_input))
            ax16.plot(mel_bins, freq_profile_input, 'b-', label='输入', linewidth=2)
            ax16.plot(mel_bins, freq_profile_output, 'r-', label='增强输出', linewidth=2)
            ax16.set_title('频率维度平均\n(肺音频率分析)', fontweight='bold')
            ax16.set_xlabel('Mel Bin')
            ax16.set_ylabel('平均幅度')
            ax16.legend()
            ax16.grid(True, alpha=0.3)
        else:
            ax16.text(0.5, 0.5, 'Invalid Data\nfor Frequency Analysis', ha='center', va='center', 
                     transform=ax16.transAxes, fontsize=12)
            ax16.set_title('频率维度平均\n(肺音频率分析)', fontweight='bold')
    except Exception as e:
        ax16.text(0.5, 0.5, f'Freq Error\n{str(e)}', ha='center', va='center', 
                 transform=ax16.transAxes, fontsize=10)
        ax16.set_title('频率维度平均\n(肺音频率分析)', fontweight='bold')
    
    # 时间维度分析
    ax17 = fig.add_subplot(gs[4, 4])
    try:
        if input_data.size > 0 and output_data.size > 0 and len(input_data.shape) == 2:
            time_profile_input = input_data.mean(axis=0)
            time_profile_output = output_data.mean(axis=0)
            
            time_frames = np.arange(len(time_profile_input))
            ax17.plot(time_frames, time_profile_input, 'b-', label='输入', linewidth=2)
            ax17.plot(time_frames, time_profile_output, 'r-', label='增强输出', linewidth=2)
            ax17.set_title('时间维度平均\n(呼吸周期分析)', fontweight='bold')
            ax17.set_xlabel('Time Frame')
            ax17.set_ylabel('平均幅度')
            ax17.legend()
            ax17.grid(True, alpha=0.3)
        else:
            ax17.text(0.5, 0.5, 'Invalid Data\nfor Time Analysis', ha='center', va='center', 
                     transform=ax17.transAxes, fontsize=12)
            ax17.set_title('时间维度平均\n(呼吸周期分析)', fontweight='bold')
    except Exception as e:
        ax17.text(0.5, 0.5, f'Time Error\n{str(e)}', ha='center', va='center', 
                 transform=ax17.transAxes, fontsize=10)
        ax17.set_title('时间维度平均\n(呼吸周期分析)', fontweight='bold')
    
    # 添加总标题
    fig.suptitle(f'真实ICBHI数据通过MSA模块的处理可视化\n文件: {filename}', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # 添加详细说明
    try:
        if input_data.size > 0 and output_data.size > 0:
            enhancement_ratio = output_data.mean() / input_data.mean() if input_data.mean() != 0 else 0
            weights_data = intermediates['attention_weights'].squeeze().numpy()
            weights_flat = weights_data.flatten() if weights_data.size > 0 else np.array([0])
            
            explanation = f"""
📋 处理流程说明:
🎵 音频文件: {filename}
📊 采样率: {sr}Hz, 音频长度: {audio.shape[-1]/sr:.1f}秒
🔄 Fbank特征: 使用icbhi_util.py的generate_fbank()函数
📐 特征维度: {fbank_2d.shape[0]}×{fbank_2d.shape[1]} (Mel bins × Time frames)
⚙️ MSA处理: 5个并行分支 + 特征融合 + 注意力加权

🏥 医学意义:
• 局部注意力: 捕获细粒度病理特征 (crackles, wheezes)
• 多尺度上下文: 分析不同时间跨度的呼吸模式  
• 全局注意力: 整体肺音特征建模
• 自适应增强: 突出重要的诊断相关特征

📈 性能指标:
• 平均增强倍数: {enhancement_ratio:.3f}
• 注意力权重范围: [{weights_flat.min():.3f}, {weights_flat.max():.3f}]
• 权重均值: {weights_flat.mean():.3f} ± {weights_flat.std():.3f}
"""
        else:
            explanation = f"""
📋 处理流程说明:
🎵 音频文件: {filename}
📊 采样率: {sr}Hz, 音频长度: {audio.shape[-1]/sr:.1f}秒
🔄 Fbank特征: 使用icbhi_util.py的generate_fbank()函数
📐 特征维度: {fbank_2d.shape[0]}×{fbank_2d.shape[1]} (Mel bins × Time frames)
⚙️ MSA处理: 5个并行分支 + 特征融合 + 注意力加权

🏥 医学意义:
• 局部注意力: 捕获细粒度病理特征 (crackles, wheezes)
• 多尺度上下文: 分析不同时间跨度的呼吸模式  
• 全局注意力: 整体肺音特征建模
• 自适应增强: 突出重要的诊断相关特征

📈 性能指标: 正在计算中...
"""
    except Exception as e:
        explanation = f"处理完成，但统计计算时出现问题: {str(e)}"
    
    fig.text(0.02, 0.02, explanation, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9),
            verticalalignment='bottom')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('icbhi_real_msa_visualization.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # 输出统计信息
    print("\n📊 MSA处理统计 (真实ICBHI数据):")
    print(f"📁 文件名: {filename}")
    print(f"🎵 原始音频: {audio.shape}")
    print(f"📊 Fbank特征: {fbank_2d.shape}")
    print(f"🔄 MSA输入: {fbank_tensor.shape}")
    print(f"✨ MSA输出: {intermediates['output'].shape}")
    
    try:
        if input_data.size > 0 and output_data.size > 0:
            enhancement_ratio = output_data.mean() / input_data.mean() if input_data.mean() != 0 else 0
            weights_data = intermediates['attention_weights'].squeeze().numpy()
            weights_flat = weights_data.flatten() if weights_data.size > 0 else np.array([0])
            
            print(f"📈 平均增强倍数: {enhancement_ratio:.3f}")
            print(f"🎯 注意力权重统计:")
            print(f"   - 范围: [{weights_flat.min():.3f}, {weights_flat.max():.3f}]")
            print(f"   - 均值: {weights_flat.mean():.3f}")
            print(f"   - 标准差: {weights_flat.std():.3f}")
        else:
            print("⚠️ 输出数据为空，无法计算统计信息")
    except Exception as e:
        print(f"❌ 统计计算出错: {e}")

def create_branch_comparison_real(specific_file=None):
    """创建真实数据的MSA分支对比图"""
    
    # 使用指定的文件
    if specific_file is None:
        specific_file = r"D:\bishe\data\ICBHI_final_database\icbhi_dataset\101_1b1_Al_sc_Meditron.wav"
    
    # 加载数据
    try:
        audio, sr, filename = load_real_icbhi_data(specific_file)
    except:
        audio, sr, filename = generate_simulated_icbhi_data()
    
    # 生成fbank特征
    fbank = generate_fbank(audio, sr, n_mels=128)
    fbank_2d = fbank.squeeze(-1)
    
    # 限制尺寸
    max_frames = 512
    if fbank_2d.shape[1] > max_frames:
        fbank_2d = fbank_2d[:, :max_frames]
    
    fbank_tensor = torch.from_numpy(fbank_2d).float().unsqueeze(0).unsqueeze(0)
    
    # MSA处理
    msa = MSA(channels=1, r=4)
    msa.eval()
    
    with torch.no_grad():
        intermediates = msa.forward_with_intermediates(fbank_tensor)
    
    # 创建对比图
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    features = [
        ('input', '原始Fbank输入'),
        ('local', '局部注意力\n(3×3卷积)'),
        ('global', '全局注意力\n(1×1池化)'),
        ('context1_resized', '上下文1\n(4×4→128×512)'),
        ('context2_resized', '上下文2\n(8×8→128×512)'),
        ('context3_resized', '上下文3\n(16×16→128×512)'),
        ('fused', '特征融合\n(所有分支相加)'),
        ('attention_weights', '注意力权重\n(Sigmoid激活)'),
        ('output', '最终输出\n(加权增强)')
    ]
    
    for idx, (feature_name, title) in enumerate(features):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        data = intermediates[feature_name].squeeze().numpy()
        
        # 选择colormap
        if 'context' in feature_name:
            cmap = 'Blues'
        elif feature_name == 'local':
            cmap = 'Reds'
        elif feature_name == 'global':
            cmap = 'Oranges'
        elif feature_name == 'attention_weights':
            cmap = 'RdYlBu'
        elif feature_name == 'fused':
            cmap = 'plasma'
        else:
            cmap = 'viridis'
        
        im = ax.imshow(data, aspect='auto', origin='lower', cmap=cmap, 
                      interpolation='bilinear')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Frames')
        ax.set_ylabel('Mel Bins')
        
        # 统计信息
        mean_val = data.mean()
        std_val = data.std()
        ax.text(0.02, 0.98, f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
               fontsize=9)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f'真实ICBHI数据MSA各分支对比\n文件: {filename}', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('icbhi_real_branch_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def main():
    """主函数"""
    print("🎨 开始生成真实ICBHI数据的MSA可视化...")
    
    # 指定您的ICBHI音频文件路径
    specific_file = r"D:\bishe\data\ICBHI_final_database\icbhi_dataset\101_1b1_Al_sc_Meditron.wav"
    
    print(f"📁 目标音频文件: {specific_file}")
    
    print("\n📊 1. 创建完整的MSA处理流程可视化...")
    create_icbhi_real_msa_visualization(specific_file)
    
    print("\n📊 2. 创建MSA各分支对比图...")
    create_branch_comparison_real(specific_file)
    
    print("\n✅ 所有可视化图像生成完成!")
    print("📁 生成的文件:")
    print("   - icbhi_real_msa_visualization.png (完整流程)")
    print("   - icbhi_real_branch_comparison.png (分支对比)")

if __name__ == "__main__":
    main()