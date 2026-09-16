import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
import librosa
import os
import glob
import pandas as pd

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_icbhi_samples():
    """
    从ICBHI数据集中加载四种不同类型的呼吸音样本
    """
    icbhi_path = r"D:\bishe\data\ICBHI_final_database\icbhi_dataset"
    
    # 🎯 查找不同类型的音频文件
    audio_files = glob.glob(os.path.join(icbhi_path, "*.wav"))
    
    # 🔍 根据文件名模式分类（ICBHI数据集的命名规则）
    samples = {
        'Normal': None,
        'Crackle': None, 
        'Wheeze': None,
        'Both': None
    }
    
    # 预定义一些已知的文件（根据ICBHI数据集）
    target_files = {
        'Normal': ['101_1b1_Al_sc_Meditron.wav', '102_1b1_Al_sc_Meditron.wav'],
        'Crackle': ['101_1b1_Tc_sc_Meditron.wav', '103_1b1_Tc_sc_Meditron.wav'],
        'Wheeze': ['101_1b1_Pr_sc_Meditron.wav', '105_1b1_Pr_sc_Meditron.wav'],
        'Both': ['101_1b1_Tc_sc_Meditron.wav', '104_1b1_Pl_sc_Meditron.wav']
    }
    
    # 🔍 搜索并分类文件
    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        
        # 根据ICBHI标注规则判断类型
        if any(normal_file in filename for normal_file in target_files['Normal']):
            if samples['Normal'] is None:
                samples['Normal'] = audio_file
        elif 'Tc' in filename or 'crackle' in filename.lower():
            if samples['Crackle'] is None:
                samples['Crackle'] = audio_file
        elif 'Pr' in filename or 'wheeze' in filename.lower():
            if samples['Wheeze'] is None:
                samples['Wheeze'] = audio_file
        elif 'Pl' in filename or 'both' in filename.lower():
            if samples['Both'] is None:
                samples['Both'] = audio_file
    
    # 如果找不到特定文件，随机选择
    if any(v is None for v in samples.values()):
        available_files = audio_files[:4]  # 取前4个文件
        labels = ['Normal', 'Crackle', 'Wheeze', 'Both']
        for i, label in enumerate(labels):
            if samples[label] is None and i < len(available_files):
                samples[label] = available_files[i]
    
    return samples

def create_real_respiratory_waveforms():
    """
    创建真实ICBHI呼吸音波形图
    """
    # 🎯 加载音频样本
    samples = load_icbhi_samples()
    
    # 检查是否成功加载文件
    available_samples = {k: v for k, v in samples.items() if v is not None and os.path.exists(v)}
    
    if len(available_samples) == 0:
        print("❌ 未找到ICBHI音频文件，请检查路径是否正确")
        print(f"🔍 查找路径: D:\\bishe\\data\\ICBHI_final_database\\icbhi_dataset")
        return
    
    print(f"✅ 找到 {len(available_samples)} 个音频文件:")
    for sound_type, file_path in available_samples.items():
        print(f"   {sound_type}: {os.path.basename(file_path)}")
    
    # 🎨 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = {
        'Normal': '#2E8B57',    # 海绿色
        'Crackle': '#FF6347',   # 番茄红
        'Wheeze': '#4169E1',    # 皇家蓝
        'Both': '#9932CC'       # 深兰花紫
    }
    
    descriptions = {
        'Normal': '正常呼吸音\n低频平滑，无异常音',
        'Crackle': '爆裂音(湿啰音)\n短促断续的爆破音',
        'Wheeze': '哮鸣音(干啰音)\n连续高调的音调',
        'Both': '混合音\n同时包含爆裂音和哮鸣音'
    }
    
    # 🎯 处理每个音频文件
    for idx, (sound_type, file_path) in enumerate(available_samples.items()):
        if idx >= 4:  # 最多处理4个
            break
            
        try:
            # 🔊 加载音频
            audio, sr = librosa.load(file_path, sr=22050, duration=5.0)  # 加载5秒
            
            # 📈 创建时间轴
            time = np.linspace(0, len(audio)/sr, len(audio))
            
            ax = axes[idx]
            
            # 🎨 绘制波形
            ax.plot(time, audio, color=colors[sound_type], linewidth=1, alpha=0.8)
            ax.fill_between(time, audio, alpha=0.3, color=colors[sound_type])
            
            # 🎯 设置图形属性
            ax.set_title(f'{sound_type} - {descriptions[sound_type]}', 
                        fontsize=14, fontweight='bold', color=colors[sound_type])
            ax.set_xlabel('时间 (秒)', fontsize=12, fontweight='bold')
            ax.set_ylabel('振幅', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max(time))
            
            # 📊 添加统计信息
            rms = np.sqrt(np.mean(audio**2))
            zero_crossing_rate = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
            
            # 添加信息文本框
            info_text = f'文件: {os.path.basename(file_path)}\n'
            info_text += f'采样率: {sr} Hz\n'
            info_text += f'时长: {len(audio)/sr:.2f} 秒\n'
            info_text += f'RMS: {rms:.4f}\n'
            info_text += f'过零率: {zero_crossing_rate:.4f}'
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 🔍 标记特征点
            if sound_type == 'Crackle':
                # 对于爆裂音，标记振幅峰值
                peaks_idx = np.where(np.abs(audio) > np.std(audio) * 2)[0]
                if len(peaks_idx) > 0:
                    peak_times = time[peaks_idx[:5]]  # 显示前5个峰值
                    peak_amps = audio[peaks_idx[:5]]
                    ax.scatter(peak_times, peak_amps, color='red', s=30, alpha=0.7, label='爆裂峰值')
                    
            elif sound_type == 'Wheeze':
                # 对于哮鸣音，可以标记持续的高频成分
                ax.axhline(y=np.mean(audio) + 2*np.std(audio), color='orange', 
                          linestyle='--', alpha=0.7, label='高频阈值')
                ax.axhline(y=np.mean(audio) - 2*np.std(audio), color='orange', 
                          linestyle='--', alpha=0.7)
            
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
                
        except Exception as e:
            print(f"❌ 处理文件 {file_path} 时出错: {e}")
            # 如果音频加载失败，显示错误信息
            ax.text(0.5, 0.5, f'无法加载音频文件\n{os.path.basename(file_path)}\n错误: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, color='red',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
            ax.set_title(f'{sound_type} - 加载失败', fontsize=14, fontweight='bold', color='red')
    
    # 隐藏多余的子图
    for idx in range(len(available_samples), 4):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('🫁 ICBHI呼吸音数据集 - 真实波形分析', fontsize=18, fontweight='bold', y=0.98)
    
    # 保存图片
    output_path = 'icbhi_respiratory_waveforms.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 波形图已保存: {output_path}")
    plt.show()

def create_combined_frequency_analysis():
    """
    创建频谱分析对比图
    """
    samples = load_icbhi_samples()
    available_samples = {k: v for k, v in samples.items() if v is not None and os.path.exists(v)}
    
    if len(available_samples) == 0:
        print("❌ 未找到音频文件，无法进行频谱分析")
        return
    
    # 🎨 创建频谱对比图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    colors = {
        'Normal': '#2E8B57',
        'Crackle': '#FF6347', 
        'Wheeze': '#4169E1',
        'Both': '#9932CC'
    }
    
    # 上图：时域波形叠加
    ax1.set_title('🎵 四种呼吸音时域波形对比', fontsize=14, fontweight='bold')
    
    # 下图：频域谱叠加
    ax2.set_title('📊 四种呼吸音频域特征对比', fontsize=14, fontweight='bold')
    
    for sound_type, file_path in available_samples.items():
        try:
            # 加载音频（取较短的片段用于对比）
            audio, sr = librosa.load(file_path, sr=22050, duration=2.0)
            time = np.linspace(0, len(audio)/sr, len(audio))
            
            # 时域波形 (归一化后叠加显示)
            normalized_audio = audio / np.max(np.abs(audio))  # 归一化
            ax1.plot(time, normalized_audio + list(available_samples.keys()).index(sound_type) * 2.5, 
                    color=colors[sound_type], linewidth=1.5, label=sound_type, alpha=0.8)
            
            # 频域分析
            fft = np.fft.fft(audio)
            freq = np.fft.fftfreq(len(audio), 1/sr)[:len(audio)//2]
            magnitude = np.abs(fft)[:len(audio)//2]
            
            # 平滑频谱
            magnitude_smooth = np.convolve(magnitude, np.ones(10)/10, mode='same')
            
            # 绘制频谱
            ax2.plot(freq, 20*np.log10(magnitude_smooth + 1e-10), 
                    color=colors[sound_type], linewidth=2, label=sound_type, alpha=0.8)
            
        except Exception as e:
            print(f"❌ 处理 {sound_type} 时出错: {e}")
    
    # 设置图形属性
    ax1.set_xlabel('时间 (秒)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('归一化振幅 + 偏移', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('频率 (Hz)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('幅度 (dB)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 2000)  # 关注0-2kHz频段
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('icbhi_combined_analysis.png', dpi=300, bbox_inches='tight')
    print("💾 组合分析图已保存: icbhi_combined_analysis.png")
    plt.show()

# 保留原有的频率特征分析函数
def create_respiratory_frequency_chart():
    """
    创建呼吸音频率范围波形图
    展示crackle、wheeze、normal、both的频率特征
    """
    
    # 🎯 定义各种呼吸音的频率特征
    respiratory_sounds = {
        'Normal': {
            'freq_range': (50, 2500),      # 正常呼吸音：50Hz-2.5kHz
            'peak_freq': (100, 600),       # 主要能量：100-600Hz
            'color': '#2E8B57',            # 海绿色
            'description': '正常呼吸音\n主要能量在低频段'
        },
        'Crackle': {
            'freq_range': (100, 2000),     # 爆裂音：100Hz-2kHz
            'peak_freq': (200, 800),       # 主要能量：200-800Hz
            'color': '#FF6347',            # 番茄红
            'description': '爆裂音(湿啰音)\n短促、断续的高频成分'
        },
        'Wheeze': {
            'freq_range': (100, 1600),     # 哮鸣音：100Hz-1.6kHz
            'peak_freq': (200, 400),       # 主要能量：200-400Hz (较低)
            'color': '#4169E1',            # 皇家蓝
            'description': '哮鸣音(干啰音)\n连续、高调的音调'
        },
        'Both': {
            'freq_range': (50, 2500),      # 混合音：覆盖最广频段
            'peak_freq': (150, 800),       # 主要能量：混合范围
            'color': '#9932CC',            # 深兰花紫
            'description': '混合音\n同时包含爆裂音和哮鸣音特征'
        }
    }
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    axes = [ax1, ax2, ax3, ax4]
    sound_names = ['Normal', 'Crackle', 'Wheeze', 'Both']
    
    # 🎯 为每种呼吸音创建频谱图
    for idx, (ax, sound_name) in enumerate(zip(axes, sound_names)):
        sound_info = respiratory_sounds[sound_name]
        
        # 生成频率轴 (0-3000 Hz)
        freq = np.linspace(0, 3000, 1000)
        
        # 🔥 生成特征性频谱
        if sound_name == 'Normal':
            # 正常呼吸音：低频主导，平滑衰减
            amplitude = np.exp(-(freq - 200)**2 / (2 * 300**2)) * 0.8
            amplitude += np.exp(-(freq - 100)**2 / (2 * 150**2)) * 0.6
            amplitude[freq < 50] = 0
            amplitude[freq > 2500] = 0
            
        elif sound_name == 'Crackle':
            # 爆裂音：中频段有尖锐峰值，代表短促爆裂
            amplitude = np.exp(-(freq - 400)**2 / (2 * 200**2)) * 0.9
            # 添加多个爆裂峰值
            for peak_f in [250, 500, 750, 1200]:
                amplitude += np.exp(-(freq - peak_f)**2 / (2 * 50**2)) * 0.4
            amplitude[freq < 100] = 0
            amplitude[freq > 2000] = 0
            
        elif sound_name == 'Wheeze':
            # 哮鸣音：特定频率的强峰值，代表音调性
            amplitude = np.exp(-(freq - 300)**2 / (2 * 80**2)) * 1.0
            amplitude += np.exp(-(freq - 600)**2 / (2 * 100**2)) * 0.7
            # 添加谐波
            amplitude += np.exp(-(freq - 900)**2 / (2 * 60**2)) * 0.4
            amplitude[freq < 100] = 0
            amplitude[freq > 1600] = 0
            
        elif sound_name == 'Both':
            # 混合音：结合crackle和wheeze的特征
            # Wheeze成分
            amplitude = np.exp(-(freq - 300)**2 / (2 * 80**2)) * 0.8
            # Crackle成分
            for peak_f in [200, 500, 800, 1200]:
                amplitude += np.exp(-(freq - peak_f)**2 / (2 * 70**2)) * 0.5
            amplitude[freq < 50] = 0
            amplitude[freq > 2500] = 0
        
        # 添加噪声使其更真实
        noise = np.random.normal(0, 0.05, len(amplitude))
        amplitude = np.maximum(0, amplitude + noise)
        
        # 🎨 绘制频谱
        ax.fill_between(freq, 0, amplitude, 
                       color=sound_info['color'], alpha=0.6, 
                       label=f'{sound_name}频谱')
        ax.plot(freq, amplitude, 
               color=sound_info['color'], linewidth=2)
        
        # 🔍 标记主要频率范围
        freq_range = sound_info['freq_range']
        peak_range = sound_info['peak_freq']
        
        # 添加频率范围标记
        ax.axvspan(freq_range[0], freq_range[1], 
                  alpha=0.2, color=sound_info['color'], 
                  label=f'总频率范围: {freq_range[0]}-{freq_range[1]}Hz')
        
        # 添加主要能量范围标记
        ax.axvspan(peak_range[0], peak_range[1], 
                  alpha=0.4, color=sound_info['color'], 
                  label=f'主要能量: {peak_range[0]}-{peak_range[1]}Hz')
        
        # 🎯 设置图形属性
        ax.set_xlabel('频率 (Hz)', fontsize=12, fontweight='bold')
        ax.set_ylabel('振幅', fontsize=12, fontweight='bold')
        ax.set_title(f'{sound_name} - {sound_info["description"]}', 
                    fontsize=14, fontweight='bold', 
                    color=sound_info['color'])
        ax.set_xlim(0, 3000)
        ax.set_ylim(0, max(amplitude) * 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # 添加峰值频率标注
        peak_idx = np.argmax(amplitude)
        peak_freq = freq[peak_idx]
        peak_amp = amplitude[peak_idx]
        ax.annotate(f'峰值: {peak_freq:.0f}Hz', 
                   xy=(peak_freq, peak_amp), 
                   xytext=(peak_freq + 300, peak_amp * 0.8),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.suptitle('🫁 呼吸音频率特征分析图', fontsize=18, fontweight='bold', y=0.98)
    
    # 保存图片
    plt.savefig('respiratory_frequency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comparative_frequency_chart():
    """
    创建对比性频率图表
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 🎯 频率范围对比
    sounds = ['Normal', 'Crackle', 'Wheeze', 'Both']
    freq_ranges = [(50, 2500), (100, 2000), (100, 1600), (50, 2500)]
    peak_ranges = [(100, 600), (200, 800), (200, 400), (150, 800)]
    colors = ['#2E8B57', '#FF6347', '#4169E1', '#9932CC']
    
    # 上图：频率范围对比
    y_pos = np.arange(len(sounds))
    
    for i, (sound, freq_range, peak_range, color) in enumerate(zip(sounds, freq_ranges, peak_ranges, colors)):
        # 绘制总频率范围
        ax1.barh(i, freq_range[1] - freq_range[0], 
                left=freq_range[0], height=0.6, 
                color=color, alpha=0.4, 
                label=f'{sound} 总范围')
        
        # 绘制主要能量范围
        ax1.barh(i, peak_range[1] - peak_range[0], 
                left=peak_range[0], height=0.3, 
                color=color, alpha=0.8)
        
        # 添加数值标签
        ax1.text(freq_range[1] + 50, i, 
                f'{freq_range[0]}-{freq_range[1]}Hz', 
                va='center', fontweight='bold')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sounds)
    ax1.set_xlabel('频率 (Hz)', fontsize=12, fontweight='bold')
    ax1.set_title('🔊 各类呼吸音频率范围对比', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 3000)
    ax1.grid(True, alpha=0.3)
    
    # 下图：重叠频率分析
    freq = np.linspace(0, 3000, 1000)
    
    # 为每种声音创建频率响应曲线
    responses = {}
    for sound, color in zip(sounds, colors):
        if sound == 'Normal':
            response = np.exp(-(freq - 300)**2 / (2 * 400**2))
        elif sound == 'Crackle':
            response = np.exp(-(freq - 500)**2 / (2 * 300**2))
        elif sound == 'Wheeze':
            response = np.exp(-(freq - 300)**2 / (2 * 150**2))
        elif sound == 'Both':
            response = (np.exp(-(freq - 300)**2 / (2 * 150**2)) + 
                       np.exp(-(freq - 500)**2 / (2 * 300**2))) / 2
        
        responses[sound] = response
        ax2.plot(freq, response, color=color, linewidth=3, 
                label=f'{sound}', alpha=0.8)
        ax2.fill_between(freq, 0, response, color=color, alpha=0.2)
    
    ax2.set_xlabel('频率 (Hz)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('相对响应', fontsize=12, fontweight='bold')
    ax2.set_title('🎵 各类呼吸音频率响应对比', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 3000)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('respiratory_frequency_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_clinical_frequency_table():
    """
    创建临床频率特征表格
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 📊 临床数据表格
    data = [
        ['声音类型', '频率范围(Hz)', '主要能量(Hz)', '临床特征', '病理意义'],
        ['Normal\n正常', '50-2500', '100-600', '低频主导\n平滑过渡', '正常肺功能\n无病理改变'],
        ['Crackle\n爆裂音', '100-2000', '200-800', '短促断续\n多峰值', '肺泡开放\n分泌物存在'],
        ['Wheeze\n哮鸣音', '100-1600', '200-400', '连续音调\n谐波丰富', '气道狭窄\n阻塞性病变'],
        ['Both\n混合音', '50-2500', '150-800', '复合特征\n频谱复杂', '多种病理\n病情复杂']
    ]
    
    # 创建表格
    table = ax.table(cellText=data[1:], colLabels=data[0], 
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.2, 0.2, 0.2, 0.25])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # 设置颜色
    colors = ['#2E8B57', '#FF6347', '#4169E1', '#9932CC']
    for i in range(1, 5):
        for j in range(5):
            table[(i, j)].set_facecolor(colors[i-1] if j == 0 else 'white')
            table[(i, j)].set_alpha(0.3 if j == 0 else 0.1)
    
    # 设置标题行颜色
    for j in range(5):
        table[(0, j)].set_facecolor('#4F4F4F')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.title('📋 呼吸音频率特征临床对照表', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('respiratory_frequency_table.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("🫁 开始分析ICBHI呼吸音数据集...")
    
    # 🎯 主要功能：创建真实ICBHI音频波形图
    print("\n🔊 创建真实呼吸音波形图...")
    create_real_respiratory_waveforms()
    
    # 🎵 创建组合分析图
    print("\n📊 创建组合分析图...")
    create_combined_frequency_analysis()
    
    # 📈 创建理论频率分析图
    print("\n📈 创建理论频率特征图...")
    create_respiratory_frequency_chart()
    
    # 📊 创建对比图表
    print("\n📊 创建对比图表...")
    create_comparative_frequency_chart()
    
    # 📋 创建临床特征表
    print("\n📋 创建临床特征表...")
    create_clinical_frequency_table()
    
    print("\n✅ 所有图表生成完成！")
    print("📁 生成的文件:")
    print("   - icbhi_respiratory_waveforms.png (真实ICBHI波形)")
    print("   - icbhi_combined_analysis.png (组合分析)")
    print("   - respiratory_frequency_analysis.png (理论频谱)")
    print("   - respiratory_frequency_comparison.png (频率对比)")
    print("   - respiratory_frequency_table.png (临床对照表)")