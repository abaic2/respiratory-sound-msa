import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import librosa
import torch
import torchaudio
from torchaudio import transforms as T
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 修改字体设置函数
def setup_chinese_fonts():
    """设置SimHei字体和白色背景"""
    import matplotlib.font_manager as fm
    import matplotlib
    
    print("🔧 配置SimHei字体和白色背景...")
    
    # 设置SimHei字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置白色背景
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.edgecolor'] = 'none'
    
    # 设置网格和轴的颜色
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.alpha'] = 0.3
    
    # 设置文本颜色
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    
    print("✅ 已设置SimHei字体和白色背景")
    
    # 验证字体设置
    try:
        font_list = [f.name for f in fm.fontManager.ttflist if 'SimHei' in f.name or 'sim' in f.name.lower()]
        if font_list:
            print(f"✅ 找到SimHei字体: {font_list[:3]}")
        else:
            print("⚠️ 未找到SimHei字体，将使用系统默认字体")
    except Exception as e:
        print(f"⚠️ 字体检查时出错: {e}")
    
    return True

# 设置字体和背景
FONT_SETUP = setup_chinese_fonts()

class ICBHISignalProcessingVisualizer:
    """ICBHI数据集信号处理可视化器"""
    
    def __init__(self, data_folder, sample_rate=16000, desired_length=8):
        self.data_folder = data_folder
        self.sample_rate = sample_rate
        self.desired_length = desired_length
        self.target_duration = desired_length * sample_rate
        
        self.output_dir = "/home/yujieyang/bishe/MVST-main/basevisualization/basepreprocessing/signal_processing_visualization"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print('🎯 初始化信号处理可视化器')
        print(f"   数据文件夹: {data_folder}")
        print(f"   目标采样率: {sample_rate} Hz")
        print(f"   目标长度: {desired_length} 秒")
        print(f"   输出目录: {self.output_dir}")

    def create_white_background_figure(self, figsize=(15, 10)):
        """创建白色背景的图形"""
        fig = plt.figure(figsize=figsize, facecolor='white')
        fig.patch.set_facecolor('white')
        return fig

    def setup_white_axes(self, ax):
        """设置坐标轴为白色背景，无网格线"""
        ax.set_facecolor('white')
        ax.grid(False)  # 关闭网格线
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')

    def load_annotations(self, filename):
        """加载音频文件的注释数据"""
        base_filename = filename.replace('.wav', '')
        annotation_file = os.path.join(self.data_folder, f"{base_filename}.txt")
        
        if not os.path.exists(annotation_file):
            print(f"⚠️ 注释文件不存在: {annotation_file}")
            return pd.DataFrame()
        
        try:
            # 读取注释文件
            annotations = pd.read_csv(annotation_file, sep='\t', header=None,
                                    names=['start', 'end', 'crackles', 'wheezes'])
            print(f"✅ 成功加载注释文件: {annotation_file}")
            print(f"   注释条目数: {len(annotations)}")
            return annotations
        except Exception as e:
            print(f"❌ 加载注释文件失败: {e}")
            return pd.DataFrame()

    def load_audio(self, filename):
        """加载音频文件"""
        audio_file = os.path.join(self.data_folder, filename)
        
        if not os.path.exists(audio_file):
            print(f"⚠️ 音频文件不存在: {audio_file}")
            return None, None
        
        try:
            # 使用torchaudio加载音频
            audio_data, sample_rate = torchaudio.load(audio_file)
            
            # 转为单声道
            if audio_data.shape[0] > 1:
                audio_data = audio_data[0:1, :]
            
            print(f"✅ 成功加载音频文件: {filename}")
            print(f"   原始采样率: {sample_rate} Hz")
            print(f"   音频长度: {audio_data.shape[1]} 样本 ({audio_data.shape[1]/sample_rate:.2f} 秒)")
            
            return audio_data.squeeze().numpy(), sample_rate
        except Exception as e:
            print(f"❌ 加载音频文件失败: {e}")
            return None, None

    def cut_pad_sample_enhanced(self, data):
        """增强版的裁剪和填充函数，支持重复填充"""
        if len(data) > self.target_duration:
            # 截断到目标长度
            return data[:self.target_duration], '截断'
        elif len(data) < self.target_duration:
            # 计算需要填充的长度
            padding_needed = self.target_duration - len(data)
            
            if len(data) == 0:
                # 如果原始数据为空，用零填充
                return np.zeros(self.target_duration), '零填充'
            
            # 重复填充策略
            repeat_count = padding_needed // len(data)
            remainder = padding_needed % len(data)
            
            # 构建重复填充的数据
            repeated_data = np.tile(data, repeat_count)
            if remainder > 0:
                repeated_data = np.concatenate([repeated_data, data[:remainder]])
            
            # 组合原始数据和重复数据
            padded_data = np.concatenate([data, repeated_data])
            
            # 在重复部分的边界应用淡化效果，避免突变
            fade_samples = min(int(0.01 * self.sample_rate), len(data) // 4)  # 10ms或原始长度的1/4
            if fade_samples > 0 and len(padded_data) > len(data) + fade_samples:
                # 在原始数据结束和重复数据开始处应用淡化
                fade_start = len(data)
                fade_end = fade_start + fade_samples
                fade_weights = np.linspace(1, 0.3, fade_samples)
                padded_data[fade_start:fade_end] *= fade_weights
            
            return padded_data, '重复填充'
        else:
            # 长度匹配
            return data, '长度匹配'

    def visualize_step1_file_structure(self, filename):
        """步骤1：文件结构和标注解析可视化"""
        print("\n" + "="*60)
        print('📁 步骤1：文件结构和标注解析')
        print("="*60)
        
        # 解析文件名
        tokens = filename.strip().split('_')
        while len(tokens) < 5:
            tokens.append('')
        tokens = tokens[:5]
        
        # 加载标注
        annotations = self.load_annotations(filename)
        
        # 创建可视化 - 白色背景
        fig = self.create_white_background_figure(figsize=(15, 10))
        gs = GridSpec(3, 2, height_ratios=[1, 1, 2], width_ratios=[1, 1])
        
        # 子图1：文件名解析
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        ax1.set_title('ICBHI文件名结构解析', fontsize=18, fontweight='bold', pad=20, color='black')
        
        labels = ['患者编号', '记录索引', '胸部位置', '获取模式', '录音设备']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        y_pos = 0.5
        x_positions = np.linspace(0.1, 0.9, len(tokens))
        
        for i, (token, label, color) in enumerate(zip(tokens, labels, colors)):
            rect = patches.FancyBboxPatch(
                (x_positions[i]-0.08, y_pos-0.15), 0.16, 0.3,
                boxstyle="round,pad=0.02", facecolor=color, alpha=0.7, edgecolor='black'
            )
            ax1.add_patch(rect)
            
            ax1.text(x_positions[i], y_pos+0.05, token, ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='black')
            ax1.text(x_positions[i], y_pos-0.25, label, ha='center', va='center', 
                    fontsize=12, color='darkblue')
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # 子图2：注释统计
        if not annotations.empty:
            ax2 = fig.add_subplot(gs[1, 0])
            self.setup_white_axes(ax2)
            
            # 统计标签分布
            label_counts = {
                'Normal': len(annotations[(annotations['crackles'] == 0) & (annotations['wheezes'] == 0)]),
                'Crackles': len(annotations[(annotations['crackles'] == 1) & (annotations['wheezes'] == 0)]),
                'Wheezes': len(annotations[(annotations['crackles'] == 0) & (annotations['wheezes'] == 1)]),
                'Both': len(annotations[(annotations['crackles'] == 1) & (annotations['wheezes'] == 1)])
            }
            
            colors = ['#2ECC71', '#E74C3C', '#F39C12', '#8E44AD']
            ax2.pie(label_counts.values(), labels=label_counts.keys(), autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax2.set_title('标签分布', fontsize=14, fontweight='bold', color='black')
        
        # 子图3：时间轴标注
        if not annotations.empty:
            ax3 = fig.add_subplot(gs[1, 1])
            self.setup_white_axes(ax3)
            
            durations = annotations['end'] - annotations['start']
            ax3.hist(durations, bins=10, color='lightblue', alpha=0.7, edgecolor='black')
            ax3.set_title('周期时长分布', fontsize=14, fontweight='bold', color='black')
            ax3.set_xlabel('时长 (秒)', fontsize=12, color='black')
            ax3.set_ylabel('频次', fontsize=12, color='black')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/step1_file_structure.png', 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()
        
        return annotations

    def visualize_step2_audio_preprocessing(self, filename):
        """步骤2：音频预处理可视化"""
        print("\n" + "="*60)
        print('🎵 步骤2：音频预处理')
        print("="*60)
        
        # 加载原始音频
        original_audio, original_sr = self.load_audio(filename)
        if original_audio is None:
            return None
        
        # 重采样
        if original_sr != self.sample_rate:
            resampler = T.Resample(original_sr, self.sample_rate)
            resampled_audio = resampler(torch.tensor(original_audio).unsqueeze(0)).squeeze().numpy()
            print(f"🔄 重采样: {original_sr} Hz → {self.sample_rate} Hz")
        else:
            resampled_audio = original_audio
            print(f"✅ 采样率已匹配: {self.sample_rate} Hz")
        
        # 应用Fade效果
        fade_samples = int(self.sample_rate / 16)  # fade长度
        fade_transform = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples)
        faded_audio = fade_transform(torch.tensor(resampled_audio).unsqueeze(0)).squeeze().numpy()
        
        # 创建可视化
        fig = self.create_white_background_figure(figsize=(15, 12))
        gs = GridSpec(3, 2, height_ratios=[1, 1, 1])
        
        # 原始音频波形
        ax1 = fig.add_subplot(gs[0, :])
        self.setup_white_axes(ax1)
        time_orig = np.arange(len(original_audio)) / original_sr
        ax1.plot(time_orig, original_audio, color='blue', alpha=0.7)
        ax1.set_title(f'原始音频波形 (采样率: {original_sr} Hz)', fontsize=14, fontweight='bold', color='black')
        ax1.set_xlabel('时间 (秒)', fontsize=12, color='black')
        ax1.set_ylabel('幅度', fontsize=12, color='black')
        
        # 重采样后音频波形
        ax2 = fig.add_subplot(gs[1, :])
        self.setup_white_axes(ax2)
        time_resamp = np.arange(len(resampled_audio)) / self.sample_rate
        ax2.plot(time_resamp, resampled_audio, color='green', alpha=0.7)
        ax2.set_title(f'重采样后音频波形 (采样率: {self.sample_rate} Hz)', fontsize=14, fontweight='bold', color='black')
        ax2.set_xlabel('时间 (秒)', fontsize=12, color='black')
        ax2.set_ylabel('幅度', fontsize=12, color='black')
        
        # Fade效果对比
        ax3 = fig.add_subplot(gs[2, :])
        self.setup_white_axes(ax3)
        time_fade = np.arange(len(faded_audio)) / self.sample_rate
        ax3.plot(time_fade, resampled_audio, color='orange', alpha=0.5, label='重采样后', linewidth=1)
        ax3.plot(time_fade, faded_audio, color='red', alpha=0.8, label='应用Fade后', linewidth=1.5)
        ax3.set_title('Fade效果对比', fontsize=14, fontweight='bold', color='black')
        ax3.set_xlabel('时间 (秒)', fontsize=12, color='black')
        ax3.set_ylabel('幅度', fontsize=12, color='black')
        ax3.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/step2_audio_preprocessing.png', 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()
        
        return faded_audio

    def visualize_step3_cycle_extraction(self, filename, preprocessed_audio):
        """步骤3：呼吸周期提取可视化"""
        print("\n" + "="*60)
        print('✂️ 步骤3：呼吸周期提取')
        print("="*60)
        
        annotations = self.load_annotations(filename)
        if annotations.empty or preprocessed_audio is None:
            return []
        
        # 提取所有周期
        cycles = []
        for idx, row in annotations.iterrows():
            start_sample = int(row['start'] * self.sample_rate)
            end_sample = int(row['end'] * self.sample_rate)
            
            # 边界检查
            start_sample = max(0, min(start_sample, len(preprocessed_audio)))
            end_sample = max(start_sample, min(end_sample, len(preprocessed_audio)))
            
            if end_sample > start_sample:
                cycle_data = preprocessed_audio[start_sample:end_sample]
                
                # 标签分类
                if row['crackles'] == 0 and row['wheezes'] == 0:
                    label = 'Normal'
                elif row['crackles'] == 1 and row['wheezes'] == 0:
                    label = 'Crackles'
                elif row['crackles'] == 0 and row['wheezes'] == 1:
                    label = 'Wheezes'
                else:
                    label = 'Both'
                
                cycles.append({
                    'data': cycle_data,
                    'start_time': row['start'],
                    'end_time': row['end'],
                    'duration': row['end'] - row['start'],
                    'label': label,
                    'index': idx
                })
        
        print(f"✅ 提取了 {len(cycles)} 个呼吸周期")
        
        # 可视化前4个周期
        fig = self.create_white_background_figure(figsize=(15, 12))
        
        colors_map = {'Normal': '#2ECC71', 'Crackles': '#E74C3C', 'Wheezes': '#F39C12', 'Both': '#8E44AD'}
        
        for i, cycle in enumerate(cycles[:4]):
            ax = fig.add_subplot(2, 2, i+1)
            self.setup_white_axes(ax)
            
            time = np.linspace(cycle['start_time'], cycle['end_time'], len(cycle['data']))
            color = colors_map.get(cycle['label'], 'blue')
            
            ax.plot(time, cycle['data'], color=color, linewidth=1.5)
            ax.set_title(f"周期{i+1}: {cycle['label']} ({cycle['duration']:.2f}s)", 
                        fontsize=12, fontweight='bold', color='black')
            ax.set_xlabel('时间 (秒)', fontsize=10, color='black')
            ax.set_ylabel('幅度', fontsize=10, color='black')
        
        plt.suptitle('提取的呼吸周期示例', fontsize=16, fontweight='bold', color='black')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/step3_cycle_extraction.png', 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()
        
        return cycles

    def visualize_step5_length_standardization(self, cycles):
        """步骤5：长度标准化可视化 - 优化版本"""
        print("\n" + "="*60)
        print('📏 步骤5：长度标准化')
        print("="*60)
        
        # 分析长度分布
        lengths = [len(cycle['data']) for cycle in cycles]
        durations = [cycle['duration'] for cycle in cycles]
        
        print(f"📊 原始长度统计:")
        print(f"   最短: {min(lengths)} 样本 ({min(durations):.2f} 秒)")
        print(f"   最长: {max(lengths)} 样本 ({max(durations):.2f} 秒)")
        print(f"   平均: {np.mean(lengths):.0f} 样本 ({np.mean(durations):.2f} 秒)")
        print(f"   标准差: {np.std(lengths):.0f} 样本 ({np.std(durations):.2f} 秒)")
        
        # 使用增强版标准化处理
        standardized_cycles = []
        truncated_count = 0
        padded_count = 0
        matched_count = 0
        
        for cycle in cycles:
            original_data = cycle['data']
            standardized_data, method = self.cut_pad_sample_enhanced(original_data)
            
            if method == '截断':
                truncated_count += 1
            elif method in ['重复填充', '零填充']:
                padded_count += 1
            else:
                matched_count += 1
            
            standardized_cycles.append({
                **cycle,
                'standardized_data': standardized_data,
                'original_length': len(original_data),
                'method': method
            })
        
        print(f"📊 标准化处理统计:")
        print(f"   截断: {truncated_count} 个周期")
        print(f"   填充: {padded_count} 个周期")
        print(f"   匹配: {matched_count} 个周期")
        
        # 创建可视化 - 白色背景
        fig = self.create_white_background_figure(figsize=(20, 18))
        gs = GridSpec(5, 2, height_ratios=[1, 1.5, 2, 2.5, 0.8], width_ratios=[1, 1])
        
        # 定义颜色 - 按照您的要求
        original_blue = '#1E3A8A'  # 深蓝色，用于原始信号
        filled_red = '#DC2626'     # 红色，用于填充后的信号
        
        # 修改 setup_white_axes 方法，去掉网格线
        def setup_clean_axes(ax):
            """设置坐标轴为白色背景，无网格线"""
            ax.set_facecolor('white')
            # 移除网格线
            ax.grid(False)
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black')
            ax.spines['right'].set_color('black')
            ax.spines['left'].set_color('black')
        
        # 子图1：长度分布直方图
        ax1 = fig.add_subplot(gs[0, 0])
        setup_clean_axes(ax1)
        n, bins, patches = ax1.hist(lengths, bins=15, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.axvline(self.target_duration, color='red', linestyle='--', linewidth=2, 
                   label=f'目标长度: {self.target_duration}')
        ax1.set_title('原始长度分布', fontsize=14, fontweight='bold', color='black')
        ax1.set_xlabel('样本数', fontsize=10, color='black')
        ax1.set_ylabel('频次', fontsize=10, color='black')
        ax1.legend(fontsize=10)
        
        # 子图2：处理方法统计
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('white')
        methods = ['截断', '填充', '匹配']
        counts = [truncated_count, padded_count, matched_count]
        colors = ['#E74C3C', '#3498DB', '#2ECC71']
        
        wedges, texts, autotexts = ax2.pie(counts, labels=methods, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax2.set_title('处理方法分布', fontsize=14, fontweight='bold', color='black')
        
        # 子图3：循环填充策略详细示例 - 无网格线，无标注遮挡
        ax3 = fig.add_subplot(gs[1, :])
        setup_clean_axes(ax3)
        ax3.set_title('循环填充策略详细示例', fontsize=16, fontweight='bold', pad=20, color='black')
        
        # 选择一个需要填充的周期作为示例
        example_cycle = None
        for cycle in standardized_cycles:
            if cycle['method'] == '重复填充' and cycle['original_length'] < self.target_duration:
                example_cycle = cycle
                break
        
        if example_cycle:
            # 创建详细的循环填充示例
            original_length = example_cycle['original_length']
            
            # 为了演示，创建一个较短的示例
            demo_original = example_cycle['data'][:min(800, original_length)]
            demo_target_length = 2400  # 演示目标长度
            
            # 手动执行循环填充过程以便可视化
            padding_needed = demo_target_length - len(demo_original)
            repeat_count = padding_needed // len(demo_original)
            remainder = padding_needed % len(demo_original)
            
            # 构建重复数据
            repeated_data = np.tile(demo_original, repeat_count)
            if remainder > 0:
                repeated_data = np.concatenate([repeated_data, demo_original[:remainder]])
            
            # 组合数据
            demo_padded = np.concatenate([demo_original, repeated_data])
            
            # 应用淡化效果
            fade_samples = min(int(0.01 * self.sample_rate), len(demo_original) // 4)
            if fade_samples > 0:
                fade_start = len(demo_original)
                fade_end = fade_start + fade_samples
                fade_weights = np.linspace(1, 0.3, fade_samples)
                demo_padded[fade_start:fade_end] *= fade_weights
            
            # 绘制示例 - 确保波形清晰可见，无任何遮挡
            x_orig = np.arange(len(demo_original))
            x_padded = np.arange(len(demo_padded))
            
            # 原始信号 - 深蓝色
            ax3.plot(x_orig, demo_original, color=original_blue, linewidth=3, 
                    alpha=0.9, zorder=3)
            
            # 完整的填充后信号 - 红色
            ax3.plot(x_padded, demo_padded, color=filled_red, linewidth=2, 
                    alpha=0.8, zorder=2)
            
            # 去掉所有可能遮挡波形的元素
            # 不添加边界线、淡化区域等
            
            # 设置坐标轴范围，确保波形完全可见
            ax3.set_xlim(0, len(demo_padded))
            
            # 计算y轴范围，留出足够空间
            y_min = np.min(demo_padded) * 1.3
            y_max = np.max(demo_padded) * 1.3
            ax3.set_ylim(y_min, y_max)
            
            ax3.set_xlabel('样本索引', fontsize=12, color='black')
            ax3.set_ylabel('幅度', fontsize=12, color='black')
            
            # 去掉右上角的图例标注
            # ax3.legend() 这行被注释掉
            
            # # 信息文本放在图外下方
            # info_text = f'原始长度: {len(demo_original)} 样本  |  目标长度: {demo_target_length} 样本  |  重复次数: {repeat_count}次 + {remainder}样本  |  淡化长度: {fade_samples} 样本'
            # fig.text(0.5, 0.72, info_text, ha='center', va='center', fontsize=11, 
            #         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9),
            #         color='black')
        
        # 子图4：循环填充过程分解图 - 无网格线
        ax4 = fig.add_subplot(gs[2, :])
        setup_clean_axes(ax4)
        ax4.set_title('循环填充过程分解', fontsize=16, fontweight='bold', pad=20, color='black')
        
        if example_cycle:
            # 创建更简单的分解示例
            simple_original = example_cycle['data'][:min(400, original_length)]
            
            # 显示原始信号和多个重复
            total_width = len(simple_original) * 4  # 显示4个周期
            x_base = np.arange(len(simple_original))
            
            # 计算合适的垂直偏移量，确保波形不重叠
            max_amp = np.max(np.abs(simple_original))
            offset_step = max_amp * 3.5
            
            # 原始信号 - 深蓝色
            ax4.plot(x_base, simple_original, color=original_blue, linewidth=3, 
                    alpha=0.9)
            
            # 重复信号1 - 红色
            x_repeat1 = x_base + len(simple_original)
            ax4.plot(x_repeat1, simple_original - offset_step, color=filled_red, linewidth=2.5, 
                    alpha=0.8, linestyle='--')
            
            # 重复信号2 - 红色
            x_repeat2 = x_base + len(simple_original) * 2
            ax4.plot(x_repeat2, simple_original - offset_step*2, color=filled_red, linewidth=2.5, 
                    alpha=0.7, linestyle=':')
            
            # 重复信号3 - 红色
            x_repeat3 = x_base + len(simple_original) * 3
            ax4.plot(x_repeat3, simple_original - offset_step*3, color=filled_red, linewidth=2.5, 
                    alpha=0.6, linestyle='-.')
            
            # 去掉连接箭头，避免遮挡波形
            
            # 设置坐标轴范围，确保所有内容可见
            ax4.set_xlim(0, total_width)
            ax4.set_ylim(-offset_step*4, offset_step)
            ax4.set_xlabel('样本索引', fontsize=12, color='black')
            ax4.set_ylabel('幅度 (垂直偏移)', fontsize=12, color='black')
            
            # 去掉图例
            # ax4.legend() 这行被注释掉
            
            # 过程说明放在图外
            process_text = '循环填充过程：1. 计算需要填充的长度  →  2. 重复原始信号直到达到目标长度  →  3. 在连接处应用淡化效果'
            fig.text(0.5, 0.47, process_text, ha='center', va='center', fontsize=11, 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                    color='black')
        
        # 子图5：标准化前后对比 - 无网格线，确保波形完全不被遮挡
        ax5 = fig.add_subplot(gs[3, :])
        setup_clean_axes(ax5)
        
        # 选择代表性周期进行展示
        example_indices = []
        methods_to_show = ['截断', '重复填充', '长度匹配']
        
        for method in methods_to_show:
            for i, cycle in enumerate(standardized_cycles):
                if cycle['method'] == method or (method == '重复填充' and '填充' in cycle['method']):
                    example_indices.append(i)
                    break
        
        example_indices = example_indices[:4]  # 最多显示4个
        
        colors_method = {'截断': '#E74C3C', '重复填充': filled_red, '零填充': filled_red, 
                        '长度匹配': '#2ECC71'}
        
        # 计算合适的垂直间距
        max_amplitude = 0
        for idx in example_indices:
            cycle = standardized_cycles[idx]
            max_amplitude = max(max_amplitude, np.max(np.abs(cycle['data'])), 
                              np.max(np.abs(cycle['standardized_data'])))
        
        # 设置足够大的垂直偏移，确保波形不重叠
        vertical_spacing = max_amplitude * 5
        
        for i, idx in enumerate(example_indices):
            cycle = standardized_cycles[idx]
            offset = i * vertical_spacing
            
            # 原始数据 - 深蓝色细线
            time_orig = np.linspace(0, cycle['duration'], cycle['original_length'])
            ax5.plot(time_orig, cycle['data'] + offset, 
                    color=original_blue, linewidth=2, alpha=0.7)
            
            # 标准化后数据 - 根据方法使用不同颜色
            time_std = np.linspace(0, self.desired_length, len(cycle['standardized_data']))
            method_color = colors_method.get(cycle['method'], filled_red)
            ax5.plot(time_std, cycle['standardized_data'] + offset, 
                    color=method_color, linewidth=2.5, alpha=0.9)
        
        # 设置标题和标签
        ax5.set_title('标准化前后对比 (各周期垂直偏移显示)', fontsize=16, fontweight='bold', color='black', pad=20)
        ax5.set_xlabel('时间 (秒)', fontsize=13, color='black')
        ax5.set_ylabel('幅度 (垂直偏移)', fontsize=13, color='black')
        
        # 设置坐标轴范围，确保所有波形可见且有足够边距
        ax5.set_xlim(0, self.desired_length)
        
        # 调整y轴范围，确保所有内容可见且不被遮挡
        y_min = -max_amplitude * 2
        y_max = (len(example_indices) - 1) * vertical_spacing + max_amplitude * 2.5
        ax5.set_ylim(y_min, y_max)
        
        # 去掉图例
        # ax5.legend() 这行被注释掉
        
        # 在图外添加周期和方法信息
        info_lines = []
        for i, idx in enumerate(example_indices):
            cycle = standardized_cycles[idx]
            length_info = f"周期{idx+1} ({cycle['method']}): {cycle['original_length']} → {len(cycle['standardized_data'])} 样本"
            info_lines.append(length_info)
        
        info_text = " | ".join(info_lines)
        fig.text(0.5, 0.18, info_text, ha='center', va='center', fontsize=11, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
                color='black')
        
        # 子图6：统计信息表格
        ax6 = fig.add_subplot(gs[4, :])
        ax6.axis('off')
        ax6.set_facecolor('white')
        
        stats_data = [
            ['处理前', f'{np.mean(lengths):.0f}', f'{min(lengths)}', f'{max(lengths)}', f'{np.std(lengths):.0f}'],
            ['处理后', f'{self.target_duration}', f'{self.target_duration}', f'{self.target_duration}', '0'],
            ['统计', f'截断: {truncated_count}', f'填充: {padded_count}', f'匹配: {matched_count}', f'总计: {len(cycles)}']
        ]
        
        table = ax6.table(cellText=stats_data,
                         colLabels=['阶段', '平均长度', '最小长度', '最大长度', '标准差/其他'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.1, 0.1, 0.8, 0.8])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # 设置表格颜色 - 白色背景
        for i in range(len(stats_data[0])):
            table[(0, i)].set_facecolor('#F0F0F0')  # 浅灰色表头
            table[(0, i)].set_text_props(weight='bold', color='black')
        
        for i in range(1, len(stats_data) + 1):
            for j in range(len(stats_data[0])):
                table[(i, j)].set_facecolor('white')
                table[(i, j)].set_text_props(color='black')
                if i == 2:  # 处理后行
                    table[(i, j)].set_facecolor('#E8F5E8')  # 浅绿色
                elif i == 3:  # 统计行
                    table[(i, j)].set_facecolor('#FFF8DC')  # 浅黄色
        
        plt.tight_layout(pad=4.0)
        plt.savefig(f'{self.output_dir}/step5_length_standardization.png', 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()
        
        return standardized_cycles

    def test_chinese_display(self):
        """测试中文显示效果"""
        print("🧪 测试中文字体显示...")
        
        fig = self.create_white_background_figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        
        ax.text(0.5, 0.5, '中文字体测试\n信号处理可视化\n呼吸周期分析', 
               ha='center', va='center', fontsize=16, color='black')
        ax.set_title('SimHei中文字体测试', fontsize=18, fontweight='bold', color='black')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        test_path = f'{self.output_dir}/chinese_test.png'
        plt.savefig(test_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ 中文测试图片已保存: {test_path}")

    def run_complete_visualization(self, target_file=None):
        """运行完整的可视化流程"""
        print("🚀 开始运行完整的信号处理可视化流程")
        print("="*60)
        
        # 如果没有指定文件，选择第一个可用的文件
        if target_file is None:
            files = [f for f in os.listdir(self.data_folder) if f.endswith('.wav')]
            if not files:
                print("❌ 数据文件夹中没有找到音频文件")
                return
            target_file = files[0]
            print(f"🎯 自动选择文件: {target_file}")
        
        try:
            # 步骤1：文件结构解析
            annotations = self.visualize_step1_file_structure(target_file)
            
            # 步骤2：音频预处理
            preprocessed_audio = self.visualize_step2_audio_preprocessing(target_file)
            
            # 步骤3：呼吸周期提取
            cycles = self.visualize_step3_cycle_extraction(target_file, preprocessed_audio)
            
            # 步骤5：长度标准化
            if cycles:
                standardized_cycles = self.visualize_step5_length_standardization(cycles)
                
                print("\n" + "="*60)
                print("🎉 完整的可视化流程已完成！")
                print(f"📁 所有图片已保存到: {self.output_dir}")
                print("="*60)
            else:
                print("⚠️ 没有提取到有效的呼吸周期，跳过长度标准化步骤")
                
        except Exception as e:
            print(f"❌ 运行过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数 - 在这里配置所有参数"""
    # 📁 数据路径配置
    data_folder = r"D:\bishe\data\ICBHI_final_database\icbhi_dataset"
    
    # 🎵 音频处理参数
    sample_rate = 16000      # 目标采样率
    desired_length = 8       # 目标时长（秒）
    
    target_file = "107_2b3_Pr_mc_AKGC417L.wav"
    
    print("🎯 文件选择配置:")
    print(f"   目标文件: {target_file if target_file else '默认（第一个文件）'}")
    print(f"   数据文件夹: {data_folder}")
    print(f"   目标采样率: {sample_rate} Hz")
    print(f"   目标时长: {desired_length} 秒")
    print("="*60)
    
    # 创建可视化器
    visualizer = ICBHISignalProcessingVisualizer(
        data_folder=data_folder,
        sample_rate=sample_rate,
        desired_length=desired_length
    )
    
    # 运行可视化
    visualizer.run_complete_visualization(target_file=target_file)

if __name__ == "__main__":
    main()