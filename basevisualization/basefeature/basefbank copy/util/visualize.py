import os
import math
import random
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

# 导入现有的工具函数 (请确保 icbhi_util.py 文件在相同目录下或Python路径中)
from icbhi_util import (
    get_annotations,
    get_individual_cycles_torchaudio,
    generate_fbank,
    get_feature_extractor,
    _get_base_feature_extractor,
    batch_extract_features
)

# 设置英文字体
def setup_english_fonts():
    """设置Times New Roman字体，并强制设定标题和基础字体大小"""
    import matplotlib.font_manager as fm

    print("🔧 配置Times New Roman字体...")
    plt.rcParams['font.family'] = 'serif'
    # 确保Times New Roman是列表中的第一个首选字体
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif', 'Arial'] 
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    plt.rcParams['axes.grid'] = False # 默认不显示网格
    plt.rcParams['axes.grid.which'] = 'both' # 针对主要和次要刻度
    plt.rcParams['grid.alpha'] = 0 # 网格透明度
    
    # ====== 强制设置字体大小：标题字号改为 24 ======
    plt.rcParams['font.size'] = 16 # 增大基础字体大小
    plt.rcParams['axes.titlesize'] = 24 # 强制设置 Axes 标题的字体大小为 24
    plt.rcParams['axes.labelsize'] = 14 # 轴标签字体大小
    plt.rcParams['xtick.labelsize'] = 12 # x轴刻度标签字体大小
    plt.rcParams['ytick.labelsize'] = 12 # y轴刻度标签字体大小
    plt.rcParams['legend.fontsize'] = 'large' # 图例字体大小
    plt.rcParams['figure.titlesize'] = 'x-large' # Figure 标题字体大小 (如果使用 fig.suptitle)
    # ===============================================

    print("✅ Times New Roman字体设置完成。")

    try:
        prop = fm.FontProperties(family='Times New Roman')
        return prop
    except Exception:
        prop = fm.FontProperties(family='serif')
        return prop

# 设置字体并获取字体属性
FONT_PROP = setup_english_fonts()

# 保持样式注释掉，如果需要seaborn的视觉效果，可以手动添加回来，但可能需要调整rcParams的顺序或再做优先级处理
# try:
#     plt.style.use('seaborn-v0_8') 
# except:
#     try:
#         plt.style.use('seaborn')
#     except:
#         pass
# sns.set_palette("husl")


# 创建模拟的args对象
class FeatureArgs:
    def __init__(self):
        self.sample_rate = 16000
        self.desired_length = 8
        self.pad_types = 'repeat'
        self.class_split = 'lungsound'

class ICBHIClassVisualizer:
    """ICBHI数据集分类可视化器 - 为指定目录中的音频文件生成Mel频带频谱图"""

    def __init__(self, data_folder, sample_rate=16000, desired_length=8, n_mels=64, freq_range=None):
        self.data_folder = data_folder
        self.sample_rate = sample_rate
        self.desired_length = desired_length
        self.n_mels = n_mels

        self.freq_range = freq_range if freq_range else (0, n_mels)
        self.font_prop = FONT_PROP # 保持这个属性，因为轴标签还在使用

        self.args = FeatureArgs()
        self.args.sample_rate = sample_rate
        self.args.desired_length = desired_length

        self.output_base_dir = os.path.join(os.getcwd(), "split_audio_spectrograms")
        os.makedirs(self.output_base_dir, exist_ok=True)

        # 用于统一特征值范围
        self.global_vmin = None
        self.global_vmax = None

        print('🎯 Initializing Classification Visualizer')
        print(f"   Data Folder: {data_folder}")
        print(f"   Target Sample Rate: {sample_rate} Hz")
        print(f"   Target Length: {desired_length} s")
        print(f"   Mel Bands: {n_mels}")
        print(f"   Frequency Range: {self.freq_range[0]}-{self.freq_range[1]}")
        print(f"   Output Directory: {self.output_base_dir}")

    def load_audio_file(self, audio_path):
        """加载音频文件"""
        try:
            audio_data, sr = librosa.load(audio_path, sr=self.sample_rate)
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
            target_samples = int(self.sample_rate * self.desired_length)
            if len(audio_tensor) > target_samples:
                audio_tensor = audio_tensor[:target_samples]
            elif len(audio_tensor) < target_samples:
                pad_length = target_samples - len(audio_tensor)
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_length), mode='constant', value=0)
            return audio_tensor
        except Exception as e:
            print(f"Failed to load audio file {audio_path}: {e}")
            return None

    def collect_split_audio_files(self):
        """收集split目录中的所有音频文件"""
        print("\n🔍 Scanning audio files in split directory...")
        if not os.path.exists(self.data_folder):
            print(f"❌ Directory does not exist: {self.data_folder}")
            return []
        audio_files = []
        for file in os.listdir(self.data_folder):
            if file.endswith('.wav'):
                audio_path = os.path.join(self.data_folder, file)
                audio_files.append((file, audio_path))
        audio_files.sort(key=lambda x: x[0])
        print(f"   Found {len(audio_files)} audio files")
        for file_name, _ in audio_files:
            print(f"   - {file_name}")
        return audio_files

    def calculate_global_range(self, audio_files):
        """计算所有音频文件的全局特征值范围"""
        print("\n📊 Calculating global feature range...")
        all_vmin = []
        all_vmax = []
        for file_name, audio_path in audio_files:
            audio_data = self.load_audio_file(audio_path)
            if audio_data is None:
                continue
            try:
                fbank_feature = generate_fbank(audio_data, self.sample_rate, self.n_mels)
                fbank_2d = fbank_feature[:, :, 0] if len(fbank_feature.shape) == 3 else fbank_feature
                freq_start, freq_end = self.freq_range
                fbank_display = fbank_2d[:, freq_start:freq_end]
                all_vmin.append(fbank_display.min())
                all_vmax.append(fbank_display.max())
            except Exception as e:
                print(f"   Error processing {file_name}: {e}")
                continue
        if all_vmin and all_vmax:
            self.global_vmin = min(all_vmin)
            self.global_vmax = max(all_vmax)
            print(f"   Global range: {self.global_vmin:.2f} to {self.global_vmax:.2f}")
        else:
            print("   Warning: Could not calculate global range")

    def generate_combined_spectrograms(self):
        """为split目录中的音频文件生成组合Mel频谱图"""
        print("\n" + "="*60)
        print('🔍 Generating combined Mel-band spectrograms for split directory audio files')
        print("="*60)

        audio_files = self.collect_split_audio_files()
        if not audio_files:
            print("❌ No audio files found")
            return

        self.calculate_global_range(audio_files)

        fig, axes = plt.subplots(2, 2, figsize=(20, 20)) 
        axes = axes.flatten() 

        for idx, (file_name, audio_path) in enumerate(audio_files[:4]): 
            print(f"\n📁 Processing file {idx+1}: {file_name}")
            audio_data = self.load_audio_file(audio_path)
            if audio_data is None:
                continue

            try:
                ax = axes[idx]
                freq_start, freq_end = self.freq_range
                fbank_feature = generate_fbank(audio_data, self.sample_rate, self.n_mels)
                fbank_2d = fbank_feature[:, :, 0] if len(fbank_feature.shape) == 3 else fbank_feature
                fbank_display = fbank_2d[:, freq_start:freq_end]

                vmin = self.global_vmin if self.global_vmin is not None else fbank_display.min()
                vmax = self.global_vmax if self.global_vmax is not None else fbank_display.max()
                
                im = ax.imshow(fbank_display.T, aspect='auto', origin='lower',
                                 cmap='viridis', interpolation='nearest',
                                 vmin=vmin, vmax=vmax)

                # 设置轴标签（英文），此处继续使用 fontproperties
                ax.set_xlabel('Time (s)', fontproperties=self.font_prop) 
                ax.set_ylabel('Mel Frequency Band', fontproperties=self.font_prop)

                # 设置y轴刻度
                y_range = freq_end - freq_start
                y_ticks = list(range(0, y_range, 8)) 
                if y_range == 64 and 63 not in y_ticks:
                    y_ticks.append(63)
                y_labels = [str(freq_start + t) for t in y_ticks]
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels, fontproperties=self.font_prop) 

                # 设置x轴刻度
                x_max = fbank_display.shape[0]
                time_total = self.desired_length
                time_intervals = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                x_ticks = [int(t * x_max / time_total) for t in time_intervals if t <= time_total]
                x_ticks = [min(tick, x_max-1) for tick in x_ticks]
                time_labels = [str(int(t)) for t in time_intervals if t <= time_total]
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(time_labels, fontproperties=self.font_prop) 

                # 设置标题（英文）
                file_name_clean = os.path.splitext(file_name)[0]  
                plot_title = f"Mel-band Spectrogram - {file_name_clean}"
                
                print(f"DEBUG: Before set_title - rcParams['axes.titlesize']: {plt.rcParams['axes.titlesize']}")
                print(f"DEBUG: Before set_title - rcParams['font.size']: {plt.rcParams['font.size']}")
                
                # ****** 最终修复处：标题字号调整为 24 ******
                title = ax.set_title(plot_title, fontsize=24, fontweight='bold', pad=20) 
                # *******************************************************

                print(f"DEBUG: After set_title - Title font size set for '{file_name_clean}': {title.get_fontsize()}")

                # 设置刻度字体 (这些通常也会被 font.size 或更具体的 rcParams 控制)
                for label in ax.get_xticklabels():
                    label.set_fontproperties(self.font_prop)
                for label in ax.get_yticklabels():
                    label.set_fontproperties(self.font_prop)

                # 为每个子图添加独立的颜色条
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Feature Value (dB)', fontsize=12, fontproperties=self.font_prop) 

                # 设置颜色条刻度
                cbar_ticks = np.linspace(vmin, vmax, 5)
                cbar.set_ticks(cbar_ticks)
                cbar.set_ticklabels([f'{tick:.1f}' for tick in cbar_ticks])
                
                # 设置颜色条刻度字体
                for label in cbar.ax.get_yticklabels():
                    label.set_fontproperties(self.font_prop)

                print(f"   ✅ Processed: {file_name}")

            except Exception as e:
                print(f"   ❌ Failed to generate spectrogram for {file_name}: {e}")
                continue

        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.92, wspace=0.3, hspace=0.4) 

        save_filename = "combined_mel_spectrograms.png"
        save_path = os.path.join(self.output_base_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"\n   ✅ Combined spectrogram saved: {save_filename}")

    def run_visualization(self):
        """运行可视化"""
        print("🚀 Starting combined split directory audio file spectrogram generation (Mel-band version)")
        print("="*60)

        try:
            self.generate_combined_spectrograms()

            print("\n" + "="*60)
            print(f"🎉 Combined spectrogram generation completed! File saved to:")
            print(f"   {self.output_base_dir}")
            print("="*60)

        except Exception as e:
            print(f"❌ Error during spectrogram generation: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """主函数 - 生成split目录中音频文件的组合Mel频带频谱图"""
    data_folder = r"D:\bishe\MVST-main\basevisualization\basefeature\basefbank\split\split"

    sample_rate = 16000
    desired_length = 8
    n_mels = 64  
    freq_range = (0, 64) 

    print("🎯 Combined split directory audio file spectrogram generation configuration (Mel-band version):")
    print(f"   Data folder: {data_folder}")
    print(f"   Target sample rate: {sample_rate} Hz")
    print(f"   Target duration: {desired_length} seconds")
    print(f"   Mel bands: {n_mels}")
    print(f"   Frequency display range: {freq_range[0]}-{freq_range[1]}")
    print("="*60)

    visualizer = ICBHIClassVisualizer(
        data_folder=data_folder,
        sample_rate=sample_rate,
        desired_length=desired_length,
        n_mels=n_mels,
        freq_range=freq_range
    )

    visualizer.run_visualization()

if __name__ == "__main__":
    main()