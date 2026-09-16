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

# 导入现有的工具函数（假设icbhi_util.py包含这些函数且在同一目录或PYTHONPATH中）
from icbhi_util import (
    get_annotations,
    get_individual_cycles_torchaudio,
    generate_fbank,
    get_feature_extractor,
    _get_base_feature_extractor,
    batch_extract_features
)

# 使用更简单的中文字体配置
def setup_chinese_fonts_simple():
    """使用通用的中文字体配置"""
    import matplotlib.font_manager as fm

    print("🔧 配置通用的中文字体...")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.grid.which'] = 'both'
    plt.rcParams['grid.alpha'] = 0
    print("✅ 通用中文字体设置完成。")

    try:
        font_name = plt.rcParams['font.sans-serif'][0]
        prop = fm.FontProperties(family=font_name)
        return prop
    except Exception:
        return None

# 设置字体并获取字体属性
FONT_PROP = setup_chinese_fonts_simple()

# 设置样式
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass

sns.set_palette("husl")

# 创建模拟的args对象
class FeatureArgs:
    def __init__(self):
        self.sample_rate = 16000
        self.desired_length = 8
        self.pad_types = 'repeat'
        self.class_split = 'lungsound'

class ICBHISpecificCombinationVisualizer:
    """ICBHI数据集特定音频组合可视化器"""

    def __init__(self, data_folder, sample_rate=16000, desired_length=8, n_mels=128, freq_range=None):
        self.data_folder = data_folder
        self.sample_rate = sample_rate
        self.desired_length = desired_length
        self.n_mels = n_mels

        self.freq_range = freq_range if freq_range else (0, n_mels)
        self.font_prop = FONT_PROP

        self.args = FeatureArgs()
        self.args.sample_rate = sample_rate
        self.args.desired_length = desired_length

        self.output_dir = os.path.join(os.getcwd(), "combination_output")
        os.makedirs(self.output_dir, exist_ok=True)

        print('🎯 初始化特定组合可视化器')
        print(f"   数据文件夹: {data_folder}")
        print(f"   目标采样率: {sample_rate} Hz")
        print(f"   目标长度: {desired_length} 秒")
        print(f"   Mel频带数: {n_mels}")
        print(f"   频率显示范围: {self.freq_range[0]}-{self.freq_range[1]}")
        print(f"   组合图将保存到: {self.output_dir}")

    def set_chinese_text(self, ax, x, y, text, **kwargs):
        """设置中文文本，使用指定字体"""
        if self.font_prop:
            kwargs['fontproperties'] = self.font_prop
        return ax.text(x, y, text, **kwargs)

    def set_chinese_label(self, ax, xlabel=None, ylabel=None, **kwargs):
        """设置中文轴标签，使用指定字体"""
        if self.font_prop:
            kwargs['fontproperties'] = self.font_prop

        if xlabel:
            ax.set_xlabel(xlabel, **kwargs)
        if ylabel:
            ax.set_ylabel(ylabel, **kwargs)

    def set_chinese_cbar_label(self, cbar, label, **kwargs):
        """设置中文颜色条标签，使用指定字体"""
        if self.font_prop:
            kwargs['fontproperties'] = self.font_prop
        cbar.set_label(label, **kwargs)
    
    def extract_and_group_samples_from_files(self, filenames):
        """从指定的音频文件中提取样本，并按类别分组，只保留第一个样本"""
        print("\n🔍 从指定文件中提取并分组样本...")

        class_names = {0: '正常', 1: '爆裂音', 2: '喘鸣音', 3: '混合音'}
        # 为了生成组合图，我们为每个类别只保留一个来自指定文件的样本
        # 如果一个指定文件有多个周期的某个类别，我们只取第一个
        # 如果多个指定文件有同一个类别，我们还是按照顺序取指定文件的第一个可用样本
        
        selected_samples = {} # 最终要用于绘图的每个类别的样本

        for audio_filename_base in filenames:
            audio_path = os.path.join(self.data_folder, audio_filename_base + '.wav')
            annotation_path = os.path.join(self.data_folder, audio_filename_base + '.txt')

            if not os.path.exists(audio_path):
                print(f"⚠️ 警告: 未找到音频文件: {audio_path}")
                continue
            if not os.path.exists(annotation_path):
                print(f"⚠️ 警告: 未找到标注文件: {annotation_path}")
                continue

            try:
                annotations = pd.read_csv(
                    annotation_path,
                    names=['start', 'end', 'crackles', 'wheezes'],
                    delimiter='\t'
                )

                cycles_data = get_individual_cycles_torchaudio(
                    self.args, annotations, self.data_folder, audio_filename_base,
                    self.sample_rate, n_cls=4
                )

                for audio_data, label in cycles_data:
                    if label not in selected_samples: # 如果该类别尚未被选中样本
                        selected_samples[label] = (audio_data, audio_filename_base)
                        # 一旦某个类别从当前文件找到了一个样本，就停止这个文件的该类别查找，继续下一个文件
                        # 这样确保了每个类别优先来自指定列表靠前的文件
                    
                # 如果所有4个类别都已找到样本，则可以提前停止扫描所有指定文件
                if len(selected_samples) == 4:
                    break

            except Exception as e:
                print(f"❌ 处理文件 {audio_filename_base} 失败: {e}")
                continue
        
        print("\n   📊 组合图样本提取结果:")
        for label, class_name in class_names.items():
            if label in selected_samples:
                print(f"   ✅ {class_name}: 来源于 {selected_samples[label][1]}.wav")
            else:
                print(f"   ❌ {class_name}: 未在指定文件中找到样本。")
        
        return selected_samples

    def visualize_specific_combination(self, specific_filenames):
        """生成特定音频文件的组合Mel频谱图"""
        print("\n" + "="*60)
        print('🔍 生成特定音频文件的组合Mel频带特征图')
        print("="*60)

        # 提取用于组合图的样本
        samples_for_combination = self.extract_and_group_samples_from_files(specific_filenames)
        class_names = {0: '正常', 1: '爆裂音', 2: '喘鸣音', 3: '混合音'}

        fig = plt.figure(figsize=(20, 6))
        gs = GridSpec(1, 4, figure=fig) # 固定为4个子图

        freq_start, freq_end = self.freq_range

        for i, label in enumerate(range(4)): # 遍历所有4个类别
            ax = fig.add_subplot(gs[0, i])
            feature_data_tuple = samples_for_combination.get(label) # (audio_data, filename)

            if feature_data_tuple is None:
                # 即使没有样本，也要显示分类标题
                self.set_chinese_text(ax, 0.5, 1.05, f"{class_names[label]}",
                                      transform=ax.transAxes, ha='center', va='bottom',
                                      fontsize=16, fontweight='bold')
                ax.text(0.5, 0.5, '无可用样本', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='gray', fontsize=12, fontproperties=self.font_prop)
                self.set_chinese_label(ax, xlabel='时间 (s)', ylabel='Mel频带', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            audio_data, source_file = feature_data_tuple
            
            try:
                fbank_feature = generate_fbank(audio_data, self.sample_rate, self.n_mels)
                fbank_2d = fbank_feature[:, :, 0] if len(fbank_feature.shape) == 3 else fbank_feature
                fbank_display = fbank_2d[:, freq_start:freq_end]

                im = ax.imshow(fbank_display.T, aspect='auto', origin='lower',
                                 cmap='viridis', interpolation='nearest')

                self.set_chinese_label(ax, xlabel='时间 (s)', ylabel='Mel频带', fontsize=12)

                y_range = freq_end - freq_start
                y_ticks_interval = max(1, y_range // 3)
                y_ticks = [0, y_ticks_interval, y_range-1] if y_range > 1 else [0]
                y_labels = [str(freq_start + t) for t in y_ticks]
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels)

                x_max = fbank_display.shape[0]
                time_total = self.desired_length
                x_ticks = [0, x_max//2, x_max-1] if x_max > 1 else [0]
                time_labels = ['0', f'{time_total/2:.1f}', f'{time_total:.0f}'] if x_max > 1 else ['0']
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(time_labels)

                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                self.set_chinese_cbar_label(cbar, '特征值 (dB)', fontsize=10)

                vmin, vmax = fbank_display.min(), fbank_display.max()
                cbar_ticks = np.linspace(vmin, vmax, 3)
                cbar.set_ticks(cbar_ticks)
                cbar.set_ticklabels([f'{tick:.1f}' for tick in cbar_ticks])

                # 只显示类别名称作为子图标题
                self.set_chinese_text(ax, 0.5, 1.05, f"{class_names[label]}",
                                     transform=ax.transAxes, ha='center', va='bottom',
                                     fontsize=16, fontweight='bold')

            except Exception as e:
                print(f"❌ 为类别 '{class_names[label]}' (源文件: {source_file}) 生成图失败: {e}")
                self.set_chinese_text(ax, 0.5, 1.05, f"{class_names[label]}",
                                      transform=ax.transAxes, ha='center', va='bottom',
                                      fontsize=16, fontweight='bold')
                ax.text(0.5, 0.5, '特征提取失败', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red', fontsize=12, fontproperties=self.font_prop)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.close(fig) # 确保关闭图表以避免内存泄漏
                continue

        # 整体标题反映特定文件的组合
        overall_title = "特定音频文件组合Mel频谱图"
        plt.suptitle(overall_title, fontsize=20, fontproperties=self.font_prop if self.font_prop else None)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(self.output_dir, 'specific_audio_combination.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print("   ✅ 特定音频组合图保存完成。")

    def run_visualization(self, specific_filenames):
        """运行特定音频组合可视化"""
        print("🚀 开始ICBHI特定音频组合可视化（Mel频带版本）")
        print("="*60)

        try:
            self.visualize_specific_combination(specific_filenames)

            print("\n" + "="*60)
            print(f"🎉 特定音频组合可视化完成！文件已保存到:")
            print(f"   {self.output_dir}")
            print(f"\n📂 生成的图命名为 'specific_audio_combination.png'")
            print("="*60)

        except Exception as e:
            print(f"❌ 特定音频组合可视化过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """主函数 - 生成特定音频组合的Mel频带频谱图"""
    data_folder = r"D:\bishe\data\ICBHI_final_database\icbhi_dataset"

    sample_rate = 16000
    desired_length = 8
    n_mels = 128

    freq_range = (0, 128)

    # 您指定的四个音频文件名（不含.wav后缀）
    specific_audios = [
        "101_1b1_Al_sc_Meditron",
        "104_1b1_Ll_sc_Litt3200",
        "131_1b1_Al_sc_Meditron", # 这个是重复的，但可以作为示例
        "141_1b2_Pr_mc_LittC2SE"
    ]

    print("🎯 特定音频组合可视化配置（Mel频带版本）:")
    print(f"   数据文件夹: {data_folder}")
    print(f"   目标采样率: {sample_rate} Hz")
    print(f"   目标时长: {desired_length} 秒")
    print(f"   Mel频带数: {n_mels}")
    print(f"   频率显示范围: {freq_range[0]}-{freq_range[1]}")
    print(f"   将为以下音频文件生成组合图: {specific_audios}")
    print("="*60)

    visualizer = ICBHISpecificCombinationVisualizer( # 更改了类名
        data_folder=data_folder,
        sample_rate=sample_rate,
        desired_length=desired_length,
        n_mels=n_mels,
        freq_range=freq_range
    )

    visualizer.run_visualization(specific_audios)

if __name__ == "__main__":
    main()