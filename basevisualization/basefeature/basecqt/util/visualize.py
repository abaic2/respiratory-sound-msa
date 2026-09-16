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

# 导入现有的工具函数
from icbhi_util import (
    get_annotations, 
    get_individual_cycles_torchaudio, 
    generate_fbank,
    get_score
)

# 使用指定的中文字体文件
def setup_chinese_fonts():
    """使用指定的中文字体文件"""
    import matplotlib.font_manager as fm
    import matplotlib
    
    print("🔧 配置指定的中文字体...")
    
    # 指定的字体文件路径
    wqy_font_path = "/home/yujieyang/.local/share/fonts/wqy-microhei.ttc"
    
    if os.path.exists(wqy_font_path):
        try:
            # 手动添加字体到matplotlib
            from matplotlib.font_manager import FontProperties, fontManager
            
            print(f"🔧 注册字体文件: {wqy_font_path}")
            fontManager.addfont(wqy_font_path)
            
            # 获取字体属性
            font_prop = FontProperties(fname=wqy_font_path)
            font_name = font_prop.get_name()
            print(f"✅ 字体注册成功，名称: {font_name}")
            
            # 设置为默认字体
            plt.rcParams['font.sans-serif'] = [font_name, 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 全局设置：去掉网格线
            plt.rcParams['axes.grid'] = False
            plt.rcParams['axes.grid.which'] = 'both'
            plt.rcParams['grid.alpha'] = 0
            
            print(f"✅ 使用字体: {font_name}")
            return wqy_font_path, font_prop
            
        except Exception as e:
            print(f"⚠️ 字体注册失败: {e}")
    else:
        print(f"❌ 字体文件不存在: {wqy_font_path}")
    
    # 备用设置
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.grid'] = False
    
    print("✅ 使用备用字体设置")
    return None, None

# 设置字体并获取字体属性
FONT_PATH, FONT_PROP = setup_chinese_fonts()

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

class ICBHIClassComparisonVisualizer:
    """ICBHI数据集分类对比可视化器 - CQT版本"""
    
    def __init__(self, data_folder, sample_rate=16000, desired_length=8, n_bins=84, 
                 bins_per_octave=12, fmin=50, fmax=8000, freq_range=None):
        self.data_folder = data_folder
        self.sample_rate = sample_rate
        self.desired_length = desired_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.fmin = fmin
        self.fmax = fmax
        
        # 频率索引范围控制
        self.freq_range = freq_range if freq_range else (0, n_bins)
        
        # 保存字体属性
        self.font_prop = FONT_PROP
        
        # 创建args对象
        self.args = FeatureArgs()
        self.args.sample_rate = sample_rate
        self.args.desired_length = desired_length
        
        self.output_dir = "/home/yujieyang/bishe/MVST-main/basevisualization/basefeature/basecqt/util/class_comparison_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print('🎯 初始化CQT分类对比可视化器')
        print(f"   数据文件夹: {data_folder}")
        print(f"   目标采样率: {sample_rate} Hz")
        print(f"   目标长度: {desired_length} 秒")
        print(f"   CQT频带数: {n_bins}")
        print(f"   每八度频带数: {bins_per_octave}")
        print(f"   频率范围: {fmin}-{fmax} Hz")
        print(f"   频率显示范围: {self.freq_range[0]}-{self.freq_range[1]}")
        print(f"   输出目录: {self.output_dir}")
    
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
    
    def load_annotations_flexible(self, annotation_path):
        """灵活加载标注文件，处理不同的列名格式"""
        try:
            # 首先检查文件的第一行，判断是否有标题行
            with open(annotation_path, 'r') as f:
                first_line = f.readline().strip()
            
            # 检查是否包含数字（数据行）或者文字（标题行）
            try:
                # 尝试将第一行按制表符分割并转换为浮点数
                parts = first_line.split('\t')
                float(parts[0])  # 如果第一个元素能转换为数字，说明没有标题行
                has_header = False
            except ValueError:
                # 如果不能转换为数字，说明有标题行
                has_header = True
            
            if has_header:
                # 有标题行，直接读取
                annotations = pd.read_csv(annotation_path, delimiter='\t')
                print(f"   📝 检测到标题行，列名: {list(annotations.columns)}")
                
                # 统一列名（转换为小写）
                annotations.columns = [col.lower() for col in annotations.columns]
                
                # 确保列名正确
                expected_cols = ['Start', 'End', 'Crackles', 'Wheezes']
                if not all(col in annotations.columns for col in expected_cols):
                    print(f"   ⚠️ 列名不匹配，当前列名: {list(annotations.columns)}")
                    # 尝试重命名列
                    if len(annotations.columns) == 4:
                        annotations.columns = expected_cols
                        print(f"   🔧 已重命名列为: {expected_cols}")
            else:
                # 没有标题行，手动指定列名
                annotations = pd.read_csv(
                    annotation_path, 
                    names=['Start', 'End', 'Crackles', 'Wheezes'], 
                    delimiter='\t'
                )
                print(f"   📝 无标题行，使用默认列名")
            
            return annotations
            
        except Exception as e:
            print(f"   ❌ 标注文件读取失败: {e}")
            return None
    
    def find_representative_samples_by_class(self):
        """自动从文件夹中为每种分类找到代表性样本"""
        print("\n🔍 自动搜索每种分类的代表性样本...")
        
        # 获取所有音频文件
        audio_files = [f for f in os.listdir(self.data_folder) if f.endswith('.wav')]
        audio_files.sort()
        
        # 为每种分类收集样本
        class_samples = {0: [], 1: [], 2: [], 3: []}  # 正常, 爆裂音, 喘鸣音, 混合音
        class_names = {0: '正常', 1: '爆裂音', 2: '喘鸣音', 3: '混合音'}
        
        print(f"   扫描 {len(audio_files)} 个音频文件...")
        
        # 扫描每个文件，提取其中的呼吸周期
        for i, audio_file in enumerate(audio_files[:50]):  # 增加扫描范围到50个文件
            filename = audio_file.replace('.wav', '')
            annotation_path = os.path.join(self.data_folder, filename + '.txt')
            
            if not os.path.exists(annotation_path):
                continue
                
            try:
                # 使用灵活的标注加载方法
                annotations = self.load_annotations_flexible(annotation_path)
                if annotations is None:
                    continue
                
                # 提取周期
                cycles_data = get_individual_cycles_torchaudio(
                    self.args, annotations, self.data_folder, filename, 
                    self.sample_rate, n_cls=4
                )
                
                # 按标签分类
                for audio_data, label in cycles_data:
                    if len(class_samples[label]) < 3:  # 每类最多收集3个样本
                        class_samples[label].append((audio_data, filename))
                
                # 检查是否每类都有足够样本
                if all(len(samples) >= 1 for samples in class_samples.values()):
                    print(f"   ✅ 已找到所有分类的样本，停止搜索（扫描了{i+1}个文件）")
                    break
                    
            except Exception as e:
                print(f"   ⚠️ 文件 {filename} 处理失败: {e}")
                continue
        
        # 报告结果
        print("   📊 分类样本搜索结果:")
        for label, samples in class_samples.items():
            if samples:
                print(f"   ✅ {class_names[label]}: 找到 {len(samples)} 个样本")
            else:
                print(f"   ❌ {class_names[label]}: 未找到样本")
        
        return class_samples
    
    def generate_cqt_feature(self, audio_data):
        """生成CQT特征"""
        try:
            # 确保音频数据是numpy数组
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.numpy()
            
            # 确保是1D数组
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            # 使用librosa生成CQT特征
            cqt = librosa.cqt(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=512,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave,
                fmin=self.fmin
            )
            
            # 转换为dB
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            
            # 转置以匹配显示格式 (time, freq)
            cqt_feature = cqt_db.T
            
            return cqt_feature
            
        except Exception as e:
            print(f"   ❌ CQT特征提取失败: {e}")
            return None
    
    def visualize_class_comparison(self):
        """生成分类对比可视化 - CQT版本"""
        print("\n" + "="*60)
        print('🔍 不同分类CQT特征对比 (自动搜索)')
        print("="*60)
        
        # 自动搜索每种分类的代表性样本
        class_samples = self.find_representative_samples_by_class()
        class_names = {0: '正常', 1: '爆裂音', 2: '喘鸣音', 3: '混合音'}
        
        # 为每个类别提取特征
        class_features = {}
        for label, samples in class_samples.items():
            if samples:
                # 选择第一个样本作为代表
                representative_audio, source_file = samples[0]
                try:
                    # 使用自定义的CQT特征提取
                    cqt_feature = self.generate_cqt_feature(representative_audio)
                    if cqt_feature is not None:
                        class_features[label] = (cqt_feature, source_file)
                        print(f"   ✅ {class_names[label]}: {cqt_feature.shape} (来源: {source_file})")
                    else:
                        class_features[label] = None
                except Exception as e:
                    print(f"   ❌ {class_names[label]} CQT特征提取失败: {e}")
                    class_features[label] = None
        
        # 计算有效分类
        valid_classes = [label for label, feat in class_features.items() if feat is not None]
        
        if len(valid_classes) == 0:
            print("⚠️ 没有成功提取的特征")
            return
        
        print("\n📊 生成CQT版本的分类对比图...")
        
        # 创建对比可视化 - CQT版本
        fig = plt.figure(figsize=(20, 6))
        gs = GridSpec(1, len(valid_classes), figure=fig)
        
        freq_start, freq_end = self.freq_range
        
        for i, label in enumerate(valid_classes):
            feature_data = class_features[label]
            if feature_data is None:
                continue
                
            feature, source_file = feature_data
            
            # 应用频率范围切片
            cqt_display = feature[:, freq_start:freq_end]
            
            # 主特征图
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(cqt_display.T, aspect='auto', origin='lower', 
                          cmap='viridis', interpolation='nearest')
            
            # 设置轴标签 - 使用中文字体
            self.set_chinese_label(ax, xlabel='时间 (s)', ylabel='CQT频带', fontsize=12)
            
            # 设置y轴刻度 - CQT频带
            y_range = freq_end - freq_start
            if y_range >= 60:
                y_ticks = [0, 30, y_range-1]
                y_labels = [str(freq_start), str(freq_start + 30), str(freq_end - 1)]
            elif y_range >= 40:
                y_ticks = [0, 20, y_range-1]
                y_labels = [str(freq_start), str(freq_start + 20), str(freq_end - 1)]
            else:
                mid = y_range // 2
                y_ticks = [0, mid, y_range-1]
                y_labels = [str(freq_start), str(freq_start + mid), str(freq_end - 1)]
            
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)
            
            # 设置x轴刻度 - 转换为时间秒数
            x_max = cqt_display.shape[0]
            time_total = self.desired_length  # 总时长8秒
            
            # 选择3个关键时间点
            x_ticks = [0, x_max//2, x_max-1]
            time_labels = ['0', f'{time_total/2:.1f}', f'{time_total:.0f}']
            
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(time_labels)
            
            # 添加颜色条，明确标注dB单位
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            self.set_chinese_cbar_label(cbar, '特征值 (dB)', fontsize=10)
            
            # 简化颜色条刻度 - 只显示3个值，保留1位小数
            vmin, vmax = cqt_display.min(), cqt_display.max()
            vmid = (vmin + vmax) / 2
            cbar_ticks = [vmin, vmid, vmax]
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels([f'{tick:.1f}' for tick in cbar_ticks])
            
            # 在图像上方添加分类标签 - 使用中文字体
            self.set_chinese_text(ax, 0.5, 1.05, class_names[label], 
                                transform=ax.transAxes, ha='center', va='bottom', 
                                fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # 为标题留出空间
        plt.savefig(f'{self.output_dir}/class_comparison_cqt.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("   ✅ CQT版本保存完成")
        
        return class_features
    
    def run_visualization(self):
        """运行分类对比可视化"""
        print("🚀 开始ICBHI分类对比可视化（CQT版本）")
        print("="*60)
        
        try:
            # 执行分类对比可视化
            class_features = self.visualize_class_comparison()
            
            print("\n" + "="*60)
            print(f"🎉 CQT分类对比可视化完成！文件已保存到:")
            print(f"   {self.output_dir}")
            print("\n📂 生成的文件:")
            print("   • class_comparison_cqt.png - CQT版本可视化")
            print("="*60)
            
        except Exception as e:
            print(f"❌ CQT分类对比可视化过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """主函数 - 生成CQT版本的分类对比可视化"""
    # 📁 数据路径配置
    data_folder = "/home/yujieyang/bishe/data/ICBHI_final_database/icbhi_dataset"
    
    # 🎵 特征提取参数
    sample_rate = 16000      # 目标采样率
    desired_length = 8       # 目标时长（秒）
    n_bins = 84             # CQT频带数
    bins_per_octave = 12    # 每八度频带数
    fmin = 50               # 最低频率
    fmax = 8000             # 最高频率
    
    # 🎯 频率索引范围控制
    freq_range = (0, 84)     # 显示全部CQT频带
    
    print("🎯 CQT分类对比可视化配置:")
    print(f"   数据文件夹: {data_folder}")
    print(f"   目标采样率: {sample_rate} Hz")
    print(f"   目标时长: {desired_length} 秒")
    print(f"   CQT频带数: {n_bins}")
    print(f"   每八度频带数: {bins_per_octave}")
    print(f"   频率范围: {fmin}-{fmax} Hz")
    print(f"   频率显示范围: {freq_range[0]}-{freq_range[1]}")
    print(f"   🔍 自动搜索各分类的代表性样本")
    print("="*60)
    
    # 创建分类对比可视化器
    visualizer = ICBHIClassComparisonVisualizer(
        data_folder=data_folder,
        sample_rate=sample_rate,
        desired_length=desired_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
        fmax=fmax,
        freq_range=freq_range
    )
    
    # 运行可视化
    visualizer.run_visualization()

if __name__ == "__main__":
    main()