from collections import namedtuple
import os
import math
import random
from tkinter import W
import pandas as pd
import numpy as np
from tqdm import tqdm

import cv2
import cmapy
import librosa
import torch
import torchaudio
from torchaudio import transforms as T
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from augmentation import augment_raw_audio
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

__all__ = ['get_annotations', 'get_individual_cycles_torchaudio', 'generate_fbank', 'get_score']


# ==========================================================================
""" ICBHI dataset information """
def _extract_lungsound_annotation(file_name, data_folder):
    tokens = file_name.strip().split('_')
    
    # 添加健壮性处理
    while len(tokens) < 5:
        tokens.append('')
    tokens = tokens[:5]  # 确保不会超过5个元素
    
    recording_info = pd.DataFrame(data=[tokens], columns=['Patient Number', 'Recording index', 'Chest location', 'Acquisition mode', 'Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(data_folder, file_name + '.txt'), names=['Start', 'End', 'Crackles', 'Wheezes'], delimiter='\t')

    return recording_info, recording_annotations


def get_annotations(args, data_folder):
    if args.class_split == 'lungsound' or args.class_split in ['lungsound_meta', 'meta']:
        filenames = [f.strip().split('.')[0] for f in os.listdir(data_folder) if '.txt' in f]

        annotation_dict = {}
        for f in filenames:
            info, ann = _extract_lungsound_annotation(f, data_folder)
            annotation_dict[f] = ann

    elif args.class_split == 'diagnosis':
        filenames = [f.strip().split('.')[0] for f in os.listdir(data_folder) if '.txt' in f]
        tmp = pd.read_csv(os.path.join(args.data_folder, 'icbhi_dataset/bing.txt'), names=['Disease'], delimiter='\t')

        annotation_dict = {}
        for f in filenames:
            info, ann = _extract_lungsound_annotation(f, data_folder)
            ann.drop(['Crackles', 'Wheezes'], axis=1, inplace=True)

            disease = tmp.loc[int(f.strip().split('_')[0]), 'Disease']
            ann['Disease'] = disease

            annotation_dict[f] = ann
            
    return annotation_dict

def _get_lungsound_label(crackle, wheeze, n_cls):
    if n_cls == 4:
        if crackle == 0 and wheeze == 0:
            return 0
        elif crackle == 1 and wheeze == 0:
            return 1
        elif crackle == 0 and wheeze == 1:
            return 2
        elif crackle == 1 and wheeze == 1:
            return 3
    
    elif n_cls == 2:
        if crackle == 0 and wheeze == 0:
            return 0
        else:
            return 1


def _get_diagnosis_label(disease, n_cls):
    if n_cls == 3:
        if disease in ['COPD', 'Bronchiectasis', 'Asthma']:
            return 1
        elif disease in ['URTI', 'LRTI', 'Pneumonia', 'Bronchiolitis']:
            return 2
        else:
            return 0

    elif n_cls == 2:
        if disease == 'Healthy':
            return 0
        else:
            return 1

def _slice_data_torchaudio(start, end, data, sample_rate):
    """
    SCL paper..
    sample_rate denotes how many sample points for one second
    """
    max_ind = data.shape[1]
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)

    return data[:, start_ind: end_ind]


def cut_pad_sample_torchaudio(data, args):
    fade_samples_ratio = 16
    fade_samples = int(args.sample_rate / fade_samples_ratio)
    fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
    target_duration = args.desired_length * args.sample_rate

    if data.shape[-1] > target_duration:
        data = data[..., :target_duration]
    else:
        if args.pad_types == 'zero':
            tmp = torch.zeros(1, target_duration, dtype=torch.float32)
            diff = target_duration - data.shape[-1]
            tmp[..., diff//2:data.shape[-1]+diff//2] = data
            data = tmp
        elif args.pad_types == 'repeat':
            ratio = math.ceil(target_duration / data.shape[-1])
            data = data.repeat(1, ratio)
            data = data[..., :target_duration]
            data = fade_out(data)
    
    return data

def get_individual_cycles_torchaudio(args, recording_annotations, data_folder, filename, sample_rate, n_cls):
    """
    SCL paper..
    used to split each individual sound file into separate sound clips containing one respiratory cycle each
    output: [(audio_chunk:np.array, label:int), (...)]
    """
    sample_data = []
    fpath = os.path.join(data_folder, filename+'.wav')
        
    sr = librosa.get_samplerate(fpath)
    data, _ = torchaudio.load(fpath)
    
    if sr != sample_rate:
        resample = T.Resample(sr, sample_rate)
        data = resample(data)

    fade_samples_ratio = 16
    fade_samples = int(sample_rate / fade_samples_ratio)

    fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')

    data = fade(data)
    for idx in recording_annotations.index:
        row = recording_annotations.loc[idx]

        start = row['Start'] # time (second)
        end = row['End'] # time (second)
        audio_chunk = _slice_data_torchaudio(start, end, data, sample_rate)

        if args.class_split == 'lungsound':
            crackles = row['Crackles']
            wheezes = row['Wheezes']            
            sample_data.append((audio_chunk, _get_lungsound_label(crackles, wheezes, n_cls)))
        elif args.class_split == 'diagnosis':
            disease = row['Disease']            
            sample_data.append((audio_chunk, _get_diagnosis_label(disease, n_cls)))

    padded_sample_data = []
    for data, label in sample_data:
        data = cut_pad_sample_torchaudio(data, args)
        padded_sample_data.append((data, label))

    return padded_sample_data


def generate_fbank(waveform, sample_rate, n_fft=1024, hop_length=512, n_mels=128, f_min=50, f_max=2000):
    """
    生成频谱图特征（Spectrogram）
    
    参数:
        waveform: 输入音频波形 (numpy数组或torch张量)
        sample_rate: 采样率
        n_fft: FFT窗口大小
        hop_length: 帧移
        n_mels: 梅尔滤波器数量（如果需要梅尔频谱图）
        f_min: 最小频率
        f_max: 最大频率
    
    返回:
        频谱图特征，形状为 (时间, 频率, 1)
    """
    import librosa
    import numpy as np
    
    try:
        # 转换为numpy数组
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        
        # 确保是一维数组
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
        
        # 检查是否有有效数据
        if len(waveform) == 0:
            raise ValueError("Empty waveform")
        
        # 计算短时傅里叶变换(STFT)
        S = librosa.stft(
            y=waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window='hann',
            center=True
        )
        
        # 转换为幅度谱并取对数
        magnitude_spec = np.abs(S)
        log_spec = librosa.amplitude_to_db(magnitude_spec, ref=np.max)
        
        # 可选：使用梅尔频谱图
        # mel_spec = librosa.feature.melspectrogram(
        #     y=waveform,
        #     sr=sample_rate,
        #     n_fft=n_fft,
        #     hop_length=hop_length,
        #     n_mels=n_mels,
        #     fmin=f_min,
        #     fmax=f_max
        # )
        # log_spec = librosa.power_to_db(mel_spec)
        
        # 标准化频谱图（可选）
        # log_spec = (log_spec - np.mean(log_spec)) / (np.std(log_spec) + 1e-8)
        
        # 转置并添加通道维度
        log_spec = log_spec.T  # (频率, 时间) -> (时间, 频率)
        log_spec = np.expand_dims(log_spec, axis=2)  # (时间, 频率) -> (时间, 频率, 1)
        
        return log_spec
        
    except Exception as e:
        print(f"生成频谱图特征时出错: {str(e)}")
        # 返回空数组作为后备
        freq_bins = n_fft // 2 + 1
        return np.zeros((128, freq_bins, 1))

# ==========================================================================
""" evaluation metric """
def get_score(hits, counts, pflag=False):
    # normal accuracy
    sp = hits[0] / (counts[0] + 1e-10) * 100
    # abnormal accuracy
    se = sum(hits[1:]) / (sum(counts[1:]) + 1e-10) * 100
    sc = (sp + se) / 2.0

    if pflag:
        # print("************* Metrics ******************")
        print("S_p: {}, S_e: {}, Score: {}".format(sp, se, sc))

    return sp, se, sc
# ==========================================================================


# 在文件末尾的测试代码部分替换为以下内容
if __name__ == "__main__":
    import argparse
    import sys
    
    print("测试频谱图 (Spectrogram) 特征提取...")
    
    # 创建简单的参数解析器用于测试
    parser = argparse.ArgumentParser(description='频谱图特征提取测试')
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'single'],
                        help='auto: 自动选择四种分类各一个样本, single: 指定单个文件')
    parser.add_argument('--filename', type=str, default='101_1b1_Al_sc_Meditron',
                        help='单个文件模式下的文件名（不含扩展名）')
    parser.add_argument('--data_folder', type=str, 
                        default='/home/yujieyang/bishe/data/ICBHI_final_database/icbhi_dataset',
                        help='ICBHI数据集路径')
    
    # 解析命令行参数或使用默认值
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        # 默认参数
        class Args:
            mode = 'auto'
            filename = '101_1b1_Al_sc_Meditron'
            data_folder = '/home/yujieyang/bishe/data/ICBHI_final_database/icbhi_dataset'
            class_split = 'lungsound'
            n_cls = 4
            sample_rate = 16000
            desired_length = 8
            pad_types = 'repeat'
        args = Args()
    
    # 添加必要的参数
    if not hasattr(args, 'class_split'):
        args.class_split = 'lungsound'
    if not hasattr(args, 'n_cls'):
        args.n_cls = 4
    if not hasattr(args, 'sample_rate'):
        args.sample_rate = 16000
    if not hasattr(args, 'desired_length'):
        args.desired_length = 8
    if not hasattr(args, 'pad_types'):
        args.pad_types = 'repeat'
    
    # 检查数据文件夹是否存在
    if not os.path.exists(args.data_folder):
        print(f"错误：数据文件夹不存在 {args.data_folder}")
        sys.exit(1)
    
    try:
        # 获取注释信息
        annotation_dict = get_annotations(args, args.data_folder)
        print(f"找到 {len(annotation_dict)} 个音频文件")
        
        # 频谱图参数设置
        n_fft = 1024
        hop_length = 512
        n_mels = 128
        f_min = 50
        f_max = 2000
        
        if args.mode == 'single':
            # 单文件模式
            print(f"处理单个文件: {args.filename}")
            
            if args.filename not in annotation_dict:
                print(f"错误：文件 {args.filename} 不存在于数据集中")
                available_files = list(annotation_dict.keys())[:10]
                print(f"可用文件示例: {available_files}")
                sys.exit(1)
            
            # 获取音频周期数据
            cycles = get_individual_cycles_torchaudio(
                args, annotation_dict[args.filename], 
                args.data_folder, args.filename, 
                args.sample_rate, args.n_cls
            )
            
            if not cycles:
                print(f"文件 {args.filename} 中没有有效的音频周期")
                sys.exit(1)
            
            # 选择第一个周期进行可视化
            audio_data, label = cycles[0]
            waveform = audio_data.squeeze().numpy() if hasattr(audio_data, 'numpy') else audio_data.squeeze()
            
            print(f"文件: {args.filename}")
            print(f"标签: {label}")
            print(f"波形形状: {waveform.shape}")
            
            # 提取频谱图特征
            spec_features = generate_fbank(waveform, args.sample_rate, 
                                         n_fft=n_fft, hop_length=hop_length, 
                                         n_mels=n_mels, f_min=f_min, f_max=f_max)
            
            # 可视化
            visualize_single_sample(waveform, spec_features, args.filename, label, args.sample_rate)
            
        elif args.mode == 'auto':
            # 自动模式：四种分类各选一个
            print("自动选择四种分类的样本进行可视化...")
            
            # 收集每种类别的样本
            class_samples = {0: [], 1: [], 2: [], 3: []}  # 正常、爆裂音、哮鸣音、两者都有
            
            for filename, annotations in annotation_dict.items():
                try:
                    cycles = get_individual_cycles_torchaudio(
                        args, annotations, args.data_folder, filename, 
                        args.sample_rate, args.n_cls
                    )
                    
                    for audio_data, label in cycles:
                        if label in class_samples and len(class_samples[label]) < 5:  # 每类最多收集5个
                            waveform = audio_data.squeeze().numpy() if hasattr(audio_data, 'numpy') else audio_data.squeeze()
                            class_samples[label].append((waveform, filename, label))
                            
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")
                    continue
            
            # 检查是否所有类别都有样本
            class_names = ['正常', '爆裂音', '哮鸣音', '两者都有']
            selected_samples = []
            
            for class_id in range(4):
                if class_samples[class_id]:
                    # 选择第一个样本
                    waveform, filename, label = class_samples[class_id][0]
                    selected_samples.append((waveform, filename, label, class_names[class_id]))
                    print(f"类别 {class_id} ({class_names[class_id]}): 文件 {filename}, 样本数: {len(class_samples[class_id])}")
                else:
                    print(f"警告：未找到类别 {class_id} ({class_names[class_id]}) 的样本")
            
            if not selected_samples:
                print("错误：未找到任何有效样本")
                sys.exit(1)
            
            # 为每个类别提取频谱图特征并可视化
            for waveform, filename, label, class_name in selected_samples:
                print(f"\n处理类别: {class_name} (标签: {label})")
                print(f"文件: {filename}")
                print(f"波形形状: {waveform.shape}")
                
                spec_features = generate_fbank(waveform, args.sample_rate, 
                                             n_fft=n_fft, hop_length=hop_length, 
                                             n_mels=n_mels, f_min=f_min, f_max=f_max)
                
                # 可视化单个样本
                visualize_single_sample(waveform, spec_features, 
                                      f"{filename}_{class_name}", label, 
                                      args.sample_rate)
            
            # 创建四类对比图
            visualize_four_classes(selected_samples, args.sample_rate, n_fft, hop_length, n_mels, f_min, f_max)
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def visualize_single_sample(waveform, spec_features, filename, label, sample_rate):
    """可视化单个样本"""
    try:
        import matplotlib.pyplot as plt
        
        # 创建时间轴
        duration = len(waveform) / sample_rate
        t = np.linspace(0, duration, len(waveform))
        
        # 生成其他特征用于对比
        S = librosa.stft(waveform, n_fft=1024, hop_length=512)
        linear_spec = np.abs(S)
        log_spec = librosa.amplitude_to_db(linear_spec, ref=np.max)
        
        # 生成梅尔频谱图
        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128)
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        # 生成功率谱密度
        f, psd = librosa.core.piptrack(y=waveform, sr=sample_rate)
        
        # 保存路径
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                f'spectrogram_single_{filename}_label{label}.png')
        
        plt.figure(figsize=(15, 20))
        
        # 波形图
        plt.subplot(5, 1, 1)
        plt.title(f'Waveform - {filename} (Label: {label})')
        plt.plot(t, waveform)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # 频谱图特征（我们的主要特征）
        plt.subplot(5, 1, 2)
        plt.title('Log-Magnitude Spectrogram (Main Feature)')
        im1 = plt.imshow(spec_features[:, :, 0].T, aspect='auto', origin='lower', cmap='viridis')
        plt.xlabel('Time Frames')
        plt.ylabel('Frequency Bins')
        plt.colorbar(im1, format='%+2.0f dB')
        
        # 线性幅度频谱图对比
        plt.subplot(5, 1, 3)
        plt.title('Linear Magnitude Spectrogram (for comparison)')
        im2 = plt.imshow(linear_spec, aspect='auto', origin='lower', cmap='viridis')
        plt.xlabel('Time Frames')
        plt.ylabel('Frequency Bins')
        plt.colorbar(im2)
        
        # 梅尔频谱图对比
        plt.subplot(5, 1, 4)
        plt.title('Log-Mel Spectrogram (for comparison)')
        im3 = plt.imshow(log_mel_spec, aspect='auto', origin='lower', cmap='viridis')
        plt.xlabel('Time Frames')
        plt.ylabel('Mel Frequency Bands')
        plt.colorbar(im3, format='%+2.0f dB')
        
        # 频率分析
        plt.subplot(5, 1, 5)
        plt.title('Frequency Analysis - Mean Power')
        mean_power = np.mean(spec_features[:, :, 0], axis=0)
        freq_bins = np.linspace(0, sample_rate/2, len(mean_power))
        plt.semilogy(freq_bins, np.exp(mean_power/10))  # 转换回线性尺度
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Mean Power')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 2000)  # 专注于肺音的主要频率范围
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存单个样本图片到 {save_path}")
        
    except ImportError:
        print("无法进行可视化，缺少matplotlib库")
    except Exception as e:
        print(f"可视化过程中出错: {e}")

def visualize_four_classes(selected_samples, sample_rate, n_fft, hop_length, n_mels, f_min, f_max):
    """可视化四种分类的对比图"""
    try:
        import matplotlib.pyplot as plt
        
        if len(selected_samples) < 4:
            print(f"样本数量不足，只有 {len(selected_samples)} 个样本")
            return
        
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                'spectrogram_four_classes_comparison.png')
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        class_names = ['正常', '爆裂音', '哮鸣音', '两者都有']
        
        for i, (waveform, filename, label, class_name) in enumerate(selected_samples[:4]):
            # 提取频谱图特征
            spec_features = generate_fbank(waveform, sample_rate, 
                                         n_fft=n_fft, hop_length=hop_length, 
                                         n_mels=n_mels, f_min=f_min, f_max=f_max)
            
            # 生成对比特征
            S = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
            linear_spec = np.abs(S)
            
            mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, 
                                                    n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            log_mel_spec = librosa.power_to_db(mel_spec)
            
            # 功率谱密度
            freqs, times, psd = librosa.reassigned_spectrogram(y=waveform, sr=sample_rate, 
                                                              n_fft=n_fft, hop_length=hop_length)
            
            # 时间轴
            duration = len(waveform) / sample_rate
            t = np.linspace(0, duration, len(waveform))
            
            # 波形图
            axes[i, 0].plot(t, waveform)
            axes[i, 0].set_title(f'{class_name} - Waveform\n{filename}')
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].grid(True, alpha=0.3)
            
            # 频谱图
            im1 = axes[i, 1].imshow(spec_features[:, :, 0].T, aspect='auto', origin='lower', cmap='viridis')
            axes[i, 1].set_title(f'{class_name} - Log Spectrogram')
            axes[i, 1].set_xlabel('Time Frames')
            axes[i, 1].set_ylabel('Frequency Bins')
            plt.colorbar(im1, ax=axes[i, 1], format='%+2.0f dB')
            
            # 线性频谱图
            im2 = axes[i, 2].imshow(linear_spec, aspect='auto', origin='lower', cmap='viridis')
            axes[i, 2].set_title(f'{class_name} - Linear Spectrogram')
            axes[i, 2].set_xlabel('Time Frames')
            axes[i, 2].set_ylabel('Frequency Bins')
            plt.colorbar(im2, ax=axes[i, 2])
            
            # 梅尔频谱图
            im3 = axes[i, 3].imshow(log_mel_spec, aspect='auto', origin='lower', cmap='viridis')
            axes[i, 3].set_title(f'{class_name} - Log-Mel')
            axes[i, 3].set_xlabel('Time Frames')
            axes[i, 3].set_ylabel('Mel Bands')
            plt.colorbar(im3, ax=axes[i, 3], format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存四类对比图片到 {save_path}")
        
        # 额外生成频谱分析图
        create_spectrum_analysis_plot(selected_samples, sample_rate, n_fft, hop_length, n_mels, f_min, f_max)
        
    except ImportError:
        print("无法进行可视化，缺少matplotlib库")
    except Exception as e:
        print(f"四类对比可视化过程中出错: {e}")

def create_spectrum_analysis_plot(selected_samples, sample_rate, n_fft, hop_length, n_mels, f_min, f_max):
    """创建频谱分析图"""
    try:
        import matplotlib.pyplot as plt
        
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                'spectrum_analysis.png')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        class_names = ['正常', '爆裂音', '哮鸣音', '两者都有']
        colors = ['blue', 'red', 'green', 'orange']
        
        # 收集所有类别的频谱统计信息
        all_mean_spec = []
        all_std_spec = []
        
        for i, (waveform, filename, label, class_name) in enumerate(selected_samples[:4]):
            spec_features = generate_fbank(waveform, sample_rate, 
                                         n_fft=n_fft, hop_length=hop_length, 
                                         n_mels=n_mels, f_min=f_min, f_max=f_max)
            
            # 计算统计信息
            mean_spec = np.mean(spec_features[:, :, 0], axis=0)
            std_spec = np.std(spec_features[:, :, 0], axis=0)
            
            all_mean_spec.append(mean_spec)
            all_std_spec.append(std_spec)
        
        # 创建频率轴
        freq_bins = np.linspace(0, sample_rate/2, len(all_mean_spec[0]))
        
        # 绘制各类别频谱均值对比
        axes[0, 0].set_title('Mean Spectrum by Class')
        for i, (mean_spec, class_name, color) in enumerate(zip(all_mean_spec, class_names, colors)):
            axes[0, 0].plot(freq_bins, mean_spec, label=class_name, color=color, linewidth=2)
        axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('Mean Log Power (dB)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(0, 2000)  # 专注于肺音频率范围
        
        # 绘制各类别频谱标准差对比
        axes[0, 1].set_title('Spectrum Standard Deviation by Class')
        for i, (std_spec, class_name, color) in enumerate(zip(all_std_spec, class_names, colors)):
            axes[0, 1].plot(freq_bins, std_spec, label=class_name, color=color, linewidth=2)
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Standard Deviation (dB)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(0, 2000)
        
        # 绘制频率重要性（方差）
        axes[1, 0].set_title('Frequency Bin Variance Across Classes')
        freq_variances = []
        for freq_idx in range(len(all_mean_spec[0])):
            freq_values = [mean_spec[freq_idx] for mean_spec in all_mean_spec]
            variance = np.var(freq_values)
            freq_variances.append(variance)
        
        axes[1, 0].plot(freq_bins, freq_variances, color='purple', linewidth=2)
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Variance Across Classes')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, 2000)
        
        # 高亮显示最重要的频率
        max_var_idx = np.argmax(freq_variances)
        max_var_freq = freq_bins[max_var_idx]
        axes[1, 0].axvline(x=max_var_freq, color='red', linestyle='--', linewidth=2)
        axes[1, 0].text(max_var_freq, freq_variances[max_var_idx], 
                       f'Most Important\n{max_var_freq:.0f} Hz', 
                       ha='center', va='bottom', fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 频率范围分析
        axes[1, 1].set_title('Frequency Range Analysis')
        freq_ranges = {
            '低频 (50-200 Hz)': (50, 200),
            '中低频 (200-600 Hz)': (200, 600),
            '中高频 (600-1200 Hz)': (600, 1200),
            '高频 (1200-2000 Hz)': (1200, 2000)
        }
        
        range_colors = ['blue', 'green', 'orange', 'red']
        x_pos = np.arange(len(class_names))
        bar_width = 0.2
        
        for i, (range_name, (f_low, f_high)) in enumerate(freq_ranges.items()):
            # 找到对应的频率索引
            freq_mask = (freq_bins >= f_low) & (freq_bins <= f_high)
            range_powers = []
            
            for mean_spec in all_mean_spec:
                range_power = np.mean(mean_spec[freq_mask])
                range_powers.append(range_power)
            
            axes[1, 1].bar(x_pos + i * bar_width, range_powers, bar_width, 
                          label=range_name, color=range_colors[i], alpha=0.7)
        
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Mean Power in Range (dB)')
        axes[1, 1].set_xticks(x_pos + bar_width * 1.5)
        axes[1, 1].set_xticklabels(class_names)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存频谱分析图到 {save_path}")
        
    except Exception as e:
        print(f"频谱分析图生成出错: {e}")