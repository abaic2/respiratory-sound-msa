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


def generate_fbank(waveform, sample_rate, hop_length=512, n_bins=84, bins_per_octave=12, fmin=50, fmax=8000):
    """
    生成常数 Q 变换 (CQT) 特征
    
    参数:
        waveform: 输入音频波形 (numpy数组或torch张量)
        sample_rate: 采样率
        hop_length: 帧移
        n_bins: CQT频率箱数
        bins_per_octave: 每个八度的箱数 (通常为12，对应12个半音)
        fmin: 最小频率
        fmax: 最大频率
    
    返回:
        CQT特征，形状为 (时间, n_bins, 1)
    """
    import numpy as np
    import librosa
    
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
        
        # 计算常数 Q 变换 (CQT)
        C = librosa.cqt(
            waveform,
            sr=sample_rate,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            filter_scale=1.0,  # 滤波器带宽
            norm=1,  # 归一化滤波器
            sparsity=0.01  # 控制滤波器的稀疏性
        )
        
        # 计算CQT的幅度谱
        C_mag = np.abs(C)
        
        # 应用对数变换以增强视觉对比度
        C_log = librosa.amplitude_to_db(C_mag, ref=np.max)
        
        # 转置并添加通道维度
        C_log = C_log.T  # (n_bins, 时间) -> (时间, n_bins)
        C_log = np.expand_dims(C_log, axis=2)  # (时间, n_bins) -> (时间, n_bins, 1)
        
        return C_log
        
    except Exception as e:
        print(f"生成CQT特征时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        # 返回空数组作为后备
        return np.zeros((128, n_bins, 1))

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


# 在文件末尾的测试代码部分
# 在文件末尾的测试代码部分
if __name__ == "__main__":
    import argparse
    import sys
    
    print("测试常数 Q 变换 (CQT) 特征提取...")
    
    # 创建简单的参数解析器用于测试
    parser = argparse.ArgumentParser(description='CQT特征提取测试')
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
        
        # CQT参数设置
        hop_length = 512
        n_bins = 84
        bins_per_octave = 12
        
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
            
            # 提取CQT特征
            cqt_features = generate_fbank(waveform, args.sample_rate, 
                                      hop_length=hop_length, 
                                      n_bins=n_bins, 
                                      bins_per_octave=bins_per_octave)
            
            # 可视化
            visualize_single_sample(waveform, cqt_features, args.filename, label, args.sample_rate, hop_length)
            
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
            
            # 为每个类别提取CQT特征并可视化
            for waveform, filename, label, class_name in selected_samples:
                print(f"\n处理类别: {class_name} (标签: {label})")
                print(f"文件: {filename}")
                print(f"波形形状: {waveform.shape}")
                
                cqt_features = generate_fbank(waveform, args.sample_rate, 
                                          hop_length=hop_length, 
                                          n_bins=n_bins, 
                                          bins_per_octave=bins_per_octave)
                
                # 可视化单个样本
                visualize_single_sample(waveform, cqt_features, 
                                      f"{filename}_{class_name}", label, 
                                      args.sample_rate, hop_length)
            
            # 创建四类对比图
            visualize_four_classes(selected_samples, args.sample_rate, hop_length, n_bins, bins_per_octave)
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def visualize_single_sample(waveform, cqt_features, filename, label, sample_rate, hop_length):
    """可视化单个样本"""
    try:
        import matplotlib.pyplot as plt
        
        # 创建时间轴
        duration = len(waveform) / sample_rate
        t = np.linspace(0, duration, len(waveform))
        
        # 同时生成梅尔频谱图以便比较
        S = librosa.stft(waveform, n_fft=1024, hop_length=hop_length)
        mel_spec = librosa.feature.melspectrogram(S=np.abs(S)**2, sr=sample_rate, n_mels=128)
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        # 保存路径
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                f'cqt_single_{filename}_label{label}.png')
        
        plt.figure(figsize=(15, 12))
        
        # 波形图
        plt.subplot(3, 1, 1)
        plt.title(f'Waveform - {filename} (Label: {label})')
        plt.plot(t, waveform)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # CQT图
        plt.subplot(3, 1, 2)
        plt.title('Constant-Q Transform (CQT)')
        plt.imshow(cqt_features[:, :, 0].T, aspect='auto', origin='lower', cmap='viridis')
        plt.xlabel('Time Frames')
        plt.ylabel('CQT Bins (log-frequency)')
        plt.colorbar(format='%+2.0f dB')
        
        # 梅尔频谱图对比
        plt.subplot(3, 1, 3)
        plt.title('Log-Mel Spectrogram (for comparison)')
        plt.imshow(log_mel_spec, aspect='auto', origin='lower', cmap='viridis')
        plt.xlabel('Time Frames')
        plt.ylabel('Mel Frequency Bands')
        plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存单个样本图片到 {save_path}")
        
    except ImportError:
        print("无法进行可视化，缺少matplotlib库")
    except Exception as e:
        print(f"可视化过程中出错: {e}")

def visualize_four_classes(selected_samples, sample_rate, hop_length, n_bins, bins_per_octave):
    """可视化四种分类的对比图"""
    try:
        import matplotlib.pyplot as plt
        
        if len(selected_samples) < 4:
            print(f"样本数量不足，只有 {len(selected_samples)} 个样本")
            return
        
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                'cqt_four_classes_comparison.png')
        
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        class_names = ['正常', '爆裂音', '哮鸣音', '两者都有']
        
        for i, (waveform, filename, label, class_name) in enumerate(selected_samples[:4]):
            # 提取CQT特征
            cqt_features = generate_fbank(waveform, sample_rate, 
                                      hop_length=hop_length, 
                                      n_bins=n_bins, 
                                      bins_per_octave=bins_per_octave)
            
            # 生成梅尔频谱图
            S = librosa.stft(waveform, n_fft=1024, hop_length=hop_length)
            mel_spec = librosa.feature.melspectrogram(S=np.abs(S)**2, sr=sample_rate, n_mels=128)
            log_mel_spec = librosa.power_to_db(mel_spec)
            
            # 时间轴
            duration = len(waveform) / sample_rate
            t = np.linspace(0, duration, len(waveform))
            
            # 波形图
            axes[i, 0].plot(t, waveform)
            axes[i, 0].set_title(f'{class_name} - Waveform\n{filename}')
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].grid(True, alpha=0.3)
            
            # CQT图
            im1 = axes[i, 1].imshow(cqt_features[:, :, 0].T, aspect='auto', origin='lower', cmap='viridis')
            axes[i, 1].set_title(f'{class_name} - CQT')
            axes[i, 1].set_xlabel('Time Frames')
            axes[i, 1].set_ylabel('CQT Bins')
            plt.colorbar(im1, ax=axes[i, 1], format='%+2.0f dB')
            
            # 梅尔频谱图
            im2 = axes[i, 2].imshow(log_mel_spec, aspect='auto', origin='lower', cmap='viridis')
            axes[i, 2].set_title(f'{class_name} - Mel Spectrogram')
            axes[i, 2].set_xlabel('Time Frames')
            axes[i, 2].set_ylabel('Mel Bands')
            plt.colorbar(im2, ax=axes[i, 2], format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存四类对比图片到 {save_path}")
        
    except ImportError:
        print("无法进行可视化，缺少matplotlib库")
    except Exception as e:
        print(f"四类对比可视化过程中出错: {e}")

# 单文件
# cd /home/yujieyang/bishe/MVST-main/basecqt/util
# python icbhi_util.py --mode single --filename 101_1b1_Al_sc_Meditron

# 四分类
# cd /home/yujieyang/bishe/MVST-main/basecqt/util
# python icbhi_util.py --mode auto