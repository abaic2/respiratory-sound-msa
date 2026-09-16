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


def generate_fbank(waveform, sample_rate, n_fft=1024, hop_length=512, n_filters=64, fmin=50, fmax=8000):
    """
    生成耳蜗图特征 (Cochleagram)
    
    参数:
        waveform: 输入音频波形 (numpy数组或torch张量)
        sample_rate: 采样率
        n_fft: FFT点数
        hop_length: 帧移
        n_filters: 耳蜗滤波器数量
        fmin: 最小频率
        fmax: 最大频率
    
    返回:
        耳蜗图特征，形状为 (时间, n_filters, 1)
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
        
        # 计算频谱图
        S = librosa.stft(waveform, 
                        n_fft=n_fft, 
                        hop_length=hop_length,
                        win_length=n_fft, 
                        window='hann',
                        center=True)
        
        # 计算功率谱
        power_spec = np.abs(S) ** 2
        
        # 创建ERB滤波器组 - 手动实现，不依赖 librosa.erb_frequencies
        # ERB比例尺 (Equivalent Rectangular Bandwidth)
        # 使用Glasberg and Moore公式: ERB = 24.7 * (4.37 * fc / 1000 + 1)
        
        # 计算ERB频率
        def erb_frequencies_manual(n_bands, fmin, fmax):
            # 将最小和最大频率转换为ERB比例
            min_erb = 9.26 * np.log(0.00437 * fmin + 1)
            max_erb = 9.26 * np.log(0.00437 * fmax + 1)
            
            # 在ERB比例上均匀分布频率点
            erb_points = np.linspace(min_erb, max_erb, n_bands)
            
            # 将ERB比例转回Hz频率
            freq_hz = (np.exp(erb_points / 9.26) - 1) / 0.00437
            
            return freq_hz
        
        # 生成ERB频率
        erb_freqs = erb_frequencies_manual(n_filters, fmin, fmax)
        
        # 创建滤波器矩阵
        filters = np.zeros((n_filters, n_fft//2 + 1))
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        
        # ERB带宽
        ear_q = 9.26449  # Q值常数
        min_bw = 24.7    # 最小带宽
        order = 1        # 滤波器阶数
        
        # 生成ERB滤波器 (简化版Gammatone滤波器)
        for i, cf in enumerate(erb_freqs):
            # 计算ERB带宽
            erb = min_bw + (cf / ear_q)
            # 定义频率响应
            response = 1.0 / (1.0 + ((freqs - cf) / (erb / 2.0)) ** (2 * order))
            filters[i] = response
        
        # 应用滤波器组到功率谱
        cochleagram = np.dot(filters, power_spec)
        
        # 应用对数压缩
        log_cochleagram = np.log(cochleagram + 1e-8)
        
        # 转置并添加通道维度
        log_cochleagram = log_cochleagram.T  # (n_filters, 时间) -> (时间, n_filters)
        log_cochleagram = np.expand_dims(log_cochleagram, axis=2)  # (时间, n_filters) -> (时间, n_filters, 1)
        
        return log_cochleagram
        
    except Exception as e:
        print(f"生成耳蜗图特征时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        # 返回空数组作为后备
        return np.zeros((128, n_filters, 1))

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


# ==========================================================================


# 在文件末尾的测试代码部分
if __name__ == "__main__":
    import argparse
    import os
    import random
    import matplotlib.pyplot as plt
    import torchaudio

    parser = argparse.ArgumentParser(description='Cochleagram Feature Extraction and Visualization Test')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'all_classes'],
                        help='Visualization mode: single-one file, all_classes-one from each class')
    parser.add_argument('--file', type=str, default='', 
                        help='Specific file name to visualize (without extension, only used when mode=single)')
    args = parser.parse_args()

    # Data paths
    data_folder = '/home/yujieyang/bishe/data/ICBHI_final_database/icbhi_dataset'
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cochleagram_visualizations')
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Feature extraction parameters
    n_fft = 1024
    hop_length = 512
    n_filters = 64
    sample_rate = 16000
    f_min = 50
    f_max = 8000
    
    print("Starting cochleagram feature extraction and visualization...")
    
    class TempArgs:
        def __init__(self):
            self.n_cls = 4
            self.class_split = 'lungsound'
            self.data_folder = os.path.dirname(data_folder)

    temp_args = TempArgs()
    
    # Get annotation information
    annotation_dict = get_annotations(temp_args, data_folder)
    
    # Process files
    if args.mode == 'single':
        # Mode 1: Visualize single file
        if args.file:
            filename = args.file
        else:
            # If no file specified, choose one randomly
            all_files = [f.split('.')[0] for f in os.listdir(data_folder) if f.endswith('.wav')]
            filename = random.choice(all_files)
            
        process_files = [filename]
        print(f"Selected file: {filename}")
    
    else:
        # Mode 2: Randomly select one from each class
        print("Selecting one file from each class...")
        
        # Collect all files and their labels
        files_by_class = [[] for _ in range(4)]  # Store files for 4 classes
        
        for filename, annotations in annotation_dict.items():
            try:
                # Get file label
                for idx in annotations.index:
                    row = annotations.loc[idx]
                    crackles = row['Crackles']
                    wheezes = row['Wheezes']
                    label = _get_lungsound_label(crackles, wheezes, 4)
                    
                    # Add file to corresponding class list
                    files_by_class[label].append(filename)
                    # Only process first label, as we just need the filename
                    break
            except Exception as e:
                print(f"Error processing file {filename} label: {str(e)}")
        
        # Select one file from each class
        process_files = []
        class_names = ['Normal', 'Crackles', 'Wheezes', 'Crackles+Wheezes']
        
        for class_idx, files in enumerate(files_by_class):
            if files:  # Ensure class has files
                selected = random.choice(files)
                process_files.append(selected)
                print(f"Selected class {class_idx} ({class_names[class_idx]}): {selected}")
            else:
                print(f"Warning: Class {class_idx} ({class_names[class_idx]}) has no available files")
    
    # Process selected files
    for filename in process_files:
        try:
            file_path = os.path.join(data_folder, f'{filename}.wav')
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File does not exist: {file_path}")
                continue
                
            # Load audio file
            original_sr = librosa.get_samplerate(file_path)
            waveform, _ = torchaudio.load(file_path)
            
            # Resample to target sample rate
            if original_sr != sample_rate:
                resample = T.Resample(original_sr, sample_rate)
                waveform = resample(waveform)
            
            # Ensure one-dimensional array or tensor
            waveform_numpy = waveform.squeeze().numpy()
            
            # Extract cochleagram features
            cochleagram_features = generate_fbank(waveform_numpy, sample_rate, 
                                               n_fft=n_fft, 
                                               hop_length=hop_length, 
                                               n_filters=n_filters,
                                               fmin=f_min,
                                               fmax=f_max)
            
            # Get classification label
            try:
                annotations = annotation_dict[filename]
                # Get first cycle label
                row = annotations.loc[0]
                crackles = row['Crackles']
                wheezes = row['Wheezes']
                label = _get_lungsound_label(crackles, wheezes, 4)
                class_names = ['Normal', 'Crackles', 'Wheezes', 'Crackles+Wheezes']
                label_name = class_names[label]
            except Exception:
                label_name = "Unknown"
            
            # Calculate time axis
            t = np.arange(len(waveform_numpy)) / sample_rate
            
            # Visualization
            plt.figure(figsize=(12, 10))
            
            # Plot waveform (first 1 second)
            show_samples = min(sample_rate, len(waveform_numpy))
            plt.subplot(3, 1, 1)
            plt.title(f'Respiratory Sound Waveform - {filename} ({label_name})')
            plt.plot(t[:show_samples], waveform_numpy[:show_samples])
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            
            # Plot cochleagram
            plt.subplot(3, 1, 2)
            plt.title('Cochleagram (ERB scale)')
            plt.imshow(cochleagram_features[:, :, 0].T, aspect='auto', origin='lower', cmap='viridis')
            plt.xlabel('Time Frames')
            plt.ylabel('ERB Frequency Bands')
            plt.colorbar(format='%.2f')
            
            # Plot regular spectrogram for comparison
            S = librosa.stft(waveform_numpy, n_fft=n_fft, hop_length=hop_length)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            
            plt.subplot(3, 1, 3)
            plt.title('Regular Spectrogram (for comparison)')
            plt.imshow(S_db, aspect='auto', origin='lower', cmap='viridis')
            plt.xlabel('Time Frames')
            plt.ylabel('Frequency Bins')
            plt.colorbar(format='%+2.0f dB')
            
            plt.tight_layout()
            
            # Save visualization result
            save_path = os.path.join(output_folder, f'{filename}_cochleagram.png')
            plt.savefig(save_path)
            print(f"Saved {filename} visualization to {save_path}")
            plt.close()
            
            # Print feature statistics
            print(f"Cochleagram shape: {cochleagram_features.shape}")
            print(f"Min: {cochleagram_features.min():.2f}, Max: {cochleagram_features.max():.2f}")
            print(f"Mean: {cochleagram_features.mean():.2f}, Std: {cochleagram_features.std():.2f}")
            
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Also show a synthetic example for comparison
    print("\nGenerating synthetic test signal for comparison...")
    
    # Create a synthetic signal with multiple frequency components
    duration = 3  # 3 seconds
    t_synth = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    
    # Create signal with multiple frequency components
    synth_waveform = (np.sin(2 * np.pi * 200 * t_synth) +      # low freq
                      np.sin(2 * np.pi * 1000 * t_synth) +      # mid freq
                      np.sin(2 * np.pi * 3000 * t_synth) +      # high freq
                      0.5 * np.sin(2 * np.pi * 5000 * t_synth)) / 4.0  # even higher
    
    # Add some noise
    synth_waveform += np.random.normal(0, 0.01, len(synth_waveform))
    
    # Extract cochleagram features
    synth_cochleagram = generate_fbank(synth_waveform, sample_rate, 
                                     n_fft=n_fft, 
                                     hop_length=hop_length, 
                                     n_filters=n_filters,
                                     fmin=f_min,
                                     fmax=f_max)
    
    # Visualization
    plt.figure(figsize=(12, 10))
    
    # Plot waveform (first 1000 samples)
    plt.subplot(3, 1, 1)
    plt.title('Synthetic Test Signal Waveform')
    plt.plot(t_synth[:1000], synth_waveform[:1000])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Plot cochleagram
    plt.subplot(3, 1, 2)
    plt.title('Cochleagram (ERB scale)')
    plt.imshow(synth_cochleagram[:, :, 0].T, aspect='auto', origin='lower', cmap='viridis')
    plt.xlabel('Time Frames')
    plt.ylabel('ERB Frequency Bands')
    plt.colorbar(format='%.2f')
    
    # Plot regular spectrogram
    S_synth = librosa.stft(synth_waveform, n_fft=n_fft, hop_length=hop_length)
    S_db_synth = librosa.amplitude_to_db(np.abs(S_synth), ref=np.max)
    
    plt.subplot(3, 1, 3)
    plt.title('Regular Spectrogram (for comparison)')
    plt.imshow(S_db_synth, aspect='auto', origin='lower', cmap='viridis')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    
    # Save synthetic example
    synth_save_path = os.path.join(output_folder, 'synthetic_cochleagram_comparison.png')
    plt.savefig(synth_save_path)
    print(f"Saved synthetic example to {synth_save_path}")
    plt.close()
    
    print("Cochleagram feature extraction and visualization completed!")