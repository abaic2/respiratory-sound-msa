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


def generate_fbank(waveform, sample_rate, n_fft=2048, hop_length=512, n_chroma=12):
    """
    生成色度图特征
    
    参数:
        waveform: 输入音频波形 (numpy数组或torch张量)
        sample_rate: 采样率
        n_fft: FFT点数
        hop_length: 帧移
        n_chroma: 色度音高类别数（通常为12，对应12个音高类）
    
    返回:
        色度图特征，形状为 (时间, n_chroma, 1)
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
        
        # 计算色度图 (Chromagram)
        # 用较大的n_fft能够提供更好的频率分辨率
        chroma = librosa.feature.chroma_stft(
            y=waveform,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_chroma=n_chroma
        )
        
        # 可选：对色度图应用对数变换增强动态范围
        # 由于色度图值域为[0,1]，添加小值避免log(0)
        log_chroma = np.log1p(chroma)
        
        # 转置并添加通道维度
        log_chroma = log_chroma.T  # (n_chroma, 时间) -> (时间, n_chroma)
        log_chroma = np.expand_dims(log_chroma, axis=2)  # (时间, n_chroma) -> (时间, n_chroma, 1)
        
        return log_chroma
        
    except Exception as e:
        print(f"生成色度图特征时出错: {str(e)}")
        # 返回空数组作为后备
        return np.zeros((128, n_chroma, 1))

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


if __name__ == "__main__":
    import argparse
    import os
    import random
    import matplotlib.pyplot as plt
    import torchaudio

    parser = argparse.ArgumentParser(description='Chromagram Feature Extraction and Visualization Test')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'all_classes'],
                        help='Visualization mode: single-one file, all_classes-one from each class')
    parser.add_argument('--file', type=str, default='', 
                        help='Specific file name to visualize (without extension, only used when mode=single)')
    args = parser.parse_args()

    # Data paths
    data_folder = '/home/yujieyang/bishe/data/ICBHI_final_database/icbhi_dataset'
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Feature extraction parameters
    n_fft = 2048 
    hop_length = 512
    n_chroma = 12
    sample_rate = 16000
    
    print("Starting chromagram feature extraction and visualization...")
    
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
            
            # Extract chromagram features
            chroma_features = generate_fbank(waveform_numpy, sample_rate, 
                                           n_fft=n_fft, 
                                           hop_length=hop_length, 
                                           n_chroma=n_chroma)
            
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
            
            # Visualization
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Waveform
            time_axis = np.arange(len(waveform_numpy)) / sample_rate
            axes[0].plot(time_axis, waveform_numpy)
            axes[0].set_title(f'File: {filename} - Class: {label_name}')
            axes[0].set_xlabel('Time (seconds)')
            axes[0].set_ylabel('Amplitude')
            
            # Chromagram
            im = axes[1].imshow(chroma_features[:, :, 0].T, aspect='auto', origin='lower', 
                               interpolation='nearest', cmap='viridis')
            axes[1].set_title('Chromagram')
            axes[1].set_xlabel('Time Frames')
            axes[1].set_ylabel('Pitch Classes')
            axes[1].set_yticks(np.arange(12))
            axes[1].set_yticklabels(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
            
            plt.colorbar(im, ax=axes[1], format='%.2f')
            plt.tight_layout()
            
            # Save visualization result
            save_path = os.path.join(output_folder, f'{filename}_chromagram.png')
            plt.savefig(save_path)
            print(f"Saved {filename} visualization to {save_path}")
            plt.close()
            
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("Chromagram feature extraction and visualization completed!")




# python icbhi_util.py --mode single --file 101_1b1_Al_sc_Meditron    单个文件
# python icbhi_util.py --mode all_classes   每个类别