from collections import namedtuple
import os
import math
import random
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

from .augmentation import augment_raw_audio

__all__ = [
    'get_annotations', 
    'get_individual_cycles_torchaudio', 
    'cut_pad_sample_torchaudio',
    'generate_fbank', 
    'get_score', 
    'generate_raw_features', 
    'extract_time_domain_features', 
    'extract_sequential_features'
]

# 保持原有代码不变...

# ==========================================================================
# 为RNN模型添加时序特征提取函数
def generate_raw_features(audio, sample_rate, frame_length=1024, hop_length=512):
    """
    将原始音频分帧处理，提取时序特征，适用于RNN模型
    """
    # 确保采样率为16kHz
    assert sample_rate == 16000, 'input audio sampling rate must be 16kHz'
    
    # 获取音频长度
    audio_len = audio.shape[1]
    num_frames = 1 + (audio_len - frame_length) // hop_length
    
    # 预分配内存
    frames = torch.zeros((num_frames, frame_length), dtype=audio.dtype)
    
    # 分帧
    for i in range(num_frames):
        start = i * hop_length
        end = min(start + frame_length, audio_len)
        frame = audio[:, start:end]
        if end - start < frame_length:  # 如果最后一帧不完整，进行填充
            frames[i, :(end-start)] = frame
        else:
            frames[i] = frame.squeeze(0)
    
    return frames  # 返回形状为[num_frames, frame_length]


def extract_time_domain_features(audio, sample_rate, segment_length=1024, hop_length=512):
    """
    从原始音频中提取时域特征，包括：
    - RMS能量
    - 过零率
    - 自相关系数
    - 短时能量
    
    这些特征更适合RNN模型进行时序分析
    """
    # 确保采样率为16kHz
    assert sample_rate == 16000, 'input audio sampling rate must be 16kHz'
    
    # 转为numpy以便使用librosa
    audio_np = audio.squeeze().numpy()
    
    # 检查音频长度，如果太短，进行填充
    if len(audio_np) < segment_length:
        # 使用重复填充使音频至少达到 segment_length 长度
        repeat_factor = int(np.ceil(segment_length / len(audio_np)))
        audio_np = np.tile(audio_np, repeat_factor)[:segment_length]
        print(f"警告: 音频太短，已重复填充到 {segment_length} 样本")
    
    # 提取时域特征
    # 1. 过零率 (Zero Crossing Rate)
    zcr = librosa.feature.zero_crossing_rate(audio_np, 
                                            frame_length=segment_length, 
                                            hop_length=hop_length,
                                            center=False).T
    
    # 2. 短时能量 (RMS Energy)
    rms = librosa.feature.rms(y=audio_np, 
                             frame_length=segment_length, 
                             hop_length=hop_length,
                             center=False).T
    
    # 3. 自相关特征 (Autocorrelation)
    # 为效率考虑只提取几个延迟点
    ac_features = []
    for lag in [1, 2, 3, 5, 10]:
        # 确保输入足够长以计算自相关
        if len(audio_np) <= lag:
            # 如果音频短于滞后值，添加零值特征
            ac_mean = np.zeros((max(1, zcr.shape[0]), 1))
        else:
            try:
                ac = np.correlate(audio_np, np.roll(audio_np, lag), mode='valid')
                # 检查ac长度是否足够进行分帧
                if len(ac) >= segment_length:
                    ac = librosa.util.frame(ac, frame_length=segment_length, hop_length=hop_length).T
                    if ac.shape[0] > 0:
                        ac_mean = np.mean(ac, axis=1, keepdims=True)
                    else:
                        ac_mean = np.zeros((max(1, zcr.shape[0]), 1))
                else:
                    # 如果ac太短，无法分帧，则使用一个平均值
                    ac_mean = np.mean(ac) * np.ones((max(1, zcr.shape[0]), 1))
            except Exception as e:
                print(f"自相关计算错误 (lag={lag}): {e}")
                ac_mean = np.zeros((max(1, zcr.shape[0]), 1))
                
        ac_features.append(ac_mean)
    
    # 确保所有自相关特征长度一致
    min_ac_len = min(f.shape[0] for f in ac_features)
    ac_features = [f[:min_ac_len] for f in ac_features]
    
    if ac_features and min_ac_len > 0:
        ac_features = np.concatenate(ac_features, axis=1)
    else:
        ac_features = np.zeros((max(1, zcr.shape[0]), 5))
    
    # 4. 包络线 (Envelope) - 通过低通滤波器获取
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    # 计算包络线 (低通滤波)
    try:
        envelope = butter_lowpass_filter(np.abs(audio_np), 150, sample_rate)
        if len(envelope) >= segment_length:
            envelope = librosa.util.frame(envelope, frame_length=segment_length, hop_length=hop_length).T
            if envelope.shape[0] > 0:
                envelope_mean = np.mean(envelope, axis=1, keepdims=True)
            else:
                envelope_mean = np.zeros((max(1, zcr.shape[0]), 1))
        else:
            # 如果包络线太短，使用一个全局平均值
            envelope_mean = np.mean(envelope) * np.ones((max(1, zcr.shape[0]), 1))
    except Exception as e:
        print(f"包络线计算错误: {e}")
        envelope_mean = np.zeros((max(1, zcr.shape[0]), 1))
    
    # 确保所有特征长度一致
    feature_shapes = [f.shape[0] for f in [zcr, rms, ac_features, envelope_mean] if f.shape[0] > 0]
    
    if not feature_shapes:  # 如果所有特征都为空
        # 创建一个基本的单帧特征
        return torch.zeros((1, 8), dtype=torch.float32)  # 1帧，8个特征
        
    min_len = min(feature_shapes)
    
    # 确保所有特征至少有一个元素
    zcr = zcr[:min_len] if zcr.shape[0] > 0 else np.zeros((min_len, 1))
    rms = rms[:min_len] if rms.shape[0] > 0 else np.zeros((min_len, 1))
    ac_features = ac_features[:min_len] if ac_features.shape[0] > 0 else np.zeros((min_len, 5))
    envelope_mean = envelope_mean[:min_len] if envelope_mean.shape[0] > 0 else np.zeros((min_len, 1))
    
    # 组合特征
    features = np.concatenate([zcr, rms, ac_features, envelope_mean], axis=1)
    
    # 标准化 - 添加异常处理
    try:
        features_mean = np.mean(features, axis=0, keepdims=True)
        features_std = np.std(features, axis=0, keepdims=True) + 1e-6
        normalized_features = (features - features_mean) / features_std
    except Exception as e:
        print(f"特征标准化错误: {e}, 使用未标准化特征")
        normalized_features = features
    
    return torch.from_numpy(normalized_features).float()  # 返回张量


def extract_sequential_features(audio, sample_rate, hop_length=160):
    """
    将音频分成固定大小的序列，适用于GRU/LSTM/TCN等序列模型
    提取一组更为丰富的时域特征
    """
    # 确保采样率为16kHz
    assert sample_rate == 16000, 'input audio sampling rate must be 16kHz'
    
    # 转为numpy以便使用librosa
    audio_np = audio.squeeze().numpy()
    
    # 检查音频长度，如果太短，进行填充
    min_length = 2048  # 最小需要的长度，确保所有特征提取函数能正常工作
    if len(audio_np) < min_length:
        # 使用重复填充使音频至少达到最小长度
        repeat_factor = int(np.ceil(min_length / max(1, len(audio_np))))
        audio_np = np.tile(audio_np if len(audio_np) > 0 else [0], repeat_factor)[:min_length]
        print(f"警告: 音频太短，已重复填充到 {min_length} 样本")
    
    # 特征列表
    feature_list = []
    
    # 提取特征时添加异常处理
    try:
        # 1. 提取过零率特征 (ZCR)
        zcr = librosa.feature.zero_crossing_rate(audio_np, hop_length=hop_length, center=False).T
        feature_list.append(zcr)
    except Exception as e:
        print(f"ZCR提取错误: {e}")
        # 创建空特征，保证后续处理不会崩溃
        zcr = np.zeros((1, 1))
        feature_list.append(zcr)
    
    try:
        # 2. 提取RMS能量
        rms = librosa.feature.rms(y=audio_np, hop_length=hop_length, center=False).T
        feature_list.append(rms)
    except Exception as e:
        print(f"RMS提取错误: {e}")
        rms = np.zeros((1, 1))
        feature_list.append(rms)
    
    try:
        # 3. 提取频谱质心 (Spectral Centroid)
        centroid = librosa.feature.spectral_centroid(y=audio_np, sr=sample_rate, 
                                                 hop_length=hop_length, center=False).T
        feature_list.append(centroid)
    except Exception as e:
        print(f"频谱质心提取错误: {e}")
        centroid = np.zeros((1, 1))
        feature_list.append(centroid)
    
    try:
        # 4. 提取频谱带宽 (Spectral Bandwidth)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio_np, sr=sample_rate, 
                                                   hop_length=hop_length, center=False).T
        feature_list.append(bandwidth)
    except Exception as e:
        print(f"频谱带宽提取错误: {e}")
        bandwidth = np.zeros((1, 1))
        feature_list.append(bandwidth)
    
    try:
        # 5. 提取频谱对比度 (Spectral Contrast)
        contrast = librosa.feature.spectral_contrast(y=audio_np, sr=sample_rate, 
                                                 hop_length=hop_length, center=False).T
        if contrast.shape[0] > 0:
            # 将多维压缩为单维均值
            contrast_mean = np.mean(contrast, axis=1, keepdims=True)
            feature_list.append(contrast_mean)
        else:
            contrast_mean = np.zeros((1, 1))
            feature_list.append(contrast_mean)
    except Exception as e:
        print(f"频谱对比度提取错误: {e}")
        contrast_mean = np.zeros((1, 1))
        feature_list.append(contrast_mean)
    
    try:
        # 6. 提取频谱平坦度 (Spectral Flatness)
        flatness = librosa.feature.spectral_flatness(y=audio_np, hop_length=hop_length, center=False).T
        feature_list.append(flatness)
    except Exception as e:
        print(f"频谱平坦度提取错误: {e}")
        flatness = np.zeros((1, 1))
        feature_list.append(flatness)
    
    # 确保所有特征长度一致，过滤掉空特征
    valid_features = [f for f in feature_list if f.shape[0] > 0]
    
    if not valid_features:
        # 如果没有有效特征，返回一个基本的单帧多特征
        return torch.zeros((1, 6), dtype=torch.float32)  # 1帧，6个特征
    
    # 找出所有有效特征中最短的长度
    min_len = min(f.shape[0] for f in valid_features)
    aligned_features = [f[:min_len] for f in valid_features]
    
    # 组合特征
    try:
        combined = np.concatenate(aligned_features, axis=1)
        
        # 标准化
        if combined.shape[0] > 0:
            mean = np.mean(combined, axis=0, keepdims=True)
            std = np.std(combined, axis=0, keepdims=True) + 1e-6
            normalized = (combined - mean) / std
        else:
            # 处理极短的音频
            normalized = np.zeros((1, sum(f.shape[1] for f in aligned_features)))
    except Exception as e:
        print(f"特征组合或标准化错误: {e}")
        # 返回基础特征作为后备
        normalized = np.zeros((1, 6))
    
    return torch.from_numpy(normalized).float()  # [time_steps, features]

# ==========================================================================
""" ICBHI dataset information """
def _extract_lungsound_annotation(file_name, data_folder):
    # 解析文件名
    tokens = file_name.strip().split('_')
    
    # 添加健壮性处理
    while len(tokens) < 5:
        tokens.append('')
    tokens = tokens[:5]  # 确保不会超过5个元素
    
    recording_info = pd.DataFrame(data=[tokens], columns=['Patient Number', 'Recording index', 'Chest location', 'Acquisition mode', 'Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(data_folder, file_name + '.txt'), names=['Start', 'End', 'Crackles', 'Wheezes'], delimiter='\t')

    return recording_info, recording_annotations


def get_annotations(args, data_folder):
    """
    从ICBHI数据集文件中提取标注信息
    
    参数:
        args: 包含类别划分信息的参数对象
        data_folder: 数据文件夹路径
    
    返回:
        annotation_dict: 文件名到标注的字典
    """
    if args.class_split == 'lungsound' or args.class_split in ['lungsound_meta', 'meta']:
        filenames = [f.strip().split('.')[0] for f in os.listdir(data_folder) if '.txt' in f]

        annotation_dict = {}
        for f in filenames:
            info, ann = _extract_lungsound_annotation(f, data_folder)
            annotation_dict[f] = ann

    elif args.class_split == 'diagnosis':
        filenames = [f.strip().split('.')[0] for f in os.listdir(data_folder) if '.txt' in f]
        try:
            tmp = pd.read_csv(os.path.join(args.data_folder, 'icbhi_dataset/bing.txt'), names=['Disease'], delimiter='\t')
        except FileNotFoundError:
            print(f"警告: 诊断文件 'bing.txt' 未找到，这在诊断分类中是必需的")
            # 创建一个空的诊断DataFrame
            tmp = pd.DataFrame(columns=['Disease'])
            tmp['Disease'] = 'Unknown'

        annotation_dict = {}
        for f in filenames:
            info, ann = _extract_lungsound_annotation(f, data_folder)
            ann.drop(['Crackles', 'Wheezes'], axis=1, inplace=True)

            try:
                patient_id = int(f.strip().split('_')[0])
                if patient_id in tmp.index:
                    disease = tmp.loc[patient_id, 'Disease']
                else:
                    disease = 'Unknown'
            except (ValueError, IndexError):
                disease = 'Unknown'
                
            ann['Disease'] = disease
            annotation_dict[f] = ann
            
    return annotation_dict


def _get_lungsound_label(crackle, wheeze, n_cls):
    """获取肺音分类标签"""
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
    """获取诊断分类标签"""
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
    """对音频数据进行切片"""
    max_ind = data.shape[1]
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)

    return data[:, start_ind: end_ind]


def cut_pad_sample_torchaudio(data, args):
    """将音频数据切割或填充到目标长度"""
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
    将每个声音文件分割成单独的包含一个呼吸周期的声音片段
    输出: [(audio_chunk:np.array, label:int), (...)]
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


def generate_fbank(audio, sample_rate, n_mels=128): 
    """
    使用torchaudio库为AST模型转换mel fbank
    """    
    assert sample_rate == 16000, 'input audio sampling rate must be 16kHz'
    fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True, sample_frequency=sample_rate, use_energy=False, window_type='hanning', num_mel_bins=n_mels, dither=0.0, frame_shift=10)
    
    mean, std =  -4.2677393, 4.5689974
    fbank = (fbank - mean) / (std * 2) # mean / std
    fbank = fbank.unsqueeze(-1).numpy()
    return fbank 

# ==========================================================================
""" evaluation metric """
def get_score(hits, counts, pflag=False):
    # normal accuracy
    sp = hits[0] / (counts[0] + 1e-10) * 100
    # abnormal accuracy
    se = sum(hits[1:]) / (sum(counts[1:]) + 1e-10) * 100
    sc = (sp + se) / 2.0

    if pflag:
        print("S_p: {:.4f}, S_e: {:.4f}, Score: {:.4f}".format(sp, se, sc))

    return sp, se, sc
# ==========================================================================