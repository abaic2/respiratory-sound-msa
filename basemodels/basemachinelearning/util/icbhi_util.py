from collections import namedtuple
import os
import math
import random
import numpy as np
from tqdm import tqdm
import pandas as pd

import librosa
import torch
import torchaudio
from torchaudio import transforms as T
from scipy.signal import butter, lfilter
from scipy.stats import skew, kurtosis

from .augmentation import augment_raw_audio

__all__ = [
    'get_annotations', 
    'get_individual_cycles_torchaudio', 
    'cut_pad_sample_torchaudio',
    'generate_fbank', 
    'get_score', 
    'generate_raw_features',  # 确保这里包含了这个函数
    'extract_time_domain_features', 
    'extract_sequential_features',
    'extract_ml_features'
]

# 保持原有代码不变...

# ==========================================================================
# 为机器学习模型添加特征提取函数
def extract_ml_features(audio, sample_rate, feature_type='statistical', show_progress=False):
    """
    从原始音频中提取适合机器学习模型的特征 - 简化版本
    
    参数:
        audio: 音频数据 [1, samples]
        sample_rate: 采样率
        feature_type: 特征类型，支持 'statistical', 'spectral', 'full'
        show_progress: 是否显示进度条
        
    返回:
        features: 形状为 [1, n_features] 的特征张量
    """
    try:
        if show_progress:
            print(f"提取 {feature_type} 特征...")
        
        # 转为numpy以便处理
        if isinstance(audio, torch.Tensor):
            audio_np = audio.squeeze().cpu().numpy()
        else:
            audio_np = np.asarray(audio).squeeze()
        
        # 处理无效数据
        if len(audio_np) == 0 or np.all(np.isnan(audio_np)) or np.all(audio_np == 0):
            audio_np = np.random.randn(16000) * 0.01
        
        # 处理极端值
        audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 确保音频长度适当
        if len(audio_np) < 16000:  # 至少1秒
            repeat_factor = int(np.ceil(16000 / max(1, len(audio_np))))
            audio_np = np.tile(audio_np, repeat_factor)[:16000]
        elif len(audio_np) > 160000:  # 最多10秒
            audio_np = audio_np[:160000]
        
        # 标准化
        if np.std(audio_np) > 0:
            audio_np = (audio_np - np.mean(audio_np)) / np.std(audio_np)
        
        # 初始化特征列表
        features = []
        
        # === 统计特征 (基本) ===
        features.extend([
            np.mean(audio_np),                 # 平均值
            np.std(audio_np),                  # 标准差
            np.max(audio_np),                  # 最大值
            np.min(audio_np),                  # 最小值
            np.median(audio_np),               # 中位数
            np.percentile(audio_np, 25),       # 25%分位数
            np.percentile(audio_np, 75),       # 75%分位数
            np.mean(np.abs(audio_np)),         # 平均绝对值
            np.mean(np.square(audio_np))       # 均方值
        ])
        
        # 过零率 (简化计算)
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_np))))
        features.append(zero_crossings / len(audio_np))
        
        # 简单自相关
        for lag in [1, 5, 10]:
            if len(audio_np) > lag:
                autocorr = np.corrcoef(audio_np[:-lag], audio_np[lag:])[0, 1]
                features.append(autocorr)
            else:
                features.append(0)
        
        # 能量统计
        energy = np.sum(audio_np**2) / len(audio_np)
        features.append(energy)
        
        # === 频域特征 (如果需要) ===
        if feature_type in ['spectral', 'full']:
            try:
                # 计算简单FFT
                fft_features = np.abs(np.fft.rfft(audio_np))
                fft_features = fft_features[:min(1000, len(fft_features))]  # 截取最多1000个点
                
                # 添加简单的频域统计特征
                features.append(np.mean(fft_features))
                features.append(np.std(fft_features))
                features.append(np.max(fft_features))
                
                # 添加几个频段的能量
                if len(fft_features) >= 100:
                    freqs = np.fft.rfftfreq(len(audio_np), 1/sample_rate)
                    freqs = freqs[:len(fft_features)]
                    
                    # 低频段 (0-500Hz)
                    low_mask = freqs < 500
                    if np.any(low_mask):
                        features.append(np.mean(fft_features[low_mask]))
                    else:
                        features.append(0)
                    
                    # 中频段 (500-2000Hz)
                    mid_mask = (freqs >= 500) & (freqs < 2000)
                    if np.any(mid_mask):
                        features.append(np.mean(fft_features[mid_mask]))
                    else:
                        features.append(0)
                    
                    # 高频段 (>2000Hz)
                    high_mask = freqs >= 2000
                    if np.any(high_mask):
                        features.append(np.mean(fft_features[high_mask]))
                    else:
                        features.append(0)
                else:
                    features.extend([0, 0, 0])
                
                # 添加更多频域特征，使维度匹配
                features.extend([0] * 24)  # 填充到所需维度
            except Exception as e:
                if show_progress:
                    print(f"频域特征提取失败: {e}")
                # 使用零填充
                features.extend([0] * 30)  # 所有频域特征的默认值
        
        # === 更多特征 (如果需要'full'特征集) ===
        if feature_type == 'full':
            try:
                # 计算短时能量
                frame_length = 1024
                hop_length = 512
                frames = np.array([audio_np[i:i+frame_length] for i in range(0, len(audio_np)-frame_length, hop_length)])
                frame_energy = np.sum(frames**2, axis=1) / frame_length
                
                # 添加短时能量统计
                features.append(np.mean(frame_energy))
                features.append(np.std(frame_energy))
                features.append(np.max(frame_energy))
                
                # 为匹配维度，添加填充
                features.extend([0] * 10)  # 填充到所需维度
            except Exception as e:
                if show_progress:
                    print(f"短时能量特征提取失败: {e}")
                # 使用零填充
                features.extend([0] * 13)  # 所有额外特征的默认值
        
        # 确保特征是有效的浮点数
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 确保特征维度符合预期
        if feature_type == 'statistical':
            expected_dim = 16
        elif feature_type == 'spectral':
            expected_dim = 34
        else:  # full
            expected_dim = 60
        
        # 调整特征维度
        if len(features) < expected_dim:
            # 添加零填充
            features = np.pad(features, (0, expected_dim - len(features)))
        elif len(features) > expected_dim:
            # 截断
            features = features[:expected_dim]
        
        # 返回特征张量
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)
        
        return features_tensor
    
    except Exception as e:
        if show_progress:
            print(f"特征提取失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 返回默认特征
        if feature_type == 'statistical':
            return torch.zeros((1, 16), dtype=torch.float32)
        elif feature_type == 'spectral':
            return torch.zeros((1, 34), dtype=torch.float32)
        else:  # full
            return torch.zeros((1, 60), dtype=torch.float32)
# 保留原有函数但优化机器学习特征提取
def extract_time_domain_features(audio, sample_rate, segment_length=1024, hop_length=512, for_ml=False):
    """
    从原始音频中提取时域特征，包括：
    - RMS能量
    - 过零率
    - 自相关系数
    - 短时能量
    
    这些特征更适合RNN模型和机器学习模型进行时序分析
    
    参数:
        audio: 音频数据
        sample_rate: 采样率
        segment_length: 分段长度
        hop_length: 帧移
        for_ml: 是否为机器学习模型提取特征
        
    返回:
        特征张量
    """
    # 如果是为机器学习模型提取特征，调用专用函数
    if for_ml:
        return extract_ml_features(audio, sample_rate, feature_type='statistical')
    
    # 以下代码保持不变...
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


def extract_sequential_features(audio, sample_rate, hop_length=160, for_ml=False):
    """
    将音频分成固定大小的序列，提取一组更为丰富的时域特征
    
    参数:
        audio: 音频数据
        sample_rate: 采样率
        hop_length: 帧移
        for_ml: 是否为机器学习模型提取特征
        
    返回:
        特征张量
    """
    # 如果是为机器学习模型提取特征，调用专用函数
    if for_ml:
        return extract_ml_features(audio, sample_rate, feature_type='full')
        
    # 以下代码保持不变...
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
def generate_raw_features(audio, sample_rate, frame_length=1024, hop_length=512):
    """
    将原始音频分帧为特征
    
    参数:
        audio: 音频数据 [1, samples]
        sample_rate: 采样率
        frame_length: 帧长度
        hop_length: 帧移
        
    返回:
        形状为 [num_frames, frame_length] 的张量
    """
    # 确保采样率为16kHz
    assert sample_rate == 16000, 'input audio sampling rate must be 16kHz'
    
    # 转为numpy以便处理
    audio_np = audio.squeeze().numpy()
    
    # 检查音频长度
    if len(audio_np) < frame_length:
        # 使用重复填充使音频至少达到frame_length长度
        repeat_factor = int(np.ceil(frame_length / max(1, len(audio_np))))
        audio_np = np.tile(audio_np if len(audio_np) > 0 else [0], repeat_factor)[:frame_length]
        print(f"警告: 音频太短，已填充到 {frame_length} 样本")
    
    # 分帧
    try:
        frames = librosa.util.frame(audio_np, frame_length=frame_length, hop_length=hop_length).T
    except Exception as e:
        print(f"分帧错误: {e}")
        # 返回一个空帧
        return torch.zeros((1, frame_length), dtype=torch.float32)
    
    # 转换为tensor
    frames_tensor = torch.from_numpy(frames).float()
    
    return frames_tensor

def save_features_to_csv(features_data, labels, filename, feature_type='full'):
    """
    将特征保存到CSV文件
    
    参数:
        features_data: 特征数组 [n_samples, n_features]
        labels: 标签数组 [n_samples]
        filename: 输出文件名
        feature_type: 特征类型
    """
    import pandas as pd
    import os
    import time
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 特征列名称
    feature_cols = [f'feat_{i}' for i in range(features_data.shape[1])]
    
    # 创建DataFrame
    df = pd.DataFrame(features_data, columns=feature_cols)
    df['label'] = labels
    
    # 添加元数据
    df.attrs['feature_type'] = feature_type
    df.attrs['creation_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # 保存到CSV
    df.to_csv(filename, index=False)
    
    print(f"特征已保存到 {filename}, 形状: {features_data.shape}")
    
    return df

def load_features_from_csv(filename):
    """
    从CSV文件加载特征
    
    参数:
        filename: CSV文件路径
        
    返回:
        features: 特征张量
        labels: 标签张量
    """
    import pandas as pd
    
    # 加载CSV
    df = pd.read_csv(filename)
    
    # 分离特征和标签
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    features = df[feature_cols].values
    labels = df['label'].values
    
    # 转换为tensor
    features_tensor = torch.from_numpy(features).float()
    labels_tensor = torch.from_numpy(labels).long()
    
    print(f"从 {filename} 加载了 {len(labels)} 个样本")
    
    return features_tensor, labels_tensor

# 添加到文件中
import multiprocessing
from joblib import Parallel, delayed

def extract_features_parallel(audio_data_list, sample_rate, feature_type='statistical', n_jobs=1, show_progress=True):
    """
    并行提取多个音频样本的特征
    
    参数:
        audio_data_list: 音频数据列表
        sample_rate: 采样率
        feature_type: 特征类型，支持 'statistical', 'spectral', 'full'
        n_jobs: 并行作业数，-1表示使用所有CPU
        show_progress: 是否显示进度条
        
    返回:
        features_list: 特征张量列表
    """
    try:
        # 如果没有多进程支持，导入相关模块
        import multiprocessing
        from joblib import Parallel, delayed
        
        # 确定CPU数量
        if n_jobs <= 0:
            n_jobs = max(1, multiprocessing.cpu_count() + n_jobs + 1)
        else:
            n_jobs = min(n_jobs, multiprocessing.cpu_count())
        
        if show_progress:
            print(f"使用 {n_jobs} 个CPU进行并行特征提取...")
        
        # 定义进度条更新处理器
        if show_progress:
            with tqdm(total=len(audio_data_list), desc="特征提取") as pbar:
                # 并行处理音频
                results = Parallel(n_jobs=n_jobs)(
                    delayed(extract_ml_features)(audio, sample_rate, feature_type, False)
                    for audio in audio_data_list
                )
                # 更新进度条
                pbar.update(len(audio_data_list))
        else:
            # 不显示进度条的版本
            results = Parallel(n_jobs=n_jobs)(
                delayed(extract_ml_features)(audio, sample_rate, feature_type, False)
                for audio in audio_data_list
            )
            
        return results
        
    except Exception as e:
        if show_progress:
            print(f"并行特征提取失败: {e}")
            import traceback
            traceback.print_exc()
            
        # 回退到串行处理
        if show_progress:
            print("回退到串行处理...")
            
        features_list = []
        for audio in tqdm(audio_data_list, desc="特征提取", disable=not show_progress):
            features = extract_ml_features(audio, sample_rate, feature_type, False)
            features_list.append(features)
            
        return features_list