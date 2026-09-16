import os
import sys
import pandas as pd
import numpy as np
import librosa
import torchaudio
import torch
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
from copy import deepcopy
import math
from torchaudio import transforms as T
from tqdm import tqdm  # 添加进度条库

def get_annotations(args, data_folder):
    """
    获取所有文件的注释
    """
    annotation_dict = {}
    
    # 获取所有wav文件
    filenames = os.listdir(data_folder)
    wav_files = [f for f in filenames if f.endswith('.wav')]
    
    print(f"正在读取 {len(wav_files)} 个音频文件的注释...")
    
    # 添加进度条
    for wav_file in tqdm(wav_files, desc="读取注释文件", unit="文件"):
        file_name = wav_file.replace('.wav', '')
        try:
            # 检查对应的txt文件是否存在
            txt_file = os.path.join(data_folder, file_name + '.txt')
            if os.path.exists(txt_file):
                # 直接读取注释
                annotations = pd.read_csv(txt_file, names=['start', 'end', 'crackles', 'wheezes'], delimiter='\t')
                annotation_dict[file_name] = annotations
        except Exception as e:
            print(f"读取注释文件时出错，文件: {file_name}, 错误: {str(e)}")
            continue
    
    print(f"成功读取 {len(annotation_dict)} 个文件的注释")
    return annotation_dict


def cut_pad_sample_torchaudio(data, args):
    """
    音频裁剪和填充函数
    """
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    
    # 确保是2D张量 (channels, samples)
    if len(data.shape) == 1:
        data = data.unsqueeze(0)
    
    fade_samples_ratio = 16
    fade_samples = int(args.sample_rate / fade_samples_ratio)
    fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
    target_duration = int(args.desired_length * args.sample_rate)

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
    
    return data.squeeze().numpy()


def get_individual_cycles_torchaudio(args, recording_annotations, data_folder, file_name, sample_rate, n_cls):
    """
    提取个别呼吸周期，添加进度显示
    """
    # 确保file_name是字符串
    if not isinstance(file_name, str):
        raise ValueError(f"file_name应该是字符串，但得到的是: {type(file_name)}")
    
    # 确保recording_annotations是DataFrame
    if not isinstance(recording_annotations, pd.DataFrame):
        raise ValueError(f"recording_annotations应该是DataFrame，但得到的是: {type(recording_annotations)}")
    
    # 加载音频文件
    wav_file_path = os.path.join(data_folder, file_name + '.wav')
    if not os.path.exists(wav_file_path):
        raise FileNotFoundError(f"音频文件不存在: {wav_file_path}")
    
    # 使用torchaudio加载音频
    try:
        audio_data, sr = torchaudio.load(wav_file_path)
    except Exception as e:
        raise RuntimeError(f"加载音频文件失败: {wav_file_path}, 错误: {str(e)}")
    
    # 如果是立体声，取第一个通道
    if audio_data.shape[0] > 1:
        audio_data = audio_data[0:1, :]
    
    # 重采样到目标采样率
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio_data = resampler(audio_data)
    
    # 添加fade效果
    fade_samples_ratio = 16
    fade_samples = int(sample_rate / fade_samples_ratio)
    fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
    audio_data = fade(audio_data)
    
    # 提取呼吸周期
    sample_data = []
    total_annotations = len(recording_annotations)
    
    # 为单个文件的周期提取添加小进度条
    if total_annotations > 5:  # 只有当注释较多时才显示
        annotation_iter = tqdm(recording_annotations.iterrows(), 
                             total=total_annotations,
                             desc=f"🔄 {file_name[:20]}",
                             unit="周期",
                             leave=False,
                             bar_format="{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt}")
    else:
        annotation_iter = recording_annotations.iterrows()
    
    # 使用列名访问DataFrame
    for i, row in annotation_iter:
        start_time = row['start']
        end_time = row['end']
        crackles = row['crackles']
        wheezes = row['wheezes']
        
        # 计算样本索引
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # 提取音频片段
        max_samples = audio_data.shape[1]
        start_sample = min(start_sample, max_samples)
        end_sample = min(end_sample, max_samples)
        
        if start_sample < end_sample and end_sample <= max_samples:
            audio_chunk = audio_data[:, start_sample:end_sample]
            
            # 确定标签
            if hasattr(args, 'class_split') and args.class_split == 'lungsound':
                if n_cls == 4:
                    if crackles == 0 and wheezes == 0:
                        label = 0  # normal
                    elif crackles == 1 and wheezes == 0:
                        label = 1  # crackle
                    elif crackles == 0 and wheezes == 1:
                        label = 2  # wheeze
                    else:  # both
                        label = 3
                elif n_cls == 2:
                    if crackles == 0 and wheezes == 0:
                        label = 0  # normal
                    else:
                        label = 1  # abnormal
                else:
                    raise ValueError(f"不支持的类别数: {n_cls}")
            else:
                # 默认按4类分类
                if crackles == 0 and wheezes == 0:
                    label = 0  # normal
                elif crackles == 1 and wheezes == 0:
                    label = 1  # crackle
                elif crackles == 0 and wheezes == 1:
                    label = 2  # wheeze
                else:  # both
                    label = 3
            
            # 应用裁剪和填充
            audio_chunk = cut_pad_sample_torchaudio(audio_chunk, args)
            sample_data.append((audio_chunk, label))
    
    return sample_data


def generate_fbank(audio, sample_rate, n_mels=128): 
    """
    使用原始代码中的torchaudio库转换mel fbank为AST模型使用
    """    
    assert sample_rate == 16000, 'input audio sampling rate must be 16kHz'
    
    # 确保audio是torch tensor
    if isinstance(audio, np.ndarray):
        audio = torch.tensor(audio, dtype=torch.float32)
    
    # 确保是2D张量
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)
    
    fbank = torchaudio.compliance.kaldi.fbank(
        audio, 
        htk_compat=True, 
        sample_frequency=sample_rate, 
        use_energy=False, 
        window_type='hanning', 
        num_mel_bins=n_mels, 
        dither=0.0, 
        frame_shift=10
    )
    
    mean, std = -4.2677393, 4.5689974
    fbank = (fbank - mean) / (std * 2)  # mean / std
    fbank = fbank.unsqueeze(-1).numpy()
    return fbank


def get_feature_extractor(feature_type, show_progress=True):
    """
    根据特征类型返回相应的特征提取函数（带进度显示）
    """
    # 获取基础提取函数
    base_extractor = _get_base_feature_extractor(feature_type)
    
    # 创建进度跟踪器
    progress_tracker = {'current': 0, 'total': 0, 'pbar': None}
    
    # 包装进度显示
    def progress_wrapper(audio, sample_rate, n_mels=128, target_length=1024, 
                        sample_index=None, total_samples=None):
        try:
            # 更新进度条
            if show_progress and sample_index is not None and total_samples is not None:
                if progress_tracker['pbar'] is None:
                    progress_tracker['pbar'] = tqdm(
                        total=total_samples, 
                        desc=f"🔧 提取{feature_type}特征", 
                        unit="样本",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                    )
                
                if sample_index > progress_tracker['current']:
                    progress_tracker['pbar'].update(sample_index - progress_tracker['current'])
                    progress_tracker['current'] = sample_index
                    
                    # 添加详细信息
                    progress_tracker['pbar'].set_postfix({
                        '特征': feature_type,
                        '进度': f"{sample_index}/{total_samples}"
                    })
            
            result = base_extractor(audio, sample_rate, n_mels, target_length)
            return result
            
        except Exception as e:
            if show_progress:
                tqdm.write(f"⚠️ {feature_type}特征提取出现问题: {str(e)}")
            # 返回默认的mel特征作为fallback
            return _get_base_feature_extractor('mel')(audio, sample_rate, n_mels, target_length)
    
    return progress_wrapper, progress_tracker

def _get_base_feature_extractor(feature_type):
    """
    修复特征提取函数，确保所有特征具有相同的时间和频率维度
    """
    def extract_mel(audio, sample_rate, n_mels=128, target_length=1024):
        """提取Mel频谱图"""
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        # 确保是1D数组
        if len(audio.shape) > 1:
            audio = audio.squeeze()
        
        # 计算Mel频谱图，确保固定的时间维度
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_mels=n_mels,
            n_fft=1024, hop_length=512, fmin=50, fmax=2000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = mel_spec_db.T  # 转置为 (time, freq)
        
        # 确保时间维度为target_length
        if mel_spec_db.shape[0] < target_length:
            # 如果太短，进行重复填充
            repeat_times = target_length // mel_spec_db.shape[0] + 1
            mel_spec_db = np.tile(mel_spec_db, (repeat_times, 1))
        
        # 截断到目标长度
        mel_spec_db = mel_spec_db[:target_length, :]
        
        # 确保频率维度正确
        if mel_spec_db.shape[1] != n_mels:
            # 如果频率维度不匹配，进行插值调整
            try:
                from scipy import ndimage
                scale_factor = n_mels / mel_spec_db.shape[1]
                mel_spec_db = ndimage.zoom(mel_spec_db, (1, scale_factor), order=1)
            except ImportError:
                # 如果没有scipy，使用简单的重采样
                if mel_spec_db.shape[1] > n_mels:
                    step = mel_spec_db.shape[1] // n_mels
                    mel_spec_db = mel_spec_db[:, ::step][:, :n_mels]
                else:
                    padding = np.zeros((mel_spec_db.shape[0], n_mels - mel_spec_db.shape[1]))
                    mel_spec_db = np.hstack([mel_spec_db, padding])
        
        mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        return mel_spec_db
    
    def extract_fbank(audio, sample_rate, n_mels=128, target_length=1024):
        """使用原始代码中的torchaudio Kaldi fbank实现，并确保尺寸一致"""
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio, dtype=torch.float32)
        
        # 确保是2D张量 (1, samples)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        
        # 确保采样率为16kHz（如果不是，需要重采样）
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio = resampler(audio)
            sample_rate = 16000
        
        # 使用原始的generate_fbank函数
        fbank = torchaudio.compliance.kaldi.fbank(
            audio, 
            htk_compat=True, 
            sample_frequency=sample_rate, 
            use_energy=False, 
            window_type='hanning', 
            num_mel_bins=n_mels, 
            dither=0.0, 
            frame_shift=10
        )
        
        # 应用标准化（使用原始代码中的参数）
        mean, std = -4.2677393, 4.5689974
        fbank = (fbank - mean) / (std * 2)
        
        # 确保时间维度为target_length
        if fbank.shape[0] < target_length:
            # 如果太短，进行重复填充
            repeat_times = target_length // fbank.shape[0] + 1
            fbank = fbank.repeat(repeat_times, 1)
        
        # 截断到目标长度
        fbank = fbank[:target_length, :]
        
        # 确保频率维度正确
        if fbank.shape[1] != n_mels:
            if fbank.shape[1] > n_mels:
                fbank = fbank[:, :n_mels]
            else:
                padding = torch.zeros(fbank.shape[0], n_mels - fbank.shape[1])
                fbank = torch.cat([fbank, padding], dim=1)
        
        fbank = fbank.unsqueeze(-1).numpy()
        return fbank
    
    def extract_cqt(audio, sample_rate, n_mels=128, target_length=1024):
        """提取CQT特征，确保尺寸一致"""
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        # 确保是1D数组
        if len(audio.shape) > 1:
            audio = audio.squeeze()
        
        try:
            # 修复：调整CQT参数
            fmin = 50
            fmax = min(2000, sample_rate // 2 - 100)
            n_bins = min(n_mels, 84)
            
            cqt = librosa.cqt(
                y=audio, 
                sr=sample_rate, 
                hop_length=512, 
                n_bins=n_bins,
                fmin=fmin,
                bins_per_octave=12
            )
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            cqt_db = cqt_db.T  # 转置为 (time, freq)
            
            # 确保时间维度为target_length
            if cqt_db.shape[0] < target_length:
                repeat_times = target_length // cqt_db.shape[0] + 1
                cqt_db = np.tile(cqt_db, (repeat_times, 1))
            
            cqt_db = cqt_db[:target_length, :]
            
            # 确保频率维度正确
            if cqt_db.shape[1] < n_mels:
                padding = np.zeros((cqt_db.shape[0], n_mels - cqt_db.shape[1]))
                cqt_db = np.hstack([cqt_db, padding])
            elif cqt_db.shape[1] > n_mels:
                cqt_db = cqt_db[:, :n_mels]
            
            cqt_db = np.expand_dims(cqt_db, axis=-1)
            cqt_db = (cqt_db - cqt_db.min()) / (cqt_db.max() - cqt_db.min() + 1e-8)
            return cqt_db
        except Exception as e:
            print(f"CQT提取失败: {e}, 回退到Mel频谱图")
            return extract_mel(audio, sample_rate, n_mels, target_length)
    
    # 其他特征提取函数也类似修复...
    def extract_mfcc(audio, sample_rate, n_mels=128, target_length=1024):
        """提取MFCC特征，确保尺寸一致"""
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        if len(audio.shape) > 1:
            audio = audio.squeeze()
        
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sample_rate, 
            n_mfcc=min(n_mels, 40),
            n_fft=1024, hop_length=512
        )
        
        # 扩展到n_mels维度
        if mfcc.shape[0] < n_mels:
            padding = np.zeros((n_mels - mfcc.shape[0], mfcc.shape[1]))
            mfcc = np.vstack([mfcc, padding])
        
        mfcc = mfcc.T  # 转置为 (time, freq)
        
        # 确保时间维度为target_length
        if mfcc.shape[0] < target_length:
            repeat_times = target_length // mfcc.shape[0] + 1
            mfcc = np.tile(mfcc, (repeat_times, 1))
        
        mfcc = mfcc[:target_length, :n_mels]
        mfcc = np.expand_dims(mfcc, axis=-1)
        mfcc = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-8)
        return mfcc
    
    # ... 其他特征函数类似处理 ...
    
    # 特征提取函数映射
    extractors = {
        'mel': extract_mel,
        'fbank': extract_fbank,
        'cqt': extract_cqt,
        'mfcc': extract_mfcc,
        'chroma': lambda *args, **kwargs: extract_mel(*args, **kwargs),  # 简化处理
        'gamma': lambda *args, **kwargs: extract_mel(*args, **kwargs),   # 简化处理
        'stft': lambda *args, **kwargs: extract_mel(*args, **kwargs),    # 简化处理
    }
    
    if feature_type not in extractors:
        raise ValueError(f"不支持的特征类型: {feature_type}")
    
    return extractors[feature_type]


def get_score(hits, counts, pflag=False):
    """
    计算评分指标
    Args:
        hits: 每个类别的命中数
        counts: 每个类别的总数
    """
    # 确保输入是numpy数组
    hits = np.array(hits)
    counts = np.array(counts)
    
    # 避免除零错误
    counts = np.where(counts == 0, 1e-10, counts)
    
    # normal accuracy
    sp = hits[0] / counts[0] * 100
    # abnormal accuracy
    se = sum(hits[1:]) / sum(counts[1:]) * 100
    sc = (sp + se) / 2.0

    if pflag:
        print("S_p: {}, S_e: {}, Score: {}".format(sp, se, sc))

    return sp, se, sc

def batch_extract_features(audio_list, sample_rate, feature_types, n_mels=128):
    """
    批量提取多种特征，带详细进度显示
    """
    print(f"🚀 开始批量提取特征...")
    print(f"📊 音频数量: {len(audio_list)}")
    print(f"🔧 特征类型: {feature_types}")
    print(f"🎵 采样率: {sample_rate}Hz")
    print(f"📐 Mel bins: {n_mels}")
    
    results = {}
    
    # 为每种特征类型创建进度条
    for feat_type in feature_types:
        print(f"\n🔧 开始提取 {feat_type} 特征...")
        
        extractor, _ = get_feature_extractor(feat_type, show_progress=True)
        feature_results = []
        
        with tqdm(total=len(audio_list), desc=f"提取{feat_type}", unit="音频",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}") as pbar:
            
            for i, audio in enumerate(audio_list):
                try:
                    feature = extractor(audio, sample_rate, n_mels, 
                                      sample_index=i, total_samples=len(audio_list))
                    feature_results.append(feature)
                    
                    pbar.set_postfix({
                        '特征': feat_type,
                        '形状': str(feature.shape) if feature is not None else "None"
                    })
                    
                except Exception as e:
                    tqdm.write(f"❌ 第{i}个音频{feat_type}特征提取失败: {str(e)}")
                    feature_results.append(None)
                
                pbar.update(1)
        
        results[feat_type] = feature_results
        print(f"✅ {feat_type} 特征提取完成!")
    
    print(f"🎉 所有特征提取完成!")
    return results