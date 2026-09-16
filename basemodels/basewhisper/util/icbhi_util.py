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

from .augmentation import augment_raw_audio

__all__ = ['get_annotations', 'get_individual_cycles_torchaudio', 'generate_fbank', 'get_score', 
           'generate_whisper_features', 'cut_pad_sample_whisper']  # 添加Whisper相关函数


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
    """原有的音频切片和填充函数"""
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


def cut_pad_sample_whisper(data, args, whisper_sample_rate=16000):
    """
    为Whisper优化的音频切片和填充函数
    确保输出符合Whisper的要求
    """
    # Whisper标准设置
    fade_samples_ratio = 16
    fade_samples = int(whisper_sample_rate / fade_samples_ratio)
    fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
    
    # 计算目标长度（使用Whisper采样率）
    target_duration = args.desired_length * whisper_sample_rate
    
    # 确保输入是正确的维度
    if data.dim() == 1:
        data = data.unsqueeze(0)
    
    if data.shape[-1] > target_duration:
        # 从中心裁剪（Whisper推荐）
        start_idx = (data.shape[-1] - target_duration) // 2
        data = data[..., start_idx:start_idx + target_duration]
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
    
    # 确保输出是单通道
    if data.shape[0] > 1:
        data = data.mean(dim=0, keepdim=True)
    
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
        # 根据是否有whisper相关属性选择处理函数
        if hasattr(args, 'use_whisper') and args.use_whisper:
            data = cut_pad_sample_whisper(data, args)
        else:
            data = cut_pad_sample_torchaudio(data, args)
        padded_sample_data.append((data, label))

    return padded_sample_data


def generate_fbank(audio, sample_rate, n_mels=128): 
    """
    use torchaudio library to convert mel fbank for AST model
    """    
    assert sample_rate == 16000, 'input audio sampling rate must be 16kHz'
    fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True, sample_frequency=sample_rate, use_energy=False, window_type='hanning', num_mel_bins=n_mels, dither=0.0, frame_shift=10)
    
    mean, std =  -4.2677393, 4.5689974
    fbank = (fbank - mean) / (std * 2) # mean / std
    fbank = fbank.unsqueeze(-1).numpy()
    return fbank 


def generate_whisper_features(audio, sample_rate=16000, n_mels=80):
    """
    为Whisper生成mel频谱特征
    使用Whisper标准的80个mel频率bins
    """
    assert sample_rate == 16000, 'Whisper requires 16kHz sampling rate'
    
    # 使用Whisper的标准参数
    fbank = torchaudio.compliance.kaldi.fbank(
        audio, 
        htk_compat=True, 
        sample_frequency=sample_rate, 
        use_energy=False, 
        window_type='hanning', 
        num_mel_bins=n_mels,  # Whisper使用80个mel bins
        dither=0.0, 
        frame_shift=10
    )
    
    # Whisper特定的归一化（可选）
    # 注意：Whisper模型通常有自己的归一化，这里可以选择是否预归一化
    whisper_mean, whisper_std = -4.2677393, 4.5689974
    fbank = (fbank - whisper_mean) / (whisper_std * 2)
    fbank = fbank.unsqueeze(-1).numpy()
    
    return fbank


def prepare_audio_for_whisper(audio_path, target_sample_rate=16000, target_length=8):
    """
    为Whisper准备音频数据的便捷函数
    
    Args:
        audio_path: 音频文件路径
        target_sample_rate: 目标采样率（Whisper使用16kHz）
        target_length: 目标长度（秒）
    
    Returns:
        audio: 处理后的音频tensor
    """
    # 加载音频
    sr = librosa.get_samplerate(audio_path)
    audio, _ = torchaudio.load(audio_path)
    
    # 重采样到16kHz
    if sr != target_sample_rate:
        resample = T.Resample(sr, target_sample_rate)
        audio = resample(audio)
    
    # 转换为单声道
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # 调整长度
    target_samples = target_length * target_sample_rate
    if audio.shape[1] > target_samples:
        # 从中心裁剪
        start_idx = (audio.shape[1] - target_samples) // 2
        audio = audio[:, start_idx:start_idx + target_samples]
    elif audio.shape[1] < target_samples:
        # 重复填充
        ratio = math.ceil(target_samples / audio.shape[1])
        audio = audio.repeat(1, ratio)
        audio = audio[:, :target_samples]
    
    return audio.squeeze(0)  # 返回1D tensor


def validate_whisper_audio(audio, sample_rate=16000, expected_length=8):
    """
    验证音频是否符合Whisper要求
    
    Args:
        audio: 音频tensor
        sample_rate: 采样率
        expected_length: 期望长度（秒）
    
    Returns:
        bool: 是否符合要求
        str: 验证信息
    """
    issues = []
    
    # 检查采样率
    if sample_rate != 16000:
        issues.append(f"采样率应为16000Hz，当前为{sample_rate}Hz")
    
    # 检查音频长度
    expected_samples = expected_length * sample_rate
    actual_samples = len(audio) if audio.dim() == 1 else audio.shape[-1]
    
    if abs(actual_samples - expected_samples) > sample_rate * 0.1:  # 允许0.1秒误差
        issues.append(f"音频长度不匹配，期望{expected_samples}个样本，实际{actual_samples}个样本")
    
    # 检查音频维度
    if audio.dim() > 2 or (audio.dim() == 2 and audio.shape[0] > 1):
        issues.append("音频应为单声道（1D tensor或shape为[1,N]的2D tensor）")
    
    # 检查数值范围
    if audio.abs().max() > 10:  # 合理的音频幅度范围
        issues.append(f"音频幅度过大，最大值为{audio.abs().max():.2f}")
    
    if len(issues) == 0:
        return True, "音频格式符合Whisper要求"
    else:
        return False, "; ".join(issues)


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
""" Whisper compatibility functions """
def create_whisper_compatible_args(args):
    """
    创建Whisper兼容的参数对象
    
    Args:
        args: 原始参数对象
    
    Returns:
        args: 修改后的参数对象
    """
    # 复制原始args以避免修改原对象
    import copy
    whisper_args = copy.deepcopy(args)
    
    # 设置Whisper特定参数
    whisper_args.sample_rate = 16000  # Whisper标准采样率
    whisper_args.use_whisper = True  # 标记使用Whisper
    
    # 如果没有设置mel频率bins数量，使用Whisper标准
    if not hasattr(whisper_args, 'whisper_n_mels'):
        whisper_args.whisper_n_mels = 80
    
    return whisper_args


def test_whisper_audio_processing():
    """测试Whisper音频处理功能"""
    print("🧪 测试Whisper音频处理功能...")
    
    try:
        # 创建测试音频
        test_audio = torch.randn(1, 16000 * 8)  # 8秒，16kHz
        
        # 测试音频验证
        is_valid, message = validate_whisper_audio(test_audio.squeeze(), 16000, 8)
        print(f"音频验证: {'✅' if is_valid else '❌'} {message}")
        
        # 测试Whisper特征生成
        features = generate_whisper_features(test_audio, 16000, 80)
        print(f"✅ Whisper特征生成成功: {features.shape}")
        
        # 创建测试参数
        class TestArgs:
            sample_rate = 16000
            desired_length = 8
            pad_types = 'repeat'
            use_whisper = True
        
        args = TestArgs()
        
        # 测试Whisper音频切片
        processed = cut_pad_sample_whisper(test_audio, args)
        print(f"✅ Whisper音频处理成功: {test_audio.shape} -> {processed.shape}")
        
        print("✅ 所有Whisper音频处理测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ Whisper音频处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 测试ICBHI工具函数...")
    
    # 运行Whisper音频处理测试
    test_whisper_audio_processing()
    
    print("\n📝 新增的Whisper支持功能:")
    print("- generate_whisper_features(): 生成Whisper兼容的mel特征")
    print("- cut_pad_sample_whisper(): Whisper优化的音频处理")
    print("- prepare_audio_for_whisper(): 音频预处理便捷函数")
    print("- validate_whisper_audio(): 音频格式验证")
    print("- create_whisper_compatible_args(): 创建Whisper兼容参数")
    
    print("\n🎉 ICBHI工具函数测试完成!")
# ==========================================================================