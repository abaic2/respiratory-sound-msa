import os
import random
import numpy as np
import traceback

import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from copy import deepcopy
from PIL import Image

from .icbhi_util import (get_annotations, generate_fbank, get_individual_cycles_torchaudio, 
                        cut_pad_sample_torchaudio, get_feature_extractor)
from .augmentation import augment_raw_audio

# 支持的所有特征类型
SUPPORTED_FEATURES = ['fbank', 'mel', 'cqt', 'gamma', 'mfcc', 'chroma', 'stft']

# 原有ICBHIDataset类保持完全不变
class ICBHIDataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True, mean_std=False):
        data_folder = os.path.join(args.data_folder, 'icbhi_dataset')
        test_fold = args.test_fold
        
        self.data_folder = data_folder
        self.train_flag = train_flag
        self.split = 'train' if train_flag else 'test'
        self.transform = transform
        self.args = args
        self.mean_std = mean_std

        # parameters for spectrograms
        self.sample_rate = args.sample_rate
        self.desired_length = args.desired_length
        self.pad_types = args.pad_types
        self.nfft = args.nfft
        self.hop = self.nfft // 2
        self.n_mels = args.n_mels
        self.f_min = 50
        self.f_max = 2000

        filenames = os.listdir(data_folder)
        # 只包含符合ICBHI命名格式的文件（形如：101_1b1_Al_sc_Meditron）
        valid_filenames = []
        for f in filenames:
            if '.wav' in f or '.txt' in f:
                base_name = f.strip().split('.')[0]
                # 检查是否符合预期格式：包含下划线且可以分割成至少4-5个部分
                if '_' in base_name:
                    parts = base_name.split('_')
                    if len(parts) >= 4:  # 至少应有患者ID、记录索引、胸部位置和采集模式
                        valid_filenames.append(base_name)

        filenames = set(valid_filenames)
        filenames = sorted(filenames)

        # 从指定的traintest.txt文件加载预定义的训练/测试集划分
        train_test_file = '/home/u202420085410012/55555/bishe/data/ICBHI_final_database/icbhi_dataset/traintest.txt'
        
        # 初始化患者字典
        patient_dict = {}
        
        if os.path.exists(train_test_file):
            if print_flag:
                print(f"使用预定义的训练/测试集划分: {train_test_file}")
            
            # 读取预定义的划分
            train_files = []
            test_files = []
            
            with open(train_test_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('//') or line.startswith('#'):
                    continue
                    
                # 使用制表符分割，而不是逗号
                parts = line.split('\t')
                if len(parts) >= 2:
                    file_name = parts[0].strip()
                    split = parts[1].strip().lower()
                    
                    if split == 'train':
                        train_files.append(file_name)
                    elif split == 'test':
                        test_files.append(file_name)
            
            # 根据当前运行模式（训练或测试）选择适当的文件
            if train_flag:
                for f in train_files:
                    patient_dict[f] = 'train'
                if print_flag:
                    print(f"从预定义文件中加载了 {len(train_files)} 个训练集文件")
            else:
                for f in test_files:
                    patient_dict[f] = 'test'
                if print_flag:
                    print(f"从预定义文件中加载了 {len(test_files)} 个测试集文件")
        else:
            # 如果找不到预定义文件，则回退到原始的60-40随机划分逻辑
            if print_flag:
                print(f"未找到预定义划分文件 {train_test_file}, 回退到60-40随机划分")
                
            indices = [i for i, file in enumerate(filenames)]
            random.Random(1).shuffle(indices)
            train_size = int(len(indices) * 0.6)
            train_idx = indices[:train_size]
            test_idx = indices[train_size:]
            train_files = [filenames[i] for i in train_idx]
            test_files = [filenames[i] for i in test_idx]
            for f in train_files:
                if train_flag:
                    patient_dict[f] = 'train'
            for f in test_files:
                if not train_flag:
                    patient_dict[f] = 'test'

        if print_flag:
            print('*' * 20)
            if os.path.exists(train_test_file):
                print('使用预定义文件划分, test_fold {}'.format(test_fold))
            else:
                print('使用随机60-40划分, test_fold {}'.format(test_fold))
            print('文件数量在 {} 数据集中: {}'.format(self.split, len(patient_dict)))

        annotation_dict = get_annotations(args, data_folder)

        self.filenames = []
        for f in filenames:
            idx = f.split('_')[0] if test_fold in ['0', '1', '2', '3', '4'] else f
            if hasattr(self, 'file_to_device') and args.stetho_id >= 0:
                if idx in patient_dict and self.file_to_device[f] == args.stetho_id:
                    self.filenames.append(f)
            else:
                if idx in patient_dict:
                    self.filenames.append(f)
        
        self.audio_data = []
        self.labels = []

        if print_flag:
            print('*' * 20)  
            print("提取单独的呼吸周期...")

        self.cycle_list = []
        self.filename_to_label = {}
        self.classwise_cycle_list = [[] for _ in range(args.n_cls)]
        self.filenames.sort()

        print(f"开始加载ICBHI数据集...")
        print(f"数据文件夹: {data_folder}")
        print(f"测试折: {test_fold}")
        print(f"类别数: {args.n_cls}")
        print(f"目标长度: {self.desired_length}s")
        print(f"采样率: {self.sample_rate}Hz")
        
        # 获取注释
        print("正在获取文件注释...")
        annotation_dict = get_annotations(args, data_folder)
        print(f"成功获取 {len(annotation_dict)} 个文件的注释")
        
        # 处理文件列表
        print(f"开始处理 {len(self.filenames)} 个音频文件...")
        
        # 添加总体进度条
        with tqdm(total=len(self.filenames), desc="处理音频文件", unit="文件") as pbar:
            for idx, filename in enumerate(self.filenames):
                pbar.set_postfix({"当前文件": filename[:20] + "..." if len(filename) > 20 else filename})
                
                self.filename_to_label[filename] = []
                try:
                    # 处理单个文件
                    sample_data = get_individual_cycles_torchaudio(
                        args, annotation_dict[filename], data_folder, filename, 
                        args.sample_rate, args.n_cls
                    )
                    
                    cycles_with_labels = [(data[0], data[1]) for data in sample_data]
                    self.cycle_list.extend(cycles_with_labels)
                    
                    for d in cycles_with_labels:
                        self.filename_to_label[filename].append(d[1])
                        self.classwise_cycle_list[d[1]].append(d)
                    
                    pbar.set_postfix({
                        "当前文件": filename[:15] + "...",
                        "周期数": len(cycles_with_labels),
                        "总周期": len(self.cycle_list)
                    })
                    
                except Exception as e:
                    tqdm.write(f"❌ 处理文件 {filename} 时出错: {str(e)}")
                    if print_flag:
                        traceback.print_exc()
                
                pbar.update(1)
        
        print(f"✅ 数据加载完成!")
        print(f"📊 总共处理了 {len(self.cycle_list)} 个音频周期")
        print(f"📂 各类别分布: {[len(self.classwise_cycle_list[i]) for i in range(args.n_cls)]}")
        
        for sample in self.cycle_list:
            self.audio_data.append(sample)

        if len(self.audio_data) == 0:
            raise ValueError("没有加载到任何音频数据，请检查数据路径和预处理过程")

        self.class_nums = np.zeros(args.n_cls)
        for sample in self.audio_data:
            self.class_nums[sample[1]] += 1
            self.labels.append(sample[1])
        self.class_ratio = self.class_nums / sum(self.class_nums) * 100
        
        if print_flag:
            print('[预处理后的 {} 数据集信息]'.format(self.split))
            print('音频数据总数: {}'.format(len(self.audio_data)))
            for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
                print('类别 {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.cls_list[i]+')', int(n), p))    
        
        if print_flag:
            print(f"准备生成 {len(self.audio_data)} 个音频图像...")
        
        self.audio_images = []
        for index in range(len(self.audio_data)):
            try:
                audio, label = self.audio_data[index][0], self.audio_data[index][1]

                audio_image = []
                for aug_idx in range(self.args.raw_augment+1): 
                    if aug_idx > 0:
                        if self.train_flag and not mean_std:
                            audio = augment_raw_audio(audio, self.sample_rate, self.args)
                            audio = cut_pad_sample_torchaudio(torch.tensor(audio), args)
                        else:
                            audio_image.append(None)
                            continue
                    
                    image = generate_fbank(audio, self.sample_rate, n_mels=self.n_mels)
                    audio_image.append(image)
                self.audio_images.append((audio_image, label))
                
                if index == 0 and print_flag:
                    print(f"第一个音频图像形状: {audio_image[0].shape}")
                    
            except Exception as e:
                print(f"处理第 {index} 个音频样本时出错: {str(e)}")
                if print_flag:
                    traceback.print_exc()
                # 跳过有问题的样本，继续处理其他样本
                continue
        
        # 检查是否至少有一个有效的音频图像
        if len(self.audio_images) == 0:
            raise ValueError("没有成功生成任何音频图像，请检查音频处理和特征提取过程")
        
        # 确保第一个图像是有效的
        if self.audio_images[0][0][0] is None:
            # 寻找第一个非None的图像
            valid_image_found = False
            for img_data in self.audio_images:
                for img in img_data[0]:
                    if img is not None:
                        self.h, self.w, _ = img.shape
                        valid_image_found = True
                        break
                if valid_image_found:
                    break
            
            if not valid_image_found:
                raise ValueError("所有音频图像均无效，无法确定图像形状")
        else:
            self.h, self.w, _ = self.audio_images[0][0][0].shape
        
        if print_flag:
            print(f"成功生成 {len(self.audio_images)} 个音频图像，形状: ({self.h}, {self.w})")

    def __getitem__(self, index):
        audio_images, label = self.audio_images[index][0], self.audio_images[index][1]

        if self.args.raw_augment and self.train_flag and not self.mean_std:
            aug_idx = random.randint(0, self.args.raw_augment)
            audio_image = audio_images[aug_idx]
        else:
            audio_image = audio_images[0]
        
        if self.transform is not None:
            audio_image = self.transform(audio_image)
        
        return audio_image, label

    def __len__(self):
        return len(self.audio_data)


# ==========================================================================
""" 新增：双特征数据集类 """

class ICBHIDualFeatureDataset(Dataset):
    """
    双特征数据集，支持任意两种特征的组合
    """
    def __init__(self, train_flag, transform, args, feature_combo='mel+cqt', 
                 print_flag=True, mean_std=False):
        
        if print_flag:
            print(f"正在初始化双特征数据集，特征组合: {feature_combo}")
        
        # 解析特征组合
        if '+' in feature_combo:
            self.feature_types = feature_combo.split('+')
        else:
            raise ValueError(f"特征组合格式错误: {feature_combo}，应为 'type1+type2' 格式")
        
        # 验证特征类型
        for feat_type in self.feature_types:
            if feat_type not in SUPPORTED_FEATURES:
                raise ValueError(f"不支持的特征类型: {feat_type}. "
                               f"支持的类型: {SUPPORTED_FEATURES}")
        
        self.feature_combo = feature_combo
        self.train_flag = train_flag
        self.transform = transform
        self.args = args
        self.mean_std = mean_std
        
        # 复制ICBHIDataset的数据准备逻辑
        data_folder = os.path.join(args.data_folder, 'icbhi_dataset')
        test_fold = args.test_fold
        
        self.data_folder = data_folder
        self.split = 'train' if train_flag else 'test'
        
        # 修复：正确初始化所有参数
        self.sample_rate = args.sample_rate
        self.desired_length = args.desired_length
        self.pad_types = args.pad_types
        self.nfft = args.nfft
        self.hop = self.nfft // 2
        self.n_mels = args.n_mels
        self.f_min = 50
        self.f_max = 2000

        filenames = os.listdir(data_folder)
        valid_filenames = []
        for f in filenames:
            if '.wav' in f or '.txt' in f:
                base_name = f.strip().split('.')[0]
                if '_' in base_name:
                    parts = base_name.split('_')
                    if len(parts) >= 4:
                        valid_filenames.append(base_name)

        filenames = set(valid_filenames)
        filenames = sorted(filenames)

        # 训练/测试集划分逻辑（与原数据集相同）
        train_test_file = '/home/yujieyang/bishe/data/ICBHI_final_database/icbhi_dataset/traintest.txt'
        patient_dict = {}
        
        if os.path.exists(train_test_file):
            if print_flag:
                print(f"使用预定义的训练/测试集划分: {train_test_file}")
            
            train_files = []
            test_files = []
            
            with open(train_test_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('//') or line.startswith('#'):
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 2:
                    file_name = parts[0].strip()
                    split = parts[1].strip().lower()
                    
                    if split == 'train':
                        train_files.append(file_name)
                    elif split == 'test':
                        test_files.append(file_name)
            
            if train_flag:
                for f in train_files:
                    patient_dict[f] = 'train'
                if print_flag:
                    print(f"从预定义文件中加载了 {len(train_files)} 个训练集文件")
            else:
                for f in test_files:
                    patient_dict[f] = 'test'
                if print_flag:
                    print(f"从预定义文件中加载了 {len(test_files)} 个测试集文件")
        else:
            if print_flag:
                print(f"未找到预定义划分文件 {train_test_file}, 回退到60-40随机划分")
                
            indices = [i for i, file in enumerate(filenames)]
            random.Random(1).shuffle(indices)
            train_size = int(len(indices) * 0.6)
            train_idx = indices[:train_size]
            test_idx = indices[train_size:]
            train_files = [filenames[i] for i in train_idx]
            test_files = [filenames[i] for i in test_idx]
            for f in train_files:
                if train_flag:
                    patient_dict[f] = 'train'
            for f in test_files:
                if not train_flag:
                    patient_dict[f] = 'test'

        if print_flag:
            print('*' * 20)
            print('双特征数据集, 特征组合: {}, test_fold {}'.format(feature_combo, test_fold))
            print('文件数量在 {} 数据集中: {}'.format(self.split, len(patient_dict)))

        annotation_dict = get_annotations(args, data_folder)

        self.filenames = []
        for f in filenames:
            idx = f.split('_')[0] if test_fold in ['0', '1', '2', '3', '4'] else f
            if hasattr(self, 'file_to_device') and args.stetho_id >= 0:
                if idx in patient_dict and self.file_to_device[f] == args.stetho_id:
                    self.filenames.append(f)
            else:
                if idx in patient_dict:
                    self.filenames.append(f)
        
        self.audio_data = []
        self.labels = []

        if print_flag:
            print('*' * 20)  
            print("提取单独的呼吸周期...")

        self.cycle_list = []
        self.filename_to_label = {}
        self.classwise_cycle_list = [[] for _ in range(args.n_cls)]
        self.filenames.sort()

        print(f"🔥 开始加载双特征数据集...")
        print(f"📁 数据文件夹: {data_folder}")
        print(f"🔍 特征组合: {feature_combo}")
        print(f"📊 类别数: {args.n_cls}")
        print(f"⏱️  目标长度: {self.desired_length}s")
        print(f"🎵 采样率: {self.sample_rate}Hz")
        
        # 获取注释
        print("📋 正在获取文件注释...")
        annotation_dict = get_annotations(args, data_folder)
        print(f"✅ 成功获取 {len(annotation_dict)} 个文件的注释")
        
        # 处理文件列表
        print(f"🚀 开始处理 {len(self.filenames)} 个音频文件...")
        
        # 添加详细的进度条
        with tqdm(total=len(self.filenames), desc="🎵 处理双特征音频", unit="文件", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}") as pbar:
            
            successful_files = 0
            total_cycles = 0
            
            for idx, filename in enumerate(self.filenames):
                pbar.set_postfix({
                    "当前": filename[:15] + "...",
                    "成功": successful_files,
                    "周期": total_cycles
                })
                
                self.filename_to_label[filename] = []
                try:
                    # 处理单个文件
                    sample_data = get_individual_cycles_torchaudio(
                        args, annotation_dict[filename], data_folder, filename, 
                        args.sample_rate, args.n_cls
                    )
                    
                    cycles_with_labels = [(data[0], data[1]) for data in sample_data]
                    self.cycle_list.extend(cycles_with_labels)
                    
                    for d in cycles_with_labels:
                        self.filename_to_label[filename].append(d[1])
                        self.classwise_cycle_list[d[1]].append(d)
                    
                    successful_files += 1
                    total_cycles += len(cycles_with_labels)
                    
                    if len(cycles_with_labels) > 0:
                        pbar.set_postfix({
                            "当前": filename[:12] + "...",
                            "成功": successful_files,
                            "周期": total_cycles,
                            "本文件": len(cycles_with_labels)
                        })
                    
                except Exception as e:
                    tqdm.write(f"❌ 处理文件 {filename} 时出错: {str(e)}")
                    if print_flag:
                        traceback.print_exc()
                
                pbar.update(1)
        
        print(f"🎉 双特征数据加载完成!")
        print(f"📊 成功处理: {successful_files}/{len(self.filenames)} 个文件")
        print(f"🔄 总音频周期: {len(self.cycle_list)} 个")
        print(f"📈 各类别分布: {[len(self.classwise_cycle_list[i]) for i in range(args.n_cls)]}")
        
        if len(self.cycle_list) == 0:
            raise ValueError("❌ 没有加载到任何音频数据，请检查数据路径和预处理过程")
        
        for sample in self.cycle_list:
            self.audio_data.append(sample)

        if len(self.audio_data) == 0:
            raise ValueError("没有加载到任何音频数据，请检查数据路径和预处理过程")

        self.class_nums = np.zeros(args.n_cls)
        for sample in self.audio_data:
            self.class_nums[sample[1]] += 1
            self.labels.append(sample[1])
        self.class_ratio = self.class_nums / sum(self.class_nums) * 100
        
        if print_flag:
            print('[预处理后的 {} 数据集信息]'.format(self.split))
            print('音频数据总数: {}'.format(len(self.audio_data)))
            for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
                print('类别 {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.cls_list[i]+')', int(n), p))    
        
        # 获取特征提取器（带进度显示）
        from .icbhi_util import get_feature_extractor
        self.feature_extractors = {}
        self.progress_trackers = {}
        
        for feat_type in self.feature_types:
            extractor, tracker = get_feature_extractor(feat_type, show_progress=True)
            self.feature_extractors[feat_type] = extractor
            self.progress_trackers[feat_type] = tracker
        
        # 生成双特征音频图像
        self._generate_dual_feature_images_with_progress(print_flag)
        
        if print_flag:
            print(f"🎉 成功创建双特征数据集，包含 {len(self.audio_images)} 个样本")
            print(f"🔧 特征组合: {' + '.join(self.feature_types)}")
    
    def _generate_dual_feature_images_with_progress(self, print_flag=True):
        """
        生成双特征音频图像（带详细进度显示）
        """
        if print_flag:
            print(f"🔧 开始生成双特征图像: {self.feature_combo}")
        
        self.audio_images = []
        total_samples = len(self.audio_data)
        
        # 计算目标长度，确保所有特征一致
        target_length = self.args.target_length if hasattr(self.args, 'target_length') else 1024
        
        # 为整体进度创建主进度条
        with tqdm(total=total_samples, desc="🎨 生成双特征图像", unit="样本",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}") as main_pbar:
            
            successful_samples = 0
            failed_samples = 0
            
            for index in range(total_samples):
                main_pbar.set_postfix({
                    '成功': successful_samples,
                    '失败': failed_samples,
                    '当前': f"{index+1}/{total_samples}"
                })
                
                try:
                    audio, label = self.audio_data[index][0], self.audio_data[index][1]
                    
                    # 为每种特征类型生成图像
                    feature_images = {}
                    
                    for feat_idx, feat_type in enumerate(self.feature_types):
                        audio_images_per_type = []
                        
                        # 为每个增强版本生成特征
                        for aug_idx in range(self.args.raw_augment + 1):
                            current_audio = audio
                            
                            # 应用数据增强
                            if aug_idx > 0:
                                if self.train_flag and not self.mean_std:
                                    from .augmentation import augment_raw_audio
                                    current_audio = augment_raw_audio(audio, self.sample_rate, self.args)
                                    current_audio = cut_pad_sample_torchaudio(torch.tensor(current_audio), self.args)
                                else:
                                    audio_images_per_type.append(None)
                                    continue
                            
                            # 根据特征类型生成相应的特征
                            try:
                                if feat_type == 'fbank':
                                    # 使用原有的fbank生成方式，但确保目标长度一致
                                    image = generate_fbank(current_audio, self.sample_rate, 
                                                         n_mels=self.n_mels)
                                    
                                    # 确保fbank图像的时间维度与target_length一致
                                    if image.shape[0] != target_length:
                                        if image.shape[0] < target_length:
                                            # 重复填充
                                            repeat_times = target_length // image.shape[0] + 1
                                            image = np.tile(image, (repeat_times, 1, 1))
                                        # 截断到目标长度
                                        image = image[:target_length, :, :]
                                else:
                                    # 使用新的特征提取函数（带target_length参数）
                                    image = self.feature_extractors[feat_type](
                                        current_audio, self.sample_rate, 
                                        n_mels=self.n_mels,
                                        target_length=target_length,
                                        sample_index=index,
                                        total_samples=total_samples
                                    )
                                
                                # 验证生成的图像尺寸
                                if image is not None:
                                    expected_shape = (target_length, self.n_mels, 1)
                                    if image.shape != expected_shape:
                                        if print_flag and index < 5:
                                            tqdm.write(f"⚠️ {feat_type}特征形状不匹配: {image.shape} vs 期望 {expected_shape}")
                                        
                                        # 调整到期望形状
                                        if image.shape[0] != target_length:
                                            if image.shape[0] < target_length:
                                                repeat_times = target_length // image.shape[0] + 1
                                                image = np.tile(image, (repeat_times, 1, 1))
                                            image = image[:target_length, :, :]
                                        
                                        if image.shape[1] != self.n_mels:
                                            if image.shape[1] > self.n_mels:
                                                image = image[:, :self.n_mels, :]
                                            else:
                                                padding = np.zeros((image.shape[0], self.n_mels - image.shape[1], 1))
                                                image = np.hstack([image, padding])
                                        
                                        if len(image.shape) == 2:
                                            image = np.expand_dims(image, axis=-1)
                                
                                audio_images_per_type.append(image)
                                
                            except Exception as e:
                                if print_flag and failed_samples < 5:
                                    tqdm.write(f"⚠️ 样本{index} {feat_type}特征提取出错: {str(e)}")
                                audio_images_per_type.append(None)
                        
                        feature_images[feat_type] = audio_images_per_type
                    
                    self.audio_images.append((feature_images, label))
                    successful_samples += 1
                    
                    # 打印第一个样本的信息
                    if index == 0 and print_flag:
                        tqdm.write("✅ 第一个样本特征形状:")
                        for feat_type in self.feature_types:
                            if feature_images[feat_type][0] is not None:
                                tqdm.write(f"   {feat_type}: {feature_images[feat_type][0].shape}")
                                
                except Exception as e:
                    failed_samples += 1
                    if print_flag and failed_samples <= 5:
                        tqdm.write(f"❌ 处理第 {index} 个音频样本时出错: {str(e)}")
                    continue
                
                main_pbar.update(1)
        
        # 关闭所有特征的进度条
        for tracker in self.progress_trackers.values():
            if tracker['pbar'] is not None:
                tracker['pbar'].close()
        
        if print_flag:
            print(f"🎉 双特征图像生成完成!")
            print(f"✅ 成功: {successful_samples}/{total_samples} 个样本")
            print(f"❌ 失败: {failed_samples}/{total_samples} 个样本")
            
            if successful_samples > 0:
                success率 = (successful_samples / total_samples) * 100
                print(f"📊 成功率: {success率:.1f}%")
                print(f"📐 目标图像形状: ({target_length}, {self.n_mels}, 1)")
        
        # 确定图像形状
        if len(self.audio_images) > 0:
            for feat_type in self.feature_types:
                first_valid_image = None
                for img_data in self.audio_images:
                    for img in img_data[0][feat_type]:
                        if img is not None:
                            first_valid_image = img
                            break
                    if first_valid_image is not None:
                        break
                
                if first_valid_image is not None:
                    self.h, self.w, _ = first_valid_image.shape
                    if print_flag:
                        print(f"📐 实际图像形状: ({self.h}, {self.w})")
                    break
    
    def __getitem__(self, index):
        feature_images_dict, label = self.audio_images[index][0], self.audio_images[index][1]
        
        # 每1000个样本显示一次在线特征提取进度
        if index % 1000 == 0:
            print(f"🔄 在线特征提取进度: {index}/{len(self.audio_images)} ({(index/len(self.audio_images)*100):.1f}%)")
        
        result_features = []  # 改为列表，而不是字典
        
        for feat_type in self.feature_types:
            audio_images = feature_images_dict[feat_type]
            
            # 选择增强版本
            if self.args.raw_augment and self.train_flag and not self.mean_std:
                aug_idx = random.randint(0, self.args.raw_augment)
                audio_image = audio_images[aug_idx]
            else:
                audio_image = audio_images[0]
            
            # 确保音频图像不为None
            if audio_image is None:
                # 创建零图像作为fallback
                audio_image = np.zeros((self.h, self.w, 1))
            
            # 应用变换
            if self.transform is not None:
                audio_image = self.transform(audio_image)
            
            # 确保是torch tensor且维度正确
            if not isinstance(audio_image, torch.Tensor):
                audio_image = torch.tensor(audio_image, dtype=torch.float32)
            
            # 确保维度是 (C, H, W) 格式
            if len(audio_image.shape) == 3 and audio_image.shape[-1] == 1:
                audio_image = audio_image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            elif len(audio_image.shape) == 2:
                audio_image = audio_image.unsqueeze(0)  # (H, W) -> (1, H, W)
            
            result_features.append(audio_image)
        
        # 返回元组：(feature1, feature2), label
        return tuple(result_features), torch.tensor(label, dtype=torch.long)
    
    def __len__(self):
        return len(self.audio_images)


# ==========================================================================
""" 其他函数保持不变 """

def create_train_test_file(output_path=None):
    """创建训练/测试集划分文件，使用制表符分隔"""
    import os
    import random
    
    if output_path is None:
        output_path = '/home/yujieyang/bishe/data/ICBHI_final_database/icbhi_dataset/traintest.txt'
    
    data_folder = os.path.dirname(output_path)
    
    # 获取所有wav文件
    filenames = os.listdir(data_folder)
    valid_filenames = []
    for f in filenames:
        if '.wav' in f:
            base_name = f.strip().split('.')[0]
            if '_' in base_name:
                parts = base_name.split('_')
                if len(parts) >= 4:
                    valid_filenames.append(base_name)
    
    filenames = sorted(list(set(valid_filenames)))
    
    # 随机划分，保持60-40比例
    indices = list(range(len(filenames)))
    random.Random(1).shuffle(indices)  # 使用固定种子保证可重复性
    train_size = int(len(indices) * 0.6)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    with open(output_path, 'w') as f:
        f.write("// 文件名\t划分类型(train/test)\n")
        for i in train_idx:
            f.write(f"{filenames[i]}\ttrain\n")
        for i in test_idx:
            f.write(f"{filenames[i]}\ttest\n")
    
    print(f"已创建训练/测试集划分文件: {output_path}")
    print(f"训练集: {len(train_idx)} 文件")
    print(f"测试集: {len(test_idx)} 文件")

    return train_size, len(indices) - train_size