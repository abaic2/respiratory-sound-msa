import os
import random
import numpy as np
import traceback

import torch
from torch.utils.data import Dataset
from copy import deepcopy
from PIL import Image

from .icbhi_util import (
    get_annotations, 
    get_individual_cycles_torchaudio, 
    cut_pad_sample_torchaudio,
    extract_time_domain_features, 
    extract_sequential_features, 
    generate_raw_features
)
from .augmentation import augment_raw_audio


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

        # parameters for audio processing
        self.sample_rate = args.sample_rate
        self.desired_length = args.desired_length
        self.pad_types = args.pad_types
        
        # 时序特征参数
        self.feature_type = args.feature_type if hasattr(args, 'feature_type') else 'time_domain'
        self.frame_length = args.frame_length if hasattr(args, 'frame_length') else 1024
        self.hop_length = args.hop_length if hasattr(args, 'hop_length') else 512

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
        train_test_file = '/home/yujieyang/bishe/data/ICBHI_final_database/icbhi_dataset/traintest.txt'
        
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

        for idx, filename in enumerate(self.filenames):
            self.filename_to_label[filename] = []
            try:
                sample_data = get_individual_cycles_torchaudio(args, annotation_dict[filename], data_folder, filename, args.sample_rate, args.n_cls)
                cycles_with_labels = [(data[0], data[1]) for data in sample_data]
                self.cycle_list.extend(cycles_with_labels)
                for d in cycles_with_labels:
                    # {filename: [label for cycle 1, ...]}
                    self.filename_to_label[filename].append(d[1])
                    self.classwise_cycle_list[d[1]].append(d)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
                if print_flag:
                    traceback.print_exc()
                
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
            print(f"准备生成 {len(self.audio_data)} 个时序特征...")
        
        # 提取时序特征代替频谱图
        self.time_features = []
        for index in range(len(self.audio_data)):
            try:
                audio, label = self.audio_data[index][0], self.audio_data[index][1]

                features_list = []
                for aug_idx in range(self.args.raw_augment+1): 
                    if aug_idx > 0:
                        if self.train_flag and not mean_std:
                            audio = augment_raw_audio(audio, self.sample_rate, self.args)
                            audio = cut_pad_sample_torchaudio(torch.tensor(audio), args)
                        else:
                            features_list.append(None)
                            continue
                    
                    # 根据选择的特征类型提取不同的时序特征
                    if self.feature_type == 'time_domain':
                        # 提取时域特征（ZCR、RMS、自相关等）
                        features = extract_time_domain_features(
                            audio, 
                            self.sample_rate, 
                            segment_length=self.frame_length, 
                            hop_length=self.hop_length
                        )
                    elif self.feature_type == 'sequential':
                        # 提取丰富的序列特征（频谱质心、带宽等）
                        features = extract_sequential_features(
                            audio, 
                            self.sample_rate, 
                            hop_length=self.hop_length
                        )
                    else:  # 默认使用原始波形帧
                        features = generate_raw_features(
                            audio, 
                            self.sample_rate, 
                            frame_length=self.frame_length, 
                            hop_length=self.hop_length
                        )
                    
                    features_list.append(features)
                    
                self.time_features.append((features_list, label))
                
                if index == 0 and print_flag:
                    print(f"第一个时序特征形状: {features_list[0].shape}")
                    
            except Exception as e:
                print(f"处理第 {index} 个音频样本时出错: {str(e)}")
                if print_flag:
                    traceback.print_exc()
                # 跳过有问题的样本，继续处理其他样本
                continue
        
        # 检查是否至少有一个有效的特征
        if len(self.time_features) == 0:
            raise ValueError("没有成功生成任何时序特征，请检查音频处理和特征提取过程")
        
        # 确保第一个特征是有效的
        if self.time_features[0][0][0] is None:
            # 寻找第一个非None的特征
            valid_feature_found = False
            for feat_data in self.time_features:
                for feat in feat_data[0]:
                    if feat is not None:
                        # 获取特征维度，用于模型参数设置
                        self.seq_len, self.feat_dim = feat.shape
                        valid_feature_found = True
                        break
                if valid_feature_found:
                    break
            
            if not valid_feature_found:
                raise ValueError("所有时序特征均无效，无法确定特征形状")
        else:
            self.seq_len, self.feat_dim = self.time_features[0][0][0].shape
        
        if print_flag:
            print(f"成功生成 {len(self.time_features)} 个时序特征，形状: (时间步: {self.seq_len}, 特征维度: {self.feat_dim})")

    def __getitem__(self, index):
        features_list, label = self.time_features[index][0], self.time_features[index][1]

        if self.args.raw_augment and self.train_flag and not self.mean_std:
            aug_idx = random.randint(0, self.args.raw_augment)
            features = features_list[aug_idx]
        else:
            features = features_list[0]
        
        # 将特征转换为张量
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        
        # 对于RNN模型，不需要额外变换，直接返回[seq_len, feat_dim]格式
        # 如果需要兼容卷积模型，可以调整为[1, seq_len, feat_dim]
        
        # 兼容原始transform
        if self.transform is not None and hasattr(self.transform, 'transforms'):
            # 检查是否有ToTensor转换
            has_to_tensor = any(isinstance(t, torch.nn.Module) for t in self.transform.transforms)
            if has_to_tensor:
                # 如果已经是张量，跳过ToTensor，但应用其他转换
                for t in self.transform.transforms:
                    if not isinstance(t, torch.nn.Module):
                        features = t(features)
            else:
                features = self.transform(features)
        
        return features, label

    def __len__(self):
        return len(self.time_features)