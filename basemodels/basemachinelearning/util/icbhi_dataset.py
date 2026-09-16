import os
import random
import numpy as np
import traceback
import math
from tqdm import tqdm
import pandas as pd  # 添加pandas导入

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
    generate_raw_features,
    extract_ml_features,
    save_features_to_csv,
    load_features_from_csv,
    extract_features_parallel  # 添加缺失的导入
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
        self.is_ml_model = hasattr(args, 'ml_model') and args.ml_model  # 添加标志以识别机器学习模型

        # 设置特征CSV保存路径
        self.feature_save_dir = os.path.join(args.data_folder, 'ml_features')
        os.makedirs(self.feature_save_dir, exist_ok=True)
        
        # 特征文件名：根据配置参数生成
        ml_model_name = args.ml_model if hasattr(args, 'ml_model') else 'none'
        ml_feature_type = args.ml_feature_type if hasattr(args, 'ml_feature_type') else 'statistical'
        self.feature_file = os.path.join(
            self.feature_save_dir, 
            f"{self.split}_{ml_model_name}_{ml_feature_type}_features.csv"
        )
        
        # 检查是否使用缓存特征
        self.use_cache = getattr(args, 'use_feature_cache', True)
        
        # parameters for audio processing
        self.sample_rate = args.sample_rate
        self.desired_length = args.desired_length
        self.pad_types = args.pad_types
        
        # 时序特征参数
        self.feature_type = args.feature_type if hasattr(args, 'feature_type') else 'time_domain'
        self.frame_length = args.frame_length if hasattr(args, 'frame_length') else 1024
        self.hop_length = args.hop_length if hasattr(args, 'hop_length') else 512

        # 添加机器学习特征类型
        self.ml_feature_type = args.ml_feature_type if hasattr(args, 'ml_feature_type') else 'statistical'

        # 先检查缓存，如果存在且使用缓存，直接加载
        if self.use_cache and os.path.exists(self.feature_file) and self.is_ml_model:
            if print_flag:
                print(f"找到缓存的特征文件: {self.feature_file}，直接加载...")
            
            try:
                features_tensor, labels_tensor = load_features_from_csv(self.feature_file)
                
                # 创建特征列表
                self.time_features = []
                for i in range(len(labels_tensor)):
                    feature = features_tensor[i].unsqueeze(0)  # 添加批次维度 [1, feature_dim]
                    label = labels_tensor[i].item()
                    self.time_features.append(([feature], label))
                
                # 设置维度信息
                self.seq_len = 1
                self.feat_dim = features_tensor.shape[1]
                
                if print_flag:
                    print(f"成功从缓存加载 {len(self.time_features)} 个机器学习特征，维度: {self.feat_dim}")
                    
                # 缓存加载成功，返回
                return
                
            except Exception as e:
                print(f"加载缓存特征失败: {e}，将重新生成特征...")
        
        # 处理音频文件和标签
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
        
        # 音频数据加载完成后，开始提取特征
        
        # 检查缓存
        if self.use_cache and os.path.exists(self.feature_file) and self.is_ml_model:
            if print_flag:
                print(f"找到缓存的特征文件: {self.feature_file}，直接加载...")
            
            try:
                features_tensor, labels_tensor = load_features_from_csv(self.feature_file)
                
                # 创建特征列表
                self.time_features = []
                for i in range(len(labels_tensor)):
                    feature = features_tensor[i].unsqueeze(0)  # 添加批次维度 [1, feature_dim]
                    label = labels_tensor[i].item()
                    self.time_features.append(([feature], label))
                
                # 设置维度信息
                self.seq_len = 1
                self.feat_dim = features_tensor.shape[1]
                
                if print_flag:
                    print(f"成功从缓存加载 {len(self.time_features)} 个机器学习特征，维度: {self.feat_dim}")
                    
                # 缓存加载成功，返回
                return
                
            except Exception as e:
                print(f"加载缓存特征失败: {e}，将重新生成特征...")
        
        # ====== 特征提取 ======
        # 初始化特征列表
        self.time_features = []
        
        # 确定是否使用并行处理
        use_parallel = self.is_ml_model and hasattr(args, 'n_jobs') and args.n_jobs > 1
        
        if print_flag:
            print(f"准备生成 {len(self.audio_data)} 个{'机器学习' if self.is_ml_model else '时序'}特征...")
            if use_parallel:
                n_jobs = getattr(args, 'n_jobs', 1)
                print(f"使用 {n_jobs} 个CPU进行并行特征提取")
        
        # 根据条件选择特征提取方法
        if use_parallel:
            # ===== 并行特征提取 =====
            n_jobs = getattr(args, 'n_jobs', 1)
            batch_extract = getattr(args, 'batch_extract', 100)
            
            # 收集所有音频数据
            all_audios = []
            all_labels = []
            for i in range(len(self.audio_data)):
                all_audios.append(self.audio_data[i][0])
                all_labels.append(self.audio_data[i][1])
            
            # 分批处理
            all_features = []
            for i in range(0, len(all_audios), batch_extract):
                if print_flag:
                    print(f"处理批次 {i//batch_extract + 1}/{(len(all_audios)+batch_extract-1)//batch_extract}")
                
                batch_audios = all_audios[i:i+batch_extract]
                
                # 使用并行处理
                batch_features = extract_features_parallel(
                    batch_audios,
                    self.sample_rate,
                    feature_type=self.ml_feature_type,
                    n_jobs=n_jobs,
                    show_progress=print_flag
                )
                
                all_features.extend(batch_features)
            
            # 构建特征列表
            self.time_features = []
            for i in range(len(all_features)):
                features_list = [all_features[i]]
                # 添加None占位符为数据增强
                for _ in range(self.args.raw_augment):
                    features_list.append(None)
                self.time_features.append((features_list, all_labels[i]))
            
            # 保存特征到CSV
            if len(all_features) > 0 and print_flag:
                try:
                    # 提取特征和标签
                    feature_array = []
                    label_array = []
                    
                    for i in range(len(all_features)):
                        feature_array.append(all_features[i].cpu().numpy().squeeze(0))
                        label_array.append(all_labels[i])
                    
                    # 转换为numpy数组
                    feature_array = np.array(feature_array)
                    label_array = np.array(label_array)
                    
                    # 保存到CSV
                    save_features_to_csv(
                        feature_array,
                        label_array,
                        self.feature_file,
                        feature_type=self.ml_feature_type
                    )
                    
                    print(f"特征已保存到 {self.feature_file}")
                except Exception as e:
                    print(f"保存特征到CSV失败: {e}")
                    
        else:
            # ===== 串行特征提取 =====
            for index in tqdm(range(len(self.audio_data)), desc="特征提取", disable=not print_flag):
                try:
                    audio, label = self.audio_data[index][0], self.audio_data[index][1]
                    
                    # 添加调试信息，检查音频数据
                    if index == 0 and print_flag:
                        print(f"音频形状: {audio.shape}, 类型: {type(audio)}")
                    
                    # 初始化特征列表
                    features_list = []
                    
                    # 提取原始特征
                    try:
                        if self.is_ml_model:
                            features = extract_ml_features(
                                audio, 
                                self.sample_rate, 
                                feature_type=self.ml_feature_type,
                                show_progress=(index == 0 and print_flag)
                            )
                        elif self.feature_type == 'time_domain':
                            features = extract_time_domain_features(
                                audio, 
                                self.sample_rate, 
                                segment_length=self.frame_length, 
                                hop_length=self.hop_length
                            )
                        elif self.feature_type == 'sequential':
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
                        
                        if index == 0 and print_flag:
                            print(f"生成的特征形状: {features.shape}")
                            
                    except Exception as e:
                        print(f"特征提取错误: {e}, 对样本 {index} 使用零填充特征")
                        
                        # 创建一个有效的替代特征
                        if self.is_ml_model:
                            if self.ml_feature_type == 'statistical':
                                features = torch.zeros((1, 16), dtype=torch.float32)
                            elif self.ml_feature_type == 'spectral':
                                features = torch.zeros((1, 34), dtype=torch.float32)
                            else:  # full
                                features = torch.zeros((1, 60), dtype=torch.float32)
                        else:
                            features = torch.zeros((8, 8), dtype=torch.float32)
                    
                    # 添加原始特征
                    features_list.append(features)
                    
                    # 添加增强特征（如果需要）
                    for aug_idx in range(self.args.raw_augment): 
                        if self.train_flag and not mean_std:
                            try:
                                aug_audio = augment_raw_audio(audio, self.sample_rate, self.args)
                                aug_audio = cut_pad_sample_torchaudio(torch.tensor(aug_audio), args)
                                
                                if self.is_ml_model:
                                    aug_features = extract_ml_features(
                                        aug_audio, 
                                        self.sample_rate, 
                                        feature_type=self.ml_feature_type
                                    )
                                elif self.feature_type == 'time_domain':
                                    aug_features = extract_time_domain_features(
                                        aug_audio, 
                                        self.sample_rate,
                                        segment_length=self.frame_length, 
                                        hop_length=self.hop_length
                                    )
                                elif self.feature_type == 'sequential':
                                    aug_features = extract_sequential_features(
                                        aug_audio, 
                                        self.sample_rate, 
                                        hop_length=self.hop_length
                                    )
                                else:  # raw
                                    aug_features = generate_raw_features(
                                        aug_audio, 
                                        self.sample_rate,
                                        frame_length=self.frame_length, 
                                        hop_length=self.hop_length
                                    )
                                
                                features_list.append(aug_features)
                            except Exception as e:
                                features_list.append(None)
                        else:
                            features_list.append(None)
                    
                    # 添加到时序特征列表
                    self.time_features.append((features_list, label))
                    
                except Exception as e:
                    if print_flag and (index == 0 or index % 500 == 0):
                        print(f"处理第 {index} 个音频样本时出错: {str(e)}")
                    
                    # 创建一个有效的默认特征列表，确保不会跳过任何样本
                    default_features = []
                    
                    # 添加原始特征
                    if self.is_ml_model:
                        if self.ml_feature_type == 'statistical':
                            default_features.append(torch.zeros((1, 16), dtype=torch.float32))
                        elif self.ml_feature_type == 'spectral':
                            default_features.append(torch.zeros((1, 34), dtype=torch.float32))
                        else:  # full
                            default_features.append(torch.zeros((1, 60), dtype=torch.float32))
                    else:
                        default_features.append(torch.zeros((8, 8), dtype=torch.float32))
                    
                    # 添加占位符（增强特征）
                    for _ in range(self.args.raw_augment):
                        default_features.append(None)
                    
                    # 添加到特征列表
                    self.time_features.append((default_features, label))
        
        # ===== 检查是否正确生成了特征 =====
        if len(self.time_features) == 0:
            if print_flag:
                print("警告: 没有生成任何特征，尝试创建默认特征")
                
            # 创建默认特征
            for index in range(len(self.audio_data)):
                label = self.audio_data[index][1]
                
                # 创建默认特征
                if self.is_ml_model:
                    if self.ml_feature_type == 'statistical':
                        features = torch.zeros((1, 16), dtype=torch.float32)
                    elif self.ml_feature_type == 'spectral':
                        features = torch.zeros((1, 34), dtype=torch.float32)
                    else:  # full
                        features = torch.zeros((1, 60), dtype=torch.float32)
                else:
                    features = torch.zeros((8, 8), dtype=torch.float32)
                
                # 添加特征到特征列表
                features_list = [features]
                for _ in range(self.args.raw_augment):
                    features_list.append(None)
                
                self.time_features.append((features_list, label))
                
            if print_flag:
                print(f"已创建 {len(self.time_features)} 个默认特征")
        
        # 确保至少有一个有效的特征
        if len(self.time_features) == 0:
            raise ValueError("没有成功生成任何特征，请检查音频处理和特征提取过程")
            
        # 设置特征维度
        if len(self.time_features) > 0:
            first_feature = None
            
            # 查找第一个有效的特征
            for feat_tuple in self.time_features:
                for feat in feat_tuple[0]:
                    if feat is not None:
                        first_feature = feat
                        break
                if first_feature is not None:
                    break
            
            # 如果找到了有效特征，设置维度
            if first_feature is not None:
                if self.is_ml_model:
                    if first_feature.dim() == 1:
                        self.seq_len = 1
                        self.feat_dim = first_feature.size(0)
                    else:
                        self.seq_len = 1
                        self.feat_dim = first_feature.size(1)
                else:
                    if first_feature.dim() == 1:
                        self.seq_len = 1
                        self.feat_dim = first_feature.size(0)
                    else:
                        self.seq_len, self.feat_dim = first_feature.shape
            else:
                # 默认维度
                self.seq_len = 1
                if self.is_ml_model:
                    if self.ml_feature_type == 'statistical':
                        self.feat_dim = 16
                    elif self.ml_feature_type == 'spectral':
                        self.feat_dim = 34
                    else:  # full
                        self.feat_dim = 60
                else:
                    self.seq_len = 8
                    self.feat_dim = 8
            
            if print_flag:
                if self.is_ml_model:
                    print(f"成功生成 {len(self.time_features)} 个机器学习特征，维度: {self.feat_dim}")
                else:
                    print(f"成功生成 {len(self.time_features)} 个时序特征，形状: [时间步: {self.seq_len}, 特征维度: {self.feat_dim}]")
        
        # 保存CSV特征（如果未保存）
        if self.is_ml_model and not os.path.exists(self.feature_file) and len(self.time_features) > 0:
            try:
                # 提取特征和标签
                feature_array = []
                label_array = []
                
                for feat_list, label in self.time_features:
                    if feat_list[0] is not None:
                        feature = feat_list[0]
                        if feature.dim() == 1:
                            feature = feature.unsqueeze(0)
                        feature_array.append(feature.cpu().numpy().squeeze(0))
                        label_array.append(label)
                
                if len(feature_array) > 0:
                    # 转换为numpy数组
                    feature_array = np.array(feature_array)
                    label_array = np.array(label_array)
                    
                    # 保存到CSV
                    save_features_to_csv(
                        feature_array,
                        label_array,
                        self.feature_file,
                        feature_type=self.ml_feature_type
                    )
                    
                    if print_flag:
                        print(f"特征已保存到 {self.feature_file}")
            except Exception as e:
                if print_flag:
                    print(f"保存特征到CSV失败: {e}")
        
    def __getitem__(self, index):
        features_list, label = self.time_features[index]

        # 处理增强特征的选择
        if self.args.raw_augment and self.train_flag and not self.mean_std:
            aug_idx = random.randint(0, self.args.raw_augment)
            if aug_idx < len(features_list) and features_list[aug_idx] is not None:
                features = features_list[aug_idx]
            else:
                features = features_list[0]  # 使用原始特征作为后备
        else:
            features = features_list[0]
        
        # 确保特征是张量格式
        if not isinstance(features, torch.Tensor):
            if self.is_ml_model:
                features = torch.zeros((1, self.feat_dim), dtype=torch.float32)
            else:
                features = torch.zeros((self.seq_len, self.feat_dim), dtype=torch.float32)
        
        # 处理机器学习模型的特征
        if self.is_ml_model:
            # 确保是一维特征向量
            if features.dim() > 2:
                features = features.reshape(1, -1)
            elif features.dim() == 1:
                features = features.unsqueeze(0)
        
        # 如果是深度学习模型，确保特征形状匹配预期
        elif features.dim() == 1 and self.seq_len > 1:
            # 如果特征是一维的但期望多时间步，则重塑为正确形状
            features = features.unsqueeze(0).repeat(self.seq_len, 1)
            features = features[:self.seq_len, :self.feat_dim]  # 确保尺寸正确
        
        # 兼容原始transform
        if self.transform is not None:
            features = self.transform(features)
        
        return features, label

    def __len__(self):
        return len(self.time_features)