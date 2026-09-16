import os
import random
import numpy as np
import traceback

import torch
from torch.utils.data import Dataset
from copy import deepcopy
from PIL import Image

from icbhi_util import get_annotations, generate_fbank, get_individual_cycles_torchaudio, cut_pad_sample_torchaudio
from augmentation import augment_raw_audio


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

        # parameters for Gammatone spectrograms (修改这部分)
        self.sample_rate = args.sample_rate
        self.desired_length = args.desired_length
        self.pad_types = args.pad_types
        
        # Gammatone特定参数 - 移除 mel 相关参数
        self.n_filters = getattr(args, 'n_filters', 64)
        self.fmin = getattr(args, 'fmin', 50)
        self.fmax = getattr(args, 'fmax', 8000)
        self.order = getattr(args, 'order', 4)
        
        # 移除或注释掉这些mel相关的参数
        # self.nfft = args.nfft
        # self.n_mels = args.n_mels
        
        # 继续处理其他参数...
        self.class_split = args.class_split
        self.n_cls = args.n_cls

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
                    
                    # 修改这行 - 使用Gammatone参数而不是mel参数
                    image = generate_fbank(audio, self.sample_rate, 
                                         n_filters=self.n_filters,
                                         fmin=self.fmin,
                                         fmax=self.fmax,
                                         order=self.order)
                    
                    audio_image.append(image)
                    
                self.audio_images.append((audio_image, label))
                
                if index == 0 and print_flag:
                    print(f"第一个Gammatone图像形状: {audio_image[0].shape}")
                    
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


if __name__ == "__main__":
    import argparse
    from icbhi_util import _extract_lungsound_annotation
    import pandas as pd
    import traceback
    
    print("开始调试ICBHI数据集加载问题...")
    
    # 创建一个简单的参数对象用于测试
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='/home/yujieyang/bishe/data/ICBHI_final_database')
    parser.add_argument('--class_split', type=str, default='lungsound')
    args = parser.parse_args([])
    
    # 为测试添加必要的参数
    args.n_cls = 4
    args.sample_rate = 16000
    args.desired_length = 8
    args.nfft = 1024
    args.n_mels = 128
    args.pad_types = 'repeat'
    args.raw_augment = 0
    args.stetho_id = -1
    args.test_fold = 'official'
    args.cls_list = ['normal', 'crackle', 'wheezing', 'both']
    
    # 检查 traintest.txt 文件是否存在，如果不存在则创建
    train_test_file = os.path.join(args.data_folder, 'icbhi_dataset', 'traintest.txt')
    if not os.path.exists(train_test_file):
        print(f"未找到划分文件，创建新的...")
        train_count, test_count = create_train_test_file(train_test_file)
        print(f"创建完成: 训练集 {train_count} 文件, 测试集 {test_count} 文件")
    else:
        print(f"找到已存在的划分文件: {train_test_file}")
        
        # 读取并分析现有的划分文件
        try:
            train_files = []
            test_files = []
            
            with open(train_test_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('//') or line.startswith('#'):
                    continue
                    
                # 使用制表符分割文件
                parts = line.split('\t')
                if len(parts) >= 2:
                    file_name = parts[0].strip()
                    split = parts[1].strip().lower()
                    
                    if split == 'train':
                        train_files.append(file_name)
                    elif split == 'test':
                        test_files.append(file_name)
            
            print(f"现有划分: 训练集 {len(train_files)} 文件, 测试集 {len(test_files)} 文件")
        except Exception as e:
            print(f"读取划分文件时发生错误: {str(e)}")
            traceback.print_exc()
    
    # 测试数据文件夹路径
    data_folder = os.path.join(args.data_folder, 'icbhi_dataset')
    print(f"数据文件夹路径: {data_folder}")
    
    # 获取所有的文件名
    try:
        filenames = os.listdir(data_folder)
        wav_files = [f.strip().split('.')[0] for f in filenames if '.wav' in f]
        print(f"发现 {len(wav_files)} 个 WAV 文件")
        
        # 测试前5个文件
        for i, file_name in enumerate(wav_files[:5]):
            print(f"\n测试文件 {i+1}: {file_name}")
            print(f"文件名分割: {file_name.strip().split('_')}")
            
            try:
                info, ann = _extract_lungsound_annotation(file_name, data_folder)
                print(f"成功解析文件: {file_name}")
                print(f"信息DataFrame形状: {info.shape}")
                print(f"注释DataFrame形状: {ann.shape}")
                
                # 显示解析后的信息
                print("\n信息内容:")
                print(info)
                
                print("\n注释内容 (前3行):")
                print(ann.head(3))
                
            except Exception as e:
                print(f"解析文件 {file_name} 时发生错误:")
                print(f"错误类型: {type(e).__name__}")
                print(f"错误信息: {str(e)}")
                traceback.print_exc()
        
        # 测试一个特定文件
        special_file = "225_1b1_Pl_sc_Meditron"
        if special_file in wav_files:
            print(f"\n特别测试文件: {special_file}")
            try:
                tokens = special_file.strip().split('_')
                print(f"文件名分割: {tokens}, 长度: {len(tokens)}")
                
                info, ann = _extract_lungsound_annotation(special_file, data_folder)
                print(f"成功解析特定文件")
                
            except Exception as e:
                print(f"解析特定文件时发生错误:")
                print(f"错误信息: {str(e)}")
                traceback.print_exc()
    
    except Exception as e:
        print(f"遍历文件夹时发生错误: {str(e)}")
        traceback.print_exc()
        
    # 尝试加载数据集
    try:
        print("\n尝试加载训练数据集...")
        from torchvision import transforms
        transform = transforms.Compose([transforms.ToTensor()])
        
        train_dataset = ICBHIDataset(train_flag=True, transform=transform, args=args, print_flag=True)
        print(f"成功加载训练数据集，样本数: {len(train_dataset)}")
        
        # 查看第一个样本
        if len(train_dataset) > 0:
            sample, label = train_dataset[0]
            print(f"第一个样本形状: {sample.shape}, 标签: {label}")
    except Exception as e:
        print(f"加载数据集时发生错误: {str(e)}")
        traceback.print_exc()