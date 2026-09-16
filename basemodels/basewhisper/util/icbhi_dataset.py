import os
import random
import numpy as np
import traceback

import torch
from torch.utils.data import Dataset
from copy import deepcopy
from PIL import Image

from .icbhi_util import get_annotations, generate_fbank, get_individual_cycles_torchaudio, cut_pad_sample_torchaudio
from .augmentation import augment_raw_audio


class ICBHIDataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True, mean_std=False, 
                 return_raw_audio=False):  # 添加原始音频返回选项
        data_folder = os.path.join(args.data_folder, 'icbhi_dataset')
        test_fold = args.test_fold
        
        self.data_folder = data_folder
        self.train_flag = train_flag
        self.split = 'train' if train_flag else 'test'
        self.transform = transform
        self.args = args
        self.mean_std = mean_std
        self.return_raw_audio = return_raw_audio  # 新增：是否返回原始音频

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
        
        # 根据返回模式处理数据
        if self.return_raw_audio:
            self._prepare_raw_audio_data(print_flag)
        else:
            self._prepare_spectrogram_data(print_flag)

    def _prepare_raw_audio_data(self, print_flag):
        """准备原始音频数据（用于Whisper）"""
        if print_flag:
            print(f"🎵 准备原始音频数据用于Whisper，共 {len(self.audio_data)} 个样本...")
        
        self.audio_features = []
        
        for index in range(len(self.audio_data)):
            try:
                audio, label = self.audio_data[index][0], self.audio_data[index][1]
                
                raw_audio_list = []
                for aug_idx in range(self.args.raw_augment + 1):
                    if aug_idx > 0:
                        if self.train_flag and not self.mean_std:
                            augmented_audio = augment_raw_audio(audio, self.sample_rate, self.args)
                            processed_audio = cut_pad_sample_torchaudio(torch.tensor(augmented_audio), self.args)
                        else:
                            raw_audio_list.append(None)
                            continue
                    else:
                        processed_audio = cut_pad_sample_torchaudio(torch.tensor(audio), self.args)
                    
                    # 确保音频是numpy数组格式
                    if isinstance(processed_audio, torch.Tensor):
                        processed_audio = processed_audio.numpy()
                    
                    raw_audio_list.append(processed_audio)
                
                self.audio_features.append((raw_audio_list, label))
                
            except Exception as e:
                print(f"处理第 {index} 个音频样本时出错: {str(e)}")
                continue
        
        if print_flag:
            print(f"✅ 成功准备 {len(self.audio_features)} 个原始音频特征")

    def _prepare_spectrogram_data(self, print_flag):
        """准备频谱图数据（传统方法）"""
        if print_flag:
            print(f"准备生成 {len(self.audio_data)} 个音频图像...")
        
        self.audio_images = []
        for index in range(len(self.audio_data)):
            try:
                audio, label = self.audio_data[index][0], self.audio_data[index][1]

                audio_image = []
                for aug_idx in range(self.args.raw_augment+1): 
                    if aug_idx > 0:
                        if self.train_flag and not self.mean_std:
                            audio = augment_raw_audio(audio, self.sample_rate, self.args)
                            audio = cut_pad_sample_torchaudio(torch.tensor(audio), self.args)
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
        if self.return_raw_audio:
            # 返回原始音频数据（用于Whisper）
            audio_list, label = self.audio_features[index][0], self.audio_features[index][1]
            
            if self.args.raw_augment and self.train_flag and not self.mean_std:
                aug_idx = random.randint(0, self.args.raw_augment)
                audio_data = audio_list[aug_idx]
            else:
                audio_data = audio_list[0]
            
            # 确保音频数据是torch tensor
            if not isinstance(audio_data, torch.Tensor):
                audio_data = torch.tensor(audio_data, dtype=torch.float32)
            
            return audio_data, label
        else:
            # 返回频谱图数据（原有逻辑）
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


# 为Whisper添加便捷的数据集创建函数
def create_whisper_dataset(train_flag, args, print_flag=True):
    """创建Whisper兼容的ICBHI数据集"""
    return ICBHIDataset(
        train_flag=train_flag,
        transform=None,  # Whisper不需要图像变换
        args=args,
        print_flag=print_flag,
        mean_std=False,
        return_raw_audio=True  # 返回原始音频数据
    )


def create_train_test_file(output_path=None):
    """创建训练/测试集划分文件，使用制表符分隔"""
    import os
    import random
    
    if output_path is None:
        output_path = r'D:\55555\bishe\data\ICBHI_final_database\icbhi_dataset\traintest.txt'
    
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
    parser.add_argument('--data_folder', type=str, default=r'D:\55555\bishe\data\ICBHI_final_database')
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
    
    # 测试标准数据集
    print("\n📊 测试标准ICBHI数据集...")
    try:
        from torchvision import transforms
        transform = transforms.Compose([transforms.ToTensor()])
        
        dataset = ICBHIDataset(train_flag=True, transform=transform, args=args, print_flag=True)
        print(f"✅ 标准数据集创建成功，样本数: {len(dataset)}")
        
        if len(dataset) > 0:
            sample, label = dataset[0]
            print(f"✅ 第一个样本形状: {sample.shape}, 标签: {label}")
    except Exception as e:
        print(f"❌ 标准数据集测试失败: {e}")
        traceback.print_exc()
    
    # 测试Whisper数据集
    print("\n🎵 测试Whisper兼容数据集...")
    try:
        whisper_dataset = create_whisper_dataset(train_flag=True, args=args, print_flag=True)
        print(f"✅ Whisper数据集创建成功，样本数: {len(whisper_dataset)}")
        
        if len(whisper_dataset) > 0:
            audio, label = whisper_dataset[0]
            print(f"✅ 第一个音频样本形状: {audio.shape}, 标签: {label}")
            print(f"✅ 音频数据类型: {type(audio)}, 数值范围: [{audio.min():.3f}, {audio.max():.3f}]")
    except Exception as e:
        print(f"❌ Whisper数据集测试失败: {e}")
        traceback.print_exc()
    
    print("\n🎉 ICBHI数据集测试完成!")