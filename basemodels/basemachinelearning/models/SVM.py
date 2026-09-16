import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import pickle
from copy import deepcopy
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class SVMModel(nn.Module):
    """
    支持向量机模型用于音频分类，使用时域或频域特征
    :param label_dim: 类别数量
    :param input_fdim: 输入时间序列的特征维度
    :param input_tdim: 输入序列的时间维度
    :param model_size: SVM的配置 ('small', 'medium', 'large')
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
             imagenet_pretrain=False, audioset_pretrain=False, model_size='medium', verbose=True, mix_beta=None,
             freeze_base=False, freeze_layers=0, C=None, kernel=None, gamma=None, svm_kernel=None, svm_c=None, svm_gamma=None):
        super(SVMModel, self).__init__()
        
        self.mix_beta = mix_beta
        self.input_fdim = input_fdim  # 输入特征维度
        self.input_tdim = input_tdim  # 输入时间维度
        self.label_dim = label_dim
        self.model_size = model_size
        self.verbose = verbose
        self.final_feat_dim = 128  # 降维后的特征维度
        self.feature_dim = 128

        if verbose:
            print('---------------SVM模型概要---------------')
            print(f'使用SVM-{model_size}配置')
            print(f'输入维度: 特征={input_fdim}, 时间步={input_tdim}')
        
        # 接受命令行参数传入的SVM参数（优先级最高）
        if svm_c is not None:
            self.C = float(svm_c)
        elif C is not None:
            self.C = float(C)
        else:
            # 根据模型大小设置SVM参数（默认配置）
            if model_size == 'small':
                self.C = 1.0
            elif model_size == 'medium':
                self.C = 10.0
            elif model_size == 'large':
                self.C = 100.0
            else:
                self.C = 1.0
        
        if svm_kernel is not None:
            self.kernel = svm_kernel
        elif kernel is not None:
            self.kernel = kernel
        else:
            # 设置默认核函数
            if model_size == 'small':
                self.kernel = 'linear'
            else:
                self.kernel = 'rbf'
        
        if svm_gamma is not None:
            self.gamma = svm_gamma
        elif gamma is not None:
            self.gamma = gamma
        else:
            self.gamma = 'scale'
        
        # 根据模型大小设置最大特征数
        if model_size == 'small':
            self.max_features = 64
        elif model_size == 'medium':
            self.max_features = 128
        else:  # large
            self.max_features = 256
            
        # 初始化SVM分类器
        svm_params = {
            'C': self.C,
            'kernel': self.kernel,
            'probability': True,
            'verbose': True if verbose else False
        }
        
        # gamma参数仅在rbf、poly或sigmoid核函数时有效
        if self.kernel in ['rbf', 'poly', 'sigmoid']:
            svm_params['gamma'] = self.gamma
        
        self.svm_model = SVC(**svm_params)
        
        # 记录SVM配置
        if verbose:
            print(f'SVM参数: kernel={self.kernel}, C={self.C}' + 
                  (f', gamma={self.gamma}' if 'gamma' in svm_params else ''))
    
        # 标准化预处理
        self.scaler = StandardScaler()
        
        # 用于保存训练数据和标签
        self.train_features = []
        self.train_labels = []
        self.is_fitted = False
        
        # 特征降维预处理层
        self.feature_projector = nn.Sequential(
            nn.Linear(input_fdim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, self.max_features),
            nn.ReLU(),
            nn.BatchNorm1d(self.max_features)
        )
        
        # 特征聚合层 - 降低时间维度复杂度
        self.time_aggregation = nn.Sequential(
            nn.AdaptiveAvgPool1d(4),  # 时间维度压缩到固定长度
            nn.Flatten()  # 展平为一维特征向量
        )
        
        self.final_feat_dim = self.max_features * 4  # 时间聚合后的特征长度
        
        # 为了与其他模型保持一致的接口
        self.mlp_head = nn.Linear(self.final_feat_dim, label_dim)
        
        # 添加patch_embed属性以兼容框架接口
        class DummyPatchEmbed:
            def __init__(self, num_patches):
                self.num_patches = num_patches
        
        self.patch_embed = DummyPatchEmbed(input_tdim)
        self.v = type('', (), {})()
        self.v.patch_embed = self.patch_embed
        
        if verbose:
            print(f'最终特征维度: {self.final_feat_dim}')
    
    def load_sl_official_weights(self):
        """
        兼容性方法，SVM不使用预训练权重
        """
        print("SVM模型不使用预训练权重")
        return
    
    def get_shape(self, input_fdim, input_tdim):
        """计算模型输出特征图尺寸"""
        return self.max_features, 4  # 特征维度和压缩后的时间维度
    
    def load_audio_pretrained(self, pretrained_path):
        """从预训练的SVM模型加载"""
        if os.path.exists(pretrained_path):
            print(f"从以下路径加载预训练SVM模型: {pretrained_path}")
            try:
                with open(pretrained_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                    
                # 加载SVM模型和特征缩放器
                if isinstance(loaded_data, dict):
                    if 'svm_model' in loaded_data and 'scaler' in loaded_data:
                        self.svm_model = loaded_data['svm_model']
                        self.scaler = loaded_data['scaler']
                        self.is_fitted = True
                        print("成功加载预训练SVM模型和缩放器")
                        return True
                    else:
                        print("加载的数据缺少SVM模型或缩放器")
                else:
                    # 尝试直接加载为SVM模型
                    self.svm_model = loaded_data
                    print("成功加载预训练SVM模型(无缩放器)")
                    self.is_fitted = True
                    return True
            except Exception as e:
                print(f"加载SVM模型出错: {e}")
                return False
        else:
            print(f"在以下路径未找到预训练模型: {pretrained_path}")
            return False

    def extract_features(self, x):
        """
        从输入数据中提取特征
        :param x: 输入数据，形状为 [B, T, F] 或 [B, 1, T, F] 或 [B, F, T]
        :return: 提取的特征，形状为 [B, max_features * 4]
        """
        # 获取设备
        device = x.device
        
        # 确保模型参数在正确的设备上
        if next(self.feature_projector.parameters()).device != device:
            self.feature_projector = self.feature_projector.to(device)
            self.time_aggregation = self.time_aggregation.to(device)
    
        # 确保输入格式正确
        if x.dim() == 4:  # [B, 1, T, F]
            x = x.squeeze(1)  # [B, T, F]
            
        batch_size = x.shape[0]
    
        # 如果输入形状是 [B, F, T]，转换为 [B, T, F]
        if x.dim() == 3 and x.size(1) == self.input_fdim:  # [B, F, T]
            x = x.transpose(1, 2)  # [B, T, F]
    
        # 确保输入在正确的设备上
        if x.device != device:
            x = x.to(device)
    
        try:
            # 对时间维度进行平均来获取每个样本的特征
            # 从 [B, T, F] 转换为 [B, F]
            x_mean = torch.mean(x, dim=1)  # [B, F]
            
            # 应用特征投影 (在特征维度上)
            # BatchNorm1d需要形状为 [B, C] 或 [B, C, L]
            x_proj = self.feature_projector(x_mean)  # [B, max_features]
            
            # 扩展维度以适应时间聚合层的输入要求
            x_expanded = x_proj.unsqueeze(-1)  # [B, max_features, 1]
            
            # 为了兼容性，我们复制几次时间维度
            x_repeated = x_expanded.repeat(1, 1, 4)  # [B, max_features, 4]
            
            # 应用时间维度聚合
            x_final = self.time_aggregation(x_repeated)  # [B, max_features * 4]
            
        except RuntimeError as e:
            if "running_mean should contain" in str(e):
                # 这是BatchNorm维度问题
                print(f"批归一化维度错误: {e}")
                print(f"输入形状: {x.shape}")
                
                # 尝试不同的方法
                # 重新定义单维度BatchNorm
                for module in self.feature_projector:
                    if isinstance(module, nn.BatchNorm1d):
                        # 获取当前处理的特征维度
                        if hasattr(module, 'num_features'):
                            num_features = module.num_features
                            # 创建新的正确维度的BatchNorm
                            new_bn = nn.BatchNorm1d(num_features).to(device)
                            # 复制权重和参数
                            if hasattr(module, 'weight') and module.weight is not None:
                                new_bn.weight.data = module.weight.data
                            if hasattr(module, 'bias') and module.bias is not None:
                                new_bn.bias.data = module.bias.data
                            # 替换
                            idx = list(self.feature_projector).index(module)
                            self.feature_projector[idx] = new_bn
                
                # 重试
                x_mean = torch.mean(x, dim=1)  # [B, F]
                x_proj = self.feature_projector(x_mean)  # [B, max_features]
                x_expanded = x_proj.unsqueeze(-1)  # [B, max_features, 1]
                x_repeated = x_expanded.repeat(1, 1, 4)  # [B, max_features, 4]
                x_final = self.time_aggregation(x_repeated)  # [B, max_features * 4]
            
            elif "Expected all tensors to be on the same device" in str(e):
                # 设备不匹配问题
                self.feature_projector = self.feature_projector.to(device)
                self.time_aggregation = self.time_aggregation.to(device)
                x = x.to(device)
                
                # 重试
                x_mean = torch.mean(x, dim=1)  # [B, F]
                x_proj = self.feature_projector(x_mean)  # [B, max_features]
                x_expanded = x_proj.unsqueeze(-1)  # [B, max_features, 1]
                x_repeated = x_expanded.repeat(1, 1, 4)  # [B, max_features, 4]
                x_final = self.time_aggregation(x_repeated)  # [B, max_features * 4]
            else:
                raise e
    
        # 最后检查输出是否在正确设备上
        if x_final.device != device:
            x_final = x_final.to(device)
    
        return x_final

    def fit_model(self):
        """训练SVM模型"""
        if len(self.train_features) > 0 and len(self.train_labels) > 0:
            # 将收集的特征和标签转换为numpy数组
            X = np.vstack(self.train_features)
            y = np.concatenate(self.train_labels)
            
            if self.verbose:
                print(f"训练SVM，数据形状: {X.shape}, 标签形状: {y.shape}")
            
            # 标准化特征
            X = self.scaler.fit_transform(X)
            
            # 训练模型
            self.svm_model.fit(X, y)
            self.is_fitted = True
            
            # 清空训练数据
            self.train_features = []
            self.train_labels = []
            
            if self.verbose:
                print(f"SVM训练完成，支持向量数量: {self.svm_model.n_support_.sum()}")
            return True
        return False

    def save_model(self, path):
        """保存SVM模型和缩放器"""
        if self.is_fitted:
            # 保存为一个包含模型和缩放器的字典
            save_dict = {
                'svm_model': self.svm_model,
                'scaler': self.scaler
            }
            with open(path, 'wb') as f:
                pickle.dump(save_dict, f)
            if self.verbose:
                print(f"SVM模型已保存到: {path}")
            return True
        else:
            print("模型尚未训练，无法保存")
            return False

    @autocast()
    def forward(self, x, y=None, patch_mix=False, time_domain=True):
        """
        :param x: 输入时序特征
        :param y: 标签，用于训练
        :param patch_mix: 不适用于SVM
        :param time_domain: 不适用于SVM
        :return: 特征表示或概率输出
        """
        # 获取输入设备
        device = x.device
    
        # 提取特征
        features = self.extract_features(x)
        
        # 如果在训练模式且有标签，收集特征和标签
        if self.training and y is not None:
            # 将tensor转换为numpy进行存储
            self.train_features.append(features.detach().cpu().numpy())
            self.train_labels.append(y.detach().cpu().numpy())
        
        # 对于评估模式，使用训练好的SVM进行预测
        if not self.training and self.is_fitted:
            try:
                # 提取numpy特征并进行标准化
                np_features = features.detach().cpu().numpy()
                np_features = self.scaler.transform(np_features)
                
                # 使用SVM进行预测
                class_probs = self.svm_model.predict_proba(np_features)
                
                # 将概率转换为张量，并明确指定设备
                probs = torch.from_numpy(class_probs).float().to(device)
                
                return probs
            except Exception as e:
                print(f"SVM预测错误: {e}")
                # 错误处理：返回均匀分布的概率
                batch_size = features.size(0)
                return torch.ones(batch_size, self.label_dim, device=device) / self.label_dim
    
        # 在训练过程中或模型尚未训练时，返回特征
        if not patch_mix:
            return features
        else:
            # SVM不支持patch_mix，但为了接口一致性，返回占位符
            batch_size = features.size(0)
            index = torch.arange(batch_size, device=device)  # 确保索引在正确设备上
            lam = 1.0
            return features, y, y, lam, index