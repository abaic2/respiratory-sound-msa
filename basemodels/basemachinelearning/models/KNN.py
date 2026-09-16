import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import pickle
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier


class KNNModel(nn.Module):
    """
    K最近邻(KNN)模型用于音频分类，使用时域特征。
    :param label_dim: 类别数量
    :param input_fdim: 输入时间序列的特征维度
    :param input_tdim: 输入序列的时间维度
    :param model_size: KNN的配置 ('small', 'medium', 'large')，影响K值和距离计算方法
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
             imagenet_pretrain=False, audioset_pretrain=False, model_size='medium', verbose=True, mix_beta=None,
             freeze_base=False, freeze_layers=0, n_neighbors=None, weights=None, metric=None, algorithm=None,
             knn_n_neighbors=None, knn_weights=None, knn_algorithm=None, knn_metric=None):
        super(KNNModel, self).__init__()
        
        self.mix_beta = mix_beta
        self.input_fdim = input_fdim  # 输入特征维度
        self.input_tdim = input_tdim  # 输入时间维度
        self.label_dim = label_dim
        self.model_size = model_size
        self.verbose = verbose
        self.final_feat_dim = input_fdim  # 最终特征维度与输入特征维度相同

        if verbose:
            print('---------------KNN模型概要---------------')
            print(f'使用KNN-{model_size}配置')
            print(f'输入维度: 特征={input_fdim}, 时间步={input_tdim}')
        
        # 处理参数优先级: knn_前缀参数 > 直接参数 > 模型大小默认值
        
        # 处理n_neighbors参数
        if knn_n_neighbors is not None:
            self.n_neighbors = int(knn_n_neighbors)
        elif n_neighbors is not None:
            self.n_neighbors = int(n_neighbors)
        else:
            # 根据模型大小设置KNN参数
            if model_size == 'small':
                self.n_neighbors = 3
            elif model_size == 'medium':
                self.n_neighbors = 5
            elif model_size == 'large':
                self.n_neighbors = 10
            else:
                self.n_neighbors = 5  # 默认中等大小
    
        # 处理weights参数
        if knn_weights is not None:
            self.weights = knn_weights
        elif weights is not None:
            self.weights = weights
        else:
            if model_size == 'small':
                self.weights = 'uniform'
            else:
                self.weights = 'distance'
    
        # 处理algorithm参数
        if knn_algorithm is not None:
            self.algorithm = knn_algorithm
        elif algorithm is not None:
            self.algorithm = algorithm
        else:
            self.algorithm = 'auto'
    
        # 处理metric参数
        if knn_metric is not None:
            self.metric = knn_metric
        elif metric is not None:
            self.metric = metric
        else:
            if model_size == 'large':
                self.metric = 'minkowski'
            else:
                self.metric = 'euclidean'
        
        # 初始化KNN分类器
        self.knn_model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            metric=self.metric
        )
        
        # 用于保存训练数据和标签
        self.train_features = []
        self.train_labels = []
        self.is_fitted = False
        
        # 特征降维处理
        self.feature_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((input_fdim, 1)),
            nn.Flatten()
        )
        
        # 特征预处理层
        self.feature_preprocessor = nn.Sequential(
            nn.Linear(input_fdim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.final_feat_dim = 128
        
        # 添加patch_embed属性以兼容框架接口
        class DummyPatchEmbed:
            def __init__(self, num_patches):
                self.num_patches = num_patches
    
        self.patch_embed = DummyPatchEmbed(input_tdim)
        self.v = type('', (), {})()
        self.v.patch_embed = self.patch_embed
        
        # 为了与其他模型保持一致的接口
        self.mlp_head = nn.Linear(self.final_feat_dim, label_dim)
        
        if verbose:
            print(f'最终特征维度: {self.final_feat_dim}')
            print(f'KNN参数: k={self.n_neighbors}, 权重={self.weights}, 度量={self.metric}, 算法={self.algorithm}')
    
    def load_sl_official_weights(self):
        """
        兼容性方法，KNN不使用预训练权重
        """
        print("KNN模型不使用预训练权重")
        return
    
    def get_shape(self, input_fdim, input_tdim):
        """计算模型输出特征图尺寸"""
        return input_fdim, 1  # 时间维度被压缩为1
    
    def load_audio_pretrained(self, pretrained_path):
        """从预训练的KNN模型加载"""
        if os.path.exists(pretrained_path):
            print(f"从以下路径加载预训练KNN模型: {pretrained_path}")
            try:
                with open(pretrained_path, 'rb') as f:
                    self.knn_model = pickle.load(f)
                self.is_fitted = True
                print("成功加载预训练KNN模型")
                return True
            except Exception as e:
                print(f"加载KNN模型出错: {e}")
                return False
        else:
            print(f"在以下路径未找到预训练模型: {pretrained_path}")
            return False

    def extract_features(self, x):
        """
        从输入数据中提取特征
        :param x: 输入数据，形状为 [B, T, F] 或 [B, 1, T, F] 或 [B, F, T]
        :return: 提取的特征，形状为 [B, F]
        """
        # 获取设备
        device = x.device
    
        # 确保模型参数在正确的设备上
        if next(self.feature_preprocessor.parameters()).device != device:
            self.feature_preprocessor = self.feature_preprocessor.to(device)
    
        # 确保输入格式正确
        if x.dim() == 4:  # [B, 1, T, F]
            x = x.squeeze(1)  # [B, T, F]
        elif x.dim() == 3 and x.size(1) != self.input_tdim:  # [B, F, T]
            x = x.transpose(1, 2)  # [B, T, F]
    
        # 确保输入在正确的设备上
        if x.device != device:
            x = x.to(device)
    
        try:
            # 1. 应用均值池化压缩时间维度
            x = torch.mean(x, dim=1)  # [B, F]
            
            # 2. 应用特征预处理
            x = self.feature_preprocessor(x)  # [B, 128]
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                # 再次尝试确保设备一致
                self.feature_preprocessor = self.feature_preprocessor.to(device)
                x = x.to(device)
                # 重新执行计算
                x = torch.mean(x, dim=1)
                x = self.feature_preprocessor(x)
            else:
                raise e
    
        # 最后检查输出是否在正确设备上
        if x.device != device:
            x = x.to(device)
    
        return x

    def fit_model(self):
        """训练KNN模型"""
        if len(self.train_features) > 0 and len(self.train_labels) > 0:
            # 将收集的特征和标签转换为numpy数组
            X = np.vstack(self.train_features)
            y = np.concatenate(self.train_labels)
            
            if self.verbose:
                print(f"训练KNN，数据形状: {X.shape}, 标签形状: {y.shape}")
            
            # 训练模型
            self.knn_model.fit(X, y)
            self.is_fitted = True
            
            # 清空训练数据
            self.train_features = []
            self.train_labels = []
            
            return True
        return False

    def save_model(self, path):
        """保存KNN模型"""
        if self.is_fitted:
            with open(path, 'wb') as f:
                pickle.dump(self.knn_model, f)
            print(f"KNN模型已保存到: {path}")
            return True
        else:
            print("模型尚未训练，无法保存")
            return False

    @autocast()
    def forward(self, x, y=None, patch_mix=False, time_domain=True):
        """
        :param x: 输入时序特征
        :param y: 标签，用于训练
        :param patch_mix: 不适用于KNN
        :param time_domain: 不适用于KNN
        :return: 特征表示
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
        
        # 对于评估模式，使用训练好的KNN进行预测
        if not self.training and self.is_fitted:
            try:
                # 提取numpy特征进行预测
                np_features = features.detach().cpu().numpy()
                
                # 获取KNN的距离和索引
                distances, indices = self.knn_model.kneighbors(np_features)
                
                # 如果使用距离作为权重，则根据距离计算概率
                batch_size = np_features.shape[0]
                
                # 重要：在同一设备上创建张量
                probs = torch.zeros(batch_size, self.label_dim, device=device)
                
                # 获取训练集标签
                train_labels = self.knn_model.classes_
                
                # 为每个样本计算类别概率
                for i in range(batch_size):
                    neighbor_indices = indices[i]
                    neighbor_distances = distances[i]
                    
                    # 防止除以零
                    neighbor_distances = np.maximum(neighbor_distances, 1e-10)
                    
                    # 计算权重
                    if self.weights == 'distance':
                        weights = 1.0 / neighbor_distances
                        weights = weights / np.sum(weights)
                    else:
                        weights = np.ones(self.n_neighbors) / self.n_neighbors
                    
                    # 获取邻居的标签
                    neighbor_labels = self.knn_model._y[neighbor_indices]
                    
                    # 为每个类别累加权重
                    for j, label in enumerate(neighbor_labels):
                        # 使用整数索引和浮点权重，确保类型兼容
                        label_idx = int(label)
                        weight_val = float(weights[j])
                        probs[i, label_idx] += weight_val
            
            except Exception as e:
                print(f"KNN预测出错: {e}")
                # 出错时返回均匀分布的概率，确保不会中断评估过程
                return torch.ones(features.size(0), self.label_dim, device=device)
    
        # 在训练过程中或模型尚未训练时，返回特征
        if not patch_mix:
            return features
        else:
            # KNN不支持patch_mix，但为了接口一致性，返回占位符
            batch_size = features.size(0)
            index = torch.arange(batch_size, device=device)  # 在同一设备上创建
            lam = 1.0
            return features, y, y, lam, index