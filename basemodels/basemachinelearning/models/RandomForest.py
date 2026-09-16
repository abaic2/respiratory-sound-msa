import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import pickle
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel(nn.Module):
    """
    随机森林模型用于音频分类，使用时域特征。
    :param label_dim: 类别数量
    :param input_fdim: 输入时间序列的特征维度
    :param input_tdim: 输入序列的时间维度
    :param model_size: 随机森林的大小 ('small', 'medium', 'large')
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
         imagenet_pretrain=False, audioset_pretrain=False, model_size='medium', verbose=True, mix_beta=None,
         freeze_base=False, freeze_layers=0, n_estimators=None, max_depth=None, rf_n_estimators=None, rf_max_depth=None,
         min_samples_split=None, min_samples_leaf=None, rf_min_samples_split=None, rf_min_samples_leaf=None, **kwargs):
        super(RandomForestModel, self).__init__()
        
        self.mix_beta = mix_beta
        self.input_fdim = input_fdim  # 输入特征维度（时序特征数量）
        self.input_tdim = input_tdim  # 输入时间维度（序列长度）
        self.label_dim = label_dim
        self.model_size = model_size
        self.verbose = verbose
        self.final_feat_dim = input_fdim  # 最终特征维度与输入特征维度相同

        if verbose:
            print('---------------随机森林模型概要---------------')
            print(f'使用 RandomForest-{model_size} 结构')
            print(f'输入维度: 特征={input_fdim}, 时间步={input_tdim}')
        
        # 优先使用直接传入的参数
        if rf_n_estimators is not None:
            self.n_estimators = int(rf_n_estimators)
        elif n_estimators is not None:
            self.n_estimators = int(n_estimators)
        else:
            # 根据模型大小设置随机森林参数
            if model_size == 'small':
                self.n_estimators = 100
            elif model_size == 'medium':
                self.n_estimators = 200
            elif model_size == 'large':
                self.n_estimators = 500
            else:
                self.n_estimators = 200

        if rf_max_depth is not None:
            self.max_depth = int(rf_max_depth)
        elif max_depth is not None:
            self.max_depth = int(max_depth)
        else:
            # 根据模型大小设置最大深度
            if model_size == 'small':
                self.max_depth = 10
            elif model_size == 'medium':
                self.max_depth = 20
            elif model_size == 'large':
                self.max_depth = 30
            else:
                self.max_depth = 20
        
        # 处理min_samples_split参数
        if rf_min_samples_split is not None:
            self.min_samples_split = int(rf_min_samples_split)
        elif min_samples_split is not None:
            self.min_samples_split = int(min_samples_split)
        else:
            # 默认值
            self.min_samples_split = 2
        
        # 处理min_samples_leaf参数
        if rf_min_samples_leaf is not None:
            self.min_samples_leaf = int(rf_min_samples_leaf)
        elif min_samples_leaf is not None:
            self.min_samples_leaf = int(min_samples_leaf)
        else:
            # 默认值
            self.min_samples_leaf = 1
    
        # 初始化随机森林分类器
        self.rf_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            n_jobs=-1,  # 使用所有可用的CPU核心
            verbose=1 if verbose else 0
        )
        
        # 用于保存训练数据和标签
        self.train_features = []
        self.train_labels = []
        self.is_fitted = False
        
        # 特征降维投影层 (用于减少时间维度)
        self.feature_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((input_fdim, 1)),
            nn.Flatten()
        )
        
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
            print(f'随机森林参数: 树数量={self.n_estimators}, 最大深度={self.max_depth}, '
                 f'最小分裂样本数={self.min_samples_split}, 最小叶节点样本数={self.min_samples_leaf}')
    
    def load_sl_official_weights(self):
        """
        兼容性方法，随机森林不使用预训练权重
        """
        print("随机森林模型不使用预训练权重")
        return
    
    def get_shape(self, input_fdim, input_tdim):
        """计算模型输出特征图尺寸"""
        return input_fdim, 1  # 时间维度被压缩为1
    
    def load_audio_pretrained(self, pretrained_path):
        """从预训练的随机森林模型加载"""
        if os.path.exists(pretrained_path):
            print(f"从以下路径加载预训练随机森林模型: {pretrained_path}")
            try:
                with open(pretrained_path, 'rb') as f:
                    self.rf_model = pickle.load(f)
                self.is_fitted = True
                print("成功加载预训练随机森林模型")
                return True
            except Exception as e:
                print(f"加载随机森林模型出错: {e}")
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
        # 确保输入格式正确
        if x.dim() == 4:  # [B, 1, T, F]
            x = x.squeeze(1)  # [B, T, F]
        elif x.dim() == 3 and x.size(1) != self.input_tdim:  # [B, F, T]
            x = x.transpose(1, 2)  # [B, T, F]
        
        # 使用平均池化压缩时间维度
        x = torch.mean(x, dim=1)  # [B, F]
        
        return x

    def fit_model(self):
        """训练随机森林模型"""
        if len(self.train_features) > 0 and len(self.train_labels) > 0:
            # 将收集的特征和标签转换为numpy数组
            X = np.vstack(self.train_features)
            y = np.concatenate(self.train_labels)
            
            if self.verbose:
                print(f"训练随机森林，数据形状: {X.shape}, 标签形状: {y.shape}")
            
            # 训练模型
            self.rf_model.fit(X, y)
            self.is_fitted = True
            
            # 清空训练数据
            self.train_features = []
            self.train_labels = []

    def save_model(self, path):
        """保存随机森林模型"""
        if self.is_fitted:
            with open(path, 'wb') as f:
                pickle.dump(self.rf_model, f)
            return True
        else:
            print("模型尚未训练，无法保存")
            return False

    @autocast()
    def forward(self, x, y=None, patch_mix=False, time_domain=True):
        """
        :param x: 输入时序特征
        :param y: 标签，用于训练
        :param patch_mix: 不适用于随机森林
        :param time_domain: 不适用于随机森林
        :return: 特征表示
        """
        # 提取特征
        features = self.extract_features(x)
        
        # 如果在训练模式且有标签，收集特征和标签
        if self.training and y is not None:
            # 将tensor转换为numpy进行存储
            self.train_features.append(features.detach().cpu().numpy())
            self.train_labels.append(y.detach().cpu().numpy())
        
        # 对于评估模式，使用训练好的随机森林进行预测
        if not self.training and self.is_fitted:
            # 提取numpy特征进行预测
            np_features = features.detach().cpu().numpy()
            
            # 使用随机森林进行预测
            predictions = torch.from_numpy(
                self.rf_model.predict_proba(np_features)
            ).float().to(features.device)
            
            # 通过线性层模拟输出，以保持与其他模型接口一致
            return predictions
        
        # 在训练过程中或模型尚未训练时，返回特征
        if not patch_mix:
            return features
        else:
            # 随机森林不支持patch_mix，但为了接口一致性，返回占位符
            batch_size = features.size(0)
            device = features.device
            index = torch.arange(batch_size).to(device)
            lam = 1.0
            return features, y, y, lam, index