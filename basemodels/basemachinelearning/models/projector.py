import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim=128, apply_bn=True):
        """
        特征投影器，用于将高维特征降维到适合传统机器学习算法的维度
        
        :param in_dim: 输入特征维度
        :param out_dim: 输出特征维度，默认为128（适合传统机器学习）
        :param apply_bn: 是否应用批量归一化
        """
        super(Projector, self).__init__()
        
        # 构建一个两层MLP
        self.linear1 = nn.Linear(in_dim, min(in_dim, 512))
        mid_dim = min(in_dim, 512)
        self.linear2 = nn.Linear(mid_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(mid_dim)
        self.relu = nn.ReLU()
        
        # 根据apply_bn参数决定是否使用批量归一化
        if apply_bn:
            self.projector = nn.Sequential(
                self.linear1, 
                self.bn1, 
                self.relu, 
                self.linear2
            )
        else:
            self.projector = nn.Sequential(
                self.linear1, 
                self.relu, 
                self.linear2
            )

    def forward(self, x):
        """
        前向传播，将输入特征投影到低维空间
        """
        # 处理可能的不同输入形状
        orig_shape = x.shape
        if len(orig_shape) > 2:
            # 如果不是2D张量，先展平
            x = x.reshape(orig_shape[0], -1)
        
        # 应用投影
        x = self.projector(x)
        
        return x