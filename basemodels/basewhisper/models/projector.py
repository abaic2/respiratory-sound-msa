import torch
import torch.nn as nn


class Projector(nn.Module):
    """标准投影器，用于特征投影"""
    def __init__(self, in_dim, out_dim):
        super(Projector, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class WhisperProjector(nn.Module):
    """专为Whisper模型设计的投影器"""
    def __init__(self, whisper_model_size='base', out_dim=768):
        super(WhisperProjector, self).__init__()
        
        # 根据Whisper模型大小确定输入维度
        whisper_dims = {
            'tiny': 384,
            'small': 768, 
            'base': 512,
            'large': 1024
        }
        
        in_dim = whisper_dims.get(whisper_model_size, 512)
        
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        
        print(f"🎵 WhisperProjector: {whisper_model_size} ({in_dim}) -> {out_dim}")

    def forward(self, x):
        return self.proj(x)


# 兼容性函数
def get_projector(model_type, in_dim=None, out_dim=768, whisper_model_size='base'):
    """根据模型类型获取合适的投影器"""
    if model_type == 'whisper':
        return WhisperProjector(whisper_model_size, out_dim)
    else:
        return Projector(in_dim, out_dim)