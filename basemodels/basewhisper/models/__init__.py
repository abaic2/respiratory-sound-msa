from .projector import Projector, WhisperProjector  # 添加WhisperProjector
from .whisper import WhisperAudioClassifier  # 添加 Whisper 音频分类器

# 导出所有模型类
WhisperAudioClassifier = WhisperAudioClassifier  # 添加 Whisper 模型

def get_backbone_class(name):
    """根据名称返回相应的模型类"""
    if name == 'whisper':  # 添加 whisper 选项
        return WhisperAudioClassifier
    if name not in globals():
        raise NotImplementedError(f"找不到模型: {name}")
    return globals()[name]

# 导出所有必要的类和函数
__all__ = [
    'get_backbone_class',
    'Projector', 
    'WhisperProjector',
    'WhisperAudioClassifier'
]