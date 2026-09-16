from .EVA import EVAModel  # 从 eva.py 导入 EVA 模型
from .projector import Projector
from .swintransformer import SwinModel  # 从 swin.py 导入 Swin Transformer 模型
from .VIT import ViTModel  # 从 vit.py 导入 Vision Transformer 模型
from .deit import DeiTModel  # 从 deit.py 导入 Data-efficient image Transformer 模型

# 模型名称到类的映射
EVAModel = EVAModel
SwinModel = SwinModel
ViTModel = ViTModel
DeiTModel = DeiTModel  # 添加 DeiT 模型

def get_backbone_class(name):
    """根据名称返回相应的模型类"""
    if name == 'eva':
        return EVAModel
    if name == 'swin':
        return SwinModel
    if name == 'vit':
        return ViTModel
    if name == 'deit':  # 添加 deit 选项
        return DeiTModel
    if name not in globals():
        raise NotImplementedError(f"找不到模型: {name}")
    return globals()[name]