from .resnet import ResNetModel  # 假设您已将 ast.py 改名为 resnet.py
from .projector import Projector  # 添加这一行导入 Projector
from .googlenet import GoogLeNetModel  # 添加这一行导入 GoogLeNetModel
from .EfficientNet import EfficientNetModel  # 添加这一行导入 EfficientNetModel

# 模型名称到类的映射
ResNetModel = ResNetModel  # 这将 'resnet' 映射到 ResNetModel 类
GoogLeNetModel = GoogLeNetModel  # 这将 'googlenet' 映射到 GoogLeNetModel 类
EfficientNetModel = EfficientNetModel  # 这将 'efficientnet' 映射到 EfficientNetModel 类

def get_backbone_class(name):
    """Return the algorithm class with the given name."""
    if name == 'resnet':
        return ResNetModel
    if name == 'googlenet':  # 添加对 googlenet 模型的支持
        return GoogLeNetModel
    if name == 'efficientnet':  # 添加对 efficientnet 模型的支持
        return EfficientNetModel
    if name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(name))
    return globals()[name]