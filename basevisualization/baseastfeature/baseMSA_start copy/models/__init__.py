from .cnn6 import CNN6
from .resnet import ResNet10, ResNet18, ResNet34, ResNet50, ResNet101
from .efficientnet import EfficientNet_B0, EfficientNet_B1, EfficientNet_B2
from .ast import ASTModel, AST_Lightweight_MSAF
from .ssast import SSASTModel
from .projector import Projector

_backbone_class_map = {
    'cnn6': CNN6,
    'resnet10': ResNet10,
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'efficientnet_b0': EfficientNet_B0,
    'efficientnet_b1': EfficientNet_B1,
    'efficientnet_b2': EfficientNet_B2,
    'ast': ASTModel,
    'ssast': SSASTModel
}


def get_backbone_class(backbone_name):
    """返回模型类"""
    if backbone_name == 'ast':
        return AST_Lightweight_MSAF  # 🆕 返回带MSA的版本
    elif backbone_name == 'ast_original':
        return ASTModel  # 如果需要原始AST，使用这个别名
    else:
        raise NotImplementedError(f'Backbone {backbone_name} not implemented')