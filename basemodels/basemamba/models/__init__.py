from .cnn6 import CNN6
from .resnet import ResNet10, ResNet18, ResNet34, ResNet50, ResNet101
from .efficientnet import EfficientNet_B0, EfficientNet_B1, EfficientNet_B2
from .ast import ASTModel
from .ssast import SSASTModel
from .projector import Projector

# 导入Mamba模型
try:
    from .mamba import ASTModel as MambaModel, create_mamba_audio_model
    MAMBA_AVAILABLE = True
    print("✅ Mamba模型已加载")
except ImportError as e:
    print(f"⚠️ Mamba模型加载失败: {e}")
    MAMBA_AVAILABLE = False
    MambaModel = None

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

# 添加Mamba模型到映射表
if MAMBA_AVAILABLE:
    _backbone_class_map['mamba'] = MambaModel
    _backbone_class_map['mamba_ast'] = MambaModel  # 别名，用于替代AST

def get_backbone_class(key):
    # 特殊处理：如果请求ast但想使用mamba，可以通过环境变量控制
    import os
    if key == 'ast' and os.getenv('USE_MAMBA', 'false').lower() == 'true' and MAMBA_AVAILABLE:
        print("🐍 使用Mamba模型替代AST")
        return MambaModel
    
    if key in _backbone_class_map:
        return _backbone_class_map[key]
    else:
        available_models = list(_backbone_class_map.keys())
        raise ValueError(f'Invalid backbone: {key}. Available models: {available_models}')

# 导出便捷函数
def create_model(model_name, **kwargs):
    """
    便捷的模型创建函数
    
    Args:
        model_name: 模型名称 ('ast', 'mamba', 'cnn6', 等)
        **kwargs: 模型参数
    
    Returns:
        创建的模型实例
    """
    if model_name == 'mamba' and MAMBA_AVAILABLE:
        return create_mamba_audio_model(**kwargs)
    else:
        model_class = get_backbone_class(model_name)
        return model_class(**kwargs)

# 导出所有需要的类和函数
__all__ = [
    'CNN6', 'ResNet10', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101',
    'EfficientNet_B0', 'EfficientNet_B1', 'EfficientNet_B2',
    'ASTModel', 'SSASTModel', 'Projector',
    'get_backbone_class', 'create_model'
]

if MAMBA_AVAILABLE:
    __all__.extend(['MambaModel', 'create_mamba_audio_model'])