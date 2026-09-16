from .projector import Projector  # 导入 Projector
from .GRU import GRUModel  # 导入 GRUModel
from .LSTM import LSTMModel  # 导入 LSTMModel
from .BiLSTM import BiLSTMModel  # 导入 BiLSTMModel 替代 TCNModel

# 模型名称到类的映射
GRUModel = GRUModel  # 这将 'gru' 映射到 GRUModel 类
LSTMModel = LSTMModel  # 这将 'lstm' 映射到 LSTMModel 类
BiLSTMModel = BiLSTMModel  # 这将 'bilstm' 映射到 BiLSTMModel 类

def get_backbone_class(name):
    """Return the algorithm class with the given name."""
    if name == 'gru':  # 添加对 gru 模型的支持
        return GRUModel
    if name == 'lstm':  # 添加对 lstm 模型的支持
        return LSTMModel
    if name == 'bilstm':  # 添加对 bilstm 模型的支持
        return BiLSTMModel
    if name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(name))
    return globals()[name]