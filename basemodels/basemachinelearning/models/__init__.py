from .projector import Projector  # 保留 Projector
from .KNN import KNNModel  # 导入 KNN 模型
from .RandomForest import RandomForestModel  # 导入随机森林模型
from .SVM import SVMModel  # 导入 SVM 模型

# 模型名称到类的映射
KNNModel = KNNModel  # 将 'knn' 映射到 KNNModel 类
RandomForestModel = RandomForestModel  # 将 'randomforest' 映射到 RandomForestModel 类
SVMModel = SVMModel  # 将 'svm' 映射到 SVMModel 类

def get_backbone_class(name):
    """返回给定名称的算法类。"""
    if name == 'knn':  # 添加对 KNN 模型的支持
        return KNNModel
    if name == 'randomforest':  # 添加对随机森林模型的支持
        return RandomForestModel
    if name == 'svm':  # 添加对 SVM 模型的支持
        return SVMModel
    if name not in globals():
        raise NotImplementedError("未找到算法: {}".format(name))
    return globals()[name]