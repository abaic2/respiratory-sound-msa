import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import warnings
from io import BytesIO
from PIL import Image as PILImage
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

"""
# 机器学习模型训练流程输入输出总结

## 数据加载与预处理流程
原始特征文件: train_[model]_full_features.csv, test_[model]_full_features.csv
     ↓
CSV数据加载: (N_train, D+1), (N_test, D+1)    (samples, features+label) 读取训练和测试特征
     ↓
特征标签分离: X_train(N_train, D), y_train(N_train,), X_test(N_test, D), y_test(N_test,)    分离特征矩阵和标签向量
     ↓
缺失值处理: SimpleImputer(strategy='mean')    (samples, features) 使用均值填充NaN值
     ↓
特征标准化: StandardScaler() -> X_scaled    (samples, features) Z-score标准化到均值0方差1
     ↓
预处理完成: X_train_final, X_test_final    (samples, features) 准备好的训练和测试特征

## SVM模型训练流程
输入特征: X_train(N, D), y_train(N,)    (samples, features) 标准化后的训练数据
     ↓
SVM超参数: kernel='rbf', C=1.0, gamma='scale'    支持向量机核函数和正则化参数
     ↓
模型训练: SVC.fit(X_train, y_train)    寻找最优超平面和支持向量
     ↓
决策边界: 超平面 w·x + b = 0    (feature_space) 高维空间中的分类边界
     ↓
模型预测: SVC.predict(X_test) -> y_pred(N_test,)    (samples,) 测试集预测结果
     ↓
评估指标: Accuracy, SP, SE, ICBHI_Score    分类性能指标计算

## Random Forest模型训练流程
输入特征: X_train(N, D), y_train(N,)    (samples, features) 训练数据
     ↓
森林超参数: n_estimators=100, max_depth=None    决策树数量和最大深度
     ↓
Bootstrap采样: 每棵树随机采样训练数据    (samples, features) 有放回抽样
     ↓
特征随机选择: 每个节点随机选择sqrt(D)个特征    (node, sqrt(features)) 减少过拟合
     ↓
决策树构建: 100棵独立决策树并行训练    (trees, nodes) 集成学习
     ↓
投票预测: 多数投票或平均概率    (samples, classes) 集成预测结果
     ↓
特征重要性: feature_importances_    (features,) 每个特征的重要性分数

## KNN模型训练流程
输入特征: X_train(N, D), y_train(N,)    (samples, features) 训练数据存储
     ↓
KNN超参数: n_neighbors=5, weights='uniform'    邻居数量和权重策略
     ↓
距离计算: 欧几里得距离或其他距离度量    (query_sample, training_samples) 计算查询点到所有训练点距离
     ↓
邻居搜索: K个最近邻居查找    (query_sample, k_neighbors) 找出最相似的K个样本
     ↓
投票预测: 邻居标签投票决定预测结果    (k_neighbors,) -> prediction 多数投票分类
     ↓
懒惰学习: 无显式训练阶段，预测时计算    实时计算，无模型参数
"""

# 设置无缓冲输出以便实时显示日志
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
print("机器学习模型训练程序启动...")

# 忽略特定警告
warnings.filterwarnings('ignore', category=UserWarning)


def get_score(y_pred, y_true, args):
    """
    计算ICBHI指标: 特异度(SP)、敏感度(SE)和平均分数(SC)
    SP - 特异度：正确预测为正常的比例
    SE - 敏感度：精确预测到具体异常类型的比例
    SC - ICBHI分数：SP和SE的平均值
    """
    # 如果是二分类问题
    if args.n_cls == 2:
        # 计算混淆矩阵元素
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # 计算特异度SP: 真阴性率 (TN / (TN + FP))
        sp = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        
        # 计算敏感度SE: 真阳性率 (TP / (TP + FN))
        se = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    
    # 多分类问题(假设类别0是正常，其他是不同类型的异常)
    else:
        # 创建混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 假设类别0是正常类
        normal_idx = 0
        
        # 计算特异度SP: 正确预测正常类的比例
        if np.sum(y_true == normal_idx) > 0:
            sp = cm[normal_idx, normal_idx] / np.sum(y_true == normal_idx) * 100
        else:
            sp = 0
        
        # 计算敏感度SE：正确预测各种异常类型的平均准确率
        abnormal_accs = []
        for cls_idx in range(1, args.n_cls):  # 遍历所有异常类别
            # 计算每种异常类别的精确识别率
            if np.sum(y_true == cls_idx) > 0:
                cls_acc = cm[cls_idx, cls_idx] / np.sum(y_true == cls_idx) * 100
                abnormal_accs.append(cls_acc)
        
        # 如果有异常类别样本，则计算平均敏感度
        if abnormal_accs:
            se = np.mean(abnormal_accs)
        else:
            se = 0
    
    # 计算ICBHI分数: SP和SE的平均值
    sc = (sp + se) / 2
    
    return sp, se, sc


def load_data(args):
    """
    加载特征CSV文件并预处理
    """
    print(f"加载{args.model}数据...")
    
    try:
        # 训练集路径
        train_csv_path = os.path.join(args.data_dir, f'train_{args.model}_full_features.csv')
        
        # 测试集路径
        test_csv_path = os.path.join(args.data_dir, f'test_{args.model}_full_features.csv')
        
        # 检查文件是否存在
        if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
            print(f"错误: 数据文件不存在。检查路径: {train_csv_path}, {test_csv_path}")
            return None, None, None, None
        
        # 检查CSV文件的第一行是否为表头
        with open(train_csv_path, 'r') as f:
            first_line = f.readline().strip()
            has_header = 'feat_0' in first_line or not first_line[0].isdigit()
        
        print(f"CSV文件{'包含' if has_header else '不包含'}表头")
        
        # 加载数据
        if has_header:
            train_data = pd.read_csv(train_csv_path, header=0)
            test_data = pd.read_csv(test_csv_path, header=0)
        else:
            train_data = pd.read_csv(train_csv_path, header=None)
            test_data = pd.read_csv(test_csv_path, header=None)
        
        print(f"训练集形状: {train_data.shape}, 测试集形状: {test_data.shape}")
        
        # 检查NaN值
        train_nan_count = train_data.isna().sum().sum()
        test_nan_count = test_data.isna().sum().sum()
        
        if train_nan_count > 0 or test_nan_count > 0:
            print(f"警告: 训练集中有{train_nan_count}个NaN值，测试集中有{test_nan_count}个NaN值")
            print("正在处理NaN值...")
            
            # 使用SimpleImputer填充NaN值
            imputer = SimpleImputer(strategy='mean')
            
            # 分离特征和标签
            X_train = train_data.iloc[:, :-1].values  # 最后一列是标签
            y_train = train_data.iloc[:, -1].values
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values
            
            # 填充NaN值
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
            
            # 检查是否还有NaN值
            if np.isnan(X_train).any() or np.isnan(X_test).any():
                print("警告: 填充后仍存在NaN值，将替换为0")
                X_train = np.nan_to_num(X_train)
                X_test = np.nan_to_num(X_test)
        else:
            # 分离特征和标签
            X_train = train_data.iloc[:, :-1].values  # 最后一列是标签
            y_train = train_data.iloc[:, -1].values
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values
        
        # 特征标准化
        if args.standardize:
            print("应用特征标准化...")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        print(f"数据加载和预处理完成。特征维度: {X_train.shape[1]}")
        return X_train, y_train, X_test, y_test
    
    except Exception as e:
        print(f"加载数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def create_model(args):
    """
    根据指定的模型类型创建机器学习模型
    """
    if args.model == 'svm':
        print(f"创建SVM模型 (kernel={args.kernel}, C={args.C})...")
        model = SVC(
            kernel=args.kernel, 
            C=args.C, 
            gamma=args.gamma,
            probability=True,
            random_state=args.seed
        )
    
    elif args.model == 'knn':
        print(f"创建KNN模型 (n_neighbors={args.n_neighbors})...")
        model = KNeighborsClassifier(
            n_neighbors=args.n_neighbors,
            weights=args.weights,
            algorithm=args.algorithm
        )
    
    elif args.model == 'randomforest':
        print(f"创建RandomForest模型 (n_estimators={args.n_estimators})...")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=args.seed,
            n_jobs=args.n_jobs
        )
    
    else:
        raise ValueError(f"不支持的模型类型: {args.model}")
    
    return model


def train_model(model, X_train, y_train, args):
    """
    训练机器学习模型
    """
    print(f"训练{args.model.upper()}模型...")
    start_time = time.time()
    
    try:
        # 使用进度条训练模型
        with tqdm.tqdm(total=100, desc=f"训练{args.model.upper()}") as pbar:
            pbar.update(10)  # 初始进度
            model.fit(X_train, y_train)
            pbar.update(90)  # 完成进度
        
        train_time = time.time() - start_time
        print(f"训练完成! 用时: {train_time:.2f}秒")
        
        # 评估训练集
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred) * 100
        
        # 计算训练集ICBHI指标
        train_sp, train_se, train_sc = get_score(y_train_pred, y_train, args)
        
        print(f"训练集准确率: {train_acc:.2f}%")
        print(f"训练集 ICBHI指标 - SP: {train_sp:.2f}%, SE: {train_se:.2f}%, Score: {train_sc:.2f}%")
        
        # 如果是随机森林，打印特征重要性
        if args.model == 'randomforest':
            feature_importance = model.feature_importances_
            # 打印前10个最重要的特征
            indices = np.argsort(feature_importance)[::-1][:10]
            print("\n前10个最重要特征索引及其重要性:")
            for i, idx in enumerate(indices):
                print(f"{i+1}. 特征 {idx}: {feature_importance[idx]:.4f}")
        
        return model, train_acc, train_sp, train_se, train_sc
    
    except Exception as e:
        print(f"模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0, 0, 0


def evaluate_model(model, X_test, y_test, args):
    """
    评估机器学习模型
    """
    print(f"评估{args.model.upper()}模型...")
    
    try:
        # 模型预测
        start_time = time.time()
        y_pred = model.predict(X_test)
        eval_time = time.time() - start_time
        
        # 计算指标
        test_acc = accuracy_score(y_test, y_pred) * 100
        
        # 计算ICBHI指标
        test_sp, test_se, test_sc = get_score(y_pred, y_test, args)
        
        # 计算F1分数
        f1 = f1_score(y_test, y_pred, average='weighted') * 100
        
        # 打印结果
        print(f"测试集准确率: {test_acc:.2f}%")
        print(f"测试集 ICBHI指标 - SP: {test_sp:.2f}%, SE: {test_se:.2f}%, Score: {test_sc:.2f}%")
        print(f"测试集 F1分数: {f1:.2f}%")
        print(f"预测用时: {eval_time:.2f}秒")
        
        # 打印分类报告
        print("\n分类报告:")
        class_names = args.cls_list if hasattr(args, 'cls_list') else None
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        return test_acc, test_sp, test_se, test_sc, f1, cm, y_pred
    
    except Exception as e:
        print(f"模型评估失败: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0, 0, 0, 0, None, None


def plot_confusion_matrix(cm, class_names, save_path, epoch=None, swan=None):
    """
    绘制混淆矩阵并保存
    """
    try:
        plt.figure(figsize=(10, 8))
        
        # 计算归一化的混淆矩阵用于显示百分比
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 创建带百分比的标注格式
        fmt = '.2f'
        thresh = cm_norm.max() / 2.
        
        # 绘制热力图
        sns.heatmap(cm_norm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'归一化混淆矩阵')
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(save_path)
        
        # 如果提供了SwanLab实例，记录到SwanLab
        if swan is not None:
            try:
                # 准备图像数据
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                
                # 使用PIL打开图像
                image = PILImage.open(buffer)
                
                # 记录到SwanLab
                try:
                    import swanlab
                    if hasattr(swanlab, 'Image'):
                        swan.log({"confusion_matrix": swanlab.Image(image)}, step=epoch)
                    else:
                        # 如果SwanLab没有Image类，尝试直接使用路径
                        swan.log({"confusion_matrix": save_path}, step=epoch)
                except Exception as e_swan:
                    print(f"SwanLab记录混淆矩阵失败: {e_swan}")
            except Exception as e_img:
                print(f"准备混淆矩阵图像失败: {e_img}")
        
        plt.close()
        print(f"混淆矩阵已保存至 {save_path}")
    
    except Exception as e:
        print(f"绘制混淆矩阵失败: {e}")
        import traceback
        traceback.print_exc()


def plot_feature_importance(model, top_n=20, save_path=None, swan=None):
    """
    绘制随机森林特征重要性
    """
    if not hasattr(model, 'feature_importances_'):
        print("模型没有feature_importances_属性，无法绘制特征重要性")
        return
    
    try:
        # 获取特征重要性
        importances = model.feature_importances_
        
        # 创建特征索引
        indices = np.argsort(importances)[::-1][:top_n]
        
        # 绘制特征重要性
        plt.figure(figsize=(12, 8))
        plt.title('特征重要性 Top {}'.format(top_n))
        plt.bar(range(top_n), importances[indices], align="center")
        plt.xticks(range(top_n), indices, rotation=90)
        plt.xlim([-1, top_n])
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path)
            print(f"特征重要性图已保存至 {save_path}")
        
        # 记录到SwanLab
        if swan is not None:
            try:
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                
                image = PILImage.open(buffer)
                
                try:
                    import swanlab
                    if hasattr(swanlab, 'Image'):
                        swan.log({"feature_importance": swanlab.Image(image)})
                    else:
                        # 使用路径
                        if save_path:
                            swan.log({"feature_importance": save_path})
                except Exception as e_swan:
                    print(f"SwanLab记录特征重要性失败: {e_swan}")
            except Exception as e_img:
                print(f"准备特征重要性图像失败: {e_img}")
        
        plt.close()
    
    except Exception as e:
        print(f"绘制特征重要性失败: {e}")
        import traceback
        traceback.print_exc()


def init_swanlab(args):
    """
    初始化SwanLab
    """
    try:
        # 尝试导入SwanLab
        import swanlab
        
        # 过滤参数，确保可序列化
        config = {}
        for k, v in vars(args).items():
            try:
                # 尝试判断是否可序列化
                if isinstance(v, (bool, int, float, str, list, dict)) or v is None:
                    config[k] = v
                else:
                    config[k] = str(v)
            except:
                # 如果有问题，转为字符串
                config[k] = str(v)
        
        # 初始化实验
        experiment = swanlab.init(
            project=f"ICBHI_{args.model}",
            experiment_name=args.exp_name,
            config=config
        )
        
        print(f"SwanLab初始化成功: {args.exp_name}")
        return experiment
    
    except ImportError:
        print("SwanLab未安装，跳过记录")
        return None
    
    except Exception as e:
        print(f"SwanLab初始化失败: {e}")
        return None


def save_results(args, model, metrics, swan=None):
    """
    保存结果
    """
    # 确保结果目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. 保存模型
    model_path = os.path.join(args.save_dir, f'{args.model}_model.pkl')
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"模型已保存至 {model_path}")
    except Exception as e:
        print(f"保存模型失败: {e}")
    
    # 2. 保存指标
    results = {
        'model_type': args.model,
        'accuracy': metrics['test_acc'],
        'f1_score': metrics['f1'],
        'specificity': metrics['sp'],
        'sensitivity': metrics['se'],
        'icbhi_score': metrics['sc'],
        'training_time': metrics['train_time'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # 将指标保存为JSON
    import json
    results_path = os.path.join(args.save_dir, f'{args.model}_results.json')
    try:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"结果已保存至 {results_path}")
    except Exception as e:
        print(f"保存结果失败: {e}")
    
    # 3. 记录结果到SwanLab
    if swan is not None:
        try:
            # 记录最终指标
            swan.log({
                "final/accuracy": float(metrics['test_acc']),
                "final/f1_score": float(metrics['f1']),
                "final/specificity": float(metrics['sp']),
                "final/sensitivity": float(metrics['se']),
                "final/icbhi_score": float(metrics['sc']),
                "final/training_time": float(metrics['train_time']),
            })
            print("最终结果已记录到SwanLab")
        except Exception as e:
            print(f"SwanLab记录结果失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='机器学习模型训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./', help='数据目录路径')
    parser.add_argument('--save_dir', type=str, default='./results', help='保存结果的目录')
    parser.add_argument('--standardize', action='store_true', help='是否标准化特征')
    parser.add_argument('--n_cls', type=int, default=4, help='类别数量')
    parser.add_argument('--cls_list', type=str, default=['normal', 'crack', 'wheeze', 'both'], nargs='+', help='类别名称列表')

    # 通用模型参数
    parser.add_argument('--model', type=str, required=True, choices=['svm', 'knn', 'randomforest'], help='模型类型')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    parser.add_argument('--exp_name', type=str, default=None, help='实验名称')
    
    # SVM参数
    parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'poly', 'rbf', 'sigmoid'], help='SVM核函数')
    parser.add_argument('--C', type=float, default=1.0, help='SVM正则化参数')
    parser.add_argument('--gamma', type=str, default='scale', help='SVM核系数')
    
    # KNN参数
    parser.add_argument('--n_neighbors', type=int, default=5, help='KNN邻居数量')
    parser.add_argument('--weights', type=str, default='uniform', choices=['uniform', 'distance'], help='KNN权重函数')
    parser.add_argument('--algorithm', type=str, default='auto', choices=['auto', 'ball_tree', 'kd_tree', 'brute'], help='KNN算法')
    
    # 随机森林参数
    parser.add_argument('--n_estimators', type=int, default=100, help='随机森林树的数量')
    parser.add_argument('--max_depth', type=int, default=None, help='随机森林树的最大深度')
    parser.add_argument('--min_samples_split', type=int, default=2, help='分裂内部节点所需的最小样本数')
    parser.add_argument('--n_jobs', type=int, default=-1, help='并行任务数')
    
    # SwanLab参数
    parser.add_argument('--use_swanlab', action='store_true', help='是否使用SwanLab记录实验')
    
    args = parser.parse_args()
    
    # 设置实验名称
    if args.exp_name is None:
        args.exp_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 设置保存目录
    args.save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 初始化SwanLab
    swan = None
    if args.use_swanlab:
        swan = init_swanlab(args)
    
    # 加载数据
    X_train, y_train, X_test, y_test = load_data(args)
    if X_train is None:
        print("数据加载失败，程序退出")
        return
    
    # 创建模型
    model = create_model(args)
    
    # 训练模型
    start_time = time.time()
    model, train_acc, train_sp, train_se, train_sc = train_model(model, X_train, y_train, args)
    train_time = time.time() - start_time
    
    if model is None:
        print("模型训练失败，程序退出")
        return
    
    # 评估模型
    test_acc, test_sp, test_se, test_sc, f1, cm, y_pred = evaluate_model(model, X_test, y_test, args)
    
    # 绘制混淆矩阵
    if cm is not None:
        cm_path = os.path.join(args.save_dir, f'{args.model}_confusion_matrix.png')
        plot_confusion_matrix(cm, args.cls_list, cm_path, swan=swan)
    
    # 如果是随机森林，绘制特征重要性
    if args.model == 'randomforest':
        importance_path = os.path.join(args.save_dir, f'{args.model}_feature_importance.png')
        plot_feature_importance(model, save_path=importance_path, swan=swan)
    
    # 保存结果
    metrics = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'sp': test_sp,
        'se': test_se,
        'sc': test_sc,
        'f1': f1,
        'train_time': train_time
    }
    save_results(args, model, metrics, swan)
    
    # 打印总结
    print("\n" + "="*50)
    print(f"训练完成! 总用时: {train_time:.2f}秒")
    print(f"模型类型: {args.model.upper()}")
    print("-"*50)
    print(f"训练集准确率: {train_acc:.2f}%")
    print(f"训练集 ICBHI指标 - SP: {train_sp:.2f}%, SE: {train_se:.2f}%, Score: {train_sc:.2f}%")
    print("-"*50)
    print(f"测试集准确率: {test_acc:.2f}%")
    print(f"测试集 ICBHI指标 - SP: {test_sp:.2f}%, SE: {test_se:.2f}%, Score: {test_sc:.2f}%")
    print(f"测试集 F1分数: {f1:.2f}%")
    print("="*50)
    
    # 结束SwanLab会话（如果有的话）
    if swan is not None:
        try:
            if hasattr(swan, 'end'):
                swan.end()
                print("SwanLab会话已结束")
        except Exception as e:
            print(f"结束SwanLab会话时出错: {e}")
    
    return model, metrics


if __name__ == "__main__":
    main()