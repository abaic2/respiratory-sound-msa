import os
import time
import sys
from copy import deepcopy
import tqdm
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from io import BytesIO
from PIL import Image as PILImage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

from util.misc import AverageMeter, accuracy, warmup_learning_rate, update_moving_average
from util.icbhi_util import get_score

# 设置matplotlib英文字体，避免中文字体问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = True


def train(train_loader, model, criterion, optimizer, epoch, args, scaler=None):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # 确保device定义
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}")):
        # 处理双特征输入
        if hasattr(args, 'use_dual_feature') and args.use_dual_feature:
            if isinstance(data, (tuple, list)):
                # 元组格式：(feature1, feature2)
                feature1, feature2 = data[0].to(device), data[1].to(device)
                feature_input = (feature1, feature2)
            elif isinstance(data, dict):
                # 字典格式：{'feature_type1': tensor1, 'feature_type2': tensor2}
                feature_input = {k: v.to(device) for k, v in data.items()}
            else:
                raise ValueError(f"不支持的双特征数据格式: {type(data)}")
        else:
            # 单特征输入
            feature_input = data.to(device)
        
        target = target.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        if scaler is not None:
            with autocast():
                output = model(feature_input)
                loss = criterion(output, target)
        else:
            output = model(feature_input)
            loss = criterion(output, target)
        
        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # 统计
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 显示进度信息
        if batch_idx % 50 == 0:
            accuracy = 100. * correct / total
            avg_loss = train_loss / (batch_idx + 1)
            tqdm.write(f'Batch {batch_idx}/{len(train_loader)}: Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%')
    
    train_accuracy = 100. * correct / total
    avg_train_loss = train_loss / len(train_loader)
    
    return avg_train_loss, train_accuracy


def validate(val_loader, model, criterion, args):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    # 确保device定义
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(val_loader, desc="Validation")):
            # 处理双特征输入
            if hasattr(args, 'use_dual_feature') and args.use_dual_feature:
                if isinstance(data, (tuple, list)):
                    # 元组格式：(feature1, feature2)
                    feature1, feature2 = data[0].to(device), data[1].to(device)
                    feature_input = (feature1, feature2)
                elif isinstance(data, dict):
                    # 字典格式：{'feature_type1': tensor1, 'feature_type2': tensor2}
                    feature_input = {k: v.to(device) for k, v in data.items()}
                else:
                    raise ValueError(f"不支持的双特征数据格式: {type(data)}")
            else:
                # 单特征输入
                feature_input = data.to(device)
            
            target = target.to(device)
            
            # 前向传播
            output = model(feature_input)
            loss = criterion(output, target)
            
            # 统计
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 收集预测结果用于详细评估
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    val_accuracy = 100. * correct / total
    avg_val_loss = val_loss / len(val_loader)
    
    return avg_val_loss, val_accuracy, all_predictions, all_targets


def plot_confusion_matrix(labels, preds, class_names, save_path, swan=None, epoch=None):
    """绘制混淆矩阵并记录到SwanLab（通过步数更新同一张图）"""
    try:
        # 计算混淆矩阵
        cm = confusion_matrix(labels, preds)
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 使用原始数量而非归一化值
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - Epoch {epoch} (Raw Counts)')
        plt.tight_layout()
        
        # 保存到文件系统
        plt.savefig(save_path)
        
        # 如果提供了swan对象，记录到SwanLab - 使用单一标识符和步数
        if swan is not None and epoch is not None:
            try:
                # 使用BytesIO缓冲区
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                
                # 转换为PIL图像
                pil_image = PILImage.open(buf)
                
                # 使用统一的图表名称，但通过step参数实现不同epoch的跟踪
                try:
                    import swanlab
                    if hasattr(swanlab, 'Image'):
                        swan.log({"confusion_matrix": swanlab.Image(pil_image)}, step=epoch)
                        print(f"已成功更新混淆矩阵至步数 {epoch} (swanlab.Image)")
                except Exception as e:
                    print(f"尝试swanlab.Image方法失败: {e}")
                    
                    try:
                        # 第二种方法：使用文件路径和步数
                        swan.log({"confusion_matrix": save_path}, step=epoch)
                        print(f"已成功更新混淆矩阵至步数 {epoch} (文件路径方法)")
                    except Exception as e2:
                        print(f"尝试文件路径方法失败: {e2}")
                        
                        # 第三种方法：如果有log_image方法
                        if hasattr(swan, 'log_image'):
                            swan.log_image("confusion_matrix", save_path, step=epoch)
                            print(f"已成功更新混淆矩阵至步数 {epoch} (log_image方法)")
            except Exception as e:
                print(f"SwanLab记录混淆矩阵失败: {e}")
        
        plt.close()
        return save_path
    except Exception as e:
        print(f"绘制混淆矩阵失败: {e}")
        return save_path


def plot_training_history(epochs, train_loss, train_acc, val_loss, val_acc, save_dir, swan=None):
    """绘制训练历史曲线并记录到SwanLab（使用单一标识符和最终步数）"""
    # 确保所有输入都是Python原生类型
    train_loss = [float(x) for x in train_loss]
    train_acc = [float(x) for x in train_acc]
    val_loss = [float(x) for x in val_loss]
    val_acc = [float(x) for x in val_acc]
    
    # 获取当前最后步数
    current_step = len(epochs)
    
    try:
        # 1. 创建损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, 'b-', label='Train Loss', marker='o')
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss', marker='o')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 保存到文件系统
        loss_path = os.path.join(save_dir, 'loss_history.png')
        plt.savefig(loss_path)
        
        # 如果提供了swan对象，尝试多种方法记录到SwanLab
        if swan is not None:
            try:
                import swanlab
                if hasattr(swanlab, 'Image'):
                    # 使用BytesIO缓冲区
                    buf = BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    pil_image = PILImage.open(buf)
                    
                    # 使用统一名称和当前步数
                    swan.log({"training/loss_curve": swanlab.Image(pil_image)}, step=current_step)
                    print(f"已成功更新损失历史曲线至步数 {current_step}")
            except Exception as e:
                print(f"记录损失曲线失败: {e}")
        
        plt.close()
        
        # 2. 创建准确率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_acc, 'b-', label='Train Accuracy', marker='o')
        plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy', marker='o')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 保存到文件系统
        acc_path = os.path.join(save_dir, 'accuracy_history.png')
        plt.savefig(acc_path)
        
        # 如果提供了swan对象，记录到SwanLab
        if swan is not None:
            try:
                import swanlab
                if hasattr(swanlab, 'Image'):
                    # 使用BytesIO缓冲区
                    buf = BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    pil_image = PILImage.open(buf)
                    
                    # 使用统一名称和当前步数
                    swan.log({"training/accuracy_curve": swanlab.Image(pil_image)}, step=current_step)
                    print(f"已成功更新准确率历史曲线至步数 {current_step}")
            except Exception as e:
                print(f"记录准确率曲线失败: {e}")
        
        plt.close()
        
        return loss_path, acc_path
    except Exception as e:
        print(f"绘制训练历史曲线失败: {e}")
        return None, None


def run_epoch(train_loader, val_loader, model, criterion, optimizer, scheduler, epoch, args, scaler=None):
    """
    运行一个epoch的训练和验证
    修复参数数量问题
    """
    print(f"\n{'='*50}")
    print(f"Epoch {epoch}/{args.epochs}")
    print(f"{'='*50}")
    
    # 训练
    train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args, scaler)
    
    # 验证
    val_loss, val_acc, val_predictions, val_targets = validate(val_loader, model, criterion, args)
    
    # 学习率调度
    if scheduler is not None:
        if hasattr(scheduler, 'step'):
            if 'ReduceLROnPlateau' in str(type(scheduler)):
                scheduler.step(val_loss)
            else:
                scheduler.step()
    
    # 计算详细指标
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(val_targets, val_predictions, 
                                     target_names=[f'Class_{i}' for i in range(args.n_cls)],
                                     output_dict=True, zero_division=0)
    except ImportError:
        print("Warning: sklearn not available, using basic metrics")
        report = {'weighted avg': {'f1-score': val_acc / 100}}
    
    # 打印结果
    print(f"\n训练结果:")
    print(f"  训练损失: {train_loss:.4f}")
    print(f"  训练精度: {train_acc:.2f}%")
    print(f"\n验证结果:")
    print(f"  验证损失: {val_loss:.4f}")
    print(f"  验证精度: {val_acc:.2f}%")
    
    # 打印各类别精度
    if 'Class_0' in report:
        print(f"\n各类别性能:")
        for i in range(args.n_cls):
            class_name = f'Class_{i}'
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                print(f"  {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    # 计算加权F1分数作为主要评估指标
    weighted_f1 = report['weighted avg']['f1-score']
    
    # 确定是否为最佳模型
    best_score = weighted_f1
    save_bool = True  # 简化，每个epoch都保存
    
    metrics = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'weighted_f1': weighted_f1,
        'detailed_report': report
    }
    
    return best_score, save_bool, metrics


def init_swanlab(args):
    """初始化SwanLab，确保config可序列化"""
    try:
        # 加载SwanLab库
        import swanlab
        
        # 打印SwanLab版本
        print(f"SwanLab版本: {swanlab.__version__}")
        
        # 过滤不可序列化的配置参数
        config = {}
        for key, value in vars(args).items():
            try:
                # 尝试判断是否可序列化
                if isinstance(value, (bool, int, float, str, list, dict)) or value is None:
                    config[key] = value
                else:
                    config[key] = str(value)
            except:
                # 如果有问题，转为字符串
                config[key] = str(value)
        
        # 创建实验名称
        experiment_name = getattr(args, 'tag', f"{args.model}_{args.feature_combo if args.use_dual_feature else 'single'}")
        
        # 初始化SwanLab
        experiment = swanlab.init(
            project=f"ICBHI_{args.model}",
            experiment_name=experiment_name,
            config=config
        )
        print(f"SwanLab实验初始化成功: {experiment_name}")
        
        # 测试记录
        experiment.log({"test_metric": 1.0})
        print("SwanLab基础记录测试成功")
        
        return experiment
    except ImportError as e:
        print(f"SwanLab未安装或导入失败: {e}")
        print("请使用 'pip install swanlab' 安装SwanLab")
        return None
    except Exception as e:
        print(f"SwanLab初始化失败: {str(e)}")
        return None


def log_final_results_to_swanlab(args, history, best_score, swan=None):
    """记录最终结果到SwanLab，使用BytesIO和PIL方式"""
    if swan is None:
        return
    
    try:
        # 准备数据
        epochs = list(range(1, len(history["train_loss"]) + 1))
        
        # 创建保存目录
        save_folder = getattr(args, 'save_folder', './save')
        os.makedirs(save_folder, exist_ok=True)
        
        # 绘制训练历史曲线
        plot_training_history(
            epochs, 
            history["train_loss"], 
            history["train_acc"], 
            history["val_loss"], 
            history["val_acc"],
            save_folder,
            swan
        )
        
        # 记录最终结果指标
        swan.log({
            "final/best_ICBHI_Score": float(best_score),
            "final/total_epochs": len(history["train_loss"])
        })
        
        print("已成功记录最终结果到SwanLab")
    except Exception as e:
        print(f"SwanLab记录最终结果失败: {e}")