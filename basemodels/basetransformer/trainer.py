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

from util.misc import AverageMeter, accuracy, warmup_learning_rate, update_moving_average
from util.icbhi_util import get_score

# 设置matplotlib英文字体，避免中文字体问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = True


def train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler=None):
    """一个 epoch 的训练"""
    model.train()
    classifier.train()
    projector.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    end = time.time()
    
    # 使用tqdm进度条替代逐批次输出
    pbar = tqdm.tqdm(total=len(train_loader), desc=f"训练 Epoch {epoch}", leave=False)
    
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        images = images.cuda(non_blocking=True)
        labels = labels.long().cuda(non_blocking=True)
        if labels.dim() > 1:
            labels = labels.squeeze()
        bsz = labels.shape[0]
        
        # 确保图像格式正确
        if images.dim() == 3:
            images = images.unsqueeze(1)
        
        # 使用混合精度
        if scaler is not None:
            with torch.cuda.amp.autocast():
                if args.method == 'ce':
                    features = model(images)
                    output = classifier(features)
                    
                    # 只在第一个batch输出调试信息
                    if epoch == 1 and idx == 0:
                        print(f"输出形状: {output.shape}, 标签形状: {labels.shape}")
                    
                    # 检查并修复输出维度
                    if output.dim() > 2:
                        if output.dim() == 3:
                            output = output[:, 0] if output.size(1) > 0 else output.mean(dim=1)
                        else:
                            output = output.reshape(output.size(0), -1, output.size(-1))
                            output = output.mean(dim=1)
                    
                    loss = criterion[0](output, labels)
                elif args.method == 'patchmix':
                    features, y_a, y_b, lam, index = model(images, labels, patch_mix=True, time_domain=args.time_domain)
                    output = classifier(features)
                    
                    # 检查并修复输出维度
                    if output.dim() > 2:
                        if output.dim() == 3:
                            output = output[:, 0] if output.size(1) > 0 else output.mean(dim=1)
                        else:
                            output = output.reshape(output.size(0), -1, output.size(-1))
                            output = output.mean(dim=1)
                            
                    loss = criterion[1](output, y_a, y_b, lam)
                elif args.method == 'patchmix_cl':
                    features, y_a, y_b, lam, index = model(images, labels, patch_mix=True, time_domain=args.time_domain)
                    output = classifier(features)
                    
                    # 检查并修复输出维度
                    if output.dim() > 2:
                        if output.dim() == 3:
                            output = output[:, 0] if output.size(1) > 0 else output.mean(dim=1)
                        else:
                            output = output.reshape(output.size(0), -1, output.size(-1))
                            output = output.mean(dim=1)
                            
                    features = projector(features)
                    loss = criterion[0](output, labels) + args.alpha * criterion[1](features, y_a, y_b, lam, args.negative_pair)
                else:
                    raise NotImplementedError('unknown method: %s' % args.method)
        else:
            if args.method == 'ce':
                features = model(images)
                output = classifier(features)
                
                # 检查并修复输出维度
                if output.dim() > 2:
                    if output.dim() == 3:
                        output = output[:, 0] if output.size(1) > 0 else output.mean(dim=1)
                    else:
                        output = output.reshape(output.size(0), -1, output.size(-1))
                        output = output.mean(dim=1)
                
                loss = criterion[0](output, labels)
            elif args.method == 'patchmix':
                features, y_a, y_b, lam, index = model(images, labels, patch_mix=True, time_domain=args.time_domain)
                output = classifier(features)
                
                # 检查并修复输出维度
                if output.dim() > 2:
                    if output.dim() == 3:
                        output = output[:, 0] if output.size(1) > 0 else output.mean(dim=1)
                    else:
                        output = output.reshape(output.size(0), -1, output.size(-1))
                        output = output.mean(dim=1)
                
                loss = criterion[1](output, y_a, y_b, lam)
            elif args.method == 'patchmix_cl':
                features, y_a, y_b, lam, index = model(images, labels, patch_mix=True, time_domain=args.time_domain)
                output = classifier(features)
                
                # 检查并修复输出维度
                if output.dim() > 2:
                    if output.dim() == 3:
                        output = output[:, 0] if output.size(1) > 0 else output.mean(dim=1)
                    else:
                        output = output.reshape(output.size(0), -1, output.size(-1))
                        output = output.mean(dim=1)
                
                features = projector(features)
                loss = criterion[0](output, labels) + args.alpha * criterion[1](features, y_a, y_b, lam, args.negative_pair)
            else:
                raise NotImplementedError('unknown method: %s' % args.method)
        
        # 更新准确率
        acc1, valid_bsz = accuracy(output, labels, topk=(1,))
        losses.update(loss.item(), bsz)
        top1.update(acc1[0].item(), valid_bsz)
        
        # 更新模型参数
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # 记录时间
        batch_time.update(time.time() - end)
        end = time.time()
        
        # 更新进度条显示，而不是打印详细信息
        pbar.update(1)
        pbar.set_postfix({
            'loss': f"{losses.avg:.4f}",
            'acc': f"{top1.avg:.2f}%"
        })
    
    # 关闭进度条
    pbar.close()
    
    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, args, best_acc, best_model=None):
    """验证函数"""
    save_bool = False
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    hits, counts = [0.0] * args.n_cls, [0.0] * args.n_cls
    
    # 用于混淆矩阵的标签和预测结果
    all_labels = []
    all_preds = []

    with torch.no_grad():
        end = time.time()
        
        # 使用tqdm进度条替代逐批次输出
        pbar = tqdm.tqdm(total=len(val_loader), desc="Validation")
        
        for idx, (images, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.long().cuda(non_blocking=True)
            if labels.dim() > 1:
                labels = labels.squeeze()
            bsz = labels.shape[0]

            with torch.cuda.amp.autocast():
                features = model(images)
                output = classifier(features)
                
                # 检查并修复输出维度
                if output.dim() > 2:
                    if output.dim() == 3:
                        output = output[:, 0] if output.size(1) > 0 else output.mean(dim=1)
                    else:
                        output = output.reshape(output.size(0), -1, output.size(-1))
                        output = output.mean(dim=1)
                
                loss = criterion[0](output, labels)

            losses.update(loss.item(), bsz)
            [acc1], _ = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            _, preds = torch.max(output, 1)
            
            # 收集标签和预测结果用于混淆矩阵
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            for i in range(preds.shape[0]):
                counts[labels[i].item()] += 1.0
                if not args.two_cls_eval:
                    if preds[i].item() == labels[i].item():
                        hits[labels[i].item()] += 1.0
                else:  # only when args.n_cls == 4
                    if labels[i].item() == 0 and preds[i].item() == labels[i].item():
                        hits[labels[i].item()] += 1.0
                    elif labels[i].item() != 0 and preds[i].item() > 0:  # abnormal
                        hits[labels[i].item()] += 1.0

            batch_time.update(time.time() - end)
            end = time.time()
            
            # 更新进度条
            pbar.update(1)
        
        pbar.close()
    
    # 计算最终指标
    sp, se, sc = get_score(hits, counts)
    
    # 判断是否是最佳模型
    if sc > best_acc[-1] and se > 5:
        save_bool = True
        best_acc = [sp, se, sc]
        best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]

    return best_acc, best_model, save_bool, {
        "val_loss": losses.avg, 
        "val_acc": top1.avg, 
        "sp": sp, 
        "se": se, 
        "sc": sc, 
        "confusion_data": {
            "labels": all_labels,
            "preds": all_preds,
            "class_names": args.cls_list if hasattr(args, 'cls_list') else [str(i) for i in range(args.n_cls)]
        }
    }


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
                
                # 使用相同的标识符，但不同的步数
                try:
                    import swanlab
                    if hasattr(swanlab, 'Image'):
                        # 使用统一的图表名称，但通过step参数实现不同epoch的跟踪
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
                import traceback
                traceback.print_exc()
        
        plt.close()
        return save_path
    except Exception as e:
        print(f"绘制混淆矩阵失败: {e}")
        import traceback
        traceback.print_exc()
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
                print(f"尝试swanlab.Image方法失败: {e}")
                try:
                    # 尝试使用文件路径
                    swan.log({"training/loss_curve": loss_path}, step=current_step)
                    print(f"已成功更新损失历史曲线至步数 {current_step} (文件路径方法)")
                except Exception as e2:
                    print(f"记录损失曲线失败: {e2}")
        
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
                    swan.log({"training/accuracy_curve": swanlab.Image(pil_image)}, step=current_step)
                    print(f"已成功更新准确率历史曲线至步数 {current_step}")
            except Exception as e:
                print(f"尝试swanlab.Image方法失败: {e}")
                try:
                    # 尝试使用文件路径
                    swan.log({"training/accuracy_curve": acc_path}, step=current_step)
                    print(f"已成功更新准确率历史曲线至步数 {current_step} (文件路径方法)")
                except Exception as e2:
                    print(f"记录准确率曲线失败: {e2}")
        
        plt.close()
        
        return loss_path, acc_path
    except Exception as e:
        print(f"绘制训练历史曲线失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def run_epoch(train_loader, val_loader, model, classifier, projector, criterion, optimizer, 
              epoch, args, scaler, best_acc, best_model, swan=None):
    """运行一个完整的epoch，包括训练和验证，并输出格式化结果"""
    epoch_start_time = time.time()
    
    # 训练阶段
    print(f"\n[Epoch {epoch}/{args.epochs}]")
    train_loss, train_acc = train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler)
    
    # 验证阶段
    best_acc, best_model, save_bool, metrics = validate(val_loader, model, classifier, criterion, args, best_acc, best_model)
    
    # 计算剩余时间
    epoch_time = time.time() - epoch_start_time
    remaining_epochs = args.epochs - epoch
    remaining_time = epoch_time * remaining_epochs
    remaining_time_str = str(timedelta(seconds=int(remaining_time)))
    epoch_time_str = str(timedelta(seconds=int(epoch_time)))
    
    # 绘制混淆矩阵
    cm_save_path = os.path.join(args.save_folder, f'confusion_matrix_epoch_{epoch}.png')
    confusion_data = metrics["confusion_data"]
    plot_confusion_matrix(
        confusion_data["labels"], 
        confusion_data["preds"], 
        confusion_data["class_names"],
        cm_save_path,
        swan,
        epoch
    )
    
    # 记录到SwanLab - 只记录数值指标
    if swan is not None:
        try:
            # 确保所有记录的值都是标量
            swan.log({
                "epoch": int(epoch),
                "train/loss": float(train_loss),
                "train/acc": float(train_acc),
                "val/loss": float(metrics["val_loss"]),
                "val/acc": float(metrics["val_acc"]),
                "val/SE": float(metrics["se"]),
                "val/SP": float(metrics["sp"]),
                "val/ICBHI_Score": float(metrics["sc"]),
                "best/ICBHI_Score": float(best_acc[2]),
                "best/SE": float(best_acc[1]),
                "best/SP": float(best_acc[0]),
                "epoch_time": float(epoch_time),
                "learning_rate": float(optimizer.param_groups[0]["lr"])
            })
            print("已成功记录训练指标到SwanLab")
        except Exception as e:
            print(f"SwanLab记录训练指标失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 格式化输出
    print(f"\nEpoch {epoch}/{args.epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {metrics['val_loss']:.4f} | Val Acc: {metrics['val_acc']:.2f}%")
    print(f"SE: {metrics['se']:.4f} | SP: {metrics['sp']:.4f} | ICBHI: {metrics['sc']:.4f}")
    print(f"混淆矩阵已保存至: {cm_save_path}")
    print(f"Epoch Time: {epoch_time_str} | Remaining: {remaining_time_str}")
    print("------------------------------------------------------------")
    
    # 保存最佳模型提示
    if save_bool:
        save_path = os.path.join(args.save_folder, 'best.pth')
        print(f"✅ 保存最佳模型至: {save_path}")
        print(f"最佳 ICBHI: {best_acc[2]:.4f}, SE: {best_acc[1]:.4f}, SP: {best_acc[0]:.4f}")
    else:
        print(f"最佳 ICBHI: {best_acc[2]:.4f}, SE: {best_acc[1]:.4f}, SP: {best_acc[0]:.4f}")
    
    return best_acc, best_model, save_bool, {"train_loss": train_loss, "train_acc": train_acc, 
                                           "val_loss": metrics["val_loss"], "val_acc": metrics["val_acc"]}


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
        
        # 初始化SwanLab
        experiment = swanlab.init(
            project=f"ICBHI_{args.model}",
            experiment_name=args.model_name,
            config=config
        )
        print(f"SwanLab实验初始化成功: {args.model_name}")
        
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
        import traceback
        traceback.print_exc()
        return None


def log_final_results_to_swanlab(args, history, best_acc, swan=None):
    """记录最终结果到SwanLab，使用BytesIO和PIL方式"""
    if swan is None:
        return
    
    try:
        # 准备数据
        epochs = list(range(1, len(history["train_loss"]) + 1))
        
        # 绘制训练历史曲线
        plot_training_history(
            epochs, 
            history["train_loss"], 
            history["train_acc"], 
            history["val_loss"], 
            history["val_acc"],
            args.save_folder,
            swan
        )
        
        # 记录最终结果指标
        swan.log({
            "final/best_ICBHI_Score": float(best_acc[2]),
            "final/best_SE": float(best_acc[1]),
            "final/best_SP": float(best_acc[0]),
            "final/total_epochs": len(history["train_loss"])
        })
        
        print("已成功记录最终结果到SwanLab")
    except Exception as e:
        print(f"SwanLab记录最终结果失败: {e}")
        import traceback
        traceback.print_exc()