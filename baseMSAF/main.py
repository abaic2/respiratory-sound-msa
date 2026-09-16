import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# 设置无缓冲输出以便实时显示日志
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
print("程序启动中...")

import math
import time
import random
import pickle
import argparse
import numpy as np
from copy import deepcopy
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms

from util.icbhi_dataset import ICBHIDataset, ICBHIDualFeatureDataset
from util.icbhi_util import get_score
from util.augmentation import SpecAugment
from models.ast import ASTModel, AST_Dual_Feature_MSAF, AST_Single_Feature_MSA
from trainer import train, validate, run_epoch, init_swanlab, log_final_results_to_swanlab


def parse_option():
    print("解析命令行参数...")
    parser = argparse.ArgumentParser('argument for supervised training', add_help=False)
    
    # 基础参数
    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--save_dir', type=str, default='./save', help='path to save linear classifier')
    parser.add_argument('--tag', type=str, help='tag for experiment')
    parser.add_argument('--resume', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--eval', action='store_true', help='eval only')
    parser.add_argument('--two_cls_eval', action='store_true', help='eval two cls only')

    # optimization
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'], help='optimizer')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0, help='warmup epochs')
    parser.add_argument('--weighted_loss', action='store_true', help='weighted loss')
    parser.add_argument('--mix_beta', type=float, default=1.0, help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--time_domain', action='store_true', help='use time domain raw data')

    # dataset
    parser.add_argument('--dataset', type=str, default='icbhi', choices=['icbhi'], help='dataset')
    parser.add_argument('--data_folder', type=str, default='/home/u202420085410012/55555/bishe/data/ICBHI_final_database', help='path to custom dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--class_split', type=str, default='lungsound', choices=['lungsound', 'diagnosis'], help='class split')
    parser.add_argument('--n_cls', type=int, default=4, help='number of classes')
    parser.add_argument('--test_fold', type=str, default='official', choices=['official', '0', '1', '2', '3', '4'], help='test fold to use')
    parser.add_argument('--weighted_sampler', action='store_true', help='use weighted sampler')
    parser.add_argument('--stetho_id', type=int, default=-1, help='stetho_id used')

    # audio
    parser.add_argument('--sample_rate', type=int, default=16000, help='sample rate')
    parser.add_argument('--butterworth_filter', type=str, default='', help='apply butterworth band pass filter')
    parser.add_argument('--desired_length', type=int, default=8, help='desired length of the audio')
    parser.add_argument('--nfft', type=int, default=1024, help='number of fft')
    parser.add_argument('--n_mels', type=int, default=128, help='number of mel filter banks')
    parser.add_argument('--concat_aug_scale', type=float, default=1.0, help='scale for concat aug')
    parser.add_argument('--pad_types', type=str, default='repeat', help='pad types')
    parser.add_argument('--resz', type=float, default=1, help='resize the input')
    parser.add_argument('--raw_augment', type=int, default=0, help='number of raw augmentations')

    # spec augmentation
    parser.add_argument('--specaug_policy', type=str, default='icbhi_ast_sup', help='policy of SpecAugment')
    parser.add_argument('--specaug_mask', type=str, default='mean', choices=['mean', 'zero'], help='specaug mask mode')

    # model
    parser.add_argument('--model', type=str, default='ast')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
    parser.add_argument('--pretrained_ckpt', type=str, default='', help='pretrained checkpoint')
    parser.add_argument('--from_sl_official', action='store_true', help='load from self-supervised learning official checkpoint')
    parser.add_argument('--ma_update', action='store_true', help='using moving average update')
    parser.add_argument('--ma_beta', type=float, default=0.5, help='moving average decay')
    parser.add_argument('--audioset_pretrained', action='store_true', help='use audioset pretrained model')

    # ssast
    parser.add_argument('--ssast_task', type=str, default='ft_avgtok', choices=['ft_avgtok', 'ft_cls'], help='ssast finetune task')
    parser.add_argument('--fshape', type=int, default=16, help='fshape')
    parser.add_argument('--tshape', type=int, default=16, help='tshape')
    parser.add_argument('--ssast_pretrained_type', type=str, default='patch', help='ssast pretrained type')

    # loss
    parser.add_argument('--method', type=str, default='ce', choices=['ce', 'supcon'], help='method')
    parser.add_argument('--proj_dim', type=int, default=128, help='project dimension')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature for supcon')
    parser.add_argument('--alpha', type=float, default=1.0, help='weight for supcon loss')
    parser.add_argument('--negative_pair', type=str, default='all', choices=['all', 'diff_label'], help='negative pair for supcon')
    parser.add_argument('--target_type', type=str, default='grad_block', choices=['grad_block', 'grad_flow', 'project_block', 'project_flow'], help='target type for supcon')

    # =============== 新增：双特征MSAF参数 ===============
    parser.add_argument('--use_dual_feature', action='store_true', help='使用双特征MSAF模型')
    parser.add_argument('--dual_feature', action='store_true', help='使用双特征MSAF模型（别名）')
    parser.add_argument('--feature_combo', type=str, default='mel+cqt', 
                       help='特征组合，支持: mel+cqt, mel+mfcc, mel+chroma, cqt+mfcc, etc.')
    parser.add_argument('--feature_types', nargs=2, default=['mel', 'cqt'],
                       help='特征类型列表，如: --feature_types mel cqt')
    parser.add_argument('--model_size', type=str, default='base384',
                       choices=['tiny224', 'small224', 'base224', 'base384'],
                       help='AST模型大小')
    parser.add_argument('--spatial_size', type=int, default=8,
                       help='MSAF空间维度大小')
    parser.add_argument('--shared_weights', action='store_true',
                       help='AST分支是否共享权重')

    opt = parser.parse_args()

    # 处理feature_combo和feature_types的兼容性
    if hasattr(opt, 'feature_combo') and '+' in opt.feature_combo:
        opt.feature_types = opt.feature_combo.split('+')
    
    # 兼容性处理
    if opt.dual_feature:
        opt.use_dual_feature = True

    # 处理学习率衰减epochs
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # 设置warmup参数
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.n_gpu = torch.cuda.device_count()

    # 添加类别列表
    if not hasattr(opt, 'cls_list'):
        opt.cls_list = ['normal', 'crackle', 'wheezing', 'both']

    # 设置保存文件夹
    opt.save_folder = opt.save_dir
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


def set_loader(opt):
    """创建数据加载器"""
    print("创建数据加载器...")
    
    # 数据增强
    if opt.dataset == 'icbhi':
        if opt.specaug_policy != '':
            print(f"使用SpecAugment: {opt.specaug_policy}")
            if opt.specaug_policy.startswith('icbhi'):
                transforms_list = [transforms.ToTensor(), SpecAugment(opt)]
            else:
                raise NotImplementedError(f"SpecAugment策略 {opt.specaug_policy} 未实现")
        else:
            transforms_list = [transforms.ToTensor()]
    else:
        raise NotImplementedError(f"数据集 {opt.dataset} 未实现")
    
    train_transform = transforms.Compose(transforms_list)
    test_transform = transforms.Compose([transforms.ToTensor()])
    
    # 根据是否使用双特征选择数据集
    if opt.use_dual_feature:
        print(f"使用双特征数据集: {opt.feature_combo}")
        
        train_dataset = ICBHIDualFeatureDataset(
            train_flag=True,
            transform=train_transform,
            args=opt,
            feature_combo=opt.feature_combo,
            print_flag=True
        )
        
        test_dataset = ICBHIDualFeatureDataset(
            train_flag=False,
            transform=test_transform,
            args=opt,
            feature_combo=opt.feature_combo,
            print_flag=True
        )
    else:
        print("使用单特征数据集")
        
        train_dataset = ICBHIDataset(
            train_flag=True,
            transform=train_transform,
            args=opt,
            print_flag=True
        )
        
        test_dataset = ICBHIDataset(
            train_flag=False,
            transform=test_transform,
            args=opt,
            print_flag=True
        )

    # 数据加载器
    train_sampler = None
    if opt.weighted_sampler:
        print("使用加权采样器")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=int(opt.num_workers/2), pin_memory=True)

    return train_loader, test_loader


def set_model(opt):
    """创建模型"""
    print("创建模型...")
    
    if opt.use_dual_feature:
        print(f"创建双特征AST-MSAF模型: {opt.feature_combo}")
        
        model = AST_Dual_Feature_MSAF(
            label_dim=opt.n_cls,
            model_size=opt.model_size,
            spatial_size=opt.spatial_size,
            feature_combo=opt.feature_combo,
            shared_weights=opt.shared_weights
        )
    else:
        print("创建单特征AST模型")
        if opt.model == 'ast':
            model = ASTModel(
                label_dim=opt.n_cls,
                model_size=getattr(opt, 'model_size', 'base384'),
                imagenet_pretrain=True,
                audioset_pretrain=opt.audioset_pretrained,
                verbose=True
            )
        else:
            raise NotImplementedError(f"模型 {opt.model} 未实现")

    # 移动到GPU
    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU")
            model = torch.nn.DataParallel(model)

    return model


def main():
    print("程序开始执行...")
    opt = parse_option()
    
    # 设置随机种子
    print(f"设置随机种子: {opt.seed}")
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
    
    # 打印配置信息
    print("=" * 50)
    print(f"实验标签: {opt.tag}")
    print(f"数据集: {opt.dataset}")
    print(f"使用双特征: {opt.use_dual_feature}")
    if opt.use_dual_feature:
        print(f"特征组合: {opt.feature_combo}")
        print(f"共享权重: {opt.shared_weights}")
        print(f"模型大小: {opt.model_size}")
        print(f"空间大小: {opt.spatial_size}")
    print(f"训练epochs: {opt.epochs}")
    print(f"批量大小: {opt.batch_size}")
    print(f"学习率: {opt.learning_rate}")
    print("=" * 50)
    
    # 初始化SwanLab
    swan = init_swanlab(opt)
    
    # 创建数据加载器
    train_loader, val_loader = set_loader(opt)
    
    # 确保device定义
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    if opt.use_dual_feature:
        print("🔧 创建双特征AST模型...")
        
        # 导入双特征模型
        from models.ast import AST_Dual_Feature_MSAF
        
        model = AST_Dual_Feature_MSAF(
            label_dim=opt.n_cls,
            model_size=opt.model_size,
            spatial_size=opt.spatial_size,
            feature_combo=opt.feature_combo,
            shared_weights=False
        )
        
        print(f"✅ 双特征模型创建成功: {opt.feature_combo}")
        
    else:
        print("🔧 创建单特征AST模型...")
        from models.ast import ASTModel
        
        model = ASTModel(
            label_dim=opt.n_cls, 
            fstride=opt.fshape, 
            tstride=opt.tshape, 
            input_fdim=opt.n_mels,
            input_tdim=opt.desired_length * opt.sample_rate // 1000, 
            imagenet_pretrain=opt.pretrained, 
            audioset_pretrain=opt.audioset_pretrained, 
            model_size=opt.model_size, 
            verbose=True
        )
        
        print(f"✅ 单特征模型创建成功")
    
    # 将模型移动到设备
    model = model.to(device)
    
    # 如果是双特征模型且需要加载AudioSet预训练权重
    if opt.use_dual_feature and opt.audioset_pretrained:
        print("🔧 为双特征模型加载AudioSet预训练权重...")
        
        # 修复预训练权重路径
        checkpoint_path = '/home/u202420085410012/55555/bishe/pretrained_models/audioset_10_10_0.4593.pth'
        
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # 获取预训练权重
                if 'model' in checkpoint:
                    pretrained_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    pretrained_dict = checkpoint['state_dict']
                else:
                    pretrained_dict = checkpoint
                
                # 获取当前模型的状态字典
                model_dict = model.state_dict()
                
                # 为每个AST分支加载权重
                loaded_keys = 0
                
                # 加载第一个分支
                for k, v in pretrained_dict.items():
                    # 移除 'module.' 前缀（如果存在）
                    clean_key = k[7:] if k.startswith('module.') else k
                    
                    # 添加第一个分支的前缀
                    branch1_key = f'ast_branch1.{clean_key}'
                    branch2_key = f'ast_branch2.{clean_key}'
                    
                    if branch1_key in model_dict and v.shape == model_dict[branch1_key].shape:
                        model_dict[branch1_key] = v.clone()
                        loaded_keys += 1
                    
                    if branch2_key in model_dict and v.shape == model_dict[branch2_key].shape:
                        model_dict[branch2_key] = v.clone()
                        loaded_keys += 1
                
                # 加载更新后的权重
                model.load_state_dict(model_dict, strict=False)
                print(f"✅ 成功加载AudioSet预训练权重: {loaded_keys} 个参数")
                print(f"📁 权重文件路径: {checkpoint_path}")
                
            except Exception as e:
                print(f"❌ 加载预训练权重失败: {str(e)}")
                print("继续使用ImageNet预训练权重...")
        else:
            print(f"❌ 预训练权重文件不存在: {checkpoint_path}")
            print("🔍 正在检查其他可能的路径...")
            
            # 检查其他可能的路径
            alternative_paths = [
                '/home/u202420085410012/55555/bishe/pretrained_models/audioset_10_10_0.4593.pth',
                '/home/u202420085410012/55555/bishe/MVST-main/pretrained_models/audioset_10_10_0.4593.pth',
                '/home/u202420085410012/55555/bishe/MVST-main/3 copy 10/pretrained_models/audioset_10_10_0.4593.pth',
                './pretrained_models/audioset_10_10_0.4593.pth'
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    print(f"✅ 找到预训练权重: {alt_path}")
                    try:
                        checkpoint = torch.load(alt_path, map_location=device)
                        
                        # 获取预训练权重
                        if 'model' in checkpoint:
                            pretrained_dict = checkpoint['model']
                        elif 'state_dict' in checkpoint:
                            pretrained_dict = checkpoint['state_dict']
                        else:
                            pretrained_dict = checkpoint
                        
                        # 获取当前模型的状态字典
                        model_dict = model.state_dict()
                        
                        # 为每个AST分支加载权重
                        loaded_keys = 0
                        
                        # 加载权重
                        for k, v in pretrained_dict.items():
                            # 移除 'module.' 前缀（如果存在）
                            clean_key = k[7:] if k.startswith('module.') else k
                            
                            # 添加分支前缀
                            branch1_key = f'ast_branch1.{clean_key}'
                            branch2_key = f'ast_branch2.{clean_key}'
                            
                            if branch1_key in model_dict and v.shape == model_dict[branch1_key].shape:
                                model_dict[branch1_key] = v.clone()
                                loaded_keys += 1
                            
                            if branch2_key in model_dict and v.shape == model_dict[branch2_key].shape:
                                model_dict[branch2_key] = v.clone()
                                loaded_keys += 1
                        
                        # 加载更新后的权重
                        model.load_state_dict(model_dict, strict=False)
                        print(f"✅ 成功从备选路径加载AudioSet预训练权重: {loaded_keys} 个参数")
                        break
                        
                    except Exception as e:
                        print(f"❌ 从备选路径 {alt_path} 加载失败: {str(e)}")
                        continue
            else:
                print("❌ 所有预训练权重路径都不存在，继续使用ImageNet预训练权重...")
                opt.audioset_pretrained = False
    
    # 如果使用多GPU
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = torch.nn.DataParallel(model)
    
    # 创建损失函数
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    
    # 创建优化器
    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    
    # 创建学习率调度器
    if opt.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_decay_epochs, gamma=opt.lr_decay_rate)
    
    # 创建scaler用于混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    # 确保args有target_length属性
    if not hasattr(opt, 'target_length'):
        opt.target_length = 1024  # 设置默认目标长度
    
    print(f"🔧 目标特征长度: {opt.target_length}")
    
    # 训练循环
    best_score = 0
    
    for epoch in range(1, opt.epochs + 1):
        try:
            # 导入trainer模块
            from trainer import run_epoch
            
            # 修复参数传递 - 移除extra_param
            current_score, save_bool, metrics = run_epoch(
                train_loader, val_loader, model, criterion, optimizer, scheduler, 
                epoch, opt, scaler  # 9个参数，正确匹配
            )
            
            # 保存最佳模型
            if current_score > best_score:
                best_score = current_score
                if save_bool:
                    save_path = f'./saved_models/{opt.tag}_best_model.pth'
                    os.makedirs('./saved_models', exist_ok=True)
                    
                    # 保存模型
                    if hasattr(model, 'module'):
                        torch.save(model.module.state_dict(), save_path)
                    else:
                        torch.save(model.state_dict(), save_path)
                    
                    print(f"💾 保存最佳模型: {save_path} (Score: {best_score:.4f})")
            
            # 记录到SwanLab（如果使用）
            try:
                import swanlab
                swanlab.log({
                    'train_loss': metrics['train_loss'],
                    'train_acc': metrics['train_acc'],
                    'val_loss': metrics['val_loss'],
                    'val_acc': metrics['val_acc'],
                    'weighted_f1': metrics['weighted_f1'],
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
            except ImportError:
                pass  # SwanLab不可用时跳过
        
        except Exception as e:
            print(f"❌ Epoch {epoch} 训练出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"🎉 训练完成! 最佳F1分数: {best_score:.4f}")
    
    # 记录最终结果
    log_final_results_to_swanlab(opt, metrics, best_score, swan)
    
    print(f'\n训练完成! 最佳Score: {best_score:.3f}')
    
    # 关闭SwanLab
    if swan is not None:
        try:
            swan.finish()
        except:
            pass


if __name__ == '__main__':
    main()