#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FluidCNN v6 Main Training Script
Based on FluidCNN v5, adding top-16-row feature enhancement parallel branch

Architecture v6:
Input (H, W, 3)
  ├─ Parallel Branch 1: Top16RowModule → Extract top 16 rows → 1xN Conv → MLP → 128-dim feature
  └─ Parallel Branch 2: Multi-branch Residual + SE Attention → (H, W, 96)
       └─ Residual + Downsample → (H/2, W/2, 96)
            └─ Residual → (H/2, W/2, 128)
                 └─ Conv Pool → (H/4, W/4, 128)
                      └─ Residual + Downsample → (H/8, W/8, 128)
                           └─ Residual → (H/8, W/8, 256)
                                └─ Residual + Downsample → (H/16, W/16, 256)
                                     └─ Residual → (H/16, W/16, 512)
                                          └─ GlobalAvgPool → 512
                                               └─ Feature Fusion: 512-dim + 128-dim → 640-dim
                                                    └─ Dense → 256→128→8

Core Features:
✓ Parallel Top16RowModule for top 16 rows
✓ 1xN convolution kernel (configurable N)
✓ MLP feature scaling
✓ Weighted feature fusion (main branch 512-dim + parallel branch 128-dim)
✓ Fixed random seed for reproducibility
✓ Residual connections (ResNet-style)
✓ SE channel attention mechanism
✓ GELU activation function
✓ Enhanced classifier (3-layer FC + BN + Dropout)
✓ AdamW optimizer + Cosine Annealing scheduler
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from datetime import datetime
import sys
from pathlib import Path
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from model import FluidCNNEnhancedV6
from data_loader import MultiSizeMeshDataset, pad_collate
from trainer import FluidCNNTrainer
from tsne_analyzer import TSNEAnalyzer
from utils import (
    setup_logging,
    cleanup_logging,
    plot_training_history,
    print_model_summary,
    format_time,
    print_separator
)


def main():
    """Main training workflow"""
    # ==================== Initialize logging ====================
    print_separator()
    print("FluidCNN v6 - Top-16 Row Feature Enhancement Parallel Branch Version")
    print("=" * 60)
    print("Core Features: Parallel Branch | 1xN Convolution | Weighted Fusion | Fixed Seed")
    print("=" * 60)

    now = datetime.now()

    # Create temporary run directory
    temp_dir_suffix = now.strftime('%m%d_%H%M%S')
    temp_run_dir = Path("logs") / f"pending_{temp_dir_suffix}"
    temp_run_dir.mkdir(parents=True, exist_ok=True)

    log_filename, original_stdout, log_file = setup_logging(temp_run_dir)

    final_test_acc = None

    try:
        # ==================== Training configuration ====================
        # v6 configuration: Add parallel branch, enhance feature extraction
        BATCH_SIZE = 32
        EPOCHS = 200
        LEARNING_RATE = 3e-4
        WEIGHT_DECAY = 1e-4
        EARLY_STOPPING_PATIENCE = 60

        # Model configuration
        BRANCH_CHANNELS = 32  # Multi-branch convolution channels
        DROPOUT_RATE = 0.3    # Dropout probability

        # v6 specific configuration: Top-16 row module parameters
        CONV_KERNEL_SIZE = 10  # 1xN convolution kernel size N (adjustable)
        TOP16_HIDDEN_DIM = 64  # Top-16 row module MLP hidden layer dimension
        FUSION_METHOD = 'weighted_sum'  # Fusion method: 'weighted_sum' or 'concat'
        RANDOM_SEED = 42  # Fixed random seed

        # Feature channel selection (use all 8 feature channels)
        FEATURES_TO_USE = [0, 1, 2, 3, 4, 5, 6, 7]

        # T-SNE analysis configuration
        TSNE_ENABLED = True
        TSNE_EPOCHS = [5, 10, 15, 20, 25, 30, 35, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        TSNE_MAX_SAMPLES = 10240
        TSNE_PERPLEXITY = 30

        print(f"\nTraining Configuration:")
        print(f"   Batch size: {BATCH_SIZE}")
        print(f"   Epochs: {EPOCHS}")
        print(f"   Learning rate: {LEARNING_RATE}")
        print(f"   Optimizer: AdamW")
        print(f"   Weight decay: {WEIGHT_DECAY}")
        print(f"   Scheduler: CosineAnnealingWarmRestarts")
        print(f"   Early stopping patience: {EARLY_STOPPING_PATIENCE}")
        print(f"   Feature indices: {FEATURES_TO_USE} (Total: {len(FEATURES_TO_USE)} features)")
        print(f"\nModel Configuration:")
        print(f"   Branch channels: {BRANCH_CHANNELS}")
        print(f"   Dropout rate: {DROPOUT_RATE}")
        print(f"   Main branch convolution kernel: 3x10 (Rectangular convolution)")
        print(f"\nv6 Specific Configuration:")
        print(f"   1xN convolution kernel size: {CONV_KERNEL_SIZE}")
        print(f"   Top16 module hidden dimension: {TOP16_HIDDEN_DIM}")
        print(f"   Fusion method: {FUSION_METHOD}")
        print(f"   Random seed: {RANDOM_SEED}")
        print(f"\nT-SNE Analysis Configuration:")
        print(f"   Enable T-SNE: {TSNE_ENABLED}")
        print(f"   Analysis epochs: {TSNE_EPOCHS}")
        print(f"   Max samples: {TSNE_MAX_SAMPLES}")
        print(f"   Perplexity: {TSNE_PERPLEXITY}")

        feature_str = "".join(map(str, FEATURES_TO_USE))
        model_filename = temp_run_dir / f"{feature_str}_best_cnn_v6.pt"

        # ==================== Load dataset ====================
        print("\nLoading dataset...")
        try:
            dataset = MultiSizeMeshDataset("../cnn_input_data", feature_indices=FEATURES_TO_USE)
            print(f"Dataset loaded successfully!")
            print(f"   Total samples: {len(dataset)}")
            print(f"   Original feature channels: {dataset.num_input_channels()}")
            print(f"   Model input channels: {dataset.num_input_channels() + 3} (including mask and coordinates)")
        except Exception as e:
            print(f"Dataset loading failed: {e}")
            import traceback
            traceback.print_exc()
            raise

        # ==================== Split dataset ====================
        print("\nSplitting dataset...")
        indices = list(range(len(dataset)))

        if len(indices) < 3:
            raise RuntimeError(f"Dataset too small ({len(indices)} samples), cannot split")

        # 8 sizes for training, 1 for testing, 1 for validation
        # 7:3 split training and (validation + testing), then split equally -> 7:1.5:1.5
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)

        print(f"   Training set: {len(train_dataset)} samples ({len(train_dataset)/len(dataset)*100:.1f}%)")
        print(f"   Validation set: {len(val_dataset)} samples ({len(val_dataset)/len(dataset)*100:.1f}%)")
        print(f"   Test set: {len(test_dataset)} samples ({len(test_dataset)/len(dataset)*100:.1f}%)")

        if len(train_dataset) == 0:
            raise RuntimeError("Training set is empty, cannot train")

        # ==================== Create DataLoader ====================
        print("\n[CONFIG] Creating DataLoader...")
        num_workers = 0
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=pad_collate,
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=pad_collate,
            num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=pad_collate,
            num_workers=num_workers
        )
        print("[OK] DataLoader created successfully!")

        # ==================== Initialize model ====================
        print("\n[INIT] Initializing FluidCNN v6 Top-16 Row Feature Enhancement Parallel Branch Version...")
        model_input_channels = dataset.num_input_channels() + 3  # mask + coord_x + coord_y

        model = FluidCNNEnhancedV6(
            input_channels=model_input_channels,
            num_classes=8,
            branch_channels=BRANCH_CHANNELS,
            dropout_rate=DROPOUT_RATE,
            conv_kernel_size=CONV_KERNEL_SIZE,
            top16_hidden_dim=TOP16_HIDDEN_DIM,
            fusion_method=FUSION_METHOD,
            seed=RANDOM_SEED
        )

        # Print model summary
        print_model_summary(model, (model_input_channels, 64, 64))

        # Display model information
        model_info = model.get_model_info()
        print(f"\n[INFO] Model Information:")
        print(f"   Name: {model_info['model_name']}")
        print(f"   Input channels: {model_info['input_channels']}")
        print(f"   Number of classes: {model_info['num_classes']}")
        print(f"   1xN convolution kernel size: {model_info['conv_kernel_size']}")
        print(f"   Top16 hidden dimension: {model_info['top16_hidden_dim']}")
        print(f"   Fusion method: {model_info['fusion_method']}")
        print(f"   Key features:")
        for feature in model_info.get("key_features", []):
            print(f"     - {feature}")

        # ==================== Initialize optimizer and scheduler ====================
        print("\n[CONFIG] Initializing optimizer and scheduler...")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        criterion = nn.CrossEntropyLoss()

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        trainer = FluidCNNTrainer(model)
        print("[OK] Model and optimizer initialized successfully!")

        # Initialize T-SNE analyzer
        tsne_dir = temp_run_dir / "tsne_analysis"
        tsne_analyzer = TSNEAnalyzer(save_dir=str(tsne_dir))
        print(f"[OK] T-SNE analyzer initialized, save directory: {tsne_dir}")

        # ==================== Training loop ====================
        best_accuracy = 0.0
        best_val_report = {}
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        epochs_no_improve = 0

        print("\n🚀 开始训练...")
        print_separator()

        # 测试数据加载和模型前向传播
        print("🔍 测试数据加载和模型前向传播...")
        test_batch = next(iter(train_loader))
        test_inputs, test_labels = test_batch
        test_inputs = test_inputs.to(trainer.device)
        test_outputs = trainer.model(test_inputs)
        print(f"   ✅ 测试成功！")
        print(f"   Batch 形状: {test_inputs.shape}")
        print(f"   输出形状: {test_outputs.shape}")

        # 记录训练开始时间
        total_start_time = time.time()

        for epoch in range(EPOCHS):
            print(f"\n📅 Epoch {epoch + 1:02d}/{EPOCHS}")

            try:
                train_loss, train_acc = trainer.train_one_epoch(train_loader, optimizer, criterion)

                if len(val_dataset) > 0:
                    val_loss, val_acc, val_report = trainer.evaluate(val_loader, criterion)
                else:
                    val_loss, val_acc, val_report = 0.0, 0.0, {}
                    print("   ⚠️  跳过验证（验证集为空）")

            except KeyboardInterrupt:
                print("\n\n⚠️  用户中断训练")
                break

            except Exception as e:
                print(f"❌ Epoch {epoch + 1} 训练失败: {e}")
                import traceback
                traceback.print_exc()
                print("   继续下一个 epoch...")
                continue

            # 记录指标
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            # 学习率调度
            scheduler.step()

            print(f"   训练 Loss: {train_loss:.4f} | 训练 Acc: {train_acc:.4f}")
            print(f"   验证 Loss: {val_loss:.4f} | 验证 Acc: {val_acc:.4f}")

            # 显示当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   学习率: {current_lr:.6f}")

            # 保存最佳模型
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                epochs_no_improve = 0
                best_val_report = val_report
                torch.save(model.state_dict(), model_filename)
                print(f"💾 已保存最佳模型至 {model_filename}")

                # 显示各类别准确率
                print("   📈 最佳验证集各类别准确率:")
                if best_val_report:
                    for class_name, metrics in best_val_report.items():
                        if class_name.startswith("Label"):
                            recall = metrics.get('recall', 0.0)
                            print(f"      - {class_name}: {recall:.2%}")
            else:
                epochs_no_improve += 1
                print(f"   (!) 验证准确率未提升, 早停计数: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"\n🛑 连续 {EARLY_STOPPING_PATIENCE} 个 epochs 验证准确率未提升, 触发早停。")
                    break

            # 每5轮保存模型和混淆矩阵
            if (epoch + 1) % 5 == 0:
                print(f"\n📦 保存第 {epoch + 1} 轮模型和混淆矩阵...")
                checkpoint_filename = temp_run_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
                torch.save(model.state_dict(), checkpoint_filename)
                print(f"   💾 模型已保存: {checkpoint_filename}")

                # 计算混淆矩阵
                print(f"   📊 计算第 {epoch + 1} 轮混淆矩阵...")
                try:
                    cm_output_dir = temp_run_dir / f"confusion_matrix_epoch_{epoch+1:03d}"
                    cm_output_dir.mkdir(exist_ok=True)

                    # 在验证集上计算混淆矩阵
                    all_preds = []
                    all_labels = []

                    model.eval()
                    with torch.no_grad():
                        for batch_idx, (inputs, labels) in enumerate(val_loader):
                            inputs, labels = inputs.to(trainer.device), labels.to(trainer.device)
                            outputs = model(inputs)
                            preds = outputs.argmax(1).cpu().numpy()
                            all_preds.extend(preds)
                            all_labels.extend(labels.cpu().numpy())

                    all_preds = np.array(all_preds)
                    all_labels = np.array(all_labels)

                    # 计算混淆矩阵
                    cm = confusion_matrix(all_labels, all_preds, labels=range(8))
                    class_names = [f"Label {i}" for i in range(8)]

                    # 保存混淆矩阵数据
                    cm_file = cm_output_dir / "confusion_matrix_data.csv"
                    np.savetxt(cm_file, cm, delimiter=',', fmt='%d')

                    # 绘制混淆矩阵热力图
                    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

                    plt.figure(figsize=(12, 10))

                    cm_labels = np.empty_like(cm, dtype=object)
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            count = cm[i, j]
                            percent = cm_percent[i, j]
                            if count > 0:
                                cm_labels[i, j] = f'{count}\n({percent:.1f}%)'
                            else:
                                cm_labels[i, j] = f'0\n(0.0%)'

                    sns.heatmap(cm, annot=cm_labels, fmt='', cmap='Blues',
                                xticklabels=class_names, yticklabels=class_names,
                                cbar_kws={'label': 'Count'}, square=True, linewidths=0.5)

                    plt.title(f'FluidCNN v6 - Confusion Matrix (Epoch {epoch+1}, Validation Set)',
                              fontsize=16, fontweight='bold', pad=20)
                    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
                    plt.ylabel('True Label', fontsize=12, fontweight='bold')
                    plt.tight_layout()

                    cm_plot_path = cm_output_dir / "confusion_matrix_heatmap.png"
                    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
                    plt.close()  # 关闭图形以释放内存

                    # 计算指标
                    precision, recall, f1, support = precision_recall_fscore_support(
                        all_labels, all_preds, labels=range(8), average=None, zero_division=0
                    )

                    # 保存报告
                    report_file = cm_output_dir / "confusion_matrix_report.txt"
                    with open(report_file, 'w', encoding='utf-8') as f:
                        f.write(f"FluidCNN v6 混淆矩阵分析报告 - 第 {epoch+1} 轮\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(f"模型文件: {checkpoint_filename}\n")
                        f.write(f"验证样本数: {len(all_labels)}\n")
                        f.write(f"验证准确率: {val_acc:.4f} ({val_acc*100:.2f}%)\n\n")

                        f.write("混淆矩阵:\n")
                        f.write("-" * 80 + "\n")
                        for i in range(8):
                            f.write(f"T{i}: ")
                            for j in range(8):
                                f.write(f"{cm[i, j]:4d} ")
                            f.write("\n")

                        f.write("\n各类别指标:\n")
                        f.write("-" * 80 + "\n")
                        f.write(f"{'类别':<12} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'支持数':<10}\n")
                        f.write("-" * 80 + "\n")
                        for i in range(8):
                            f.write(f"{class_names[i]:<12} {precision[i]:<10.4f} {recall[i]:<10.4f} "
                                  f"{f1[i]:<10.4f} {support[i]:<10}\n")

                    print(f"   ✅ 第 {epoch + 1} 轮混淆矩阵已保存至: {cm_output_dir}")
                    print(f"      - 热力图: {cm_plot_path}")
                    print(f"      - 数据: {cm_file}")
                    print(f"      - 报告: {report_file}")

                except Exception as e:
                    print(f"   ⚠️  第 {epoch + 1} 轮混淆矩阵计算失败: {e}")
                    import traceback
                    traceback.print_exc()

            # T-SNE分析
            if TSNE_ENABLED and (epoch + 1) in TSNE_EPOCHS:
                print(f"\n🔬 执行 T-SNE 分析 (Epoch {epoch + 1})...")
                try:
                    # 使用验证集进行T-SNE分析
                    if len(val_dataset) > 0:
                        features, labels, predictions = trainer.collect_features(
                            val_loader, max_samples=TSNE_MAX_SAMPLES
                        )

                        # 添加到分析器
                        tsne_analyzer.epoch_features[epoch + 1] = features
                        tsne_analyzer.epoch_labels[epoch + 1] = labels
                        tsne_analyzer.epoch_predictions[epoch + 1] = predictions

                        # 运行T-SNE
                        tsne_result = tsne_analyzer.run_tsne(
                            epoch + 1,
                            perplexity=TSNE_PERPLEXITY
                        )

                        if tsne_result:
                            # 生成并显示报告
                            print("\n" + tsne_analyzer.generate_report(epoch + 1))

                            # 绘制可视化
                            tsne_analyzer.plot_tsne(epoch + 1, save=True, show=False)

                            # 保存分析结果
                            tsne_analyzer.save_analysis(epoch + 1)

                            print(f"✅ T-SNE 分析完成")
                        else:
                            print(f"⚠️  T-SNE 分析失败")
                    else:
                        print(f"   跳过T-SNE（验证集为空）")

                except Exception as e:
                    print(f"   ⚠️  T-SNE 分析过程中出错: {e}")
                    import traceback
                    traceback.print_exc()

        # 计算总训练时间
        total_time = time.time() - total_start_time
        print(f"\n⏱️  总训练时间: {format_time(total_time)}")

        # ==================== 加载最佳模型进行最终分析 ====================
        print("\n🧪 加载最佳模型进行最终测试...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            model.load_state_dict(torch.load(model_filename, map_location=trainer.device, weights_only=False))

        # ==================== 最终T-SNE分析（使用测试集） ====================
        if TSNE_ENABLED and len(test_dataset) > 0:
            print(f"\n🔬 执行最终 T-SNE 分析（使用测试集）...")
            try:
                final_epoch = EPOCHS

                # 收集测试集特征
                features, labels, predictions = trainer.collect_features(
                    test_loader, max_samples=TSNE_MAX_SAMPLES
                )

                # 添加到分析器
                tsne_analyzer.epoch_features[final_epoch] = features
                tsne_analyzer.epoch_labels[final_epoch] = labels
                tsne_analyzer.epoch_predictions[final_epoch] = predictions

                # 运行T-SNE
                tsne_result = tsne_analyzer.run_tsne(
                    final_epoch,
                    perplexity=TSNE_PERPLEXITY
                )

                if tsne_result:
                    # 生成并显示最终报告
                    print("\n" + "="*80)
                    print("最终 T-SNE 分析报告（测试集）")
                    print("="*80)
                    print(tsne_analyzer.generate_report(final_epoch))

                    # 绘制可视化
                    tsne_analyzer.plot_tsne(final_epoch, save=True, show=False)

                    # 保存分析结果
                    tsne_analyzer.save_analysis(final_epoch)

                    print(f"✅ 最终 T-SNE 分析完成")
                else:
                    print(f"⚠️  最终 T-SNE 分析失败")

            except Exception as e:
                print(f"   ⚠️  最终 T-SNE 分析过程中出错: {e}")
                import traceback
                traceback.print_exc()

        # ==================== 测试最佳模型 ====================
        print("\n🧪 测试最佳模型...")

        test_loss, test_acc, test_report = trainer.evaluate(test_loader, criterion)
        final_test_acc = test_acc

        print(f"   测试准确率: {test_acc:.4f}")
        print(f"   最佳验证准确率: {best_accuracy:.4f}")

        # 显示最终性能评估
        print("\n📊 最终模型性能评估:")
        print("   📈 最佳验证集各类别准确率:")
        if best_val_report:
            for class_name, metrics in best_val_report.items():
                if class_name.startswith("Label"):
                    recall = metrics.get('recall', 0.0)
                    print(f"      - {class_name}: {recall:.2%}")

        print("   📈 测试集各类别准确率:")
        if test_report:
            for class_name, metrics in test_report.items():
                if class_name.startswith("Label"):
                    recall = metrics.get('recall', 0.0)
                    print(f"      - {class_name}: {recall:.2%}")

        # ==================== 混淆矩阵分析 ====================
        print("\n📊 执行混淆矩阵分析...")
        try:
            import os

            # 创建混淆矩阵输出目录
            cm_output_dir = temp_run_dir / "confusion_matrix_analysis"
            cm_output_dir.mkdir(exist_ok=True)

            # 收集所有预测和真实标签
            print("   🔍 收集预测结果...")
            all_preds = []
            all_labels = []

            model.eval()
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(test_loader):
                    inputs, labels = inputs.to(trainer.device), labels.to(trainer.device)
                    outputs = model(inputs)
                    preds = outputs.argmax(1).cpu().numpy()

                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            print(f"   ✅ 收集完成! 总样本数: {len(all_labels)}")

            # 计算混淆矩阵
            cm = confusion_matrix(all_labels, all_preds, labels=range(8))

            # 类别名称
            class_names = [f"Label {i}" for i in range(8)]

            # 打印混淆矩阵
            print("\n" + "=" * 80)
            print("📊 混淆矩阵 (数值)")
            print("=" * 80)
            print("行=真实标签, 列=预测标签")
            print("-" * 80)
            print(f"{'类别':<12}", end="")
            for i in range(8):
                print(f"{f'P{i}':<8}", end="")
            print()
            print("-" * 80)

            for i in range(8):
                print(f"{f'T{i}':<12}", end="")
                for j in range(8):
                    print(f"{cm[i, j]:<8}", end="")
                print()

            # 保存原始混淆矩阵数据
            cm_file = cm_output_dir / "confusion_matrix_data.csv"
            np.savetxt(cm_file, cm, delimiter=',', fmt='%d')
            print(f"✅ 混淆矩阵原始数据已保存至: {cm_file}")

            # 绘制混淆矩阵热力图
            print("\n🎨 绘制混淆矩阵热力图...")

            # 计算百分比
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

            # 设置图表大小
            plt.figure(figsize=(12, 10))

            # 创建标签（显示数量和百分比）
            cm_labels = np.empty_like(cm, dtype=object)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    count = cm[i, j]
                    percent = cm_percent[i, j]
                    if count > 0:
                        cm_labels[i, j] = f'{count}\n({percent:.1f}%)'
                    else:
                        cm_labels[i, j] = f'0\n(0.0%)'

            # 绘制热力图
            sns.heatmap(cm, annot=cm_labels, fmt='', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        cbar_kws={'label': 'Count'}, square=True, linewidths=0.5)

            plt.title('FluidCNN v6 - Confusion Matrix (Test Set)', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
            plt.ylabel('True Label', fontsize=12, fontweight='bold')

            # 调整布局
            plt.tight_layout()

            # 保存图表
            cm_plot_path = cm_output_dir / "confusion_matrix_heatmap.png"
            plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ 混淆矩阵热力图已保存至: {cm_plot_path}")

            # 计算各类别指标
            precision, recall, f1, support = precision_recall_fscore_support(
                all_labels, all_preds, labels=range(8), average=None, zero_division=0
            )

            # 分析结果
            print("\n📈 各类别详细指标:")
            print("-" * 80)
            print(f"{'类别':<12} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'支持数':<10}")
            print("-" * 80)

            for i in range(8):
                print(f"{class_names[i]:<12} {precision[i]:<10.4f} {recall[i]:<10.4f} "
                      f"{f1[i]:<10.4f} {support[i]:<10}")

            # 分析混淆情况
            print("\n🔍 分类错误分析:")
            print("-" * 80)

            for i in range(8):
                class_name = class_names[i]
                true_positives = cm[i, i]
                false_positives = cm[:, i].sum() - true_positives
                false_negatives = cm[i, :].sum() - true_positives

                # 找出最容易被误分类为该类别的其他类别
                misclassified_to = []
                for j in range(8):
                    if i != j and cm[j, i] > 0:
                        misclassified_to.append((class_names[j], cm[j, i]))

                # 找出该类别最容易被误分类成的其他类别
                misclassified_from = []
                for j in range(8):
                    if i != j and cm[i, j] > 0:
                        misclassified_from.append((class_names[j], cm[i, j]))

                print(f"\n{class_name}:")
                print(f"  ✓ 正确分类: {true_positives}")
                print(f"  ✗ 被其他类别误分类为该类别: {false_positives}")

                if misclassified_to:
                    misclassified_to.sort(key=lambda x: x[1], reverse=True)
                    print(f"    主要误分类来源: {', '.join([f'{name}({count})' for name, count in misclassified_to[:3]])}")

                if misclassified_from:
                    misclassified_from.sort(key=lambda x: x[1], reverse=True)
                    print(f"  ✗ 被该类别误分类为其他类别: {false_negatives}")
                    print(f"    主要误分类目标: {', '.join([f'{name}({count})' for name, count in misclassified_from[:3]])}")

            # 保存详细分析报告
            report_file = cm_output_dir / "confusion_matrix_analysis_report.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("FluidCNN v6 混淆矩阵分析报告\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"模型文件: {model_filename}\n")
                f.write(f"测试样本数: {len(all_labels)}\n")
                f.write(f"总体准确率: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)\n\n")

                f.write("混淆矩阵:\n")
                f.write("-" * 80 + "\n")
                f.write("行=真实标签, 列=预测标签\n")
                f.write("-" * 80 + "\n")
                for i in range(8):
                    f.write(f"T{i}: ")
                    for j in range(8):
                        f.write(f"{cm[i, j]:4d} ")
                    f.write("\n")

                f.write("\n各类别指标:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'类别':<12} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'支持数':<10}\n")
                f.write("-" * 80 + "\n")
                for i in range(8):
                    f.write(f"{class_names[i]:<12} {precision[i]:<10.4f} {recall[i]:<10.4f} "
                          f"{f1[i]:<10.4f} {support[i]:<10}\n")

            print(f"\n✅ 混淆矩阵详细分析报告已保存至: {report_file}")
            print("=" * 80)
            print(f"📁 混淆矩阵分析结果保存在: {cm_output_dir}")
            print(f"   - 混淆矩阵热力图: {cm_plot_path}")
            print(f"   - 混淆矩阵数据: {cm_file}")
            print(f"   - 详细分析报告: {report_file}")
            print("=" * 80)

        except Exception as e:
            print(f"   ⚠️  混淆矩阵分析过程中出错: {e}")
            import traceback
            traceback.print_exc()

        # ==================== 绘制训练曲线 ====================
        print("\n📈 绘制训练历史曲线...")
        plot_filename = temp_run_dir / "training_history.png"
        plot_training_history(
            train_losses, val_losses,
            train_accuracies, val_accuracies,
            plot_filename
        )

        print("\n✅ FluidCNN v6 顶部16行特征增强并行分支版本训练完成!")
        print(f"🎉 最终测试准确率: {final_test_acc:.4f}")

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        # ==================== 清理和保存日志 ====================
        print(f"\n📝 训练日志和结果处理...")
        sys.stdout = original_stdout
        log_file.close()

        # 清理日志并重命名文件
        cleanup_logging(original_stdout, log_file, temp_run_dir, final_test_acc)


if __name__ == "__main__":
    main()
