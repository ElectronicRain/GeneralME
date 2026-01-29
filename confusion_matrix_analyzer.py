#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confusion Matrix Analyzer
Generate confusion matrix and perform visual analysis for FluidCNN v6 best model
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import os
from datetime import datetime
from pathlib import Path

# Import custom modules
from model import FluidCNNEnhancedV6
from data_loader import MultiSizeMeshDataset, pad_collate
from trainer import FluidCNNTrainer


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, title="Confusion Matrix"):
    """
    Plot confusion matrix heatmap

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Save path
        title: Chart title
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Set figure size
    plt.figure(figsize=(10, 8))

    # Create labels (show count and percentage)
    cm_labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = cm_percent[i, j]
            if count > 0:
                cm_labels[i, j] = f'{count}\n({percent:.1f}%)'
            else:
                cm_labels[i, j] = f'0\n(0.0%)'

    # Plot heatmap
    sns.heatmap(cm, annot=cm_labels, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, square=True, linewidths=0.5)

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    # Show figure
    plt.show()


def analyze_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Detailed analysis of confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        dict: Analysis results
    """
    cm = confusion_matrix(y_true, y_pred)
    num_classes = len(class_names) if class_names else cm.shape[0]

    # Calculate metrics for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), average=None, zero_division=0
    )

    # Calculate overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate TP, FP, FN, TN for each class
    results = {
        'overall_accuracy': accuracy,
        'confusion_matrix': cm,
        'class_metrics': []
    }

    print("\n" + "=" * 80)
    print("Confusion Matrix Detailed Analysis")
    print("=" * 80)

    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\nPer-Class Detailed Metrics:")
    print("-" * 80)
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Support':<10}")
    print("-" * 80)

    for i in range(num_classes):
        class_name = class_names[i] if class_names else f"Class {i}"
        results['class_metrics'].append({
            'class_name': class_name,
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': support[i]
        })

        print(f"{class_name:<12} {precision[i]:<10.4f} {recall[i]:<10.4f} "
              f"{f1[i]:<10.4f} {support[i]:<10}")

    # Analyze confusion
    print("\nClassification Error Analysis:")
    print("-" * 80)

    for i in range(num_classes):
        class_name = class_names[i] if class_names else f"Class {i}"
        true_positives = cm[i, i]
        false_positives = cm[:, i].sum() - true_positives
        false_negatives = cm[i, :].sum() - true_positives

        # Find other classes most easily misclassified as this class
        misclassified_to = []
        for j in range(num_classes):
            if i != j and cm[j, i] > 0:
                misclassified_to.append((class_names[j] if class_names else f"Class {j}", cm[j, i]))

        # Find other classes this class is most easily misclassified as
        misclassified_from = []
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                misclassified_from.append((class_names[j] if class_names else f"Class {j}", cm[i, j]))

        print(f"\n{class_name}:")
        print(f"  Correctly classified: {true_positives}")
        print(f"  Misclassified as this class by others: {false_positives}")

        if misclassified_to:
            misclassified_to.sort(key=lambda x: x[1], reverse=True)
            print(f"    Main misclassification sources: {', '.join([f'{name}({count})' for name, count in misclassified_to[:3]])}")

        if misclassified_from:
            misclassified_from.sort(key=lambda x: x[1], reverse=True)
            print(f"  Misclassified as other classes: {false_negatives}")
            print(f"    Main misclassification targets: {', '.join([f'{name}({count})' for name, count in misclassified_from[:3]])}")

    return results


def main():
    """Main function"""
    print("=" * 80)
    print("FluidCNN v6 Confusion Matrix Analyzer")
    print("=" * 80)

    # Create output directory
    output_dir = Path("confusion_matrix_analysis")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Class names
    class_names = [f"Label {i}" for i in range(8)]

    # Model file path
    model_path = Path("01234567_best_cnn_v6.pt")

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        print("Please ensure the best model file has been saved")
        return

    try:
        # ==================== Load dataset ====================
        print("\nLoading dataset...")
        FEATURES_TO_USE = [0, 1, 2, 3, 4, 5, 6, 7]
        dataset = MultiSizeMeshDataset("../cnn_input_data", feature_indices=FEATURES_TO_USE)
        print(f"Dataset loaded successfully!")
        print(f"   Total samples: {len(dataset)}")

        # ==================== Create data loader ====================
        print("\nCreating test data loader...")
        test_indices = list(range(len(dataset)))  # Use all data for testing
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=pad_collate,
            num_workers=0
        )
        print(f"   Test set samples: {len(test_dataset)}")

        # ==================== Initialize model ====================
        print("\nInitializing model...")
        model_input_channels = dataset.num_input_channels() + 3

        model = FluidCNNEnhancedV6(
            input_channels=model_input_channels,
            num_classes=8,
            branch_channels=32,
            dropout_rate=0.3,
            conv_kernel_size=10,
            top16_hidden_dim=64,
            fusion_method='weighted_sum',
            seed=42
        )

        # ==================== Load best model ====================
        print(f"\nLoading best model: {model_path}")
        trainer = FluidCNNTrainer(model, device="auto")
        model.load_state_dict(torch.load(model_path, map_location=trainer.device, weights_only=False))
        print("Model loaded successfully!")

        # ==================== Predict on test set ====================
        print("\nPredicting on test set...")
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

        print(f"Prediction completed!")
        print(f"   Total samples: {len(all_labels)}")
        print(f"   Predicted label range: {all_preds.min()} - {all_preds.max()}")
        print(f"   True label range: {all_labels.min()} - {all_labels.max()}")

        # ==================== Generate classification report ====================
        print("\nGenerating classification report...")
        report = classification_report(
            all_labels,
            all_preds,
            labels=range(8),
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        # Save classification report to file
        report_file = output_dir / f"classification_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("FluidCNN v6 Classification Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(classification_report(
                all_labels,
                all_preds,
                labels=range(8),
                target_names=class_names,
                zero_division=0
            ))
        print(f"Classification report saved to: {report_file}")

        # ==================== Plot confusion matrix ====================
        print("\nPlotting confusion matrix...")
        cm = confusion_matrix(all_labels, all_preds)

        # Save raw confusion matrix data
        cm_file = output_dir / f"confusion_matrix_data_{timestamp}.csv"
        np.savetxt(cm_file, cm, delimiter=',', fmt='%d')
        print(f"Confusion matrix raw data saved to: {cm_file}")

        # Print confusion matrix
        print("\n" + "=" * 80)
        print("Confusion Matrix (Numerical)")
        print("=" * 80)
        print("Row=True Label, Column=Predicted Label")
        print("-" * 80)
        print(f"{'Class':<12}", end="")
        for i in range(8):
            print(f"{f'P{i}':<8}", end="")
        print()
        print("-" * 80)

        for i in range(8):
            print(f"{f'T{i}':<12}", end="")
            for j in range(8):
                print(f"{cm[i, j]:<8}", end="")
            print()

        # Plot and save confusion matrix
        cm_plot_path = output_dir / f"confusion_matrix_heatmap_{timestamp}.png"
        plot_confusion_matrix(
            all_preds,  # Note: order swapped because in heatmap x-axis is prediction, y-axis is true label
            all_labels,
            class_names=class_names,
            save_path=cm_plot_path,
            title="FluidCNN v6 - Confusion Matrix (Test Set)"
        )

        # Analyze confusion matrix
        results = analyze_confusion_matrix(all_labels, all_preds, class_names)

        # ==================== Save analysis results ====================
        print("\nSaving analysis results...")
        results_file = output_dir / f"confusion_matrix_analysis_{timestamp}.txt"
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("FluidCNN v6 Confusion Matrix Analysis Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model File: {model_path}\n")
            f.write(f"Test Samples: {len(all_labels)}\n")
            f.write(f"Overall Accuracy: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)\n\n")

            f.write("Confusion Matrix:\n")
            f.write("-" * 80 + "\n")
            f.write("Row=True Label, Column=Predicted Label\n")
            f.write("-" * 80 + "\n")
            for i in range(8):
                f.write(f"T{i}: ")
                for j in range(8):
                    f.write(f"{cm[i, j]:4d} ")
                f.write("\n")

            f.write("\nPer-Class Metrics:\n")
            f.write("-" * 80 + "\n")
            for metric in results['class_metrics']:
                f.write(f"{metric['class_name']}:\n")
                f.write(f"  Precision: {metric['precision']:.4f}\n")
                f.write(f"  Recall: {metric['recall']:.4f}\n")
                f.write(f"  F1 Score: {metric['f1_score']:.4f}\n")
                f.write(f"  Support: {metric['support']}\n\n")

        print(f"Detailed analysis results saved to: {results_file}")

        print("\n" + "=" * 80)
        print("Confusion Matrix Analysis Completed!")
        print("=" * 80)
        print(f"All results saved in: {output_dir}")
        print(f"   - Confusion matrix heatmap: {cm_plot_path}")
        print(f"   - Confusion matrix data: {cm_file}")
        print(f"   - Classification report: {report_file}")
        print(f"   - Detailed analysis: {results_file}")
        print("=" * 80)

    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
