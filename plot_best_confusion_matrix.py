#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Confusion Matrix based on best.txt data
Use the same plotting method as confusion_matrix_analyzer.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from datetime import datetime


def parse_confusion_matrix(file_path):
    """
    Parse confusion matrix data from best.txt file

    Args:
        file_path: Path to best.txt file

    Returns:
        tuple: (confusion_matrix, accuracy, sample_count)
    """
    print(f"Reading file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract test sample count and accuracy
    accuracy_match = re.search(r'Overall Accuracy:\s*([\d.]+)\s*\((\d+\.\d+)%\)', content)
    sample_match = re.search(r'Test Samples:\s*(\d+)', content)

    accuracy = float(accuracy_match.group(1)) if accuracy_match else 0.0
    sample_count = int(sample_match.group(1)) if sample_match else 0

    # Extract confusion matrix data
    cm_pattern = r'T(\d+):\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)'
    matches = re.findall(cm_pattern, content)

    # Build confusion matrix
    cm = np.zeros((8, 8), dtype=int)
    for row_match in matches:
        row_idx = int(row_match[0])
        values = [int(x) for x in row_match[1:]]
        cm[row_idx, :] = values

    print(f"Parsing successful!")
    print(f"   Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Test Samples: {sample_count}")
    print(f"   Confusion Matrix Shape: {cm.shape}")

    return cm, accuracy, sample_count


def plot_confusion_matrix(cm, class_names=None, save_path=None,
                         title="Confusion Matrix", show_values=True):
    """
    Plot confusion matrix heatmap (same as function in confusion_matrix_analyzer.py)

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Save path
        title: Chart title
        show_values: Whether to show values
    """
    # Use confusion matrix directly
    cm_to_plot = cm

    # Calculate percentages
    cm_percent = cm_to_plot.astype('float') / cm_to_plot.sum(axis=1)[:, np.newaxis] * 100

    # Set figure size
    plt.figure(figsize=(12, 10))

    # If showing values, create labels
    if show_values:
        cm_labels = np.empty_like(cm_to_plot, dtype=object)
        for i in range(cm_to_plot.shape[0]):
            for j in range(cm_to_plot.shape[1]):
                count = cm_to_plot[i, j]
                percent = cm_percent[i, j]
                if count > 0:
                    cm_labels[i, j] = f'{count}\n({percent:.1f}%)'
                else:
                    cm_labels[i, j] = f'0\n(0.0%)'

        # Plot heatmap
        sns.heatmap(cm_to_plot, annot=cm_labels, fmt='', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'}, square=True, linewidths=0.5)
    else:
        # Plot heatmap (without values)
        sns.heatmap(cm_to_plot, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'}, square=True, linewidths=0.5)

    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    # Show figure
    plt.show()


def analyze_confusion_matrix(cm, class_names=None):
    """
    Detailed analysis of confusion matrix (same as function in confusion_matrix_analyzer.py)

    Args:
        cm: Confusion matrix
        class_names: List of class names
    """
    num_classes = cm.shape[0]

    print("\n" + "=" * 80)
    print("Confusion Matrix Detailed Analysis")
    print("=" * 80)

    # Calculate overall accuracy
    total_samples = cm.sum()
    correct_predictions = np.trace(cm)
    accuracy = correct_predictions / total_samples

    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Total Samples: {total_samples}")
    print(f"   Correct Predictions: {correct_predictions}")

    # Calculate metrics for each class
    print("\nPer-Class Detailed Metrics:")
    print("-" * 80)
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Support':<10}")
    print("-" * 80)

    for i in range(num_classes):
        class_name = class_names[i] if class_names else f"Class {i}"

        # Calculate TP, FP, FN
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = cm[i, :].sum()

        print(f"{class_name:<12} {precision:<10.4f} {recall:<10.4f} "
              f"{f1:<10.4f} {support:<10}")

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


def main():
    """Main function"""
    print("=" * 80)
    print("FluidCNN v6 - Plot Confusion Matrix based on best.txt data")
    print("=" * 80)

    # Input and output paths
    input_file = Path("best.txt")
    output_dir = Path("best_confusion_matrix")
    output_dir.mkdir(exist_ok=True)

    # Class names
    class_names = [f"Label {i}" for i in range(8)]

    # Parse confusion matrix
    cm, accuracy, sample_count = parse_confusion_matrix(input_file)

    # Print raw confusion matrix
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

    # Plot confusion matrix heatmap (with values)
    print("\nPlotting confusion matrix heatmap (with values)...")
    cm_plot_path = output_dir / f"confusion_matrix_with_values_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        save_path=cm_plot_path,
        title=f"FluidCNN v6 - Confusion Matrix (Best Model)\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%) | Samples: {sample_count}",
        show_values=True
    )

    # Plot confusion matrix heatmap (without values)
    print("\nPlotting confusion matrix heatmap (without values)...")
    cm_plot_path2 = output_dir / f"confusion_matrix_no_values_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        save_path=cm_plot_path2,
        title=f"FluidCNN v6 - Confusion Matrix (Best Model)\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%) | Samples: {sample_count}",
        show_values=False
    )

    # Analyze confusion matrix
    analyze_confusion_matrix(cm, class_names)

    # Save analysis results to file
    print("\nSaving analysis results...")
    results_file = output_dir / f"best_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("FluidCNN v6 Confusion Matrix Analysis Report (based on best.txt)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Test Samples: {sample_count}\n\n")

        f.write("Confusion Matrix:\n")
        f.write("-" * 80 + "\n")
        f.write("Row=True Label, Column=Predicted Label\n")
        f.write("-" * 80 + "\n")
        for i in range(8):
            f.write(f"T{i}: ")
            for j in range(8):
                f.write(f"{cm[i, j]:4d} ")
            f.write("\n")

    print(f"Detailed analysis results saved to: {results_file}")

    print("\n" + "=" * 80)
    print("Confusion Matrix Plotting Completed!")
    print("=" * 80)
    print(f"All results saved in: {output_dir}")
    print(f"   - Heatmap with values: {cm_plot_path}")
    print(f"   - Heatmap without values: {cm_plot_path2}")
    print(f"   - Detailed analysis report: {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
