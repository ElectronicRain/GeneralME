#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-SNE Analysis Module
Used for feature extraction, T-SNE dimensionality reduction analysis and visualization during training
"""

import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pathlib import Path
import time
from collections import defaultdict
import json


class TSNEAnalyzer:
    """
    T-SNE Analyzer
    Used for neural network feature extraction and dimensionality reduction visualization analysis
    """

    def __init__(self, save_dir="tsne_analysis"):
        """
        Initialize T-SNE analyzer

        Args:
            save_dir: Directory to save analysis results
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Store features and labels
        self.features = []
        self.labels = []
        self.predictions = []
        self.epoch_features = defaultdict(list)
        self.epoch_labels = defaultdict(list)
        self.epoch_predictions = defaultdict(list)

        self.tsne_results = {}

    def extract_features(self, model, inputs, layer_name='global_pool'):
        """
        Extract intermediate layer features from model

        Args:
            model: Neural network model
            inputs: Input tensor
            layer_name: Name of layer to extract features from

        Returns:
            Extracted feature vector
        """
        model.eval()

        # Register forward hook to extract features
        features = None

        def hook_fn(module, input, output):
            nonlocal features
            features = input[0] if isinstance(input, tuple) else input

        # Find target layer
        target_layer = getattr(model, layer_name, None)
        if target_layer is None:
            # If not found, use features after global average pooling
            if hasattr(model, 'global_pool'):
                target_layer = model.global_pool

        if target_layer is not None:
            handle = target_layer.register_forward_hook(hook_fn)
            with torch.no_grad():
                _ = model(inputs)
            handle.remove()
            features = features.view(features.size(0), -1)  # Flatten
        else:
            # If no suitable layer found, use entire model output
            with torch.no_grad():
                outputs = model(inputs)
                features = outputs

        return features.cpu().numpy()

    def collect_epoch_data(self, epoch, model, data_loader, device, max_samples=1000):
        """
        Collect feature data for one epoch

        Args:
            epoch: Current epoch number
            model: Model
            data_loader: Data loader
            device: Device
            max_samples: Maximum number of samples
        """
        print(f"\n   Collecting feature data for Epoch {epoch}...")

        collected = 0
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            if collected >= max_samples:
                break

            inputs = inputs.to(device)

            # Extract features
            features = self.extract_features(model, inputs)

            # Get predictions
            with torch.no_grad():
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            # Store data
            batch_size = features.shape[0]
            for i in range(min(batch_size, max_samples - collected)):
                self.epoch_features[epoch].append(features[i])
                self.epoch_labels[epoch].append(labels[i].item())
                self.epoch_predictions[epoch].append(predictions[i])

            collected += batch_size
            if batch_idx % 10 == 0:
                print(f"      Collected: {collected}/{max_samples} samples")

        print(f"   Collection completed: {len(self.epoch_features[epoch])} samples")

    def run_tsne(self, epoch, n_components=2, perplexity=30, random_state=42):
        """
        Run T-SNE on data from specified epoch

        Args:
            epoch: Epoch number
            n_components: Dimension after T-SNE reduction
            perplexity: T-SNE perplexity parameter
            random_state: Random seed

        Returns:
            dict: Dictionary containing T-SNE results
        """
        if epoch not in self.epoch_features or len(self.epoch_features[epoch]) == 0:
            print(f"   No feature data available for Epoch {epoch}")
            return None

        features = np.array(self.epoch_features[epoch])
        labels = np.array(self.epoch_labels[epoch])
        predictions = np.array(self.epoch_predictions[epoch])

        print(f"\n   Running T-SNE analysis on Epoch {epoch}...")
        print(f"      Feature dimension: {features.shape}")
        print(f"      Sample count: {len(labels)}")
        print(f"      Class count: {len(np.unique(labels))}")

        # Standardize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

        # Run T-SNE
        print(f"      Running T-SNE (perplexity={perplexity})...")
        start_time = time.time()
        tsne = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, len(features) - 1),
            random_state=random_state,
            verbose=0,
            max_iter=1000
        )
        tsne_results = tsne.fit_transform(features)
        elapsed = time.time() - start_time

        print(f"      T-SNE completed, time: {elapsed:.2f}s")

        # Calculate clustering quality metrics
        if len(np.unique(labels)) > 1:
            try:
                silhouette = silhouette_score(tsne_results, labels)
                print(f"      Silhouette score: {silhouette:.4f}")
            except Exception as e:
                print(f"      Silhouette score calculation failed: {e}")
                silhouette = None
        else:
            silhouette = None

        # Analyze misclassifications
        misclassified = predictions != labels
        misclassified_count = np.sum(misclassified)
        misclassified_rate = misclassified_count / len(labels) if len(labels) > 0 else 0

        print(f"      Misclassified: {misclassified_count}/{len(labels)} ({misclassified_rate:.2%})")

        # Analyze cluster situation for each class
        cluster_analysis = self._analyze_clusters(labels, predictions, tsne_results)

        result = {
            'epoch': epoch,
            'features': tsne_results,
            'labels': labels,
            'predictions': predictions,
            'misclassified': misclassified,
            'silhouette_score': silhouette,
            'misclassified_rate': misclassified_rate,
            'cluster_analysis': cluster_analysis,
            'total_samples': len(labels),
            'n_classes': len(np.unique(labels)),
            'perplexity': perplexity,
            'time_elapsed': elapsed
        }

        self.tsne_results[epoch] = result
        return result

    def _analyze_clusters(self, labels, predictions, tsne_features):
        """
        Analyze each cluster composition and find mixed erroneous data points

        Args:
            labels: True labels
            predictions: Predicted labels
            tsne_features: T-SNE reduced features

        Returns:
            dict: Cluster analysis results
        """
        analysis = {}

        # Create clusters for each predicted class
        unique_preds = np.unique(predictions)

        for pred_class in unique_preds:
            # Convert to Python native type
            pred_class = int(pred_class)

            # Find all samples predicted as this class
            pred_mask = predictions == pred_class
            pred_indices = np.where(pred_mask)[0]

            # Count true label distribution in this cluster
            cluster_labels = labels[pred_mask]
            unique_labels_in_cluster = np.unique(cluster_labels)

            # Count and calculate proportion of each true class in this cluster
            class_counts = {}
            for true_class in unique_labels_in_cluster:
                # Convert to Python native type
                true_class = int(true_class)

                count = int(np.sum(cluster_labels == true_class))
                total_in_cluster = len(cluster_labels)
                percentage = float(count / total_in_cluster) if total_in_cluster > 0 else 0.0
                class_counts[true_class] = {
                    'count': count,
                    'percentage': percentage
                }

            # Find majority class (class with highest proportion)
            if class_counts:
                majority_class = max(class_counts.keys(), key=lambda k: class_counts[k]['count'])
                majority_count = class_counts[majority_class]['count']
                majority_percentage = class_counts[majority_class]['percentage']

                # Count of other classes misclassified here
                mixed_in_count = len(cluster_labels) - majority_count

                analysis[pred_class] = {
                    'cluster_size': int(len(cluster_labels)),
                    'majority_class': majority_class,
                    'majority_count': majority_count,
                    'majority_percentage': majority_percentage,
                    'mixed_in_count': int(mixed_in_count),
                    'class_distribution': class_counts,
                    'is_mixed': bool(mixed_in_count > 0 and majority_percentage < 0.8),  # Considered severe mixing if majority class proportion < 80%
                    'indices': pred_indices.tolist()  # Sample indices in this cluster
                }

        return analysis

    def plot_tsne(self, epoch, save=True, show=False):
        """
        Plot T-SNE visualization

        Args:
            epoch: Epoch number
            save: Whether to save image
            show: Whether to display image
        """
        if epoch not in self.tsne_results:
            print(f"   T-SNE results for Epoch {epoch} not found")
            return

        result = self.tsne_results[epoch]
        features_2d = result['features']
        labels = result['labels']
        predictions = result['predictions']
        misclassified = result['misclassified']

        # Debug info: output class statistics
        unique_labels = np.unique(labels)
        unique_preds = np.unique(predictions)
        print(f"   Debug Info:")
        print(f"      True label types: {len(unique_labels)} (range: {min(unique_labels)}-{max(unique_labels)})")
        print(f"      Predicted label types: {len(unique_preds)} (range: {min(unique_preds) if len(unique_preds) > 0 else 'N/A'}-{max(unique_preds) if len(unique_preds) > 0 else 'N/A'})")

        for label in unique_labels:
            count = np.sum(labels == label)
            print(f"      Class {label}: {count} samples")

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        # Use more colors (Set3 has max 12 colors)
        colors = plt.cm.Set3(np.linspace(0, 1, 12))

        # Subplot 1: True labels
        ax1 = axes[0]
        for i, label in enumerate(unique_labels):
            mask = labels == label
            count = np.sum(mask)
            ax1.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=[colors[i % len(colors)]], label=f'Class {label} ({count})',
                       alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        ax1.set_title(f'Epoch {epoch} - True Labels\n{result["total_samples"]} samples, {len(unique_labels)} classes', fontsize=14)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Predicted labels
        ax2 = axes[1]
        for i, pred in enumerate(unique_preds):
            mask = predictions == pred
            count = np.sum(mask)
            ax2.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=[colors[i % len(colors)]], label=f'Pred {pred} ({count})',
                       alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        ax2.set_title(f'Epoch {epoch} - Predicted Labels\n{len(unique_preds)} clusters, Silhouette: {result.get("silhouette_score", "N/A"):.4f}', fontsize=14)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Misclassified labels
        ax3 = axes[2]
        correct_mask = ~misclassified
        correct_count = np.sum(correct_mask)
        wrong_count = np.sum(misclassified)
        ax3.scatter(features_2d[correct_mask, 0], features_2d[correct_mask, 1],
                   c='green', label=f'Correct ({correct_count})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax3.scatter(features_2d[misclassified, 0], features_2d[misclassified, 1],
                   c='red', label=f'Misclassified ({wrong_count})', alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
        ax3.set_title(f'Epoch {epoch} - Misclassified Analysis\nError Rate: {result["misclassified_rate"]:.2%}', fontsize=14)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save image
        if save:
            save_path = self.save_dir / f"tsne_epoch_{epoch:03d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   T-SNE visualization saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        # Additional info
        if len(unique_preds) < len(unique_labels):
            print(f"   Warning: Model only predicted {len(unique_preds)} classes, but there are {len(unique_labels)} true classes")
            print(f"      Missing classes: {set(unique_labels) - set(unique_preds)}")
            print(f"      Possible causes: insufficient training, class imbalance, or insufficient model capacity")

    def generate_report(self, epoch):
        """
        Generate T-SNE analysis report

        Args:
            epoch: Epoch number

        Returns:
            str: Analysis report text
        """
        if epoch not in self.tsne_results:
            return f"T-SNE analysis results for Epoch {epoch} not found\n"

        result = self.tsne_results[epoch]
        analysis = result['cluster_analysis']

        report = []
        report.append("=" * 80)
        report.append(f"T-SNE Cluster Analysis Report - Epoch {epoch}")
        report.append("=" * 80)
        report.append(f"\nOverall Statistics:")
        report.append(f"  Total Samples: {result['total_samples']}")
        report.append(f"  Class Count: {result['n_classes']}")
        report.append(f"  Misclassified Count: {np.sum(result['misclassified'])} / {result['total_samples']}")
        report.append(f"  Misclassification Rate: {result['misclassified_rate']:.2%}")
        if result['silhouette_score']:
            report.append(f"  Silhouette Score: {result['silhouette_score']:.4f}")

        report.append("\nDetailed Analysis by Cluster:")
        report.append("-" * 80)

        # Sort by cluster size
        sorted_clusters = sorted(analysis.items(), key=lambda x: x[1]['cluster_size'], reverse=True)

        for cluster_id, info in sorted_clusters:
            report.append(f"\nCluster {cluster_id} (Predicted Label):")
            report.append(f"  Cluster Size: {info['cluster_size']}")
            report.append(f"  Majority Class: Class {info['majority_class']} ({info['majority_count']} samples, {info['majority_percentage']:.1%})")

            # Show class distribution
            report.append("  Class Distribution:")
            for class_id in sorted(info['class_distribution'].keys()):
                count = info['class_distribution'][class_id]['count']
                pct = info['class_distribution'][class_id]['percentage']
                report.append(f"    - Class {class_id}: {count} samples ({pct:.1%})")

            # Label issues
            if info['is_mixed']:
                report.append(f"  Warning: This cluster is severely mixed (majority class proportion {info['majority_percentage']:.1%} < 80%)")
                report.append(f"      Wrong samples mixed in: {info['mixed_in_count']} data points")

                # List wrong classes mixed in
                wrong_classes = [c for c in info['class_distribution'].keys() if c != info['majority_class']]
                if wrong_classes:
                    report.append(f"      Wrong classes mixed in: {wrong_classes}")
            else:
                report.append(f"  Cluster is relatively pure")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def save_analysis(self, epoch):
        """
        Save analysis results to file

        Args:
            epoch: Epoch number
        """
        if epoch not in self.tsne_results:
            return

        result = self.tsne_results[epoch]

        # Save detailed data
        save_file = self.save_dir / f"tsne_analysis_epoch_{epoch:03d}.json"

        # Convert numpy types to Python native types
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        serializable_result = {
            'epoch': result['epoch'],
            'features_2d': convert_numpy_types(result['features']),
            'labels': convert_numpy_types(result['labels']),
            'predictions': convert_numpy_types(result['predictions']),
            'misclassified': convert_numpy_types(result['misclassified']),
            'silhouette_score': convert_numpy_types(result['silhouette_score']),
            'misclassified_rate': convert_numpy_types(result['misclassified_rate']),
            'cluster_analysis': convert_numpy_types(result['cluster_analysis']),
            'total_samples': result['total_samples'],
            'n_classes': result['n_classes'],
            'perplexity': result['perplexity'],
            'time_elapsed': result['time_elapsed']
        }

        with open(save_file, 'w') as f:
            json.dump(serializable_result, f, indent=2)

        print(f"   Analysis data saved: {save_file}")

        # Save text report
        report_file = self.save_dir / f"tsne_report_epoch_{epoch:03d}.txt"
        with open(report_file, 'w') as f:
            f.write(self.generate_report(epoch))

        print(f"   Analysis report saved: {report_file}")

    def compare_epochs(self, epochs):
        """
        Compare T-SNE results across multiple epochs

        Args:
            epochs: List of epochs to compare

        Returns:
            dict: Comparison results
        """
        comparison = {
            'epochs': epochs,
            'metrics': {
                'silhouette_scores': {},
                'misclassified_rates': {},
                'total_samples': {}
            }
        }

        for epoch in epochs:
            if epoch in self.tsne_results:
                result = self.tsne_results[epoch]
                comparison['metrics']['silhouette_scores'][epoch] = result['silhouette_score']
                comparison['metrics']['misclassified_rates'][epoch] = result['misclassified_rate']
                comparison['metrics']['total_samples'][epoch] = result['total_samples']

        return comparison


if __name__ == "__main__":
    # Simple test
    print("T-SNE Analyzer Test")

    # Create simulated data
    np.random.seed(42)
    features = np.random.randn(500, 50)
    labels = np.random.randint(0, 8, 500)
    predictions = labels.copy()

    # Add some noise
    noise_indices = np.random.choice(500, 50, replace=False)
    predictions[noise_indices] = np.random.randint(0, 8, 50)

    # Initialize analyzer
    analyzer = TSNEAnalyzer()

    # Add data
    analyzer.epoch_features[0] = list(features)
    analyzer.epoch_labels[0] = list(labels)
    analyzer.epoch_predictions[0] = list(predictions)

    # Run T-SNE
    result = analyzer.run_tsne(0)

    if result:
        # Generate report
        print(analyzer.generate_report(0))

        # Plot figure
        analyzer.plot_tsne(0, save=True, show=False)

        # Save analysis
        analyzer.save_analysis(0)

    print("\nTest completed")
