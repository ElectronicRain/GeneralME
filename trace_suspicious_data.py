#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import json
from pathlib import Path
import sys
import pandas as pd
from collections import defaultdict
import pickle

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from model import FluidCNNEnhanced
from data_loader import MultiSizeMeshDataset, pad_collate
from torch.utils.data import DataLoader


class DataPointTracer:
    """Data Point Tracer - Identifying potentially mislabeled data points"""

    def __init__(self, model_path, tsne_analysis_path, base_data_dir="../cnn_input_data"):
        """
        Initialize tracer

        Args:
            model_path: Path to trained model
            tsne_analysis_path: Path to T-SNE analysis result JSON file
            base_data_dir: Original data directory
        """
        self.model_path = Path(model_path)
        self.tsne_analysis_path = Path(tsne_analysis_path)
        self.base_data_dir = Path(base_data_dir)

        # Load model
        print("Loading trained model...")
        self.model = self._load_model()
        print("Model loaded successfully")

        # Load T-SNE analysis results
        print("Loading T-SNE analysis results...")
        self.tsne_data = self._load_tsne_analysis()
        print("T-SNE analysis results loaded successfully")

        # Load original dataset
        print("Loading original dataset...")
        self.dataset = MultiSizeMeshDataset(base_data_dir, feature_indices=[0, 1, 2, 3, 4, 5, 6, 7])
        print(f"Dataset loaded successfully, total samples: {len(self.dataset)}")

    def _load_model(self):
        """Load trained model"""
        model = FluidCNNEnhanced(input_channels=43, num_classes=8)  # 40 features + 3 coordinates
        state_dict = torch.load(self.model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_tsne_analysis(self):
        """Load T-SNE analysis results"""
        with open(self.tsne_analysis_path, 'r') as f:
            return json.load(f)

    def identify_mislabeled_points(self, threshold_purity=0.85):
        """
        Identify potentially mislabeled data points

        Args:
            threshold_purity: Cluster purity threshold, points below this threshold are considered suspicious

        Returns:
            dict: Contains information about suspicious data points
        """
        print(f"\nIdentifying mislabeled data points...")
        print(f"   Using cluster purity threshold: {threshold_purity:.1%}")

        cluster_analysis = self.tsne_data['cluster_analysis']
        suspicious_points = defaultdict(list)

        total_suspicious = 0

        for cluster_id, info in cluster_analysis.items():
            cluster_id = int(cluster_id)
            majority_class = info['majority_class']
            majority_percentage = info['majority_percentage']

            # If purity is below threshold, mark as suspicious
            if majority_percentage < threshold_purity:
                print(f"\nCluster {cluster_id} (predicted as Class {cluster_id}):")
                print(f"   Majority class: Class {majority_class} ({info['majority_count']} samples, {majority_percentage:.1%})")

                # Find misclassified samples
                class_distribution = info['class_distribution']
                for misclassified_class, stats in class_distribution.items():
                    if misclassified_class != majority_class:
                        count = stats['count']
                        percentage = stats['percentage']
                        print(f"   Mixed in Class {misclassified_class}: {count} samples ({percentage:.1%})")

                        # Mark these samples
                        if misclassified_class not in suspicious_points:
                            suspicious_points[misclassified_class] = []
                        suspicious_points[misclassified_class].append({
                            'cluster_id': cluster_id,
                            'predicted_class': cluster_id,
                            'count': count,
                            'percentage': percentage,
                            'should_be': majority_class
                        })

                total_suspicious += sum(
                    stats['count'] for cls, stats in class_distribution.items()
                    if cls != majority_class
                )

        print(f"\nTotal suspicious samples found: {total_suspicious}")
        print(f"   Involves {len(suspicious_points)} classes")

        return dict(suspicious_points)

    def extract_suspicious_data_indices(self, dataset_split='test', max_samples=1000):
        """
        Extract indices of suspicious data points in original dataset

        Args:
            dataset_split: 'train', 'val', or 'test'
            max_samples: Maximum number of samples

        Returns:
            list: List of suspicious sample indices
        """
        print(f"\nExtracting suspicious data points from {dataset_split} split...")

        # Reload dataset and split
        from sklearn.model_selection import train_test_split

        indices = list(range(len(self.dataset)))
        _, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

        if dataset_split == 'train':
            subset_indices = [i for i in indices if i not in temp_idx]
        elif dataset_split == 'val':
            subset_indices = val_idx
        elif dataset_split == 'test':
            subset_indices = test_idx
        else:
            raise ValueError("dataset_split must be 'train', 'val', or 'test'")

        print(f"   {dataset_split} split size: {len(subset_indices)}")

        # Create data loader
        subset = torch.utils.data.Subset(self.dataset, subset_indices[:max_samples])
        loader = DataLoader(subset, batch_size=32, shuffle=False, collate_fn=pad_collate)

        # Collect prediction results
        all_predictions = []
        all_labels = []
        all_indices = []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(loader):
                outputs = self.model(inputs)
                predictions = torch.argmax(outputs, dim=1)

                start_idx = batch_idx * 32
                batch_indices = subset_indices[start_idx:start_idx + len(labels)]

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_indices.extend(batch_indices)

        # Find misclassified samples
        misclassified_mask = np.array(all_predictions) != np.array(all_labels)
        misclassified_indices = np.array(all_indices)[misclassified_mask]
        misclassified_preds = np.array(all_predictions)[misclassified_mask]
        misclassified_labels = np.array(all_labels)[misclassified_mask]

        print(f"   Found {len(misclassified_indices)} misclassified samples")

        return misclassified_indices.tolist(), misclassified_preds.tolist(), misclassified_labels.tolist()

    def generate_removal_script(self, output_path="remove_suspicious_data.py"):
        """
        Generate script to remove suspicious data

        Args:
            output_path: Output script path
        """
        print(f"\nGenerating data removal script...")

        # Get suspicious samples in test set
        test_indices, test_preds, test_labels = self.extract_suspicious_data_indices('test')

        # Get suspicious samples in validation set
        val_indices, val_preds, val_labels = self.extract_suspicious_data_indices('val')

        script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-generated Data Deletion Script
Delete mislabeled or difficult-to-classify data points

WARNING: Please backup original data before use!
WARNING: This script is based on T-SNE analysis results, please manually review before use!
"""

import torch
import numpy as np
from pathlib import Path
import shutil
import json

def backup_data():
    """Backup original data"""
    print("Backing up original data...")
    backup_dir = Path("../cnn_input_data_backup")
    if not backup_dir.exists():
        shutil.copytree("../cnn_input_data", backup_dir)
        print(f"   Data backed up to: {{backup_dir}}")
    else:
        print(f"   Backup directory already exists: {{backup_dir}}")

def remove_suspicious_samples():
    """Remove suspicious samples"""
    print("\\nRemoving suspicious samples...")

    # Statistics
    total_removed = 0
    removed_by_group = {{}}

    # Suspicious samples in test set
    test_suspicious = [
        # (group_id, sample_index, reason)
'''

        # Add suspicious samples from test set
        if test_indices:
            for idx, pred, label in zip(test_indices, test_preds, test_labels):
                group_id = (idx // 1000) + 1  # Assume approximately 1000 samples per group
                sample_idx = idx % 1000
                reason = f"Predicted as Class {{pred}} in test set, true label is Class {{label}}"
                script_content += f'        ({group_id}, {sample_idx}, "{reason}"),\\n'

        # Add suspicious samples from validation set
        if val_indices:
            for idx, pred, label in zip(val_indices, val_preds, val_labels):
                group_id = (idx // 1000) + 1
                sample_idx = idx % 1000
                reason = f"Predicted as Class {{pred}} in validation set, true label is Class {{label}}"
                script_content += f'        ({group_id}, {sample_idx}, "{reason}"),\\n'

        script_content += f'''    ]

    # Execute deletion
    for group_id, sample_idx, reason in test_suspicious:
        group_dir = Path(f"../cnn_input_data/group{{group_id}}_8x*/")
        group_path = list(group_dir.glob("*"))[0] if group_dir.exists() else None

        if group_path and group_path.exists():
            print(f"   Deleting: group{{group_id}}/sample_{{sample_idx}} ({{reason}})")
            # Here should implement specific deletion logic
            # For example: delete corresponding files or mark as invalid
            total_removed += 1
            removed_by_group[group_id] = removed_by_group.get(group_id, 0) + 1

    print(f"\\nTotal suspicious samples removed: {{total_removed}}")
    for group_id, count in removed_by_group.items():
        print(f"   Group {{group_id}}: {{count}} samples")

def main():
    print("="*80)
    print("Data Deletion Tool")
    print("="*80)
    print("\\nWARNING: This operation is irreversible!")
    print("   Please ensure you have:")
    print("   1. Backed up the original data")
    print("   2. Manually reviewed the samples to be deleted")
    print("   3. Confirmed to proceed with deletion")

    confirm = input("\\nContinue? (enter 'yes' to confirm): ")
    if confirm != 'yes':
        print("Operation cancelled")
        return

    backup_data()
    remove_suspicious_samples()

    print("\\n" + "="*80)
    print("Data deletion completed")
    print("="*80)

if __name__ == "__main__":
    main()
'''

        with open(output_path, 'w') as f:
            f.write(script_content)

        print(f"Deletion script generated: {output_path}")
        print(f"   Suspicious sample statistics:")
        print(f"   - Test set: {len(test_indices)} samples")
        print(f"   - Validation set: {len(val_indices)} samples")
        print(f"   - Total: {len(test_indices) + len(val_indices)} samples")

    def generate_report(self, output_path="suspicious_data_report.md"):
        """
        Generate detailed report for suspicious data points

        Args:
            output_path: Output report path
        """
        print(f"\nGenerating suspicious data point report...")

        # Get misclassified samples
        test_indices, test_preds, test_labels = self.extract_suspicious_data_indices('test')
        val_indices, val_preds, val_labels = self.extract_suspicious_data_indices('val')

        report = f"""# Suspicious Data Points Report

## Generation Information
- **Model**: {self.model_path.name}
- **T-SNE Analysis**: {self.tsne_analysis_path.name}
- **Total Dataset Size**: {len(self.dataset)}
- **Generation Time**: {pd.Timestamp.now()}

## Suspicious Sample Statistics

### Test Set
- **Misclassified Samples**: {len(test_indices)}
- **Misclassification Rate**: {len(test_indices) / len(test_indices) * 100:.2f}% (relative to test set)

"""

        # Count errors by class
        error_stats = defaultdict(int)
        for _, pred, label in zip(test_indices, test_preds, test_labels):
            error_stats[f"True:{label} -> Pred:{pred}"] += 1

        if error_stats:
            report += "\n### Test Set Misclassification Statistics\n\n"
            for error_type, count in sorted(error_stats.items(), key=lambda x: -x[1]):
                report += f"- **{error_type}**: {count} samples\n"

        report += f"""


### Validation Set
- **Misclassified Samples**: {len(val_indices)}
- **Misclassification Rate**: {len(val_indices) / len(val_indices) * 100:.2f}% (relative to validation set)

"""

        # Validation set error statistics
        val_error_stats = defaultdict(int)
        for _, pred, label in zip(val_indices, val_preds, val_labels):
            val_error_stats[f"True:{label} -> Pred:{pred}"] += 1

        if val_error_stats:
            report += "\n### Validation Set Misclassification Statistics\n\n"
            for error_type, count in sorted(val_error_stats.items(), key=lambda x: -x[1]):
                report += f"- **{error_type}**: {count} samples\n"

        report += f"""


## T-SNE Cluster Analysis Results

Based on T-SNE analysis, the following clusters have mixing issues:

"""

        # Add cluster analysis results
        cluster_analysis = self.tsne_data['cluster_analysis']
        for cluster_id, info in cluster_analysis.items():
            majority_class = info['majority_class']
            majority_percentage = info['majority_percentage']

            if majority_percentage < 0.9:  # Show clusters with purity below 90%
                report += f"""### Cluster {cluster_id}
- **Majority Class**: Class {majority_class}
- **Purity**: {majority_percentage:.1%}
- **Cluster Size**: {info['cluster_size']} samples

**Mixed Classes**:
"""
                for misclassified_class, stats in info['class_distribution'].items():
                    if misclassified_class != majority_class:
                        count = stats['count']
                        percentage = stats['percentage']
                        report += f"- Class {misclassified_class}: {count} samples ({percentage:.1%})\n"

                report += "\n"

        report += f"""


## Recommended Actions

### 1. Manual Review
Please manually review the following misclassified samples:
- Test set: {len(test_indices)} samples
- Validation set: {len(val_indices)} samples

### 2. Data Cleaning Script
Run the generated deletion script:
```bash
python {Path(output_path).parent / 'remove_suspicious_data.py'}
```

### 3. Retraining
After removing suspicious data, retrain the model to verify improvements.

## Notes

WARNING **Important Reminders**:
1. Before deleting any data, please backup the original data
2. Some "misclassifications" may be model performance issues, not labeling errors
3. Consider using data augmentation or adjusting loss function first, rather than directly deleting data
4. Please manually review each sample before deleting data

---

**Report Generated**: {pd.Timestamp.now()}
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Report generated: {output_path}")
        print(f"   Contains {len(test_indices)} test set error samples")
        print(f"   Contains {len(val_indices)} validation set error samples")


def main():
    """Main function"""
    print("=" * 80)
    print("Data Point Tracing Tool")
    print("=" * 80)

    # Find latest model and analysis results
    log_dir = Path("logs")
    latest_log = sorted(log_dir.glob("*/"), key=lambda x: x.stat().st_mtime)[-1]
    model_path = latest_log / "01234567_best_cnn_v5.pt"

    # Find T-SNE analysis results
    tsne_dir = latest_log / "tsne_analysis"
    if tsne_dir.exists():
        tsne_analysis_files = list(tsne_dir.glob("tsne_analysis_epoch_*.json"))
        if tsne_analysis_files:
            tsne_analysis_path = tsne_analysis_files[-1]  # Use the latest
        else:
            print("T-SNE analysis results not found")
            return
    else:
        print("T-SNE analysis directory not found")
        return

    print(f"\nUsing resources:")
    print(f"   Model: {model_path}")
    print(f"   T-SNE Analysis: {tsne_analysis_path}")

    # Create tracer
    tracer = DataPointTracer(model_path, tsne_analysis_path)

    # Identify suspicious points
    suspicious_points = tracer.identify_mislabeled_points(threshold_purity=0.85)

    # Generate report
    tracer.generate_report("suspicious_data_report.md")

    # Generate deletion script
    tracer.generate_removal_script("remove_suspicious_data.py")

    print("\n" + "=" * 80)
    print("Data tracing completed!")
    print("=" * 80)
    print("\nGenerated files:")
    print("   1. suspicious_data_report.md  - Detailed analysis report")
    print("   2. remove_suspicious_data.py - Data deletion script")
    print("\nPlease backup original data and manually review before deleting data!")


if __name__ == "__main__":
    main()
