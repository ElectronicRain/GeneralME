#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd


def analyze_tsne_results(tsne_json_path):

    print(f"Analyzing T-SNE results: {tsne_json_path}")

    with open(tsne_json_path, 'r') as f:
        data = json.load(f)

    # Basic information
    total_samples = data['total_samples']
    n_classes = data['n_classes']
    misclassified_count = int(np.sum(data['misclassified']))
    misclassified_rate = data['misclassified_rate']

    print(f"   Total samples: {total_samples}")
    print(f"   Misclassified count: {misclassified_count}")
    print(f"   Misclassification rate: {misclassified_rate:.2%}")

    # Analyze clusters
    cluster_analysis = data['cluster_analysis']
    suspicious_clusters = []
    confusion_pairs = []

    for cluster_id, info in cluster_analysis.items():
        cluster_id = int(cluster_id)
        majority_class = info['majority_class']
        majority_percentage = info['majority_percentage']
        cluster_size = info['cluster_size']

        # Purity below 85% is considered suspicious
        if majority_percentage < 0.85:
            suspicious_clusters.append({
                'cluster_id': cluster_id,
                'predicted_class': cluster_id,
                'actual_majority': majority_class,
                'purity': majority_percentage,
                'size': cluster_size,
                'mixed_samples': cluster_size - info['majority_count']
            })

        # Record confusion pairs
        for misclass, stats in info['class_distribution'].items():
            if misclass != majority_class:
                confusion_pairs.append({
                    'true_class': majority_class,
                    'predicted_class': cluster_id,
                    'misclass_as': misclass,
                    'count': stats['count'],
                    'percentage': stats['percentage']
                })

    return {
        'total_samples': total_samples,
        'misclassified_count': misclassified_count,
        'misclassified_rate': misclassified_rate,
        'suspicious_clusters': suspicious_clusters,
        'confusion_pairs': confusion_pairs,
        'cluster_analysis': cluster_analysis
    }


def analyze_error_patterns(test_set_results):
    """Analyze error patterns"""
    print("\nAnalyzing misclassification patterns...")

    # Count errors by true class
    error_by_true_class = defaultdict(list)
    for error in test_set_results['confusion_pairs']:
        true_class = error['true_class']
        misclass_as = error['misclass_as']
        count = error['count']
        error_by_true_class[true_class].append({
            'mistaken_as': misclass_as,
            'count': count
        })

    # Find most easily confused classes
    most_confused = []
    for true_class, errors in error_by_true_class.items():
        total_errors = sum(e['count'] for e in errors)
        if total_errors > 10:  # Only focus on classes with more than 10 errors
            most_confused.append({
                'true_class': true_class,
                'total_errors': total_errors,
                'confusions': sorted(errors, key=lambda x: -x['count'])
            })

    most_confused.sort(key=lambda x: -x['total_errors'])

    return most_confused


def find_problematic_samples(log_dir, threshold_purity=0.85):
    """
    Find problematic samples

    Args:
        log_dir: Log directory path
        threshold_purity: Purity threshold

    Returns:
        dict: Analysis results
    """
    log_dir = Path(log_dir)
    tsne_analysis_dir = log_dir / "tsne_analysis"

    if not tsne_analysis_dir.exists():
        print(f"T-SNE analysis directory does not exist: {tsne_analysis_dir}")
        return None

    # Find final T-SNE results (usually epoch 150 or the largest epoch)
    tsne_files = list(tsne_analysis_dir.glob("tsne_analysis_epoch_*.json"))
    if not tsne_files:
        print("T-SNE analysis files not found")
        return None

    # Use the largest epoch
    tsne_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    final_tsne = tsne_files[-1]

    print(f"Using T-SNE analysis file: {final_tsne.name}")

    # Analyze final results
    final_results = analyze_tsne_results(final_tsne)

    # Analyze early results (for comparison)
    if len(tsne_files) > 1:
        early_tsne = tsne_files[0]
        print(f"\nComparing with early T-SNE analysis: {early_tsne.name}")
        early_results = analyze_tsne_results(early_tsne)

        # Compare confusion improvement
        final_confusion = Counter()
        for cp in final_results['confusion_pairs']:
            final_confusion[(cp['true_class'], cp['misclass_as'])] = cp['count']

        early_confusion = Counter()
        for cp in early_results['confusion_pairs']:
            early_confusion[(cp['true_class'], cp['misclass_as'])] = cp['count']

        # Find most persistent confusions
        persistent_confusion = []
        for key in final_confusion:
            if key in early_confusion:
                final_count = final_confusion[key]
                early_count = early_confusion[key]
                if final_count > 5 and final_count / max(early_count, 1) > 0.5:
                    persistent_confusion.append({
                        'true_class': key[0],
                        'mistaken_as': key[1],
                        'final_count': final_count,
                        'early_count': early_count,
                        'improvement': (early_count - final_count) / early_count
                    })

        persistent_confusion.sort(key=lambda x: -x['final_count'])
        final_results['persistent_confusion'] = persistent_confusion[:10]

    # Analyze error patterns
    most_confused = analyze_error_patterns(final_results)
    final_results['most_confused_classes'] = most_confused

    return final_results


def generate_deletion_list(analysis_results, output_path="deletion_list.json"):
    """
    Generate list of data points to delete

    Args:
        analysis_results: Analysis results
        output_path: Output file path

    Returns:
        list: List of data points to delete
    """
    print(f"\nGenerating deletion list...")

    deletion_list = []

    # Extract samples from suspicious clusters
    for cluster in analysis_results['suspicious_clusters']:
        cluster_id = cluster['cluster_id']
        actual_class = cluster['actual_majority']

        # Simulated data: these samples should be marked in original data
        # In actual use, need to map to specific file locations
        deletion_list.append({
            'reason': f'Predicted as Class {cluster_id} in cluster {cluster_id}, but actually belongs to Class {actual_class}',
            'predicted_class': cluster_id,
            'actual_class': actual_class,
            'purity': cluster['purity'],
            'mixed_samples': cluster['mixed_samples'],
            'recommendation': 'Requires manual review to confirm if it is a labeling error'
        })

    # Extract from persistent confusions
    if 'persistent_confusion' in analysis_results:
        for confusion in analysis_results['persistent_confusion']:
            deletion_list.append({
                'reason': f'Class {confusion["true_class"]} is consistently misclassified as Class {confusion["mistaken_as"]}',
                'true_class': confusion['true_class'],
                'mistaken_as': confusion['mistaken_as'],
                'error_count': confusion['final_count'],
                'recommendation': 'Check if these two classes are confused in original labeling'
            })

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(deletion_list, f, indent=2, ensure_ascii=False)

    print(f"Deletion list saved: {output_path}")
    print(f"   Total suspicious data points/patterns: {len(deletion_list)}")

    return deletion_list


def create_cleanup_script(deletion_list, script_path="cleanup_suspicious_data.py"):
    """
    Create data cleanup script

    Args:
        deletion_list: Deletion list
        script_path: Script path
    """
    print(f"\nGenerating cleanup script...")

    script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Cleanup Script - Based on T-SNE Analysis Results

WARNING: Please backup data before use!
WARNING: This script is based on model prediction results, for reference only, please manually review before use!
"""

import json
import shutil
from pathlib import Path

def backup_original_data():
    """Backup original data"""
    print("Backing up original data...")
    backup_dir = Path("../cnn_input_data_backup")
    if not backup_dir.exists():
        shutil.copytree("../cnn_input_data", backup_dir)
        print(f"   Data backed up to: {{backup_dir}}")
    else:
        print(f"   Backup directory already exists: {{backup_dir}}")
    return backup_dir

def create_problematic_samples_list():
    """Create problematic samples list"""
    print("\\nCreating problematic samples list...")

    deletion_list = {json.dumps(deletion_list, indent=2, ensure_ascii=False)}

    # Save as separate file
    with open("problematic_samples.json", 'w', encoding='utf-8') as f:
        json.dump(deletion_list, f, indent=2, ensure_ascii=False)

    return deletion_list

def analyze_confusion_patterns(deletion_list):
    """Analyze confusion patterns"""
    print("\\nAnalyzing confusion patterns...")

    confusion_stats = {{}}
    for item in deletion_list:
        if 'reason' in item:
            if 'predicted as' in item['reason'].lower():
                # Extract class information
                parts = item['reason'].split('predicted as')
                if len(parts) > 1:
                    true_part = parts[0].replace('in cluster', '').strip()
                    if ' ' in true_part:
                        true_class = true_part.split(' ')[0].replace('Class', '').strip()
                        confusion_key = f"Class {{true_class}} confusion"
                        confusion_stats[confusion_key] = confusion_stats.get(confusion_key, 0) + 1

    print("\\nConfusion Statistics:")
    for pattern, count in sorted(confusion_stats.items(), key=lambda x: -x[1]):
        print(f"   {{pattern}}: {{count}} instances")

def main():
    print("="*80)
    print("Data Cleanup Tool")
    print("="*80)

    print("\\nImportant Reminders:")
    print("   1. This tool is based on model prediction results, may have false positives")
    print("   2. Some 'issues' may be caused by data augmentation or preprocessing")
    print("   3. Consider using other methods (like data augmentation) to improve model performance first")
    print("   4. Only delete data if confirmed to be labeling errors")

    backup_dir = backup_original_data()
    deletion_list = create_problematic_samples_list()
    analyze_confusion_patterns(deletion_list)

    print("\\n" + "="*80)
    print("Cleanup preparation completed")
    print("="*80)
    print("\\nGenerated files:")
    print("   - problematic_samples.json  # Problematic samples/patterns list")
    print(f"   - {{backup_dir}}             # Original data backup")
    print("\\nNext Steps:")
    print("   1. Manually review problems in problematic_samples.json")
    print("   2. Confirm which are true labeling errors")
    print("   3. If needed, write specific deletion scripts")
    print("   4. Retrain model to verify improvements")

if __name__ == "__main__":
    main()
'''

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    print(f"Cleanup script generated: {script_path}")


def generate_report(analysis_results, output_path="data_quality_analysis.md"):
    """
    Generate detailed analysis report

    Args:
        analysis_results: Analysis results
        output_path: Output report path
    """
    print(f"\nGenerating analysis report...")

    report = f"""# Data Quality Deep Analysis Report

## Executive Summary

Based on T-SNE cluster analysis, we have identified potential data quality issues.

### Key Findings
- **Total Samples**: {analysis_results['total_samples']}
- **Misclassified Count**: {analysis_results['misclassified_count']}
- **Misclassification Rate**: {analysis_results['misclassified_rate']:.2%}
- **Suspicious Clusters**: {len(analysis_results['suspicious_clusters'])}

## Detailed Analysis

### 1. Suspicious Clusters

The following clusters have purity below 85% and may have labeling errors:

"""

    if analysis_results['suspicious_clusters']:
        for cluster in analysis_results['suspicious_clusters']:
            report += f"""
#### Cluster {cluster['cluster_id']}
- **Predicted Class**: Class {cluster['predicted_class']}
- **Actual Majority Class**: Class {cluster['actual_majority']}
- **Cluster Purity**: {cluster['purity']:.1%}
- **Cluster Size**: {cluster['size']} samples
- **Mixed Samples**: {cluster['mixed_samples']} samples
- **Recommendation**: Manual review needed for these mixed samples to check for labeling errors

"""
    else:
        report += "\nAll clusters have purity above 85%, no obvious labeling errors found.\n"

    # Confusion analysis
    report += "\n### 2. Confusion Pattern Analysis\n\n"

    if 'most_confused_classes' in analysis_results:
        for item in analysis_results['most_confused_classes'][:5]:
            true_class = item['true_class']
            total_errors = item['total_errors']
            report += f"""
#### Class {true_class}
- **Total Errors**: {total_errors}
- **Main Confusion Targets**:
"""
            for conf in item['confusions'][:3]:
                report += f"  - Misclassified as Class {conf['mistaken_as']}: {conf['count']} times\n"

    # Persistent confusions
    if 'persistent_confusion' in analysis_results:
        report += "\n### 3. Persistent Confusion Patterns\n\n"
        report += "The following confusion patterns persist throughout training and may be labeling issues:\n\n"

        for confusion in analysis_results['persistent_confusion'][:5]:
            report += f"""- **Class {confusion['true_class']} <-> Class {confusion['mistaken_as']}**
  - Final error count: {confusion['final_count']}
  - Improvement: {confusion['improvement']:.1%}
  - Recommendation: Check boundary samples of these two classes

"""

    # Recommendations
    report += """
## Recommended Actions

### 1. Immediate Actions
- [ ] Manually review mixed samples in suspicious clusters above
- [ ] Check boundary samples of Class 0-7 confusion
- [ ] Verify labeling consistency

### 2. Data Cleaning
If labeling errors are confirmed:
- [ ] Backup original data
- [ ] Delete or correct mislabeled data points
- [ ] Retrain model to verify improvements

### 3. Alternative Approaches
Before deleting data, consider:
- [ ] Use stronger data augmentation
- [ ] Adjust loss function (weighted cross-entropy, Focal Loss, etc.)
- [ ] Hard Example Mining
- [ ] Increase model capacity or training time

### 4. Preventive Measures
- [ ] Establish stricter data labeling process
- [ ] Add multi-annotator consistency checks
- [ ] Use active learning to discover difficult samples

## Notes

WARNING **Important Reminders**:
1. Model predictions are not the same as true labels
2. Some "errors" may be caused by data preprocessing
3. Please backup and confirm with multiple sources before deleting data
4. Consider data augmentation as the first choice

---

**Report Generation Time**: {pd.Timestamp.now()}
**Based on Model**: FluidCNN v5 trained
**Analysis Based on**: T-SNE cluster analysis results
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Analysis report saved: {output_path}")


def main():
    """Main function"""
    print("=" * 80)
    print("Data Quality Issue Analysis Tool")
    print("=" * 80)

    # Find latest training log
    log_dirs = list(Path("logs").glob("*/"))
    if not log_dirs:
        print("Training log directory not found")
        return

    # Sort by modification time, get the latest
    log_dirs.sort(key=lambda x: x.stat().st_mtime)
    latest_log = log_dirs[-1]

    print(f"\nAnalyzing log directory: {latest_log.name}")

    # Analyze problematic samples
    results = find_problematic_samples(latest_log)

    if results is None:
        print("Analysis failed")
        return

    # Generate report
    generate_report(results, f"{latest_log.name}_data_quality_analysis.md")

    # Generate deletion list
    deletion_list = generate_deletion_list(results, f"{latest_log.name}_deletion_list.json")

    # Generate cleanup script
    create_cleanup_script(deletion_list, "cleanup_suspicious_data.py")

    print("\n" + "=" * 80)
    print("Analysis completed!")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"   1. {latest_log.name}_data_quality_analysis.md  # Detailed analysis report")
    print(f"   2. {latest_log.name}_deletion_list.json        # Deletion list")
    print("   3. cleanup_suspicious_data.py                   # Cleanup script")
    print("\nPlease read the report carefully and be cautious when deleting data!")


if __name__ == "__main__":
    main()
