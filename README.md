# FluidCNN v6 - Mesh Quality Assessment Model

## Project Overview

FluidCNN v6 is a deep learning-based 2D structured mesh quality assessment classification model, specifically designed for classifying mesh cell quality levels in Computational Fluid Dynamics (CFD). This project is trained and evaluated on the **NACA-Market dataset**.

## Key Features

- **8-class Quality Classification**: Classifies mesh cells into 8 quality levels (Label 0 - Label 7)
- **Multi-size Support**: Supports dynamic input size mesh data
- **Innovative Architecture**:
  - Top-16-row Feature Enhancement Parallel Branch (Top16RowModule)
  - Multi-scale Residual Block (MultiBranchResidualBlock)
  - Channel Attention Mechanism (SEBlock)
  - Weighted Feature Fusion (WeightedFeatureFusion)
- **Complete Training Pipeline**: Including data augmentation, early stopping, learning rate scheduling
- **Rich Analysis Tools**: T-SNE visualization, confusion matrix analysis, data quality detection

## Model Architecture

```
Input (H, W, 11)
    │
    ├──────────────────────────────────────────┐
    │                                          │
    ▼                                          ▼
[Parallel Branch 1: Top16RowModule]   [Main Branch: Multi-branch Residual Network]
    │                                          │
    │  Extract top 16 rows                     │
    │  1×N convolution                         │
    │  MLP feature scaling                     │
    │  Output: 128-dim feature                 │
    │                                          │
    │                                          ▼
    │                               Multi-stage Residual Network (8 stages)
    │                               Output: 512-dim global feature
    │                                          │
    └──────────────────────────────────────────┤
                                               ▼
                                    [Weighted Feature Fusion]
                                    640-dim fused feature
                                               │
                                               ▼
                                    [Classifier]
                                    640 → 256 → 128 → 8
                                               │
                                               ▼
                                    Output: 8-class classification
```

## Installation

```bash
pip install torch numpy scikit-learn matplotlib seaborn scipy pandas
```

## Data Format

### Input Data Location
```
../cnn_input_data/
└── batch_data.pt  # Organized by group*_8x*/batch_data.pt format
```

### Input Feature Channels (11 channels)
| Channel | Description |
|---------|-------------|
| 0-7 | Original features (8 channels) |
| 8 | Mask channel |
| 9 | X coordinate |
| 10 | Y coordinate |

### Mesh Quality Assessment Metrics
- **Skew Angle (theta)**: Key metric for mesh quality
- **Cell Area (area)**
- **Aspect Ratio**
- **Deviation Angle**
- **Edge Lengths and Edge Vectors**

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| BATCH_SIZE | 32 | Batch size |
| EPOCHS | 200 | Number of training epochs |
| LEARNING_RATE | 3e-4 | Learning rate |
| WEIGHT_DECAY | 1e-4 | Weight decay |
| DROPOUT_RATE | 0.3 | Dropout ratio |
| CONV_KERNEL_SIZE | 10 | 1×N convolution kernel size |
| PATIENCE | 60 | Early stopping patience |

## Usage

### 1. Run Complete Training Pipeline

```python
python main.py
```

### 2. Main Functions

- **Model Training**: Complete training, validation, and testing pipeline
- **T-SNE Analysis**: Feature dimensionality reduction visualization at specified epochs
- **Confusion Matrix**: Save and analyze classification results every 5 epochs
- **Model Saving**: Save best model based on validation accuracy

### 3. Analysis Tools

```python
# T-SNE visualization analysis
python tsne_analyzer.py

# Confusion matrix analysis
python confusion_matrix_analyzer.py

# Data quality analysis
python analyze_data_quality.py

# Suspicious data tracing
python trace_suspicious_data.py

# Plot confusion matrix based on best model
python plot_best_confusion_matrix.py
```

## Project Structure

```
fluid_cnn_v6/
├── __init__.py                 # Package initialization
├── main.py                     # Main training script
├── model.py                    # FluidCNN v6 model definition
├── trainer.py                  # Trainer module
├── data_loader.py              # Data loading module
├── utils.py                    # Utility functions
├── preprocess.py               # Mesh preprocessing module
├── turntohw.py                 # Data format conversion tool
├── tsne_analyzer.py            # T-SNE analysis module
├── confusion_matrix_analyzer.py # Confusion matrix analyzer
├── plot_best_confusion_matrix.py # Plot confusion matrix from best.txt
├── trace_suspicious_data.py    # Suspicious data tracing script
├── analyze_data_quality.py     # Data quality analysis script
└── README.md                   # This documentation
```

## Core Components

### Top16RowModule
A parallel branch module specifically designed to process the top 16 rows of mesh, using 1×N convolution to extract features, particularly suitable for capturing flow characteristics in the mesh inlet region.

### MultiBranchResidualBlock
Multi-scale feature extraction module with 3 parallel branches of different convolution kernel sizes: (2,7), (3,10), (4,14), capable of capturing mesh features at different scales.

### SEBlock
Squeeze-and-Excitation channel attention mechanism that adaptively adjusts channel weights to enhance responses of important feature channels.

### WeightedFeatureFusion
Learnable weighted feature fusion mechanism supporting two fusion methods:
- `weighted_sum`: Weighted summation
- `concat`: Concatenation followed by linear transformation

## Output Description

After training, the following files will be generated:

- `best_model.pth`: Best model checkpoint
- `training_history.png`: Training history curves
- `confusion_matrix_*.png`: Confusion matrix plots
- `tsne_*.png`: T-SNE visualization plots
- `best.txt`: Best metrics record

## License

This project is for research and educational purposes only.

## References

This model is based on research related to CFD mesh quality assessment, focusing on geometric quality metrics of mesh cells including skew angle, area, aspect ratio, and other key parameters.
