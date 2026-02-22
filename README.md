# GeneralME -  generalizable mesh quality evaluation method using dual-branch networks

## About
A generalizable mesh quality evaluator suitable for datasets of all sizes.

## Project Overview

GeneralME is a deep learning-based structured mesh quality classification model, specifically designed for evaluating mesh cell quality in Computational Fluid Dynamics (CFD). This project is trained and evaluated on the **NACA-Market dataset**. The datasets during the current study are available in https://github.com/chenxinhai1234/NACA-Market.

##

![image](https://github.com/ElectronicRain/GeneralME/blob/main/model.png)

## Installation

```bash
pip install torch numpy scikit-learn matplotlib seaborn scipy pandas
```

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
GeneralME/
├── __init__.py                 # Package initialization
├── main.py                     # Main training script
├── model.py                    # GeneralME model definition
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












