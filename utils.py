#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions module
Contains logging output, data visualization and other auxiliary functions
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import os


class TeeOutput:
    """
    Auxiliary class to write standard output to multiple targets (console + log file) simultaneously
    """

    def __init__(self, *files):
        """
        Initialize TeeOutput

        Args:
            *files: List of file handles to write to
        """
        self.files = files

    def write(self, text):
        """Write text content to all files and flush immediately"""
        for file in self.files:
            file.write(text)
            file.flush()

    def flush(self):
        """Ensure all file handles are flushed"""
        for file in self.files:
            file.flush()


def setup_logging(run_dir=None):
    """
    Set up logging output

    Args:
        run_dir: Log directory, automatically created if None

    Returns:
        tuple: (log_file_path, original_stdout, log_file_handle)
    """
    if run_dir is None:
        now = datetime.now()
        temp_dir_suffix = now.strftime('%m%d_%H%M%S')
        run_dir = Path("logs") / f"pending_{temp_dir_suffix}"
        run_dir.mkdir(parents=True, exist_ok=True)

    log_filename = run_dir / "training.log"
    log_file = open(log_filename, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(original_stdout, log_file)

    return log_filename, original_stdout, log_file


def cleanup_logging(original_stdout, log_file_handle, run_dir, final_test_acc, base_name=None):
    """
    Clean up logging output and rename log file

    Args:
        original_stdout: Original stdout
        log_file_handle: Log file handle
        run_dir: Run directory
        final_test_acc: Final test accuracy
        base_name: Base file name (generated based on test accuracy if None)
    """
    sys.stdout = original_stdout
    log_file_handle.close()

    if final_test_acc is not None:
        if base_name is None:
            now = datetime.now()
            dir_timestamp = now.strftime('%y.%m.%d-%H-%M')
            num_samples = count_samples_in_dir(run_dir)
            scale_indicator = ''

            if 1000 <= num_samples < 10000:
                scale_indicator = 'K'
            elif num_samples >= 10000:
                scale_indicator = 'W'

            base_name = f"{final_test_acc:.4f}{scale_indicator} cnn_v2_{dir_timestamp}"

        final_run_dir = Path("logs") / base_name

        # Move log and plot files
        log_files = list(run_dir.glob("*.log"))
        for log_file in log_files:
            log_file.rename(run_dir / f"{base_name}.log")

        plot_files = list(run_dir.glob("*.png"))
        for plot_file in plot_files:
            plot_file.rename(run_dir / f"{base_name}.png")

        # Rename directory
        if not final_run_dir.exists():
            run_dir.rename(final_run_dir)
            print(f"   Results saved at: {final_run_dir}")
        else:
            print(f"   Error: Target folder {final_run_dir} already exists.")
            print(f"   Results retained in temporary folder: {run_dir}")
    else:
        print(f"   Training failed or incomplete, logs saved at: {run_dir}")


def count_samples_in_dir(run_dir):
    """
    Count the number of samples in directory (estimated)

    Args:
        run_dir: Run directory

    Returns:
        int: Estimated sample count
    """
    # This is a simple estimate, more precise methods may be needed in practice
    # Infer based on directory name or log content
    return 0


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, save_path):
    """
    Plot training history curves

    Args:
        train_losses: Training loss list
        val_losses: Validation loss list
        train_accuracies: Training accuracy list
        val_accuracies: Validation accuracy list
        save_path: Save path
    """
    plt.figure(figsize=(12, 4))

    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue', alpha=0.7)
    plt.plot(val_losses, label='Val Loss', color='red', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc', color='blue', alpha=0.7)
    plt.plot(val_accuracies, label='Val Acc', color='red', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def print_model_summary(model, input_size):
    """
    Print model summary information

    Args:
        model: PyTorch model
        input_size: Input size (C, H, W)
    """
    print("\n" + "=" * 60)
    print("Model Summary (FluidCNN v2)")
    print("=" * 60)

    # Model information
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        print(f"\nModel Name: {info.get('model_name', 'N/A')}")
        print(f"Input Channels: {info.get('input_channels', 'N/A')}")
        print(f"Output Classes: {info.get('num_classes', 'N/A')}")

    # Parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)

    print(f"\nParameter Statistics:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Model Size: {model_size_mb:.2f} MB")

    # Architecture information
    print(f"\nNetwork Architecture:")
    if hasattr(model, 'get_model_info'):
        arch = info.get('architecture', {})
        for stage, details in arch.items():
            if isinstance(details, dict):
                print(f"  {stage}:")
                for key, value in details.items():
                    if isinstance(value, list):
                        print(f"    {key}: {value}")
                    else:
                        print(f"    {key}: {value}")
            else:
                print(f"  {stage}: {details}")

    print("=" * 60)


def format_time(seconds):
    """
    Format time display

    Args:
        seconds: Number of seconds

    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_separator(char='=', length=60):
    """
    Print separator line

    Args:
        char: Separator character
        length: Separator line length
    """
    print(char * length)


if __name__ == "__main__":
    # Simple test
    print("Testing utility functions...")

    # Test TeeOutput
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        temp_file = f.name

    with open(temp_file, 'w', encoding='utf-8') as f:
        tee = TeeOutput(f)
        tee.write("This is a test message\n")
        tee.flush()

    print("TeeOutput test completed")

    # Cleanup
    os.unlink(temp_file)
