#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
import time
from datetime import datetime


class FluidCNNTrainer:
    """
    FluidCNN v3 Enhanced Trainer
    Responsible for complete model training, validation, and testing workflow
    """

    def __init__(self, model: nn.Module, device="auto"):
        """
        Initialize trainer

        Args:
            model: FluidCNN model instance
            device: Device type ("auto", "cuda", "cpu", etc.)
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "auto"
            else device
        )
        self.model = model.to(self.device)
        print(f"Using device: {self.device}")

    def train_one_epoch(self, loader, optimizer, criterion):
        """
        Single epoch training

        Args:
            loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function

        Returns:
            tuple: (epoch_loss, epoch_accuracy)
        """
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        start_time = time.time()
        start_dt_str = datetime.now().strftime('%H:%M:%S')

        print("   Training...", end="", flush=True)
        batch_count = 0

        for batch_idx, (inputs, labels) in enumerate(loader):
            try:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Check data validity
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    print(f"\n   WARNING: Batch {batch_idx} contains NaN or Inf, skipping")
                    continue

                optimizer.zero_grad()
                outputs = self.model(inputs)

                # Check output validity
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"\n   [WARN] Batch {batch_idx} model output contains NaN or Inf, skipping")
                    continue

                loss = criterion(outputs, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n   WARNING: Batch {batch_idx} loss is NaN or Inf, skipping")
                    continue

                loss.backward()

                # Gradient clipping to prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                batch_count += 1

            except Exception as e:
                print(f"\n   [FAIL] Batch {batch_idx} training failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        if batch_count == 0:
            raise RuntimeError("Training failed: no batch was successfully processed")

        end_time = time.time()
        end_dt_str = datetime.now().strftime('%H:%M:%S')
        duration = end_time - start_time
        print(f"   Training: start {start_dt_str} | complete {end_dt_str} | duration {duration:.2f}s")

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, loader, criterion):
        """
        Evaluate model

        Args:
            loader: Data loader (validation or test set)
            criterion: Loss function

        Returns:
            tuple: (epoch_loss, epoch_accuracy, classification_report_dict)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []

        start_time = time.time()
        start_dt_str = datetime.now().strftime('%H:%M:%S')

        batch_count = 0

        for batch_idx, (inputs, labels) in enumerate(loader):
            try:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Check data validity
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    print(f"\n   [WARN] Batch {batch_idx} contains NaN or Inf, skipping")
                    continue

                outputs = self.model(inputs)

                # Check output validity
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"\n   WARNING: Batch {batch_idx} model output contains NaN or Inf, skipping")
                    continue

                loss = criterion(outputs, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n   [WARN] Batch {batch_idx} loss is NaN or Inf, skipping")
                    continue

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                batch_count += 1

            except Exception as e:
                print(f"\n   [FAIL] Batch {batch_idx} evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        if batch_count == 0:
            print("   [WARN] Warning: no batch was successfully processed during evaluation")
            return 0.0, 0.0, {}

        end_time = time.time()
        end_dt_str = datetime.now().strftime('%H:%M:%S')
        duration = end_time - start_time
        print(f"   Evaluation: start {start_dt_str} | complete {end_dt_str} | duration {duration:.2f}s")

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)

        # Generate classification report
        num_classes = self._get_num_classes()
        target_names = [f"Label {i}" for i in range(num_classes)]
        try:
            report = classification_report(
                all_labels,
                all_preds,
                labels=range(num_classes),
                target_names=target_names,
                output_dict=True,
                zero_division=0
            )
        except Exception:
            report = {}

        return epoch_loss, epoch_acc, report

    @torch.no_grad()
    def collect_features(self, loader, max_samples=1000):
        """
        Collect features from samples in the data loader (for T-SNE analysis)

        Args:
            loader: Data loader
            max_samples: Maximum number of samples to collect

        Returns:
            tuple: (features_list, labels_list, predictions_list)
        """
        self.model.eval()
        features_list = []
        labels_list = []
        predictions_list = []

        collected = 0

        for batch_idx, (inputs, labels) in enumerate(loader):
            if collected >= max_samples:
                break

            try:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Extract features
                features = self.model.extract_features(inputs)

                # Get predictions
                outputs = self.model(inputs)
                preds = outputs.argmax(1)

                # Store
                batch_size = features.size(0)
                for i in range(min(batch_size, max_samples - collected)):
                    features_list.append(features[i].cpu().numpy())
                    labels_list.append(labels[i].item())
                    predictions_list.append(preds[i].item())

                collected += batch_size

            except Exception as e:
                print(f"   [WARN] Batch {batch_idx} feature collection failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"   Collected features from {len(features_list)} samples")
        return features_list, labels_list, predictions_list

    def _get_num_classes(self):
        """Get the number of output classes from the model"""
        # Try to get from the last layer of classifier
        classifier = getattr(self.model, "classifier", None)
        if classifier:
            last_layer = classifier[-1]
            if hasattr(last_layer, "out_features"):
                return last_layer.out_features
            elif hasattr(last_layer, "out_channels"):
                return last_layer.out_channels

        # Default return 8
        return 8

    def get_model_parameters(self):
        """
        Get model parameter statistics

        Returns:
            dict: Dictionary containing parameter statistics
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }

    def save_checkpoint(self, filepath, epoch, optimizer, loss):
        """
        Save training checkpoint

        Args:
            filepath: Save path
            epoch: Current epoch
            optimizer: Optimizer
            loss: Current loss
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "model_info": self.model.get_model_info()
        }
        torch.save(checkpoint, filepath)
        print(f"[DISK] Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath, strict=True):
        """
        Load training checkpoint

        Args:
            filepath: Checkpoint path
            strict: Whether to strictly load weights

        Returns:
            dict: Checkpoint information
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        print(f"[FOLDER] Checkpoint loaded: {filepath}")
        return checkpoint


if __name__ == "__main__":
    # Simple test
    print("Testing FluidCNNTrainer...")

    # Create test model
    from .model import FluidCNN

    model = FluidCNN(input_channels=11, num_classes=8)
    trainer = FluidCNNTrainer(model, device="cpu")

    # Display model information
    print("\nModel info:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Display parameter statistics
    print("\nParameter statistics:")
    params = trainer.get_model_parameters()
    for key, value in params.items():
        print(f"  {key}: {value}")

    print("\nTest completed")
