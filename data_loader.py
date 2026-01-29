#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading module
Contains dataset class and data processing functions
Adapted to FluidCNN v2 input/output requirements
"""

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from pathlib import Path
import warnings
import sys


class MultiSizeMeshDataset(Dataset):
    """
    Multi-size mesh dataset (adapted for FluidCNN v2)
    - Load all batch_data.pt from group* directories into memory at once
    - Support channel selection, global normalization, and multi-scale feature augmentation
    - After augmentation, mask and coordinate channels are automatically added
    """

    def __init__(self, base_dir="../cnn_input_data", group_indices=None, feature_indices=None):
        """
        Initialize dataset

        Args:
            base_dir: Dataset root directory
            group_indices: Group indices to load, default is range(1, 11)
            feature_indices: Feature channel indices to use, default is [0, 1, 2, 3, 4, 5, 6, 7]
        """
        self.base_dir = Path(base_dir)
        self.samples = []  # Raw (input, label)
        self.feature_indices = feature_indices
        self.channel_mean = None
        self.channel_std = None
        self.augmented_samples = []  # Normalized + augmented samples

        if group_indices is None:
            group_indices = range(1, 11)

        # Load data
        self._load_samples(group_indices)

        if not self.samples:
            raise RuntimeError("Dataset is empty, cannot continue.")

        print(f"   Loaded {len(self.samples)} raw samples")
        print("   Computing channel statistics...")

        # Compute channel statistics
        try:
            self.channel_mean, self.channel_std = self._compute_channel_stats()
            print(f"   [OK] Channel statistics computation completed (mean shape: {self.channel_mean.shape}, std shape: {self.channel_std.shape})")
        except Exception as e:
            print(f"   [FAIL] Channel statistics computation failed: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Augment samples
        try:
            self._augment_samples()
        except Exception as e:
            print(f"   [FAIL] Feature augmentation failed: {e}")
            import traceback
            traceback.print_exc()
            raise

        if not self.augmented_samples:
            raise RuntimeError("[FAIL] Dataset is empty after feature augmentation, cannot continue.")

        # Note: FluidCNN v2 input channels = original feature channels (after augmentation) + 3 (mask + coord_x + coord_y)
        # The pad_collate function automatically adds these 3 channels
        self.input_channels = self.augmented_samples[0][0].shape[0]
        print(f"   [OK] Final dataset ready, original feature channels: {self.input_channels}")
        print(f"   [INFO] FluidCNN v2 actual input channels: {self.input_channels + 3} (includes mask and coordinates)")

    def _load_samples(self, group_indices):
        """Load sample data"""
        print("Loading multi-size dataset into memory...")
        total_loaded = 0
        failed_groups = []

        for idx in group_indices:
            try:
                # Find group directory
                group_dirs = list(self.base_dir.glob(f"group{idx}_8x*"))
                if not group_dirs:
                    print(f"[WARN] Group {idx} not found, skipping")
                    failed_groups.append(idx)
                    continue

                group_dir = group_dirs[0]
                batch_file = group_dir / "batch_data.pt"

                if not batch_file.exists():
                    print(f"[WARN] batch_data.pt for group {idx} does not exist, skipping")
                    failed_groups.append(idx)
                    continue

                print(f"   Loading group {idx}: {group_dir.name}")

                # Load batch_data.pt (suppress FutureWarning)
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        batch_data = torch.load(batch_file, map_location="cpu", weights_only=False)
                except Exception as e:
                    print(f"   [FAIL] Group {idx} file loading failed: {e}")
                    failed_groups.append(idx)
                    continue

                # Check data structure
                if "inputs" not in batch_data or "labels" not in batch_data:
                    print(f"   [FAIL] Group {idx} data format incorrect (missing 'inputs' or 'labels' key), skipping")
                    failed_groups.append(idx)
                    continue

                inputs = batch_data["inputs"]
                labels = batch_data["labels"]

                # Check if input and label lengths match
                if len(inputs) != len(labels):
                    print(f"   [WARN] Group {idx} input and label count mismatch ({len(inputs)} vs {len(labels)}), using shorter length")
                    min_len = min(len(inputs), len(labels))
                    inputs = inputs[:min_len]
                    labels = labels[:min_len]

                # Process each sample
                group_samples = 0
                for i, (x, y) in enumerate(zip(inputs, labels)):
                    try:
                        # Feature selection
                        if self.feature_indices:
                            if x.dim() == 3 and x.shape[0] <= max(self.feature_indices):
                                print(f"   [WARN] Group {idx} sample {i} has insufficient feature channels, skipping")
                                continue
                            x = x[self.feature_indices]

                        # Ensure label is a scalar
                        if isinstance(y, torch.Tensor):
                            if y.dim() > 0:
                                y = y.squeeze().item() if y.numel() == 1 else y.squeeze()
                            else:
                                y = y.item()

                        # Check data validity
                        if torch.isnan(x).any() or torch.isinf(x).any():
                            print(f"   [WARN] Group {idx} sample {i} contains NaN or Inf, skipping")
                            continue

                        self.samples.append((x.float(), y))
                        group_samples += 1
                        total_loaded += 1

                    except Exception as e:
                        print(f"   [WARN] Group {idx} sample {i} processing failed: {e}, skipping")
                        continue

                if group_samples > 0:
                    print(f"      [OK] Group {idx} successfully loaded {group_samples} samples")
                else:
                    print(f"      [WARN] Group {idx} failed to load any valid samples")
                    failed_groups.append(idx)

            except Exception as e:
                print(f"   [FAIL] Error occurred during group {idx} loading: {e}")
                import traceback
                traceback.print_exc()
                failed_groups.append(idx)
                continue

        print(f"   [OK] Data loading completed, total {total_loaded} samples loaded")
        if failed_groups:
            print(f"   [WARN] The following groups failed or were skipped: {failed_groups}")

    def _compute_channel_stats(self):
        """Compute global mean and standard deviation for each channel"""
        if not self.samples:
            raise RuntimeError("Cannot compute statistics: sample list is empty")

        c = self.samples[0][0].shape[0]
        channel_sum = torch.zeros(c)
        channel_sumsq = torch.zeros(c)
        total_pixels = 0
        processed_samples = 0

        for idx, (x, _) in enumerate(self.samples):
            try:
                if x.shape[0] != c:
                    print(f"   [WARN] Sample {idx} channel count ({x.shape[0]}) does not match other samples ({c}), skipping")
                    continue

                pixels = x.view(c, -1)
                channel_sum += pixels.sum(dim=1)
                channel_sumsq += (pixels ** 2).sum(dim=1)
                total_pixels += pixels.size(1)
                processed_samples += 1

            except Exception as e:
                print(f"   [WARN] Sample {idx} statistics computation failed: {e}, skipping")
                continue

        if processed_samples == 0:
            raise RuntimeError("Statistics computation failed: no valid samples available for computation")

        if total_pixels == 0:
            raise RuntimeError("Statistics computation failed: total pixel count is zero")

        mean = channel_sum / total_pixels
        var = channel_sumsq / total_pixels - mean ** 2
        std = torch.sqrt(torch.clamp(var, min=1e-6))

        return mean, std

    def _augment_samples(self):
        """Normalize all samples and add local/global statistical features"""
        mean = self.channel_mean.view(-1, 1, 1)
        std = self.channel_std.view(-1, 1, 1)
        augmented = []

        print("   Augmenting sample features (normalization + statistical features)...")
        for idx, (x, y) in enumerate(self.samples):
            try:
                x_norm = (x - mean) / std

                # Check input dimensions
                h, w = x_norm.shape[1], x_norm.shape[2]
                if h >= 3 and w >= 3:
                    # Compute local statistical features (3x3 window)
                    local_mean = F.avg_pool2d(x_norm.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
                    local_var = F.avg_pool2d((x_norm.unsqueeze(0) - local_mean.unsqueeze(0)) ** 2, kernel_size=3, stride=1, padding=1).squeeze(0)
                    local_std = torch.sqrt(local_var + 1e-6)
                else:
                    # For too small samples, use global statistics instead of local statistics
                    local_mean = x_norm.mean(dim=(1, 2), keepdim=True).expand_as(x_norm)
                    local_std = torch.zeros_like(x_norm) + 1e-6

                # Compute global statistical features
                global_mean = x_norm.mean(dim=(1, 2), keepdim=True)
                global_std = x_norm.std(dim=(1, 2), keepdim=True, unbiased=False) + 1e-6

                # Concatenate all features
                broadcast_mean = global_mean.expand_as(x_norm)
                broadcast_std = global_std.expand_as(x_norm)

                augmented_x = torch.cat([x_norm, local_mean, local_std, broadcast_mean, broadcast_std], dim=0)
                augmented.append((augmented_x, y))

            except Exception as e:
                print(f"   [WARN] Sample {idx} feature augmentation failed: {e}, shape: {x.shape if hasattr(x, 'shape') else 'N/A'}")
                import traceback
                traceback.print_exc()
                raise

        self.augmented_samples = augmented
        self.samples = []  # Free memory
        print(f"   [OK] Feature augmentation completed, processed {len(augmented)} samples")

    def __len__(self):
        return len(self.augmented_samples)

    def __getitem__(self, idx):
        """Get a sample"""
        return self.augmented_samples[idx]

    def num_input_channels(self):
        """Return the number of original feature channels (excluding mask and coordinates)"""
        return self.input_channels


def pad_collate(batch):
    """
    Custom DataLoader collate function (adapted for FluidCNN v2)
    - Since different samples have different mesh sizes, zero-padding is needed within the batch to the maximum size
    - Additional mask/coordinate channels are concatenated for the model to distinguish real regions and positions
    - Return input tensors and label tensors with unified dimensions

    The returned input tensor channel count = original feature channels + 3 (mask + coord_x + coord_y)

    Args:
        batch: List containing (input, label) tuples

    Returns:
        tuple: (inputs_tensor, labels_tensor)
            - inputs_tensor: (batch_size, C, H_max, W_max)
            - labels_tensor: (batch_size,)
    """
    if not batch:
        raise ValueError("pad_collate received empty batch")

    try:
        inputs, labels = zip(*batch)
    except Exception as e:
        raise ValueError(f"pad_collate failed to parse batch: {e}")

    if not inputs:
        raise ValueError("pad_collate received empty input list")

    try:
        # Check shape consistency of all inputs
        for i, x in enumerate(inputs):
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Input {i} is not torch.Tensor type: {type(x)}")
            if x.dim() != 3:
                raise ValueError(f"Input {i} dimension incorrect (expected 3D [C,H,W], actually {x.dim()}D): {x.shape}")

        # Find the maximum size in the batch
        h_max = max(x.shape[1] for x in inputs)
        w_max = max(x.shape[2] for x in inputs)

        # Check if sizes are valid
        if h_max <= 0 or w_max <= 0:
            raise ValueError(f"Invalid maximum size: h_max={h_max}, w_max={w_max}")

        dtype = inputs[0].dtype
        device = inputs[0].device

        # Create coordinate tensors
        coord_x_base = torch.linspace(-1, 1, w_max, dtype=dtype, device=device).view(1, 1, w_max)
        coord_x_base = coord_x_base.expand(1, h_max, w_max)
        coord_y_base = torch.linspace(-1, 1, h_max, dtype=dtype, device=device).view(1, h_max, 1)
        coord_y_base = coord_y_base.expand(1, h_max, w_max)

        # Pad each sample
        padded_inputs = []
        for i, x in enumerate(inputs):
            try:
                pad_h = h_max - x.shape[1]
                pad_w = w_max - x.shape[2]

                if pad_h < 0 or pad_w < 0:
                    raise ValueError(f"Sample {i} size exceeds maximum size: {x.shape[1:]} vs ({h_max}, {w_max})")

                # Create mask
                mask = torch.ones(1, x.shape[1], x.shape[2], dtype=x.dtype, device=x.device)

                # Pad input
                x_padded = F.pad(x, (0, pad_w, 0, pad_h))
                mask_padded = F.pad(mask, (0, pad_w, 0, pad_h))

                # Add coordinates
                coord_x = coord_x_base * mask_padded
                coord_y = coord_y_base * mask_padded

                # Concatenate: original features + mask + coord_x + coord_y
                enriched = torch.cat([x_padded, mask_padded, coord_x, coord_y], dim=0)
                padded_inputs.append(enriched)

            except Exception as e:
                raise RuntimeError(f"Failed to process sample {i}: {e}")

        # Stack into batch
        inputs_tensor = torch.stack(padded_inputs)

        # Process labels
        labels_list = []
        for i, label in enumerate(labels):
            try:
                if isinstance(label, torch.Tensor):
                    if label.dim() > 0:
                        label = label.squeeze()
                    labels_list.append(label.item() if label.numel() == 1 else int(label))
                else:
                    labels_list.append(int(label))
            except Exception as e:
                raise ValueError(f"Failed to process label {i} (value: {label}, type: {type(label)}): {e}")

        labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=device)

        return inputs_tensor, labels_tensor

    except Exception as e:
        raise RuntimeError(f"pad_collate execution failed: {e}")


if __name__ == "__main__":
    # Simple test
    print("Testing MultiSizeMeshDataset...")

    try:
        # Create test dataset (actual data required for real usage)
        print("Note: This test requires actual data from cnn_input_data directory")
        print("If no data is available, this test will fail")

        # dataset = MultiSizeMeshDataset("cnn_input_data", feature_indices=[0, 1, 2, 3, 4, 5, 6, 7])
        # print(f"Dataset size: {len(dataset)}")
        # print(f"Input channels: {dataset.num_input_channels()}")

        # Test collate function
        from torch.utils.data import DataLoader
        # collate_test = pad_collate([dataset[0], dataset[1]])
        # print(f"Collate test successful, output shape: {collate_test[0].shape}")

        print("Test completed (skipped actual data loading)")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
