#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert .pt format graph data to CNN input format
Support 2D grid conversion for any number of node features (original example: 6 features)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm


class PT2CNNConverter:
    """Convert PyTorch Geometric .pt files to CNN input format"""

    def __init__(self,
                 input_dir: str = "element_graph/原始数据集",
                 output_dir: str = "cnn_input_data_3x80x380",
                 target_size: Tuple[int, int] = (80, 380)):
        """
        Initialize converter
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Initializing converter:")
        print(f"   Input directory: {self.input_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Target size: {self.target_size}")

    def load_pt_file(self, file_path: Path) -> Optional[torch.Tensor]:
        """
        Load .pt file and return node features
        """
        try:
            data = torch.load(file_path, map_location='cpu')

            if not hasattr(data, 'x') or data.x is None:
                print(f"Warning: {file_path.name} - Missing node features")
                return None

            features = data.x

            # ------------------- Modification Point -------------------
            # Now extract the first 8 features
            if features.shape[1] < 8:
                print(f"Warning: {file_path.name} - Expected at least 8-dimensional features, got {features.shape[1]} dimensions")
                return None

            selected_features = features[:, :8]
            # ------------------------------------------------

            return selected_features

        except Exception as e:
            print(f"Failed to load {file_path.name}: {e}")
            return None

    def features_to_cnn_grid(self,
                             features: torch.Tensor,
                             method: str = 'spatial_aware') -> torch.Tensor:
        """
        Convert 1D node features to 2D grid for CNN
        """
        N, feature_dim = features.shape
        H, W = self.target_size
        cnn_grid = torch.zeros(feature_dim, H, W)

        if method == 'simple_reshape':
            total_pixels = H * W
            if N <= total_pixels:
                for idx in range(N):
                    i, j = idx // W, idx % W
                    if i < H:
                        cnn_grid[:, i, j] = features[idx]
            else:
                step = N // total_pixels
                sampled_indices = np.arange(0, N, step)[:total_pixels]
                for pixel_idx, feat_idx in enumerate(sampled_indices):
                    i, j = pixel_idx // W, pixel_idx % W
                    cnn_grid[:, i, j] = features[feat_idx]

        elif method == 'spatial_aware':
            sqrt_N = int(np.sqrt(N))
            if sqrt_N * sqrt_N == N or abs(sqrt_N * sqrt_N - N) < 10:
                grid_size = min(sqrt_N, min(H, W))
                for idx in range(min(N, grid_size * grid_size)):
                    i, j = idx // grid_size, idx % grid_size
                    if i < H and j < W:
                        cnn_grid[:, i, j] = features[idx]
            else:
                idx = 0
                for i in range(H):
                    if i % 2 == 0:
                        for j in range(W):
                            if idx < N:
                                cnn_grid[:, i, j] = features[idx]
                                idx += 1
                    else:
                        for j in range(W - 1, -1, -1):
                            if idx < N:
                                cnn_grid[:, i, j] = features[idx]
                                idx += 1

        elif method == 'interpolation':
            scale_factor = max(H, W) / np.sqrt(N)
            for i in range(H):
                for j in range(W):
                    orig_i = int(i / scale_factor)
                    orig_j = int(j / scale_factor)
                    orig_idx = min(orig_i * int(np.sqrt(N)) + orig_j, N - 1)
                    cnn_grid[:, i, j] = features[orig_idx]

        return cnn_grid

    def normalize_features(self, cnn_grid: torch.Tensor) -> torch.Tensor:
        """
        Normalize CNN grid features
        """
        for channel in range(cnn_grid.shape[0]):
            channel_data = cnn_grid[channel]
            non_zero_mask = channel_data != 0
            if non_zero_mask.any():
                channel_min = channel_data[non_zero_mask].min()
                channel_max = channel_data[non_zero_mask].max()
                if channel_max > channel_min:
                    channel_data[non_zero_mask] = (channel_data[non_zero_mask] - channel_min) / (
                                channel_max - channel_min)
        return cnn_grid

    def convert_single_file(self,
                            input_path: Path,
                            method: str = 'spatial_aware') -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert single .pt file to CNN input
        """
        features = self.load_pt_file(input_path)
        if features is None:
            return None

        try:
            data = torch.load(input_path, map_location='cpu')
            label = data.y if hasattr(data, 'y') else torch.tensor([0])
        except:
            label = torch.tensor([0])

        cnn_grid = self.features_to_cnn_grid(features, method)
        cnn_grid = self.normalize_features(cnn_grid)

        return cnn_grid, label

    def convert_all_files(self, method: str = 'spatial_aware') -> dict:
        """
        Convert all .pt files
        """
        if not self.input_dir.exists():
            print(f"Input directory does not exist: {self.input_dir}")
            return {'success': 0, 'failed': 0, 'total': 0}

        pt_files = list(self.input_dir.glob("*.pt"))
        if not pt_files:
            print(f"No .pt files found in {self.input_dir}")
            return {'success': 0, 'failed': 0, 'total': 0}

        print(f"\nStarting conversion of {len(pt_files)} files...")
        print(f"   Conversion method: {method}")
        print("=" * 60)

        success_count, failed_count = 0, 0
        all_inputs, all_labels = [], []

        for input_path in tqdm(pt_files, desc="Converting files"):
            result = self.convert_single_file(input_path, method)
            if result is not None:
                cnn_input, label = result
                output_path = self.output_dir / f"cnn_{input_path.stem}.pt"
                torch.save({'input': cnn_input, 'label': label, 'original_file': input_path.name}, output_path)
                all_inputs.append(cnn_input)
                all_labels.append(label)
                success_count += 1
            else:
                failed_count += 1

        if all_inputs:
            batch_data = {
                'inputs': torch.stack(all_inputs),
                'labels': torch.stack(all_labels),
                'input_shape': all_inputs[0].shape,
                'target_size': self.target_size,
                'method': method
            }
            batch_path = self.output_dir / "batch_data.pt"
            torch.save(batch_data, batch_path)
            print(f"Batch data saved: {batch_path}")

        print("\n" + "=" * 60)
        print(f"Conversion Statistics:")
        print(f"    Success: {success_count}")
        print(f"    Failed: {failed_count}")
        print(f"    Total: {len(pt_files)}")
        print(f"    Output directory: {self.output_dir}")
        if success_count > 0:
            print(f"    CNN input shape: {all_inputs[0].shape}")

        return {
            'success': success_count,
            'failed': failed_count,
            'total': len(pt_files),
            'output_dir': str(self.output_dir),
            'input_shape': all_inputs[0].shape if success_count > 0 else None
        }

    def visualize_sample(self, sample_count: int = 3):
        """Visualize first few conversion results"""
        output_files = list(self.output_dir.glob("cnn_*.pt"))
        if not output_files:
            print("No conversion results available for visualization")
            return

        sample_count = min(sample_count, len(output_files))
        fig, axes = plt.subplots(2, sample_count, figsize=(4 * sample_count, 6))

        for i, file_path in enumerate(output_files[:sample_count]):
            try:
                data = torch.load(file_path, map_location='cpu')
                cnn_input = data['input']
                # Visualize first two channels
                axes[0, i].imshow(cnn_input[0].numpy(), cmap='viridis')
                axes[0, i].set_title(f'Feature 0\n{file_path.stem}')
                axes[0, i].axis('off')
                axes[1, i].imshow(cnn_input[1].numpy(), cmap='plasma')
                axes[1, i].set_title(f'Feature 1\n{file_path.stem}')
                axes[1, i].axis('off')
            except Exception as e:
                print(f"Failed to visualize {file_path.name}: {e}")

        plt.tight_layout()
        plt.savefig(self.output_dir / "visualization_samples.png", dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Visualization results saved: {self.output_dir / 'visualization_samples.png'}")


def main():
    """Main function"""
    print("PyTorch Geometric .pt -> CNN Input Conversion Tool")
    print("=" * 60)

    converter = PT2CNNConverter(
        input_dir="element_graph/原始数据集",
        output_dir="cnn_input_data_3x80x380",
        target_size=(80, 380)
    )

    results = converter.convert_all_files(method='spatial_aware')

    if results['success'] > 0:
        converter.visualize_sample()

    print("\nConversion completed!")
    print(f"Input directory: {converter.input_dir}")
    print(f"Output directory: {converter.output_dir}")
    print(f"Feature dimensions: {results['input_shape'][0]} dimensions")
    print(f"CNN input format: {results['input_shape']}")


if __name__ == "__main__":
    main()
