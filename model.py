#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


def set_seed(seed=42):
    """Set random seed for experiment reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_padding_size(kernel_size, stride=1):
    """Calculate convolution padding size for 'same' padding"""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    # Standard 'same' padding: padding = (kernel_size - 1) // 2
    padding_h = (kernel_size[0] - 1) // 2
    padding_w = (kernel_size[1] - 1) // 2

    return (padding_h, padding_w)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) channel attention module
    Adaptively adjust channel weights to enhance important features
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """
    Residual block (supports identity or projection mapping)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        padding1 = get_padding_size((3, 10), stride)
        padding2 = get_padding_size((3, 10), 1)

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 10), stride=stride,
            padding=padding1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 10), stride=1,
            padding=padding2, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Check input size, upsample or pad if too small
        _, _, h, w = x.shape
        min_size = 10  # Ensure at least 10x10 for 3x10 convolution

        if h < min_size or w < min_size:
            # Use adaptive pooling to adjust input to minimum size
            x = F.adaptive_avg_pool2d(x, (min_size, min_size))

        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        # Ensure shortcut has same dimensions as output
        shortcut = self.shortcut(x)
        if out.shape[2] != shortcut.shape[2] or out.shape[3] != shortcut.shape[3]:
            shortcut = F.adaptive_avg_pool2d(shortcut, (out.shape[2], out.shape[3]))

        out += shortcut
        out = F.gelu(out)
        return out


class MultiBranchResidualBlock(nn.Module):
    """
    Multi-branch residual convolution block
    Combine multi-scale feature extraction and residual learning
    """

    def __init__(self, in_channels, branch_channels=32, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.branch_channels = branch_channels

        # Three different scale convolution branches
        padding1 = get_padding_size((2, 7), 1)
        padding2 = get_padding_size((3, 10), 1)
        padding3 = get_padding_size((4, 14), 1)

        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels, branch_channels, kernel_size=(2, 7),
                stride=1, padding=padding1, bias=False
            ),
            nn.BatchNorm2d(branch_channels),
            nn.GELU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels, branch_channels, kernel_size=(3, 10),
                stride=1, padding=padding2, bias=False
            ),
            nn.BatchNorm2d(branch_channels),
            nn.GELU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels, branch_channels, kernel_size=(4, 14),
                stride=1, padding=padding3, bias=False
            ),
            nn.BatchNorm2d(branch_channels),
            nn.GELU()
        )

        # Channel attention
        self.se = SEBlock(branch_channels * 3)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != branch_channels * 3:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, branch_channels * 3, 1, bias=False),
                nn.BatchNorm2d(branch_channels * 3)
            )

    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)

        # Ensure all branch outputs have same dimensions
        target_h = min(branch1_out.shape[2], branch2_out.shape[2], branch3_out.shape[2])
        target_w = min(branch1_out.shape[3], branch2_out.shape[3], branch3_out.shape[3])

        if branch1_out.shape[2] != target_h or branch1_out.shape[3] != target_w:
            branch1_out = F.adaptive_avg_pool2d(branch1_out, (target_h, target_w))
        if branch2_out.shape[2] != target_h or branch2_out.shape[3] != target_w:
            branch2_out = F.adaptive_avg_pool2d(branch2_out, (target_h, target_w))
        if branch3_out.shape[2] != target_h or branch3_out.shape[3] != target_w:
            branch3_out = F.adaptive_avg_pool2d(branch3_out, (target_h, target_w))

        # Channel concatenation
        concatenated = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)

        # Apply attention
        concatenated = self.se(concatenated)

        # Residual connection
        shortcut = self.shortcut(x)
        # Ensure shortcut has same dimensions as concatenated
        if shortcut.shape[2] != target_h or shortcut.shape[3] != target_w:
            shortcut = F.adaptive_avg_pool2d(shortcut, (target_h, target_w))

        out = concatenated + shortcut
        out = F.gelu(out)

        return out


class Top16RowModule(nn.Module):
    """
    Top-16-row feature enhancement module
    Specifically process top 16 rows of matrix using 1xN convolution
    """

    def __init__(self, input_channels, conv_kernel_size=5, mlp_hidden_dim=64, output_dim=128):
        """
        Initialize top-16-row module

        Args:
            input_channels: Number of input channels
            conv_kernel_size: Size N of 1xN convolution kernel
            mlp_hidden_dim: MLP hidden layer dimension
            output_dim: Output dimension (for fusion with main branch)
        """
        super().__init__()
        self.input_channels = input_channels
        self.conv_kernel_size = conv_kernel_size
        self.mlp_hidden_dim = mlp_hidden_dim
        self.output_dim = output_dim

        # Convolution layer: use 1xN kernel
        padding = (conv_kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                input_channels, 32,
                kernel_size=(1, conv_kernel_size),
                stride=1,
                padding=(0, padding),
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(
                32, 64,
                kernel_size=(1, conv_kernel_size),
                stride=1,
                padding=(0, padding),
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        # 1x1 convolution for channel adjustment
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # MLP for feature scaling
        self.mlp = nn.Sequential(
            nn.Linear(128, mlp_hidden_dim, bias=False),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.GELU()
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor, shape (batch_size, input_channels, H, W)

        Returns:
            Processed features, shape (batch_size, output_dim)
        """
        b, c, h, w = x.shape

        # Extract top 16 rows
        if h >= 16:
            top_16 = x[:, :, :16, :]  # (batch_size, input_channels, 16, W)
        else:
            # If height < 16 rows, use adaptive padding
            pad_size = 16 - h
            top_16 = F.pad(x, (0, 0, 0, pad_size), mode='constant', value=0)

        # Convolution processing
        conv_out = self.conv(top_16)  # (batch_size, 64, 16, W)

        # Channel adaptation
        adapted = self.channel_adapter(conv_out)  # (batch_size, 128, 16, W)

        # Global average pooling
        pooled = self.global_pool(adapted)  # (batch_size, 128, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (batch_size, 128)

        # MLP scaling
        output = self.mlp(pooled)  # (batch_size, output_dim)

        return output


class WeightedFeatureFusion(nn.Module):
    """
    Weighted feature fusion module
    Performs weighted fusion of features from two branches
    """

    def __init__(self, feature_dim1, feature_dim2, output_dim, fusion_method='weighted_sum'):
        """
        Initialize fusion module

        Args:
            feature_dim1: Main branch feature dimension
            feature_dim2: Top-16-row branch feature dimension
            output_dim: Output dimension
            fusion_method: Fusion method ('weighted_sum', 'concat')
        """
        super().__init__()
        self.fusion_method = fusion_method

        if fusion_method == 'weighted_sum':
            # Learnable weight parameters
            self.weight_main = nn.Parameter(torch.tensor(0.9))
            self.weight_top16 = nn.Parameter(torch.tensor(0.1))

            # Feature adapter layers
            self.adapter_main = nn.Linear(feature_dim1, output_dim)
            self.adapter_top16 = nn.Linear(feature_dim2, output_dim)

            # Final BatchNorm and activation
            self.bn = nn.BatchNorm1d(output_dim)
            self.activation = nn.GELU()

        elif fusion_method == 'concat':
            # Direct concatenation
            combined_dim = feature_dim1 + feature_dim2
            self.fusion_layer = nn.Sequential(
                nn.Linear(combined_dim, output_dim, bias=False),
                nn.BatchNorm1d(output_dim),
                nn.GELU()
            )

    def forward(self, main_features, top16_features):
        """
        Forward pass

        Args:
            main_features: Main branch features (batch_size, feature_dim1)
            top16_features: Top-16-row branch features (batch_size, feature_dim2)

        Returns:
            Fused features (batch_size, output_dim)
        """
        if self.fusion_method == 'weighted_sum':
            # Feature adaptation
            adapted_main = self.adapter_main(main_features)
            adapted_top16 = self.adapter_top16(top16_features)

            # Weighted fusion
            fused = self.weight_main * adapted_main + self.weight_top16 * adapted_top16

            # Normalization and activation
            fused = self.bn(fused)
            fused = self.activation(fused)

            return fused

        elif self.fusion_method == 'concat':
            # Feature concatenation
            combined = torch.cat([main_features, top16_features], dim=1)
            fused = self.fusion_layer(combined)
            return fused


class ConvPoolBlock(nn.Module):
    """
    Improved convolutional pooling block (with residual)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        padding = get_padding_size((3, 10), stride)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=(3, 10),
                stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.se = SEBlock(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)

        # Ensure shortcut has same dimensions as output
        shortcut = self.shortcut(x)
        if out.shape[2] != shortcut.shape[2] or out.shape[3] != shortcut.shape[3]:
            shortcut = F.adaptive_avg_pool2d(shortcut, (out.shape[2], out.shape[3]))

        out += shortcut
        out = F.gelu(out)
        # Max pooling
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        return out


class FluidCNNEnhancedV6(nn.Module):
    """
    FluidCNN v6 - Top-16-row Feature Enhancement Parallel Branch Version
    High-performance CNN specifically optimized for mesh quality assessment

    Core improvements:
    - Add Top16RowModule parallel branch, specifically processing top 16 rows of matrix
    - Use 1xN convolution kernels (configurable parameter N)
    - Feature scaling through MLP
    - Weighted fusion with original features
    - Fixed random seed for reproducibility

    Architecture design:
    Main branch:
      Stage 1: MultiBranchResidualBlock -> (H, W, 96)
      Stage 2: ResidualBlock (stride=2) -> (H/2, W/2, 96)
      Stage 3: ResidualBlock -> (H/2, W/2, 128)
      Stage 4: ConvPoolBlock -> (H/4, W/4, 128)
      Stage 5: ResidualBlock (stride=2) -> (H/8, W/8, 128)
      Stage 6: ResidualBlock -> (H/8, W/8, 256)
      Stage 7: ResidualBlock (stride=2) -> (H/16, W/16, 256)
      Stage 8: ResidualBlock -> (H/16, W/16, 512)
      Global Pool -> 512

    Parallel branch (Top16RowModule):
      Extract top 16 rows -> 1xN convolution -> MLP -> 128-dimensional features

    Fusion and classification:
      WeightedFeatureFusion: Fuse main branch (512-dim) and parallel branch (128-dim)
      Classifier: 640 -> 256 -> 128 -> 8 (Multi-layer fully connected + Dropout)
    """

    def __init__(self, input_channels, num_classes=8, branch_channels=32,
                 dropout_rate=0.3, conv_kernel_size=5, top16_hidden_dim=64,
                 fusion_method='weighted_sum', seed=42):
        """
        Initialize FluidCNN v6 model

        Args:
            input_channels: Number of input channels
            num_classes: Number of classification categories
            branch_channels: Branch channel count for multi-branch module
            dropout_rate: Dropout rate
            conv_kernel_size: Size N of 1xN convolution kernel
            top16_hidden_dim: MLP hidden layer dimension for top-16-row module
            fusion_method: Fusion method ('weighted_sum', 'concat')
            seed: Random seed
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.branch_channels = branch_channels
        self.dropout_rate = dropout_rate
        self.conv_kernel_size = conv_kernel_size
        self.top16_hidden_dim = top16_hidden_dim
        self.fusion_method = fusion_method

        # Fix random seed
        set_seed(seed)

        # Main branch: Multi-branch residual convolution (no downsampling)
        self.stage1 = MultiBranchResidualBlock(
            in_channels=input_channels,
            branch_channels=branch_channels
        )
        # Output: (H, W, 96)

        # Parallel branch: Top-16-row feature enhancement module
        self.top16_module = Top16RowModule(
            input_channels=input_channels,
            conv_kernel_size=conv_kernel_size,
            mlp_hidden_dim=top16_hidden_dim,
            output_dim=128  # Dimension for fusion with subsequent features
        )
        # Output: (batch_size, 128)

        # Stage 2: Residual block (2x downsampling)
        self.stage2 = ResidualBlock(
            in_channels=branch_channels * 3,
            out_channels=branch_channels * 3,
            stride=2
        )
        # Output: (H/2, W/2, 96)

        # Stage 3: Residual block (maintain size)
        self.stage3 = ResidualBlock(
            in_channels=branch_channels * 3,
            out_channels=128,
            stride=1
        )
        # Output: (H/2, W/2, 128)

        # Stage 4: Convolutional pooling block (2x downsampling)
        self.stage4 = ConvPoolBlock(
            in_channels=128,
            out_channels=128,
            stride=1
        )
        # Output: (H/4, W/4, 128)

        # Stage 5: Residual block (2x downsampling)
        self.stage5 = ResidualBlock(
            in_channels=128,
            out_channels=128,
            stride=2
        )
        # Output: (H/8, W/8, 128)

        # Stage 6: Residual block (maintain size)
        self.stage6 = ResidualBlock(
            in_channels=128,
            out_channels=256,
            stride=1
        )
        # Output: (H/8, W/8, 256)

        # Stage 7: Residual block (2x downsampling)
        self.stage7 = ResidualBlock(
            in_channels=256,
            out_channels=256,
            stride=2
        )
        # Output: (H/16, W/16, 256)

        # Stage 8: Residual block (maintain size)
        self.stage8 = ResidualBlock(
            in_channels=256,
            out_channels=512,
            stride=1
        )
        # Output: (H/16, W/16, 512)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Feature fusion module
        self.feature_fusion = WeightedFeatureFusion(
            feature_dim1=512,
            feature_dim2=128,
            output_dim=640,  # Fused dimension
            fusion_method=fusion_method
        )

        # Enhanced classifier (multi-layer fully connected + Dropout)
        self.classifier = nn.Sequential(
            nn.Linear(640, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights (optimized for ReLU)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor, shape (batch_size, input_channels, H, W)

        Returns:
            Classification logits, shape (batch_size, num_classes)
        """
        # Parallel processing: main branch and top-16-row branch
        # Main branch: Multi-branch residual -> (H, W, 96)
        main_branch = self.stage1(x)

        # Parallel branch: Top-16-row feature enhancement
        top16_features = self.top16_module(x)  # (batch_size, 128)

        # Continue processing main branch
        # Stage 2: Residual + downsampling -> (H/2, W/2, 96)
        main_branch = self.stage2(main_branch)

        # Stage 3: Residual -> (H/2, W/2, 128)
        main_branch = self.stage3(main_branch)

        # Stage 4: Convolutional pooling -> (H/4, W/4, 128)
        main_branch = self.stage4(main_branch)

        # Stage 5: Residual + downsampling -> (H/8, W/8, 128)
        main_branch = self.stage5(main_branch)

        # Stage 6: Residual -> (H/8, W/8, 256)
        main_branch = self.stage6(main_branch)

        # Stage 7: Residual + downsampling -> (H/16, W/16, 256)
        main_branch = self.stage7(main_branch)

        # Stage 8: Residual -> (H/16, W/16, 512)
        main_branch = self.stage8(main_branch)

        # Global average pooling -> (batch_size, 512, 1, 1)
        main_features = self.global_pool(main_branch)
        # Flatten -> (batch_size, 512)
        main_features = main_features.view(main_features.size(0), -1)

        # Feature fusion: Main branch (512-dim) + Parallel branch (128-dim) -> (batch_size, 640)
        fused_features = self.feature_fusion(main_features, top16_features)

        # Classification -> (batch_size, num_classes)
        output = self.classifier(fused_features)

        return output

    def extract_features(self, x):
        """
        Extract features before the fully connected layer (for T-SNE analysis)

        Args:
            x: Input tensor, shape (batch_size, input_channels, H, W)

        Returns:
            Fused feature vector, shape (batch_size, 640)
        """
        # Parallel processing: main branch and top-16-row branch
        main_branch = self.stage1(x)
        top16_features = self.top16_module(x)

        main_branch = self.stage2(main_branch)
        main_branch = self.stage3(main_branch)
        main_branch = self.stage4(main_branch)
        main_branch = self.stage5(main_branch)
        main_branch = self.stage6(main_branch)
        main_branch = self.stage7(main_branch)
        main_branch = self.stage8(main_branch)

        main_features = self.global_pool(main_branch)
        main_features = main_features.view(main_features.size(0), -1)

        # Feature fusion
        fused_features = self.feature_fusion(main_features, top16_features)

        return fused_features

    def get_model_info(self):
        """Get model information"""
        info = {
            "model_name": "FluidCNN v6 - Top-16-row Feature Enhancement Parallel Branch Version",
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "conv_kernel_size": self.conv_kernel_size,
            "top16_hidden_dim": self.top16_hidden_dim,
            "fusion_method": self.fusion_method,
            "architecture": {
                "parallel_branch": "Top16RowModule (128 channels)",
                "main_branch": {
                    "stage1": "MultiBranchResidualBlock (96 channels)",
                    "stage2": "ResidualBlock + downsample (96 channels)",
                    "stage3": "ResidualBlock (128 channels)",
                    "stage4": "ConvPoolBlock (128 channels)",
                    "stage5": "ResidualBlock + downsample (128 channels)",
                    "stage6": "ResidualBlock (256 channels)",
                    "stage7": "ResidualBlock + downsample (256 channels)",
                    "stage8": "ResidualBlock (512 channels)",
                },
                "fusion": "WeightedFeatureFusion (640 channels)",
                "classifier": "640->256->128->8 with BatchNorm+Dropout"
            },
            "key_features": [
                "Parallel Top16RowModule for top-region enhancement",
                "1xN convolution kernels (configurable N)",
                "MLP-based feature scaling for top16 branch",
                "Weighted feature fusion",
                "Residual connections (ResNet-style)",
                "SE Channel Attention",
                "GELU activation",
                "Multi-scale feature extraction",
                "Strong regularization",
                "Fixed random seed for reproducibility"
            ],
            "improvements": [
                "Added Top16RowModule for top-region feature enhancement",
                "Parallel processing of top 16 rows with 1xN conv",
                "Feature fusion through learnable weighted sum or concatenation",
                "Enhanced classification with larger input dimension (640 vs 512)",
                "Configurable hyperparameters for fine-tuning",
                "Reproducible results with fixed seed"
            ]
        }
        return info


if __name__ == "__main__":
    # Test model
    batch_size = 4
    input_channels = 11
    H, W = 64, 64

    model = FluidCNNEnhancedV6(
        input_channels=input_channels,
        num_classes=8,
        conv_kernel_size=5,
        top16_hidden_dim=64,
        fusion_method='weighted_sum',
        seed=42
    )

    x = torch.randn(batch_size, input_channels, H, W)
    output = model(x)

    print("=" * 70)
    print("FluidCNN v6 - Top-16-row Feature Enhancement Parallel Branch Version")
    print("=" * 70)
    info = model.get_model_info()
    for key, value in info.items():
        print(f"\n{key}:")
        if isinstance(value, dict):
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, list):
            for v in value:
                print(f"  - {v}")
        else:
            print(f"  {value}")

    print("\n" + "=" * 70)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {info['num_classes']})")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameter statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 70)
