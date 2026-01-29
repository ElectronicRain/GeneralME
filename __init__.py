#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fluid CNN v2 - Multi-branch Convolutional Mesh Quality Assessment Model
Adopts a multi-branch convolutional architecture to extract features at different scales
"""

from .model import FluidCNN
from .trainer import FluidCNNTrainer
from .data_loader import MultiSizeMeshDataset, pad_collate

__version__ = "2.0.0"
__all__ = ["FluidCNN", "FluidCNNTrainer", "MultiSizeMeshDataset", "pad_collate"]
