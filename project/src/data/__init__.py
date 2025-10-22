"""
Data Loading and Augmentation Package

This package provides:
- ARCDataset: PyTorch Dataset for ARC grids
- Data augmentation transforms
- Data loader utilities
"""

from .arc_dataset import ARCDataset, ARCDatasetWithSplits
from .transforms import (
    RandomRotation,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Compose,
    ToTensor,
    Identity,
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
    create_augmentation_pipeline,
)
from .data_loaders import (
    create_data_loaders,
    create_data_loader,
    print_data_loader_summary,
)

__all__ = [
    # Datasets
    'ARCDataset',
    'ARCDatasetWithSplits',
    # Transforms
    'RandomRotation',
    'RandomHorizontalFlip',
    'RandomVerticalFlip',
    'Compose',
    'ToTensor',
    'Identity',
    'get_train_transforms',
    'get_val_transforms',
    'get_test_transforms',
    'create_augmentation_pipeline',
    # Data loaders
    'create_data_loaders',
    'create_data_loader',
    'print_data_loader_summary',
]
