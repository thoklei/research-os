"""
Data Loader Utilities - Task 2.3

Utilities for creating PyTorch DataLoaders for train/val/test splits.
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict
from pathlib import Path

from .arc_dataset import ARCDataset, ARCDatasetWithSplits
from .transforms import get_train_transforms, get_val_transforms, get_test_transforms
from .single_batch_dataset import create_single_batch_dataset


def create_data_loader(
    dataset: ARCDataset,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a single DataLoader for a dataset.

    Args:
        dataset: ARCDataset instance
        batch_size: Batch size (default: 128)
        shuffle: Whether to shuffle data (default: True)
        num_workers: Number of workers for data loading (default: 4)
        pin_memory: Whether to pin memory for GPU transfer (default: True)
        drop_last: Whether to drop last incomplete batch (default: False)

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_data_loaders(
    npz_path: str,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_augmentation: bool = True,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Create train, validation, and test data loaders from a corpus.

    Args:
        npz_path: Path to .npz corpus file
        batch_size: Batch size for all loaders (default: 128)
        num_workers: Number of workers for data loading (default: 4)
        pin_memory: Whether to pin memory for GPU transfer (default: True)
        use_augmentation: Whether to use data augmentation for training (default: True)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        Any may be None if not available in corpus.
    """
    # Get transforms
    train_transform = get_train_transforms() if use_augmentation else get_val_transforms()
    val_transform = get_val_transforms()
    test_transform = get_test_transforms()

    # Create datasets
    datasets = ARCDatasetWithSplits(
        npz_path=npz_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
    )

    train_dataset, val_dataset, test_dataset = datasets.get_datasets()

    # Create data loaders
    train_loader = None
    if train_dataset is not None:
        train_loader = create_data_loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,  # Drop last incomplete batch for training
        )

    val_loader = None
    if val_dataset is not None:
        val_loader = create_data_loader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = create_data_loader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle test
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader


def get_data_loader_info(loader: DataLoader) -> Dict:
    """
    Get information about a data loader.

    Args:
        loader: DataLoader instance

    Returns:
        Dictionary with loader information
    """
    dataset = loader.dataset

    return {
        'num_samples': len(dataset),
        'batch_size': loader.batch_size,
        'num_batches': len(loader),
        'shuffle': loader.sampler is not None,
        'num_workers': loader.num_workers,
        'pin_memory': loader.pin_memory,
        'drop_last': loader.drop_last,
    }


def print_data_loader_summary(
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
):
    """
    Print a summary of data loaders.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
    """
    print("=" * 60)
    print("Data Loader Summary")
    print("=" * 60)

    if train_loader is not None:
        info = get_data_loader_info(train_loader)
        print(f"\nTraining Set:")
        print(f"  Samples: {info['num_samples']}")
        print(f"  Batch size: {info['batch_size']}")
        print(f"  Batches: {info['num_batches']}")
        print(f"  Shuffle: {info['shuffle']}")

    if val_loader is not None:
        info = get_data_loader_info(val_loader)
        print(f"\nValidation Set:")
        print(f"  Samples: {info['num_samples']}")
        print(f"  Batch size: {info['batch_size']}")
        print(f"  Batches: {info['num_batches']}")
        print(f"  Shuffle: {info['shuffle']}")

    if test_loader is not None:
        info = get_data_loader_info(test_loader)
        print(f"\nTest Set:")
        print(f"  Samples: {info['num_samples']}")
        print(f"  Batch size: {info['batch_size']}")
        print(f"  Batches: {info['num_batches']}")
        print(f"  Shuffle: {info['shuffle']}")

    print("=" * 60)


def create_overfit_data_loaders(
    npz_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    num_repeats: int = 50,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Create data loaders that all use a single batch for overfitting sanity checks.

    This is useful for verifying that:
    1. The model has sufficient capacity to represent the data
    2. The loss function is working correctly
    3. Gradients are flowing properly

    All three loaders (train/val/test) use the same single batch from the training set,
    repeated num_repeats times to create proper epoch boundaries.

    Args:
        npz_path: Path to .npz corpus file
        batch_size: Size of the single batch to extract (default: 32)
        num_workers: Number of workers for data loading (default: 0, recommended for single batch)
        pin_memory: Whether to pin memory for GPU transfer (default: True)
        num_repeats: How many times to repeat the batch per epoch (default: 50)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        All loaders use the same single batch.
    """
    print(f"[OVERFIT MODE] Creating single-batch data loaders")
    print(f"  Batch size: {batch_size}")
    print(f"  Repeats per epoch: {num_repeats}")
    print(f"  Batches per epoch: {num_repeats}")

    # Load full dataset to extract a single batch
    # Use validation transform (no augmentation) for consistency
    val_transform = get_val_transforms()

    datasets = ARCDatasetWithSplits(
        npz_path=npz_path,
        train_transform=val_transform,
        val_transform=val_transform,
        test_transform=val_transform,
    )

    train_dataset, _, _ = datasets.get_datasets()

    if train_dataset is None:
        print("[ERROR] Could not load training dataset")
        return None, None, None

    # Create single batch dataset
    single_batch_dataset = create_single_batch_dataset(
        full_dataset=train_dataset,
        batch_size=batch_size,
        num_repeats=num_repeats,
    )

    print(f"  Single batch extracted: {batch_size} samples")
    print(f"  Total dataset size: {len(single_batch_dataset)} (batch × repeats)")

    # Create data loaders - all use the same single batch dataset
    train_loader = create_data_loader(
        single_batch_dataset,
        batch_size=batch_size,
        shuffle=True,  # Still shuffle for variety in batch ordering
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = create_data_loader(
        single_batch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = create_data_loader(
        single_batch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
