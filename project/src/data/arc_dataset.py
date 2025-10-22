"""
PyTorch Dataset for ARC Grids - Task 2.1

Dataset wrapper for loading 16×16 ARC grids from .npz files
with support for data augmentation and train/val/test splits.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent))
from visualization import load_corpus


class ARCDataset(Dataset):
    """
    PyTorch Dataset for ARC 16×16 grids.

    Args:
        npz_path (str): Path to .npz corpus file
        split (str): Which split to use - 'train', 'val', 'test', or 'all'
        transform (callable, optional): Optional transform to apply to grids
    """

    def __init__(
        self,
        npz_path: str,
        split: str = 'all',
        transform: Optional[Callable] = None
    ):
        super(ARCDataset, self).__init__()

        self.npz_path = npz_path
        self.split = split
        self.transform = transform

        # Load corpus
        corpus = load_corpus(npz_path)

        # Get images based on split
        if split == 'all':
            # Use all images (or 'images' key if no splits)
            if 'images' in corpus:
                self.images = corpus['images']
            elif 'train' in corpus:
                # Concatenate all splits
                splits = []
                for key in ['train', 'val', 'test']:
                    if key in corpus:
                        splits.append(corpus[key])
                self.images = np.concatenate(splits, axis=0)
            else:
                raise ValueError(f"Corpus at {npz_path} has unexpected format")
        elif split in corpus:
            self.images = corpus[split]
        else:
            raise ValueError(
                f"Split '{split}' not found in corpus. "
                f"Available keys: {list(corpus.keys())}"
            )

        # Convert to torch tensor (N, 16, 16)
        self.images = torch.from_numpy(self.images).long()

        print(f"Loaded {len(self.images)} images from split '{split}'")

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single grid sample.

        Args:
            idx: Index of sample to retrieve

        Returns:
            Grid tensor with shape (16, 16) and dtype torch.long
        """
        grid = self.images[idx]  # (16, 16)

        # Apply transform if provided
        if self.transform is not None:
            grid = self.transform(grid)

        return grid

    def get_batch(self, indices: list) -> torch.Tensor:
        """
        Get a batch of grids by indices.

        Args:
            indices: List of indices to retrieve

        Returns:
            Batch tensor with shape (batch_size, 16, 16)
        """
        return torch.stack([self[i] for i in indices])

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return shape of dataset (N, H, W)."""
        return self.images.shape

    @property
    def num_samples(self) -> int:
        """Return number of samples."""
        return len(self.images)

    def get_stats(self) -> Dict[str, float]:
        """
        Compute statistics about the dataset.

        Returns:
            Dictionary with statistics (color distribution, etc.)
        """
        # Color distribution
        color_counts = torch.bincount(self.images.flatten(), minlength=10)
        color_freq = color_counts.float() / color_counts.sum()

        # Sparsity (fraction of background pixels)
        sparsity = (self.images == 0).float().mean().item()

        return {
            'num_samples': len(self.images),
            'shape': self.images.shape,
            'color_distribution': color_freq.tolist(),
            'sparsity': sparsity,
            'background_ratio': color_freq[0].item(),
        }


class ARCDatasetWithSplits:
    """
    Convenience wrapper for creating train/val/test datasets from a corpus.

    Args:
        npz_path (str): Path to .npz corpus file
        train_transform (callable, optional): Transform for training set
        val_transform (callable, optional): Transform for validation set
        test_transform (callable, optional): Transform for test set
    """

    def __init__(
        self,
        npz_path: str,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
    ):
        self.npz_path = npz_path

        # Load corpus to check available splits
        corpus = load_corpus(npz_path)
        self.available_splits = list(corpus.keys())

        # Create datasets for each split
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        if 'train' in corpus:
            self.train_dataset = ARCDataset(
                npz_path, split='train', transform=train_transform
            )

        if 'val' in corpus:
            self.val_dataset = ARCDataset(
                npz_path, split='val', transform=val_transform
            )

        if 'test' in corpus:
            self.test_dataset = ARCDataset(
                npz_path, split='test', transform=test_transform
            )

        # If no splits, create from 'images' key with manual split
        if self.train_dataset is None and 'images' in corpus:
            print("No splits found, creating manual splits (80/10/10)")
            self._create_manual_splits(corpus['images'], train_transform, val_transform, test_transform)

    def _create_manual_splits(
        self,
        images: np.ndarray,
        train_transform: Optional[Callable],
        val_transform: Optional[Callable],
        test_transform: Optional[Callable],
    ):
        """Create manual train/val/test splits from images array."""
        n = len(images)
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)

        # Shuffle indices with fixed seed for reproducibility
        np.random.seed(42)
        indices = np.random.permutation(n)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Create temporary .npz with splits (in-memory)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(
                f.name,
                train=images[train_indices],
                val=images[val_indices],
                test=images[test_indices]
            )
            temp_path = f.name

        # Create datasets
        self.train_dataset = ARCDataset(temp_path, split='train', transform=train_transform)
        self.val_dataset = ARCDataset(temp_path, split='val', transform=val_transform)
        self.test_dataset = ARCDataset(temp_path, split='test', transform=test_transform)

    def get_datasets(self) -> Tuple[Optional[ARCDataset], Optional[ARCDataset], Optional[ARCDataset]]:
        """
        Get train, validation, and test datasets.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
            Any may be None if not available.
        """
        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_stats(self) -> Dict[str, Dict]:
        """
        Get statistics for all splits.

        Returns:
            Dictionary with stats for each split
        """
        stats = {}

        if self.train_dataset is not None:
            stats['train'] = self.train_dataset.get_stats()

        if self.val_dataset is not None:
            stats['val'] = self.val_dataset.get_stats()

        if self.test_dataset is not None:
            stats['test'] = self.test_dataset.get_stats()

        return stats
