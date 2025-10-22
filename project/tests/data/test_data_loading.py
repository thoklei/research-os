"""
Test suite for data loading pipeline - Task 2.4

Tests for:
- ARCDataset loading from .npz files
- Data augmentation transforms
- DataLoader creation and batching
- Train/val/test splits
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path


class TestARCDataset:
    """Test ARCDataset class."""

    @pytest.fixture
    def sample_corpus(self):
        """Create a sample corpus for testing."""
        # Create sample data (100 images, 16x16)
        images = np.random.randint(0, 10, (100, 16, 16), dtype=np.uint8)

        # Create temporary .npz file
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f.name, images=images)
            yield f.name

        # Cleanup
        os.unlink(f.name)

    @pytest.fixture
    def sample_corpus_with_splits(self):
        """Create a sample corpus with train/val/test splits."""
        # Create sample data
        train = np.random.randint(0, 10, (80, 16, 16), dtype=np.uint8)
        val = np.random.randint(0, 10, (10, 16, 16), dtype=np.uint8)
        test = np.random.randint(0, 10, (10, 16, 16), dtype=np.uint8)

        # Create temporary .npz file
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f.name, train=train, val=val, test=test)
            yield f.name

        # Cleanup
        os.unlink(f.name)

    def test_dataset_load_all(self, sample_corpus):
        """Test loading all images from corpus."""
        from data.arc_dataset import ARCDataset

        dataset = ARCDataset(sample_corpus, split='all')

        assert len(dataset) == 100
        assert dataset.shape == (100, 16, 16)

    def test_dataset_getitem(self, sample_corpus):
        """Test getting single item from dataset."""
        from data.arc_dataset import ARCDataset

        dataset = ARCDataset(sample_corpus, split='all')

        grid = dataset[0]

        assert isinstance(grid, torch.Tensor)
        assert grid.shape == (16, 16)
        assert grid.dtype == torch.long
        assert grid.min() >= 0
        assert grid.max() < 10

    def test_dataset_with_splits(self, sample_corpus_with_splits):
        """Test loading specific splits."""
        from data.arc_dataset import ARCDataset

        train_dataset = ARCDataset(sample_corpus_with_splits, split='train')
        val_dataset = ARCDataset(sample_corpus_with_splits, split='val')
        test_dataset = ARCDataset(sample_corpus_with_splits, split='test')

        assert len(train_dataset) == 80
        assert len(val_dataset) == 10
        assert len(test_dataset) == 10

    def test_dataset_get_batch(self, sample_corpus):
        """Test getting batch of items."""
        from data.arc_dataset import ARCDataset

        dataset = ARCDataset(sample_corpus, split='all')

        batch = dataset.get_batch([0, 1, 2, 3])

        assert batch.shape == (4, 16, 16)
        assert batch.dtype == torch.long

    def test_dataset_stats(self, sample_corpus):
        """Test dataset statistics."""
        from data.arc_dataset import ARCDataset

        dataset = ARCDataset(sample_corpus, split='all')

        stats = dataset.get_stats()

        assert 'num_samples' in stats
        assert 'color_distribution' in stats
        assert 'sparsity' in stats
        assert stats['num_samples'] == 100
        assert len(stats['color_distribution']) == 10


class TestDatasetWithSplits:
    """Test ARCDatasetWithSplits wrapper."""

    @pytest.fixture
    def sample_corpus_with_splits(self):
        """Create a sample corpus with splits."""
        train = np.random.randint(0, 10, (80, 16, 16), dtype=np.uint8)
        val = np.random.randint(0, 10, (10, 16, 16), dtype=np.uint8)
        test = np.random.randint(0, 10, (10, 16, 16), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f.name, train=train, val=val, test=test)
            yield f.name

        os.unlink(f.name)

    def test_dataset_with_splits_creation(self, sample_corpus_with_splits):
        """Test creating datasets with splits."""
        from data.arc_dataset import ARCDatasetWithSplits

        datasets = ARCDatasetWithSplits(sample_corpus_with_splits)

        train, val, test = datasets.get_datasets()

        assert train is not None
        assert val is not None
        assert test is not None
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_dataset_with_splits_stats(self, sample_corpus_with_splits):
        """Test getting stats for all splits."""
        from data.arc_dataset import ARCDatasetWithSplits

        datasets = ARCDatasetWithSplits(sample_corpus_with_splits)

        stats = datasets.get_stats()

        assert 'train' in stats
        assert 'val' in stats
        assert 'test' in stats


class TestTransforms:
    """Test data augmentation transforms."""

    def test_random_rotation(self):
        """Test random rotation transform."""
        from data.transforms import RandomRotation

        # Create grid with markers at each corner
        grid = torch.zeros(16, 16)
        grid[0, 0] = 1    # top-left
        grid[0, 15] = 2   # top-right
        grid[15, 0] = 3   # bottom-left
        grid[15, 15] = 4  # bottom-right

        transform = RandomRotation(angles=[90], p=1.0)
        rotated = transform(grid)

        # After 90° counter-clockwise rotation:
        # top-left (0,0) → bottom-left (15,0)
        # top-right (0,15) → top-left (0,0)
        # bottom-right (15,15) → top-right (0,15)
        # bottom-left (15,0) → bottom-right (15,15)
        assert rotated.shape == (16, 16)
        assert rotated[15, 0] == 1   # Original top-left
        assert rotated[0, 0] == 2    # Original top-right
        assert rotated[0, 15] == 4   # Original bottom-right
        assert rotated[15, 15] == 3  # Original bottom-left

    def test_random_rotation_preserves_values(self):
        """Test that rotation preserves all color values."""
        from data.transforms import RandomRotation

        grid = torch.randint(0, 10, (16, 16))

        transform = RandomRotation(angles=[90, 180, 270], p=1.0)
        rotated = transform(grid)

        # Should preserve all unique values
        assert set(grid.flatten().tolist()) == set(rotated.flatten().tolist())

    def test_horizontal_flip(self):
        """Test horizontal flip transform."""
        from data.transforms import RandomHorizontalFlip

        grid = torch.zeros(16, 16)
        grid[0, 0] = 1  # Mark top-left

        transform = RandomHorizontalFlip(p=1.0)
        flipped = transform(grid)

        # Horizontal flip: top-left → top-right
        assert flipped.shape == (16, 16)
        assert flipped[0, 15] == 1

    def test_vertical_flip(self):
        """Test vertical flip transform."""
        from data.transforms import RandomVerticalFlip

        grid = torch.zeros(16, 16)
        grid[0, 0] = 1  # Mark top-left

        transform = RandomVerticalFlip(p=1.0)
        flipped = transform(grid)

        # Vertical flip: top-left → bottom-left
        assert flipped.shape == (16, 16)
        assert flipped[15, 0] == 1

    def test_compose_transforms(self):
        """Test composing multiple transforms."""
        from data.transforms import Compose, RandomRotation, RandomHorizontalFlip

        grid = torch.randint(0, 10, (16, 16))

        transform = Compose([
            RandomRotation(angles=[0, 90], p=1.0),
            RandomHorizontalFlip(p=1.0),
        ])

        transformed = transform(grid)

        assert transformed.shape == (16, 16)
        assert transformed.dtype == torch.long

    def test_identity_transform(self):
        """Test identity transform (no-op)."""
        from data.transforms import Identity

        grid = torch.randint(0, 10, (16, 16))

        transform = Identity()
        result = transform(grid)

        assert torch.equal(grid, result)

    def test_get_train_transforms(self):
        """Test getting standard training transforms."""
        from data.transforms import get_train_transforms

        transform = get_train_transforms()

        grid = torch.randint(0, 10, (16, 16))
        transformed = transform(grid)

        assert transformed.shape == (16, 16)
        assert transformed.dtype == torch.long

    def test_transforms_preserve_dtype(self):
        """Test that transforms preserve torch.long dtype."""
        from data.transforms import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip

        grid = torch.randint(0, 10, (16, 16)).long()

        transforms = [
            RandomRotation(angles=[90], p=1.0),
            RandomHorizontalFlip(p=1.0),
            RandomVerticalFlip(p=1.0),
        ]

        for transform in transforms:
            result = transform(grid)
            assert result.dtype == torch.long


class TestDataLoaders:
    """Test DataLoader creation."""

    @pytest.fixture
    def sample_corpus_with_splits(self):
        """Create a sample corpus with splits."""
        train = np.random.randint(0, 10, (80, 16, 16), dtype=np.uint8)
        val = np.random.randint(0, 10, (10, 16, 16), dtype=np.uint8)
        test = np.random.randint(0, 10, (10, 16, 16), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f.name, train=train, val=val, test=test)
            yield f.name

        os.unlink(f.name)

    def test_create_single_data_loader(self, sample_corpus_with_splits):
        """Test creating a single data loader."""
        from data.arc_dataset import ARCDataset
        from data.data_loaders import create_data_loader

        dataset = ARCDataset(sample_corpus_with_splits, split='train')

        loader = create_data_loader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,  # Use 0 for testing
        )

        assert len(loader) > 0
        assert loader.batch_size == 16

    def test_create_data_loaders(self, sample_corpus_with_splits):
        """Test creating train/val/test loaders."""
        from data.data_loaders import create_data_loaders

        train_loader, val_loader, test_loader = create_data_loaders(
            sample_corpus_with_splits,
            batch_size=16,
            num_workers=0,
            use_augmentation=False,
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_data_loader_batching(self, sample_corpus_with_splits):
        """Test that data loader produces correct batch shapes."""
        from data.data_loaders import create_data_loaders

        train_loader, _, _ = create_data_loaders(
            sample_corpus_with_splits,
            batch_size=8,
            num_workers=0,
            use_augmentation=False,
        )

        # Get first batch
        batch = next(iter(train_loader))

        assert batch.shape[0] <= 8  # Batch size (may be smaller for last batch)
        assert batch.shape[1:] == (16, 16)
        assert batch.dtype == torch.long

    def test_data_loader_iteration(self, sample_corpus_with_splits):
        """Test iterating through data loader."""
        from data.data_loaders import create_data_loaders

        train_loader, _, _ = create_data_loaders(
            sample_corpus_with_splits,
            batch_size=16,
            num_workers=0,
            use_augmentation=False,
        )

        total_samples = 0
        for batch in train_loader:
            total_samples += batch.shape[0]
            assert batch.shape[1:] == (16, 16)
            assert batch.min() >= 0
            assert batch.max() < 10

        # Should iterate over all samples (with drop_last=True, may lose a few)
        assert total_samples >= 64  # At least 80% of 80 samples


class TestEndToEnd:
    """Test end-to-end data loading workflow."""

    def test_full_pipeline_with_real_data(self):
        """Test loading real data from test-100k dataset."""
        from data.data_loaders import create_data_loaders
        import os

        # Path to test dataset
        corpus_path = Path(__file__).parent.parent / "datasets" / "test-100k" / "corpus.npz"

        if not corpus_path.exists():
            pytest.skip("test-100k corpus not found")

        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            str(corpus_path),
            batch_size=32,
            num_workers=0,
            use_augmentation=True,
        )

        # Test that we can iterate
        batch = next(iter(train_loader))

        assert batch.shape[0] <= 32
        assert batch.shape[1:] == (16, 16)
        assert batch.dtype == torch.long
        assert batch.min() >= 0
        assert batch.max() < 10
