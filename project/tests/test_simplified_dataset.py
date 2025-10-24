"""
Test suite for simplified dataset generation.

This module tests:
- SimplifiedImageGenerator creates shape-only images (no blobs)
- Dataset generation with correct format and splits
- File size is approximately 25MB for 100k samples
- Output .npz file has correct structure
"""

import numpy as np
import pytest
import tempfile
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from generate_simplified_dataset import (
    SimplifiedImageGenerator,
    generate_simplified_corpus,
    split_dataset,
    estimate_memory,
)
from atomic_generator import validate_object_size


class TestSimplifiedImageGenerator:
    """Test simplified image generator (shapes only, no blobs)."""

    def test_generator_creates_grid(self):
        """SimplifiedImageGenerator should create a Grid with proper dimensions."""
        generator = SimplifiedImageGenerator()
        grid = generator.generate()

        assert grid.data.shape == (16, 16)
        assert grid.data.dtype in [np.int64, np.uint8]

    def test_generator_excludes_blobs(self):
        """SimplifiedImageGenerator should not have BlobGenerator."""
        generator = SimplifiedImageGenerator()

        # Check that generator only has shape generators
        assert hasattr(generator, 'rectangle_generator')
        assert hasattr(generator, 'line_generator')
        assert hasattr(generator, 'pattern_generator')
        assert not hasattr(generator, 'blob_generator')

    def test_generator_uses_correct_color_range(self):
        """Generator should use 10-color palette (0=background, 1-9=objects)."""
        generator = SimplifiedImageGenerator(num_objects=3)
        grid = generator.generate()

        # Check that all values are in range [0, 9]
        assert np.all(grid.data >= 0)
        assert np.all(grid.data <= 9)

    def test_generator_respects_num_objects(self):
        """Generator should attempt to place specified number of objects."""
        generator = SimplifiedImageGenerator(num_objects=2)
        grid = generator.generate()

        # Count non-background pixels (should have at least some objects)
        non_background = np.sum(grid.data != 0)
        assert non_background > 0  # At least some objects placed

    def test_generator_random_num_objects(self):
        """Generator with num_objects=None should use random 1-6 range."""
        generator = SimplifiedImageGenerator(num_objects=None)

        # Generate multiple grids and check they have varying object counts
        non_zero_counts = []
        for _ in range(10):
            grid = generator.generate()
            non_zero_counts.append(np.sum(grid.data != 0))

        # Should have variation in object counts
        assert len(set(non_zero_counts)) > 1


class TestDatasetGeneration:
    """Test full dataset generation pipeline."""

    def test_generate_small_corpus(self):
        """Should generate small corpus with correct shape."""
        corpus_size = 10
        images = generate_simplified_corpus(corpus_size, show_progress=False)

        assert images.shape == (corpus_size, 16, 16)
        assert images.dtype == np.uint8

    def test_corpus_contains_shapes_only(self):
        """Generated corpus should contain images with shapes (not all black)."""
        images = generate_simplified_corpus(5, show_progress=False)

        for img in images:
            # Each image should have at least some non-background pixels
            non_background = np.sum(img != 0)
            assert non_background > 0, "Image should not be all background"

    def test_corpus_uses_10_color_palette(self):
        """Corpus images should use 10-color palette (0-9)."""
        images = generate_simplified_corpus(5, show_progress=False)

        for img in images:
            assert np.all(img >= 0)
            assert np.all(img <= 9)


class TestDatasetSplitting:
    """Test dataset splitting functionality."""

    def test_split_dataset_correct_ratios(self):
        """Dataset should split into correct train/val/test ratios."""
        images = np.random.randint(0, 10, size=(1000, 16, 16), dtype=np.uint8)
        train, val, test = split_dataset(images)

        # Check approximate ratios (80/10/10)
        assert len(train) == 800
        assert len(val) == 100
        assert len(test) == 100

    def test_split_dataset_no_data_loss(self):
        """Split should preserve all samples."""
        images = np.random.randint(0, 10, size=(100, 16, 16), dtype=np.uint8)
        train, val, test = split_dataset(images)

        total_samples = len(train) + len(val) + len(test)
        assert total_samples == 100

    def test_split_dataset_shuffles_data(self):
        """Split should shuffle data (not just sequential splits)."""
        # Create dataset with sequential pattern
        images = np.arange(100).reshape(100, 1, 1) * np.ones((100, 16, 16), dtype=np.uint8)

        train, val, test = split_dataset(images)

        # Check that train doesn't just contain first 80 samples
        train_means = [img.mean() for img in train[:10]]
        # If shuffled, means should not be strictly sequential 0, 1, 2, ...
        is_sequential = all(train_means[i] < train_means[i+1] for i in range(len(train_means)-1))
        assert not is_sequential or len(set(train_means)) < len(train_means), \
            "Data should be shuffled, not sequential"


class TestMemoryEstimation:
    """Test memory estimation functionality."""

    def test_estimate_memory_100k(self):
        """Should correctly estimate ~25MB for 100k uint8 images."""
        bytes_needed, mb_estimate = estimate_memory(100000, dtype=np.uint8)

        # 100k * 16 * 16 * 1 byte = 25,600,000 bytes = 25.6 MB
        expected_bytes = 100000 * 16 * 16 * 1
        assert bytes_needed == expected_bytes

        expected_mb = expected_bytes / (1024 * 1024)
        assert abs(mb_estimate - expected_mb) < 0.01

    def test_estimate_memory_scales_correctly(self):
        """Memory estimate should scale linearly with num_images."""
        bytes_1k, mb_1k = estimate_memory(1000, dtype=np.uint8)
        bytes_10k, mb_10k = estimate_memory(10000, dtype=np.uint8)

        # 10k should be exactly 10x 1k
        assert bytes_10k == bytes_1k * 10
        assert abs(mb_10k - mb_1k * 10) < 0.01


class TestOutputFormat:
    """Test .npz output file format."""

    def test_npz_file_creation(self):
        """Should create .npz file with correct structure."""
        # Generate small dataset
        images = generate_simplified_corpus(100, show_progress=False)
        train, val, test = split_dataset(images)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            np.savez_compressed(tmp_path, train=train, val=val, test=test)

            # Load and verify
            data = np.load(tmp_path)

            assert 'train' in data
            assert 'val' in data
            assert 'test' in data

            assert data['train'].shape == (80, 16, 16)
            assert data['val'].shape == (10, 16, 16)
            assert data['test'].shape == (10, 16, 16)

        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)

    def test_npz_file_size_small_dataset(self):
        """Small dataset file size should be reasonable."""
        # Generate 1000 images
        images = generate_simplified_corpus(1000, show_progress=False)
        train, val, test = split_dataset(images)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            np.savez_compressed(tmp_path, train=train, val=val, test=test)

            # Check file size
            file_size = Path(tmp_path).stat().st_size

            # 1000 images * 256 bytes = 256KB (uncompressed)
            # Compressed should be similar or smaller
            expected_size = 1000 * 16 * 16
            # Allow generous tolerance for compression variance
            assert file_size < expected_size * 2, \
                f"File size {file_size} too large (expected ~{expected_size})"

        finally:
            Path(tmp_path).unlink(missing_ok=True)


@pytest.mark.slow
class TestFullDatasetGeneration:
    """
    Integration tests for full 100k dataset generation.

    These tests are marked as 'slow' and should be run separately:
        pytest test_simplified_dataset.py -m slow
    """

    def test_generate_100k_dataset_size(self):
        """100k dataset should be approximately 25MB."""
        images = generate_simplified_corpus(100000, show_progress=False)
        train, val, test = split_dataset(images)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            np.savez_compressed(tmp_path, train=train, val=val, test=test)

            file_size = Path(tmp_path).stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            # Expected: 25.6 MB uncompressed, compressed should be similar
            # Allow ±5MB tolerance for compression variance
            assert 20 <= file_size_mb <= 30, \
                f"File size {file_size_mb:.2f}MB outside expected range (20-30MB)"

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_100k_dataset_sample_diversity(self):
        """100k dataset should have diverse color usage."""
        images = generate_simplified_corpus(100000, show_progress=False)

        # Count color occurrences across all images
        color_counts = np.bincount(images.flatten(), minlength=10)

        # All 10 colors should appear (0-9)
        for color in range(10):
            assert color_counts[color] > 0, \
                f"Color {color} not found in dataset (no diversity)"

        # Background (0) should be most common
        assert color_counts[0] > color_counts[1:].sum() * 0.5, \
            "Background should be majority of pixels"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
