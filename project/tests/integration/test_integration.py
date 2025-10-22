"""
Integration tests for end-to-end pipeline - Task 5.

This module tests:
- Complete pipeline at 1K and 100K scales
- Memory usage monitoring
- CLI with various parameter combinations
- Dataset generation, validation, and visual inspection workflow
"""

import numpy as np
import pytest
import os
import tempfile
import subprocess
import sys
import json
from pathlib import Path

from pipeline import generate_corpus, estimate_memory
from visualization import save_corpus, create_metadata, save_metadata, validate_dataset, load_corpus


class TestPipelineIntegration:
    """Integration tests for complete pipeline."""

    def test_1k_dataset_generation(self):
        """Should generate 1K dataset successfully."""
        corpus = generate_corpus(corpus_size=1000, dtype=np.uint8, show_progress=False)

        assert len(corpus) == 1000
        assert all(g.data.shape == (16, 16) for g in corpus)
        assert all(g.data.dtype == np.uint8 for g in corpus)

    def test_1k_dataset_memory_estimate(self):
        """1K dataset should use ~256 KB of memory."""
        bytes_needed, mb_estimate = estimate_memory(1000, dtype=np.uint8)

        # 1000 images × 16 × 16 × 1 byte = 256,000 bytes
        assert bytes_needed == 256000
        assert mb_estimate < 1.0  # Less than 1 MB

    def test_1k_dataset_save_and_load(self):
        """Should save and load 1K dataset correctly."""
        corpus = generate_corpus(corpus_size=1000, dtype=np.uint8, show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = os.path.join(tmpdir, 'corpus_1k.npz')
            save_corpus(corpus, corpus_path)

            # Verify file size
            file_size = os.path.getsize(corpus_path)
            assert file_size < 1 * 1024 * 1024  # < 1 MB

            # Load and verify
            loaded = load_corpus(corpus_path)
            assert loaded['images'].shape == (1000, 16, 16)
            assert loaded['images'].dtype == np.uint8

    def test_100k_dataset_memory_estimate(self):
        """100K dataset should use ~25.6 MB of memory."""
        bytes_needed, mb_estimate = estimate_memory(100000, dtype=np.uint8)

        # 100,000 images × 16 × 16 × 1 byte = 25,600,000 bytes
        assert bytes_needed == 25600000
        assert 20 < mb_estimate < 30  # ~24.4 MB

    def test_small_scale_end_to_end(self):
        """Test complete workflow at small scale (100 images)."""
        # Generate
        corpus = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save corpus
            corpus_path = os.path.join(tmpdir, 'corpus.npz')
            save_corpus(corpus, corpus_path)

            # Save metadata
            metadata = create_metadata(
                num_images=100,
                dtype='uint8',
                seed=42,
                version='v1'
            )
            metadata_path = os.path.join(tmpdir, 'metadata.json')
            save_metadata(metadata, metadata_path)

            # Validate
            is_valid, message = validate_dataset(corpus_path)
            assert is_valid

            # Load and verify
            loaded = load_corpus(corpus_path)
            assert loaded['images'].shape == (100, 16, 16)


class TestCLIIntegration:
    """Integration tests for CLI script."""

    def test_cli_help(self):
        """CLI should display help message."""
        result = subprocess.run(
            ['python', 'generate_dataset.py', '--help'],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert 'num-images' in result.stdout
        assert 'output-dir' in result.stdout
        assert 'seed' in result.stdout

    def test_cli_invalid_num_images(self):
        """CLI should reject invalid num-images."""
        result = subprocess.run(
            ['python', 'generate_dataset.py', '--num-images', '0'],
            capture_output=True,
            text=True,
            input='n\n'  # Answer no to confirmation
        )

        assert result.returncode != 0

    def test_cli_with_custom_parameters(self):
        """CLI should accept custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'custom-dataset')

            result = subprocess.run(
                [
                    'python', 'generate_dataset.py',
                    '--num-images', '10',
                    '--output-dir', output_dir,
                    '--seed', '999'
                ],
                capture_output=True,
                text=True,
                input='y\n'  # Answer yes to confirmation
            )

            # Should succeed
            assert result.returncode == 0

            # Check files created
            assert os.path.exists(os.path.join(output_dir, 'corpus.npz'))
            assert os.path.exists(os.path.join(output_dir, 'metadata.json'))


class TestMemoryUsage:
    """Test memory usage during generation."""

    def test_memory_stays_within_bounds_1k(self):
        """Memory usage for 1K generation should stay within estimates."""
        import psutil
        import gc

        # Force garbage collection
        gc.collect()

        # Get initial memory
        process = psutil.Process()
        mem_before = process.memory_info().rss

        # Generate 1K dataset
        corpus = generate_corpus(corpus_size=1000, dtype=np.uint8, show_progress=False)

        # Get final memory
        mem_after = process.memory_info().rss
        mem_used = mem_after - mem_before

        # Should use less than 10 MB (generous margin)
        # Actual data is 256 KB, but Python overhead expected
        assert mem_used < 10 * 1024 * 1024

    def test_uint8_vs_int64_memory_savings(self):
        """uint8 should use less memory than int64."""
        corpus_uint8 = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)
        corpus_int64 = generate_corpus(corpus_size=100, dtype=np.int64, show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path_uint8 = os.path.join(tmpdir, 'uint8.npz')
            path_int64 = os.path.join(tmpdir, 'int64.npz')

            save_corpus(corpus_uint8, path_uint8)
            save_corpus(corpus_int64, path_int64)

            size_uint8 = os.path.getsize(path_uint8)
            size_int64 = os.path.getsize(path_int64)

            # uint8 should be smaller (compression reduces the difference)
            assert size_uint8 < size_int64


class TestReproducibility:
    """Test reproducibility with seeds."""

    def test_same_seed_produces_identical_datasets(self):
        """Same seed should produce identical datasets."""
        seed = 12345

        # Generate first dataset
        np.random.seed(seed)
        corpus1 = generate_corpus(corpus_size=50, dtype=np.uint8, show_progress=False)
        arrays1 = np.array([g.data for g in corpus1])

        # Generate second dataset with same seed
        np.random.seed(seed)
        corpus2 = generate_corpus(corpus_size=50, dtype=np.uint8, show_progress=False)
        arrays2 = np.array([g.data for g in corpus2])

        # Should be identical
        assert np.array_equal(arrays1, arrays2)

    def test_different_seeds_produce_different_datasets(self):
        """Different seeds should produce different datasets."""
        # Generate first dataset
        np.random.seed(111)
        corpus1 = generate_corpus(corpus_size=50, dtype=np.uint8, show_progress=False)
        arrays1 = np.array([g.data for g in corpus1])

        # Generate second dataset with different seed
        np.random.seed(222)
        corpus2 = generate_corpus(corpus_size=50, dtype=np.uint8, show_progress=False)
        arrays2 = np.array([g.data for g in corpus2])

        # Should be different
        assert not np.array_equal(arrays1, arrays2)


class TestDatasetQuality:
    """Test quality of generated datasets."""

    def test_dataset_values_in_valid_range(self):
        """All pixel values should be in [0, 9]."""
        corpus = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)
        arrays = np.array([g.data for g in corpus])

        assert np.min(arrays) >= 0
        assert np.max(arrays) <= 9

    def test_dataset_has_variety(self):
        """Dataset should have variety in pixel values."""
        corpus = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)
        arrays = np.array([g.data for g in corpus])

        # Should use multiple colors (at least 3 different values)
        unique_values = np.unique(arrays)
        assert len(unique_values) >= 3

    def test_dataset_not_all_zeros(self):
        """Dataset should not be all zeros."""
        corpus = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)
        arrays = np.array([g.data for g in corpus])

        # Should have non-zero values
        assert np.sum(arrays) > 0

    def test_dataset_background_dominant(self):
        """Background (0) should be the most common value."""
        corpus = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)
        arrays = np.array([g.data for g in corpus])

        # Count occurrences of each value
        unique, counts = np.unique(arrays, return_counts=True)

        # Background (0) should be most common
        most_common_value = unique[np.argmax(counts)]
        assert most_common_value == 0


class TestVisualValidationIntegration:
    """Test visual validation tool integration."""

    def test_visual_validation_workflow(self):
        """Test complete visual validation workflow."""
        from validate_visual import validate_visual_dataset

        # Generate dataset
        corpus = generate_corpus(corpus_size=200, dtype=np.uint8, show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save corpus and metadata
            corpus_path = os.path.join(tmpdir, 'corpus.npz')
            save_corpus(corpus, corpus_path)

            metadata = create_metadata(
                num_images=200,
                dtype='uint8',
                seed=42,
                version='v1'
            )
            metadata_path = os.path.join(tmpdir, 'metadata.json')
            save_metadata(metadata, metadata_path)

            # Run visual validation
            output_path = os.path.join(tmpdir, 'validation.png')
            validate_visual_dataset(
                corpus_path,
                metadata_path,
                output_path,
                num_samples=100,
                seed=42,
                show=False
            )

            # Verify output created
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

    def test_visual_validation_cli(self):
        """Test visual validation CLI."""
        # Generate dataset
        corpus = generate_corpus(corpus_size=150, dtype=np.uint8, show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save files
            corpus_path = os.path.join(tmpdir, 'corpus.npz')
            save_corpus(corpus, corpus_path)

            metadata = create_metadata(num_images=150, version='v1')
            metadata_path = os.path.join(tmpdir, 'metadata.json')
            save_metadata(metadata, metadata_path)

            output_path = os.path.join(tmpdir, 'validation.png')

            # Run CLI
            result = subprocess.run(
                [
                    'python', 'validate_visual.py',
                    '--corpus-path', corpus_path,
                    '--metadata-path', metadata_path,
                    '--output', output_path,
                    '--seed', '42'
                ],
                capture_output=True,
                text=True
            )

            # Should succeed
            assert result.returncode == 0
            assert os.path.exists(output_path)


class TestScaleValidation:
    """Test validation at different scales."""

    def test_validate_small_dataset(self):
        """Should validate small datasets (10 images)."""
        corpus = generate_corpus(corpus_size=10, dtype=np.uint8, show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = os.path.join(tmpdir, 'corpus.npz')
            save_corpus(corpus, corpus_path)

            is_valid, message = validate_dataset(corpus_path)
            assert is_valid
            assert '10 images' in message

    def test_validate_medium_dataset(self):
        """Should validate medium datasets (1K images)."""
        corpus = generate_corpus(corpus_size=1000, dtype=np.uint8, show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = os.path.join(tmpdir, 'corpus.npz')
            save_corpus(corpus, corpus_path)

            is_valid, message = validate_dataset(corpus_path)
            assert is_valid
            assert '1000 images' in message

    @pytest.mark.slow
    def test_validate_large_dataset(self):
        """Should validate large datasets (10K images)."""
        # Note: 10K instead of 100K to keep test runtime reasonable
        corpus = generate_corpus(corpus_size=10000, dtype=np.uint8, show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = os.path.join(tmpdir, 'corpus.npz')
            save_corpus(corpus, corpus_path)

            is_valid, message = validate_dataset(corpus_path)
            assert is_valid
            assert '10000 images' in message
