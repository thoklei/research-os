"""
Test suite for CLI script and directory structure - Task 3.

This module tests:
- Hierarchical directory creation (datasets/YYYY-MM-DD-vN/)
- CLI argument parsing
- Memory estimation with user confirmation flow
- End-to-end generation workflow
"""

import numpy as np
import pytest
import os
import tempfile
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class TestDirectoryStructure:
    """Test hierarchical directory creation."""

    def test_create_dataset_directory(self):
        """Should create hierarchical directory structure."""
        from generate_dataset import create_dataset_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = create_dataset_directory(
                base_dir=tmpdir,
                version='v1'
            )

            assert os.path.exists(output_dir)
            assert os.path.isdir(output_dir)

    def test_directory_naming_format(self):
        """Directory should follow YYYY-MM-DD-vN format."""
        from generate_dataset import create_dataset_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = create_dataset_directory(
                base_dir=tmpdir,
                version='v1'
            )

            dirname = os.path.basename(output_dir)
            # Should match format: YYYY-MM-DD-v1
            parts = dirname.split('-')
            assert len(parts) == 4  # YYYY, MM, DD, v1
            assert parts[3] == 'v1'

            # Verify date is valid
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            datetime(year, month, day)  # Should not raise

    def test_custom_version_naming(self):
        """Should support custom version numbers."""
        from generate_dataset import create_dataset_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = create_dataset_directory(
                base_dir=tmpdir,
                version='v5'
            )

            dirname = os.path.basename(output_dir)
            assert dirname.endswith('-v5')

    def test_datasets_subdirectory(self):
        """Should create datasets subdirectory if specified."""
        from generate_dataset import create_dataset_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = create_dataset_directory(
                base_dir=tmpdir,
                version='v1',
                create_datasets_subdir=True
            )

            # Should have datasets/ in path
            assert 'datasets' in output_dir

    def test_directory_already_exists(self):
        """Should handle case where directory already exists."""
        from generate_dataset import create_dataset_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first time
            output_dir1 = create_dataset_directory(
                base_dir=tmpdir,
                version='v1'
            )

            # Create again - should either return same or increment
            output_dir2 = create_dataset_directory(
                base_dir=tmpdir,
                version='v1'
            )

            # Both should be valid directories
            assert os.path.exists(output_dir1)
            assert os.path.exists(output_dir2)


class TestCLIArgumentParsing:
    """Test CLI argument parsing."""

    def test_parse_args_default_values(self):
        """Should use default values when args not provided."""
        from generate_dataset import parse_args

        args = parse_args(['--num-images', '100'])

        assert args.num_images == 100
        # Defaults
        assert args.seed is None
        assert args.output_dir is None

    def test_parse_args_num_images(self):
        """Should parse num-images argument."""
        from generate_dataset import parse_args

        args = parse_args(['--num-images', '1000'])
        assert args.num_images == 1000

    def test_parse_args_output_dir(self):
        """Should parse output-dir argument."""
        from generate_dataset import parse_args

        args = parse_args(['--num-images', '100', '--output-dir', '/tmp/test'])
        assert args.output_dir == '/tmp/test'

    def test_parse_args_seed(self):
        """Should parse seed argument."""
        from generate_dataset import parse_args

        args = parse_args(['--num-images', '100', '--seed', '42'])
        assert args.seed == 42

    def test_parse_args_all_options(self):
        """Should parse all CLI options together."""
        from generate_dataset import parse_args

        args = parse_args([
            '--num-images', '1000',
            '--output-dir', '/tmp/datasets',
            '--seed', '999'
        ])

        assert args.num_images == 1000
        assert args.output_dir == '/tmp/datasets'
        assert args.seed == 999


class TestMemoryEstimation:
    """Test memory estimation and user confirmation."""

    def test_format_memory_estimate(self):
        """Should format memory estimate in human-readable form."""
        from generate_dataset import format_memory_estimate

        # Test various sizes
        assert 'KB' in format_memory_estimate(256000)  # ~256 KB
        assert 'MB' in format_memory_estimate(25600000)  # ~25 MB
        assert 'bytes' in format_memory_estimate(500)  # Small size

    def test_memory_estimate_1k_images(self):
        """Memory estimate for 1K images should be correct."""
        from pipeline import estimate_memory

        bytes_needed, mb_estimate = estimate_memory(1000, dtype=np.uint8)
        assert bytes_needed == 256000
        assert mb_estimate < 1.0  # Less than 1 MB

    def test_memory_estimate_100k_images(self):
        """Memory estimate for 100K images should be correct."""
        from pipeline import estimate_memory

        bytes_needed, mb_estimate = estimate_memory(100000, dtype=np.uint8)
        assert bytes_needed == 25600000
        assert 20 < mb_estimate < 30  # ~24-25 MB


class TestGenerationWorkflow:
    """Test end-to-end generation workflow."""

    def test_generate_small_dataset(self):
        """Should generate small dataset successfully."""
        from generate_dataset import generate_and_save_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'test-dataset')
            os.makedirs(output_dir)

            generate_and_save_dataset(
                num_images=10,
                output_dir=output_dir,
                seed=42,
                version='v1',
                show_progress=False
            )

            # Check files created
            corpus_file = os.path.join(output_dir, 'corpus.npz')
            metadata_file = os.path.join(output_dir, 'metadata.json')

            assert os.path.exists(corpus_file)
            assert os.path.exists(metadata_file)

    def test_generated_dataset_valid(self):
        """Generated dataset should pass validation."""
        from generate_dataset import generate_and_save_dataset
        from visualization import validate_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'test-dataset')
            os.makedirs(output_dir)

            generate_and_save_dataset(
                num_images=20,
                output_dir=output_dir,
                seed=123,
                version='v1',
                show_progress=False
            )

            corpus_file = os.path.join(output_dir, 'corpus.npz')
            is_valid, message = validate_dataset(corpus_file)

            assert is_valid
            assert '20 images' in message

    def test_metadata_matches_generation_params(self):
        """Metadata should match generation parameters."""
        import json
        from generate_dataset import generate_and_save_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'test-dataset')
            os.makedirs(output_dir)

            generate_and_save_dataset(
                num_images=50,
                output_dir=output_dir,
                seed=999,
                version='v2',
                show_progress=False
            )

            metadata_file = os.path.join(output_dir, 'metadata.json')
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            assert metadata['num_images'] == 50
            assert metadata['seed'] == 999
            assert metadata['version'] == 'v2'
            assert metadata['dtype'] == 'uint8'

    def test_reproducible_with_seed(self):
        """Generation should be reproducible with same seed."""
        from generate_dataset import generate_and_save_dataset
        from visualization import load_corpus

        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate first dataset
            output_dir1 = os.path.join(tmpdir, 'dataset1')
            os.makedirs(output_dir1)
            generate_and_save_dataset(
                num_images=10,
                output_dir=output_dir1,
                seed=42,
                version='v1',
                show_progress=False
            )

            # Generate second dataset with same seed
            output_dir2 = os.path.join(tmpdir, 'dataset2')
            os.makedirs(output_dir2)
            generate_and_save_dataset(
                num_images=10,
                output_dir=output_dir2,
                seed=42,
                version='v1',
                show_progress=False
            )

            # Load both
            corpus1 = load_corpus(os.path.join(output_dir1, 'corpus.npz'))
            corpus2 = load_corpus(os.path.join(output_dir2, 'corpus.npz'))

            # Should be identical
            assert np.array_equal(corpus1['images'], corpus2['images'])


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_num_images(self):
        """Should handle invalid num_images."""
        from generate_dataset import parse_args

        with pytest.raises(SystemExit):
            parse_args(['--num-images', '0'])  # Should fail

    def test_negative_num_images(self):
        """Should handle negative num_images."""
        from generate_dataset import parse_args

        with pytest.raises(SystemExit):
            parse_args(['--num-images', '-10'])  # Should fail

    def test_nonexistent_output_dir_created(self):
        """Should create output directory if it doesn't exist."""
        from generate_dataset import generate_and_save_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'new', 'nested', 'dir')

            generate_and_save_dataset(
                num_images=5,
                output_dir=output_dir,
                version='v1',
                show_progress=False
            )

            assert os.path.exists(output_dir)
            assert os.path.exists(os.path.join(output_dir, 'corpus.npz'))


class TestIntegration:
    """Integration tests for complete CLI workflow."""

    def test_full_workflow_small_scale(self):
        """Test complete workflow at small scale."""
        from generate_dataset import generate_and_save_dataset
        from visualization import load_corpus, validate_dataset
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'full-test')
            os.makedirs(output_dir)

            # Generate dataset
            generate_and_save_dataset(
                num_images=30,
                output_dir=output_dir,
                seed=777,
                version='v1',
                show_progress=False
            )

            # Verify files exist
            corpus_file = os.path.join(output_dir, 'corpus.npz')
            metadata_file = os.path.join(output_dir, 'metadata.json')
            assert os.path.exists(corpus_file)
            assert os.path.exists(metadata_file)

            # Validate dataset
            is_valid, message = validate_dataset(corpus_file)
            assert is_valid

            # Check metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            assert metadata['num_images'] == 30
            assert metadata['seed'] == 777

            # Load and verify corpus
            corpus = load_corpus(corpus_file)
            assert corpus['images'].shape == (30, 16, 16)
            assert corpus['images'].dtype == np.uint8
