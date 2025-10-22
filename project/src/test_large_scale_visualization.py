"""
Test suite for large-scale visualization enhancements - Task 2.

This module tests:
- Metadata generation and storage
- Dataset validation
- uint8 handling in visualization functions
"""

import numpy as np
import pytest
import json
import os
import tempfile
from datetime import datetime
from pipeline import generate_corpus, split_corpus
from visualization import (
    create_metadata,
    save_metadata,
    validate_dataset,
    save_corpus,
    load_corpus,
)


class TestMetadataGeneration:
    """Test metadata generation."""

    def test_create_metadata_basic(self):
        """create_metadata should generate basic metadata."""
        corpus = generate_corpus(corpus_size=10)
        metadata = create_metadata(
            num_images=10,
            dtype='uint8',
            seed=42,
            version='v1'
        )

        assert 'num_images' in metadata
        assert 'dtype' in metadata
        assert 'seed' in metadata
        assert 'version' in metadata
        assert 'timestamp' in metadata
        assert 'grid_size' in metadata

    def test_create_metadata_includes_timestamp(self):
        """Metadata should include ISO format timestamp."""
        metadata = create_metadata(num_images=100)

        assert 'timestamp' in metadata
        # Verify it's valid ISO format
        datetime.fromisoformat(metadata['timestamp'])

    def test_create_metadata_includes_version(self):
        """Metadata should include version."""
        metadata = create_metadata(num_images=100, version='v2')

        assert metadata['version'] == 'v2'

    def test_create_metadata_includes_parameters(self):
        """Metadata should include generation parameters."""
        metadata = create_metadata(
            num_images=1000,
            dtype='uint8',
            seed=123,
            version='v1',
            object_types=['blob', 'rectangle', 'line', 'pattern'],
            object_distribution=[0.4, 0.2, 0.2, 0.2]
        )

        assert metadata['num_images'] == 1000
        assert metadata['dtype'] == 'uint8'
        assert metadata['seed'] == 123
        assert 'object_types' in metadata
        assert 'object_distribution' in metadata

    def test_create_metadata_includes_grid_size(self):
        """Metadata should include grid dimensions."""
        metadata = create_metadata(num_images=100)

        assert 'grid_size' in metadata
        assert metadata['grid_size'] == [16, 16]

    def test_create_metadata_optional_parameters(self):
        """Metadata should work with minimal parameters."""
        metadata = create_metadata(num_images=50)

        # Should have defaults
        assert metadata['num_images'] == 50
        assert 'timestamp' in metadata
        assert 'version' in metadata


class TestMetadataSaving:
    """Test metadata file saving."""

    def test_save_metadata_creates_file(self):
        """save_metadata should create JSON file."""
        metadata = create_metadata(num_images=100, version='v1')

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'metadata.json')
            save_metadata(metadata, filepath)

            assert os.path.exists(filepath)

    def test_save_metadata_valid_json(self):
        """Saved metadata should be valid JSON."""
        metadata = create_metadata(num_images=100, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'metadata.json')
            save_metadata(metadata, filepath)

            # Load and verify
            with open(filepath, 'r') as f:
                loaded = json.load(f)

            assert loaded['num_images'] == 100
            assert loaded['seed'] == 42

    def test_save_metadata_pretty_printed(self):
        """Saved JSON should be human-readable (indented)."""
        metadata = create_metadata(num_images=50)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'metadata.json')
            save_metadata(metadata, filepath)

            # Read raw content
            with open(filepath, 'r') as f:
                content = f.read()

            # Should have indentation (multiple lines)
            assert '\n' in content
            assert '  ' in content  # Has indentation

    def test_save_and_load_metadata_roundtrip(self):
        """Metadata should survive save/load roundtrip."""
        original = create_metadata(
            num_images=1000,
            dtype='uint8',
            seed=999,
            version='v3'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'metadata.json')
            save_metadata(original, filepath)

            with open(filepath, 'r') as f:
                loaded = json.load(f)

            # All fields should match
            assert loaded['num_images'] == original['num_images']
            assert loaded['dtype'] == original['dtype']
            assert loaded['seed'] == original['seed']
            assert loaded['version'] == original['version']


class TestDatasetValidation:
    """Test dataset validation functions."""

    def test_validate_dataset_valid_corpus(self):
        """validate_dataset should pass for valid corpus."""
        corpus = generate_corpus(corpus_size=20, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'corpus.npz')
            save_corpus(corpus, filepath)

            # Should not raise
            is_valid, message = validate_dataset(filepath)
            assert is_valid
            assert 'valid' in message.lower()

    def test_validate_dataset_checks_shape(self):
        """validate_dataset should verify grid shapes."""
        corpus = generate_corpus(corpus_size=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'corpus.npz')
            save_corpus(corpus, filepath)

            is_valid, message = validate_dataset(filepath)
            assert is_valid

    def test_validate_dataset_checks_dtype(self):
        """validate_dataset should verify dtype is uint8."""
        corpus = generate_corpus(corpus_size=5, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'corpus.npz')
            save_corpus(corpus, filepath)

            loaded = load_corpus(filepath)
            assert loaded['images'].dtype == np.uint8

            is_valid, message = validate_dataset(filepath)
            assert is_valid

    def test_validate_dataset_checks_value_range(self):
        """validate_dataset should verify values in [0, 9]."""
        corpus = generate_corpus(corpus_size=10, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'corpus.npz')
            save_corpus(corpus, filepath)

            is_valid, message = validate_dataset(filepath)
            assert is_valid

    def test_validate_dataset_returns_tuple(self):
        """validate_dataset should return (bool, str) tuple."""
        corpus = generate_corpus(corpus_size=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'corpus.npz')
            save_corpus(corpus, filepath)

            result = validate_dataset(filepath)
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], bool)
            assert isinstance(result[1], str)

    def test_validate_dataset_with_splits(self):
        """validate_dataset should work with train/val/test splits."""
        corpus = generate_corpus(corpus_size=20)
        train, val, test = split_corpus(corpus)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'corpus.npz')
            save_corpus(corpus, filepath, train=train, val=val, test=test)

            is_valid, message = validate_dataset(filepath)
            assert is_valid


class TestUint8Visualization:
    """Test uint8 handling in visualization functions."""

    def test_save_corpus_preserves_uint8(self):
        """save_corpus should preserve uint8 dtype."""
        corpus = generate_corpus(corpus_size=10, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'corpus.npz')
            save_corpus(corpus, filepath)

            loaded = load_corpus(filepath)
            assert loaded['images'].dtype == np.uint8

    def test_save_corpus_uint8_space_efficiency(self):
        """uint8 corpus should use less disk space."""
        corpus_uint8 = generate_corpus(corpus_size=100, dtype=np.uint8)
        corpus_int64 = generate_corpus(corpus_size=100, dtype=np.int64)

        with tempfile.TemporaryDirectory() as tmpdir:
            path_uint8 = os.path.join(tmpdir, 'uint8.npz')
            path_int64 = os.path.join(tmpdir, 'int64.npz')

            save_corpus(corpus_uint8, path_uint8)
            save_corpus(corpus_int64, path_int64)

            size_uint8 = os.path.getsize(path_uint8)
            size_int64 = os.path.getsize(path_int64)

            # uint8 should be significantly smaller
            assert size_uint8 < size_int64

    def test_load_corpus_handles_uint8(self):
        """load_corpus should handle uint8 correctly."""
        corpus = generate_corpus(corpus_size=15, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'corpus.npz')
            save_corpus(corpus, filepath)

            loaded = load_corpus(filepath)

            # Check dtype preserved
            assert loaded['images'].dtype == np.uint8

            # Check values in range
            assert np.min(loaded['images']) >= 0
            assert np.max(loaded['images']) <= 9


class TestIntegration:
    """Integration tests for enhanced visualization."""

    def test_full_workflow_with_metadata(self):
        """Complete workflow: generate -> save with metadata -> validate."""
        corpus = generate_corpus(corpus_size=50, dtype=np.uint8)
        train, val, test = split_corpus(corpus)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save corpus
            corpus_path = os.path.join(tmpdir, 'corpus.npz')
            save_corpus(corpus, corpus_path, train=train, val=val, test=test)

            # Save metadata
            metadata = create_metadata(
                num_images=50,
                dtype='uint8',
                seed=42,
                version='v1'
            )
            metadata_path = os.path.join(tmpdir, 'metadata.json')
            save_metadata(metadata, metadata_path)

            # Validate dataset
            is_valid, message = validate_dataset(corpus_path)
            assert is_valid

            # Verify files exist
            assert os.path.exists(corpus_path)
            assert os.path.exists(metadata_path)

    def test_1k_dataset_workflow(self):
        """Test workflow for 1K dataset scale."""
        from pipeline import estimate_memory

        # Estimate memory
        bytes_needed, mb_estimate = estimate_memory(num_images=1000, dtype=np.uint8)
        assert mb_estimate < 1.0  # Should be < 1 MB

        # Generate (smaller sample for test speed)
        corpus = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = os.path.join(tmpdir, 'corpus_1k.npz')
            save_corpus(corpus, corpus_path)

            # Create and save metadata
            metadata = create_metadata(
                num_images=100,
                dtype='uint8',
                version='v1'
            )
            metadata_path = os.path.join(tmpdir, 'metadata.json')
            save_metadata(metadata, metadata_path)

            # Validate
            is_valid, message = validate_dataset(corpus_path)
            assert is_valid

    def test_metadata_includes_all_required_fields(self):
        """Metadata should include all required fields for tracking."""
        metadata = create_metadata(
            num_images=1000,
            dtype='uint8',
            seed=42,
            version='v1'
        )

        required_fields = ['num_images', 'dtype', 'seed', 'version', 'timestamp', 'grid_size']
        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"
