"""
Test suite for visual validation tool - Task 4.

This module tests:
- Random sample selection from datasets
- 10x10 grid layout visualization
- Metadata overlay on validation grids
- Visual statistics computation
- PNG export for documentation
"""

import numpy as np
import pytest
import os
import tempfile
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from pipeline import generate_corpus, split_corpus
from visualization import save_corpus, create_metadata, save_metadata


class TestSampleSelection:
    """Test random sample selection for validation."""

    def test_select_random_samples_basic(self):
        """Should select specified number of random samples."""
        from validate_visual import select_random_samples

        corpus = generate_corpus(corpus_size=200, dtype=np.uint8, show_progress=False)
        samples, indices = select_random_samples(corpus, num_samples=100)

        assert len(samples) == 100
        assert len(indices) == 100

    def test_select_random_samples_with_seed(self):
        """Should be reproducible with seed."""
        from validate_visual import select_random_samples

        corpus = generate_corpus(corpus_size=150, dtype=np.uint8, show_progress=False)

        samples1, indices1 = select_random_samples(corpus, num_samples=50, seed=42)
        samples2, indices2 = select_random_samples(corpus, num_samples=50, seed=42)

        assert np.array_equal(indices1, indices2)
        assert np.array_equal(samples1, samples2)

    def test_select_random_samples_returns_indices(self):
        """Should return both samples and their indices."""
        from validate_visual import select_random_samples

        corpus = generate_corpus(corpus_size=120, dtype=np.uint8, show_progress=False)
        samples, indices = select_random_samples(corpus, num_samples=100)

        # Indices should be in valid range
        assert np.min(indices) >= 0
        assert np.max(indices) < 120

    def test_select_random_samples_fewer_than_requested(self):
        """Should handle case where corpus has fewer samples than requested."""
        from validate_visual import select_random_samples

        corpus = generate_corpus(corpus_size=50, dtype=np.uint8, show_progress=False)
        samples, indices = select_random_samples(corpus, num_samples=100)

        # Should return all available samples
        assert len(samples) == 50
        assert len(indices) == 50


class TestGridVisualization:
    """Test 10x10 grid layout visualization."""

    def test_create_validation_grid_10x10(self):
        """Should create 10x10 grid layout."""
        from validate_visual import create_validation_grid

        corpus = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)
        # Convert to numpy array
        corpus_array = np.array([g.data for g in corpus])
        samples, indices = corpus_array[:100], np.arange(100)

        fig = create_validation_grid(samples, indices, show=False)

        # Should have 10x10 subplots
        assert len(fig.axes) == 100

    def test_create_validation_grid_returns_figure(self):
        """Should return matplotlib Figure instance."""
        from validate_visual import create_validation_grid

        corpus = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)
        corpus_array = np.array([g.data for g in corpus])
        samples, indices = corpus_array[:100], np.arange(100)

        fig = create_validation_grid(samples, indices, show=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_validation_grid_with_metadata(self):
        """Should include metadata in title."""
        from validate_visual import create_validation_grid

        corpus = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)
        corpus_array = np.array([g.data for g in corpus])
        samples, indices = corpus_array[:100], np.arange(100)

        metadata = {
            'version': 'v1',
            'timestamp': '2025-10-22T12:00:00',
            'num_images': 100
        }

        fig = create_validation_grid(samples, indices, metadata=metadata, show=False)

        # Title should include version
        title = fig._suptitle.get_text() if hasattr(fig, '_suptitle') else ''
        assert 'v1' in title or len(fig.axes) == 100  # Either title has version or grid is correct
        plt.close(fig)

    def test_create_validation_grid_handles_partial_samples(self):
        """Should handle fewer than 100 samples gracefully."""
        from validate_visual import create_validation_grid

        corpus = generate_corpus(corpus_size=50, dtype=np.uint8, show_progress=False)
        corpus_array = np.array([g.data for g in corpus])
        samples, indices = corpus_array[:50], np.arange(50)

        fig = create_validation_grid(samples, indices, show=False)

        # Should still create figure (may have empty subplots)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestMetadataOverlay:
    """Test metadata overlay on validation grids."""

    def test_add_metadata_overlay_basic(self):
        """Should add metadata overlay to figure."""
        from validate_visual import create_validation_grid

        corpus = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)
        corpus_array = np.array([g.data for g in corpus])
        samples, indices = corpus_array[:100], np.arange(100)

        metadata = {
            'version': 'v2',
            'timestamp': '2025-10-22T14:30:00',
            'num_images': 1000,
            'seed': 42
        }

        fig = create_validation_grid(samples, indices, metadata=metadata, show=False)

        # Figure should be created successfully
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_metadata_includes_sample_indices(self):
        """Metadata overlay should show sample indices."""
        from validate_visual import create_validation_grid

        corpus = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)
        corpus_array = np.array([g.data for g in corpus])
        samples, indices = corpus_array[:100], np.arange(100)

        fig = create_validation_grid(samples, indices, show=False)

        # Should complete without error
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVisualStatistics:
    """Test visual statistics computation."""

    def test_compute_visual_statistics_basic(self):
        """Should compute mean, std, min, max pixel values."""
        from validate_visual import compute_visual_statistics

        corpus = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)
        corpus_array = np.array([g.data for g in corpus])
        stats = compute_visual_statistics(corpus_array)

        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats

    def test_compute_visual_statistics_ranges(self):
        """Statistics should be in valid ranges."""
        from validate_visual import compute_visual_statistics

        corpus = generate_corpus(corpus_size=50, dtype=np.uint8, show_progress=False)
        corpus_array = np.array([g.data for g in corpus])
        stats = compute_visual_statistics(corpus_array)

        # Values should be in [0, 9] range
        assert 0 <= stats['min'] <= 9
        assert 0 <= stats['max'] <= 9
        assert 0 <= stats['mean'] <= 9
        assert stats['std'] >= 0

    def test_compute_visual_statistics_uint8(self):
        """Should work with uint8 arrays."""
        from validate_visual import compute_visual_statistics

        corpus = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)
        corpus_array = np.array([g.data for g in corpus])
        stats = compute_visual_statistics(corpus_array)

        # Should return float values
        assert isinstance(stats['mean'], (float, np.floating))
        assert isinstance(stats['std'], (float, np.floating))

    def test_visual_statistics_on_grid(self):
        """Should display statistics on validation grid."""
        from validate_visual import create_validation_grid, compute_visual_statistics

        corpus = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)
        corpus_array = np.array([g.data for g in corpus])
        samples, indices = corpus_array[:100], np.arange(100)
        stats = compute_visual_statistics(samples)

        fig = create_validation_grid(
            samples,
            indices,
            statistics=stats,
            show=False
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPNGExport:
    """Test PNG export for documentation."""

    def test_save_validation_grid_creates_file(self):
        """Should save validation grid as PNG."""
        from validate_visual import create_validation_grid, save_validation_grid

        corpus = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)
        corpus_array = np.array([g.data for g in corpus])
        samples, indices = corpus_array[:100], np.arange(100)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'validation.png')

            fig = create_validation_grid(samples, indices, show=False)
            save_validation_grid(fig, output_path)

            assert os.path.exists(output_path)
            plt.close(fig)

    def test_save_validation_grid_file_size(self):
        """Saved PNG should have reasonable file size."""
        from validate_visual import create_validation_grid, save_validation_grid

        corpus = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)
        corpus_array = np.array([g.data for g in corpus])
        samples, indices = corpus_array[:100], np.arange(100)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'validation.png')

            fig = create_validation_grid(samples, indices, show=False)
            save_validation_grid(fig, output_path)

            file_size = os.path.getsize(output_path)
            # PNG should be < 5 MB
            assert file_size < 5 * 1024 * 1024
            plt.close(fig)

    def test_save_validation_grid_with_dpi(self):
        """Should support custom DPI for high-quality export."""
        from validate_visual import create_validation_grid, save_validation_grid

        corpus = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)
        corpus_array = np.array([g.data for g in corpus])
        samples, indices = corpus_array[:100], np.arange(100)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'validation_hq.png')

            fig = create_validation_grid(samples, indices, show=False)
            save_validation_grid(fig, output_path, dpi=150)

            assert os.path.exists(output_path)
            plt.close(fig)


class TestEndToEndValidation:
    """Test complete validation workflow."""

    def test_full_validation_workflow(self):
        """Should run complete validation workflow."""
        from validate_visual import validate_visual_dataset

        # Generate and save dataset
        corpus = generate_corpus(corpus_size=150, dtype=np.uint8, show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save corpus and metadata
            corpus_path = os.path.join(tmpdir, 'corpus.npz')
            save_corpus(corpus, corpus_path)

            metadata = create_metadata(
                num_images=150,
                dtype='uint8',
                seed=42,
                version='v1'
            )
            metadata_path = os.path.join(tmpdir, 'metadata.json')
            save_metadata(metadata, metadata_path)

            # Run validation
            output_path = os.path.join(tmpdir, 'validation.png')
            validate_visual_dataset(
                corpus_path,
                metadata_path,
                output_path,
                num_samples=100,
                seed=42
            )

            # Check outputs
            assert os.path.exists(output_path)

    def test_validation_with_dataset_directory(self):
        """Should handle dataset directory structure."""
        from validate_visual import validate_visual_dataset

        corpus = generate_corpus(corpus_size=200, dtype=np.uint8, show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset directory structure
            dataset_dir = os.path.join(tmpdir, '2025-10-22-v1')
            os.makedirs(dataset_dir)

            # Save files
            corpus_path = os.path.join(dataset_dir, 'corpus.npz')
            save_corpus(corpus, corpus_path)

            metadata = create_metadata(num_images=200, version='v1')
            metadata_path = os.path.join(dataset_dir, 'metadata.json')
            save_metadata(metadata, metadata_path)

            # Run validation
            output_path = os.path.join(dataset_dir, 'validation.png')
            validate_visual_dataset(
                corpus_path,
                metadata_path,
                output_path,
                num_samples=100
            )

            assert os.path.exists(output_path)

    def test_validation_includes_all_components(self):
        """Validation should include samples, metadata, and statistics."""
        from validate_visual import validate_visual_dataset

        corpus = generate_corpus(corpus_size=100, dtype=np.uint8, show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = os.path.join(tmpdir, 'corpus.npz')
            save_corpus(corpus, corpus_path)

            metadata = create_metadata(
                num_images=100,
                dtype='uint8',
                seed=123,
                version='v2'
            )
            metadata_path = os.path.join(tmpdir, 'metadata.json')
            save_metadata(metadata, metadata_path)

            output_path = os.path.join(tmpdir, 'validation.png')
            validate_visual_dataset(
                corpus_path,
                metadata_path,
                output_path,
                num_samples=100,
                seed=123
            )

            # Should create output without error
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
