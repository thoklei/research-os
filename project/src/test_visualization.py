"""
Test suite for visualization and output serialization - Task 5.

This module tests:
- Matplotlib grid visualization with ARC color scheme
- .npz compression and serialization
- Numpy array export with train/val/test splits
- End-to-end pipeline integration
"""

import numpy as np
import pytest
import os
import tempfile
from atomic_generator import Grid
from pipeline import generate_corpus, split_corpus
from visualization import (
    visualize_grid,
    visualize_gallery,
    save_corpus,
    load_corpus,
    ARC_COLORMAP,
)


class TestARCColorScheme:
    """Test ARC color scheme definition."""

    def test_arc_colormap_has_10_colors(self):
        """ARC colormap should have 10 colors (0-9)."""
        assert len(ARC_COLORMAP) == 10

    def test_arc_colormap_includes_background(self):
        """First color should be black background."""
        assert ARC_COLORMAP[0] == '#000000'

    def test_arc_colormap_valid_hex(self):
        """All colors should be valid hex codes."""
        for color in ARC_COLORMAP:
            assert color.startswith('#')
            assert len(color) == 7
            # Verify it's valid hex
            int(color[1:], 16)


class TestGridVisualization:
    """Test matplotlib grid visualization."""

    def test_visualize_grid_accepts_grid(self):
        """visualize_grid should accept Grid instance."""
        grid = Grid()
        grid.data[0, 0] = 3
        # Should not raise
        fig = visualize_grid(grid, show=False)
        assert fig is not None

    def test_visualize_grid_accepts_numpy_array(self):
        """visualize_grid should accept numpy array."""
        array = np.zeros((16, 16), dtype=int)
        array[0, 0] = 5
        # Should not raise
        fig = visualize_grid(array, show=False)
        assert fig is not None

    def test_visualize_grid_with_title(self):
        """visualize_grid should accept custom title."""
        grid = Grid()
        fig = visualize_grid(grid, title="Test Grid", show=False)
        assert fig is not None

    def test_visualize_grid_returns_figure(self):
        """visualize_grid should return matplotlib figure."""
        grid = Grid()
        fig = visualize_grid(grid, show=False)
        # Check it's a matplotlib figure
        assert hasattr(fig, 'savefig')


class TestGalleryVisualization:
    """Test multi-image gallery visualization."""

    def test_visualize_gallery_single_grid(self):
        """Gallery should work with single grid."""
        grids = [Grid()]
        fig = visualize_gallery(grids, show=False)
        assert fig is not None

    def test_visualize_gallery_multiple_grids(self):
        """Gallery should display multiple grids."""
        grids = [Grid() for _ in range(6)]
        fig = visualize_gallery(grids, show=False)
        assert fig is not None

    def test_visualize_gallery_custom_layout(self):
        """Gallery should accept custom grid layout."""
        grids = [Grid() for _ in range(6)]
        fig = visualize_gallery(grids, rows=2, cols=3, show=False)
        assert fig is not None

    def test_visualize_gallery_with_titles(self):
        """Gallery should accept custom titles."""
        grids = [Grid() for _ in range(4)]
        titles = ["Grid 1", "Grid 2", "Grid 3", "Grid 4"]
        fig = visualize_gallery(grids, titles=titles, show=False)
        assert fig is not None


class TestNPZSerialization:
    """Test .npz compression and serialization."""

    def test_save_corpus_creates_file(self):
        """save_corpus should create .npz file."""
        corpus = generate_corpus(corpus_size=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_corpus.npz")
            save_corpus(corpus, filepath)

            assert os.path.exists(filepath)

    def test_save_corpus_with_splits(self):
        """save_corpus should save train/val/test splits."""
        corpus = generate_corpus(corpus_size=10)
        train, val, test = split_corpus(corpus)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_corpus.npz")
            save_corpus(corpus, filepath, train=train, val=val, test=test)

            assert os.path.exists(filepath)

    def test_load_corpus_returns_dict(self):
        """load_corpus should return dictionary of arrays."""
        corpus = generate_corpus(corpus_size=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_corpus.npz")
            save_corpus(corpus, filepath)

            loaded = load_corpus(filepath)
            assert isinstance(loaded, dict)

    def test_save_and_load_corpus(self):
        """Saved and loaded corpus should match."""
        corpus = generate_corpus(corpus_size=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_corpus.npz")
            save_corpus(corpus, filepath)

            loaded = load_corpus(filepath)
            assert 'images' in loaded

            # Check shape
            assert loaded['images'].shape[0] == 5
            assert loaded['images'].shape[1:] == (16, 16)

            # Check values match
            for i, grid in enumerate(corpus):
                assert np.array_equal(loaded['images'][i], grid.data)

    def test_save_and_load_with_splits(self):
        """Saved and loaded splits should match."""
        corpus = generate_corpus(corpus_size=20)
        train, val, test = split_corpus(corpus)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_corpus.npz")
            save_corpus(corpus, filepath, train=train, val=val, test=test)

            loaded = load_corpus(filepath)

            assert 'train' in loaded
            assert 'val' in loaded
            assert 'test' in loaded

            # Check shapes
            assert loaded['train'].shape[0] == len(train)
            assert loaded['val'].shape[0] == len(val)
            assert loaded['test'].shape[0] == len(test)

            # Check total
            total = loaded['train'].shape[0] + loaded['val'].shape[0] + loaded['test'].shape[0]
            assert total == 20


class TestOutputFormat:
    """Test output format compatibility."""

    def test_corpus_array_shape(self):
        """Corpus array should have shape (N, 16, 16)."""
        corpus = generate_corpus(corpus_size=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_corpus.npz")
            save_corpus(corpus, filepath)

            loaded = load_corpus(filepath)
            assert loaded['images'].shape == (5, 16, 16)

    def test_corpus_array_dtype(self):
        """Corpus array should use integer dtype."""
        corpus = generate_corpus(corpus_size=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_corpus.npz")
            save_corpus(corpus, filepath)

            loaded = load_corpus(filepath)
            assert loaded['images'].dtype in [np.int32, np.int64]

    def test_corpus_values_in_range(self):
        """All values should be in range [0, 9]."""
        corpus = generate_corpus(corpus_size=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_corpus.npz")
            save_corpus(corpus, filepath)

            loaded = load_corpus(filepath)
            assert np.min(loaded['images']) >= 0
            assert np.max(loaded['images']) <= 9

    def test_compression_reduces_size(self):
        """Compressed file should be smaller than uncompressed."""
        corpus = generate_corpus(corpus_size=50)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save compressed
            compressed_path = os.path.join(tmpdir, "compressed.npz")
            save_corpus(corpus, compressed_path)

            compressed_size = os.path.getsize(compressed_path)

            # Compressed should be reasonably small
            # 50 images * 16 * 16 * 8 bytes = 102.4 KB uncompressed
            # Compressed should be significantly smaller due to sparse data
            assert compressed_size < 100000  # Less than 100 KB


class TestEndToEndIntegration:
    """Integration tests for complete pipeline."""

    def test_generate_visualize_save_load(self):
        """Complete workflow: generate -> visualize -> save -> load."""
        # Generate
        corpus = generate_corpus(corpus_size=10)
        train, val, test = split_corpus(corpus)

        # Visualize (should not raise)
        fig = visualize_gallery(corpus[:4], show=False)
        assert fig is not None

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "corpus.npz")
            save_corpus(corpus, filepath, train=train, val=val, test=test)

            # Load
            loaded = load_corpus(filepath)

            # Verify
            assert 'images' in loaded
            assert 'train' in loaded
            assert 'val' in loaded
            assert 'test' in loaded

            # Verify split sizes
            assert loaded['train'].shape[0] + loaded['val'].shape[0] + loaded['test'].shape[0] == 10

    def test_reproducible_pipeline(self):
        """Pipeline should be reproducible with seed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run 1
            np.random.seed(42)
            corpus1 = generate_corpus(corpus_size=5)
            path1 = os.path.join(tmpdir, "corpus1.npz")
            save_corpus(corpus1, path1)

            # Run 2
            np.random.seed(42)
            corpus2 = generate_corpus(corpus_size=5)
            path2 = os.path.join(tmpdir, "corpus2.npz")
            save_corpus(corpus2, path2)

            # Load and compare
            loaded1 = load_corpus(path1)
            loaded2 = load_corpus(path2)

            assert np.array_equal(loaded1['images'], loaded2['images'])

    def test_large_corpus_workflow(self):
        """Workflow should handle larger corpus."""
        corpus = generate_corpus(corpus_size=100)
        train, val, test = split_corpus(corpus)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "large_corpus.npz")
            save_corpus(corpus, filepath, train=train, val=val, test=test)

            loaded = load_corpus(filepath)

            # Verify splits
            assert loaded['train'].shape[0] >= 75  # ~80%
            assert loaded['val'].shape[0] >= 5     # ~10%
            assert loaded['test'].shape[0] >= 5    # ~10%

    def test_visualize_loaded_corpus(self):
        """Should be able to visualize loaded corpus."""
        corpus = generate_corpus(corpus_size=6)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "corpus.npz")
            save_corpus(corpus, filepath)

            loaded = load_corpus(filepath)

            # Visualize loaded arrays
            fig = visualize_gallery(loaded['images'][:4], show=False)
            assert fig is not None
