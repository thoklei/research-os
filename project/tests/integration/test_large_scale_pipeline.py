"""
Test suite for large-scale pipeline enhancements - Task 1.

This module tests:
- Memory estimation function
- uint8 dtype conversion
- Progress tracking integration
- Backward compatibility
"""

import numpy as np
import pytest
import io
import sys
from pipeline import estimate_memory, generate_corpus


class TestMemoryEstimation:
    """Test memory estimation function."""

    def test_estimate_memory_1k_images(self):
        """Memory estimate for 1,000 images should be correct."""
        bytes_needed, mb_estimate = estimate_memory(num_images=1000, dtype=np.uint8)

        # 1000 images × 16 × 16 × 1 byte = 256,000 bytes
        assert bytes_needed == 256000
        assert mb_estimate == pytest.approx(0.244, abs=0.01)  # ~0.244 MB

    def test_estimate_memory_100k_images(self):
        """Memory estimate for 100,000 images should be correct."""
        bytes_needed, mb_estimate = estimate_memory(num_images=100000, dtype=np.uint8)

        # 100000 images × 16 × 16 × 1 byte = 25,600,000 bytes
        assert bytes_needed == 25600000
        assert mb_estimate == pytest.approx(24.41, abs=0.1)  # ~24.41 MB

    def test_estimate_memory_int64_dtype(self):
        """Memory estimate for int64 should be 8x larger."""
        bytes_uint8, mb_uint8 = estimate_memory(num_images=1000, dtype=np.uint8)
        bytes_int64, mb_int64 = estimate_memory(num_images=1000, dtype=np.int64)

        assert bytes_int64 == bytes_uint8 * 8
        assert mb_int64 == pytest.approx(mb_uint8 * 8, abs=0.01)

    def test_estimate_memory_single_image(self):
        """Memory estimate for single image should be correct."""
        bytes_needed, mb_estimate = estimate_memory(num_images=1, dtype=np.uint8)

        # 1 image × 16 × 16 × 1 byte = 256 bytes
        assert bytes_needed == 256
        assert mb_estimate < 0.001  # Very small MB value

    def test_estimate_memory_returns_tuple(self):
        """estimate_memory should return (bytes, mb) tuple."""
        result = estimate_memory(num_images=100, dtype=np.uint8)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)

    def test_estimate_memory_different_dtypes(self):
        """Memory estimate should work with different dtypes."""
        dtypes = [np.uint8, np.int16, np.int32, np.int64]
        sizes = [1, 2, 4, 8]

        for dtype, size in zip(dtypes, sizes):
            bytes_needed, _ = estimate_memory(num_images=1000, dtype=dtype)
            expected = 1000 * 16 * 16 * size
            assert bytes_needed == expected


class TestUint8Conversion:
    """Test uint8 dtype conversion in generate_corpus."""

    def test_generate_corpus_uint8_dtype(self):
        """Generated corpus should use uint8 dtype."""
        corpus = generate_corpus(corpus_size=5, dtype=np.uint8)

        for grid in corpus:
            assert grid.data.dtype == np.uint8

    def test_generate_corpus_default_dtype(self):
        """Default dtype should be uint8."""
        corpus = generate_corpus(corpus_size=3)

        for grid in corpus:
            assert grid.data.dtype == np.uint8

    def test_generate_corpus_int64_dtype(self):
        """Should support int64 dtype for backward compatibility."""
        corpus = generate_corpus(corpus_size=3, dtype=np.int64)

        for grid in corpus:
            assert grid.data.dtype == np.int64

    def test_uint8_preserves_values(self):
        """uint8 conversion should preserve color values (0-9)."""
        corpus = generate_corpus(corpus_size=10, dtype=np.uint8)

        for grid in corpus:
            unique_values = set(grid.data.flatten())
            # All values should be in range [0, 9]
            for value in unique_values:
                assert 0 <= value <= 9

    def test_uint8_memory_savings(self):
        """uint8 should use less memory than int64."""
        np.random.seed(42)
        corpus_uint8 = generate_corpus(corpus_size=100, dtype=np.uint8)

        np.random.seed(42)
        corpus_int64 = generate_corpus(corpus_size=100, dtype=np.int64)

        # Check memory usage
        uint8_bytes = sum(g.data.nbytes for g in corpus_uint8)
        int64_bytes = sum(g.data.nbytes for g in corpus_int64)

        # int64 should use 8x more memory
        assert int64_bytes == uint8_bytes * 8


class TestProgressTracking:
    """Test progress tracking with tqdm."""

    def test_generate_corpus_with_progress(self):
        """Generate corpus with progress bar should work."""
        corpus = generate_corpus(corpus_size=10, show_progress=True)

        assert len(corpus) == 10
        for grid in corpus:
            assert grid.data.shape == (16, 16)

    def test_generate_corpus_without_progress(self):
        """Generate corpus without progress bar should work."""
        corpus = generate_corpus(corpus_size=10, show_progress=False)

        assert len(corpus) == 10
        for grid in corpus:
            assert grid.data.shape == (16, 16)

    def test_progress_default_disabled(self):
        """Progress bar should be disabled by default for backward compatibility."""
        # This should not show progress bar
        corpus = generate_corpus(corpus_size=5)

        assert len(corpus) == 5

    def test_progress_tracking_large_corpus(self):
        """Progress tracking should work with larger corpus."""
        corpus = generate_corpus(corpus_size=100, show_progress=True)

        assert len(corpus) == 100


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_existing_demo_still_works(self):
        """Existing demo.py usage should still work unchanged."""
        # This is the existing usage pattern
        corpus = generate_corpus(corpus_size=10)

        assert len(corpus) == 10
        for grid in corpus:
            assert grid.data.shape == (16, 16)

    def test_no_required_parameters_added(self):
        """No new required parameters should break existing code."""
        # Should work with just corpus_size
        corpus = generate_corpus(corpus_size=5)
        assert len(corpus) == 5

    def test_existing_visualizations_work(self):
        """Existing visualization code should still work."""
        from visualization import save_corpus, load_corpus
        import tempfile
        import os

        corpus = generate_corpus(corpus_size=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.npz")
            save_corpus(corpus, filepath)

            loaded = load_corpus(filepath)
            assert 'images' in loaded
            assert loaded['images'].shape[0] == 3

    def test_split_corpus_still_works(self):
        """Existing split_corpus function should still work."""
        from pipeline import split_corpus

        corpus = generate_corpus(corpus_size=20)
        train, val, test = split_corpus(corpus)

        assert len(train) + len(val) + len(test) == 20

    def test_reproducibility_maintained(self):
        """Random seed behavior should be maintained."""
        np.random.seed(42)
        corpus1 = generate_corpus(corpus_size=5)

        np.random.seed(42)
        corpus2 = generate_corpus(corpus_size=5)

        for g1, g2 in zip(corpus1, corpus2):
            assert np.array_equal(g1.data, g2.data)


class TestIntegration:
    """Integration tests for enhanced pipeline."""

    def test_full_workflow_with_uint8(self):
        """Complete workflow with uint8 dtype."""
        # Estimate memory
        bytes_needed, mb_estimate = estimate_memory(num_images=50, dtype=np.uint8)
        assert mb_estimate < 1.0  # Should be < 1 MB

        # Generate corpus with progress
        corpus = generate_corpus(corpus_size=50, dtype=np.uint8, show_progress=True)
        assert len(corpus) == 50

        # Verify dtype
        for grid in corpus:
            assert grid.data.dtype == np.uint8

    def test_memory_estimate_accuracy(self):
        """Memory estimate should match actual usage."""
        num_images = 100
        bytes_needed, _ = estimate_memory(num_images=num_images, dtype=np.uint8)

        corpus = generate_corpus(corpus_size=num_images, dtype=np.uint8)
        actual_bytes = sum(g.data.nbytes for g in corpus)

        # Should match closely
        assert actual_bytes == bytes_needed

    def test_large_corpus_feasibility(self):
        """Verify 1K corpus is feasible."""
        bytes_needed, mb_estimate = estimate_memory(num_images=1000, dtype=np.uint8)

        # Should be very reasonable memory usage
        assert mb_estimate < 1.0
        assert bytes_needed == 256000
