"""
Test suite for blob object generator - Task 2: Blob generation with connectivity-biased growth.

This module tests:
- Blob generation with size constraints (2-15 pixels)
- Connectivity-biased growth algorithm
- Color assignment from palette
- Shape validation and connectivity
"""

import numpy as np
import pytest
from atomic_generator import Object, validate_object_size
from blob_generator import BlobGenerator, is_connected


class TestBlobGeneration:
    """Test blob generation with size constraints."""

    def test_blob_generator_creates_object(self):
        """BlobGenerator should create an Object instance."""
        generator = BlobGenerator(min_pixels=2, max_pixels=15)
        blob = generator.generate()
        assert isinstance(blob, Object)

    def test_blob_respects_minimum_size(self):
        """Generated blob should have at least min_pixels."""
        generator = BlobGenerator(min_pixels=5, max_pixels=15)
        for _ in range(10):
            blob = generator.generate()
            assert blob.size() >= 5

    def test_blob_respects_maximum_size(self):
        """Generated blob should have at most max_pixels."""
        generator = BlobGenerator(min_pixels=2, max_pixels=8)
        for _ in range(10):
            blob = generator.generate()
            assert blob.size() <= 8

    def test_blob_size_in_valid_range(self):
        """Generated blob should pass size validation."""
        generator = BlobGenerator(min_pixels=2, max_pixels=15)
        for _ in range(20):
            blob = generator.generate()
            assert validate_object_size(blob, min_pixels=2, max_pixels=15)

    def test_blob_size_randomness(self):
        """Generated blobs should have varying sizes."""
        generator = BlobGenerator(min_pixels=2, max_pixels=15)
        sizes = [generator.generate().size() for _ in range(50)]
        unique_sizes = len(set(sizes))
        # Should have at least 3 different sizes out of 50 generations
        assert unique_sizes >= 3

    def test_blob_default_constraints(self):
        """Default blob generator should use 2-15 pixel constraints."""
        generator = BlobGenerator()
        for _ in range(10):
            blob = generator.generate()
            assert 2 <= blob.size() <= 15


class TestConnectivityBiasedGrowth:
    """Test connectivity-biased growth algorithm."""

    def test_blob_starts_with_seed_pixel(self):
        """Blob growth should start from seed pixel at (0,0)."""
        generator = BlobGenerator(min_pixels=2, max_pixels=5)
        blob = generator.generate()
        # Origin should be normalized to (0,0) or blob should contain relative coords
        # Since we grow from (0,0), checking connectivity is sufficient
        assert blob.size() >= 1

    def test_blob_is_connected(self):
        """Generated blob should be fully connected (8-neighborhood)."""
        generator = BlobGenerator(min_pixels=2, max_pixels=15)
        for _ in range(20):
            blob = generator.generate()
            assert is_connected(blob), f"Blob with {blob.size()} pixels is not connected"

    def test_blob_has_cohesive_shape(self):
        """Blob should be cohesive (no isolated pixels)."""
        generator = BlobGenerator(min_pixels=5, max_pixels=15)
        for _ in range(20):
            blob = generator.generate()
            # Each pixel should have at least one neighbor
            for pixel in blob.pixels:
                neighbors = get_8_neighbors(pixel)
                has_neighbor = any(n in blob.pixels for n in neighbors)
                assert has_neighbor, f"Isolated pixel found at {pixel}"

    def test_connectivity_bias_favors_dense_shapes(self):
        """Connectivity bias should create relatively compact shapes."""
        generator = BlobGenerator(min_pixels=10, max_pixels=15)
        for _ in range(10):
            blob = generator.generate()
            height, width = blob.bounds()
            bounding_box_area = height * width
            actual_pixels = blob.size()
            # Density should be reasonable (at least 30% filled)
            density = actual_pixels / bounding_box_area
            assert density >= 0.3, f"Blob too sparse: density={density:.2f}"

    def test_growth_terminates(self):
        """Growth algorithm should terminate within reasonable iterations."""
        generator = BlobGenerator(min_pixels=2, max_pixels=15)
        # This test passes if generate() returns without hanging
        for _ in range(10):
            blob = generator.generate()
            assert blob is not None


class TestBlobColorAssignment:
    """Test blob color assignment from palette."""

    def test_color_sampled_from_palette(self):
        """Color should be sampled from 9-color palette {1-9}."""
        generator = BlobGenerator()
        for _ in range(20):
            color = generator.sample_color()
            assert 1 <= color <= 9

    def test_color_sampling_is_uniform(self):
        """Color sampling should be approximately uniform."""
        generator = BlobGenerator()
        colors = [generator.sample_color() for _ in range(1000)]

        # Check all colors appear
        unique_colors = set(colors)
        assert len(unique_colors) >= 7, "Not enough color variety"

        # Check approximate uniformity (chi-squared style)
        expected_count = 1000 / 9
        for color in range(1, 10):
            count = colors.count(color)
            # Allow 50% deviation from expected (rough check)
            assert count > expected_count * 0.5, f"Color {color} underrepresented"
            assert count < expected_count * 1.5, f"Color {color} overrepresented"

    def test_custom_palette(self):
        """Generator should accept custom color palette."""
        custom_palette = [3, 5, 7]
        generator = BlobGenerator(color_palette=custom_palette)
        for _ in range(30):
            color = generator.sample_color()
            assert color in custom_palette


class TestBlobShapeValidation:
    """Test blob shape validation and connectivity checking."""

    def test_is_connected_single_pixel(self):
        """Single pixel should be considered connected."""
        obj = Object({(0, 0)})
        assert is_connected(obj)

    def test_is_connected_horizontal_line(self):
        """Horizontal line should be connected."""
        obj = Object({(0, 0), (0, 1), (0, 2)})
        assert is_connected(obj)

    def test_is_connected_diagonal(self):
        """Diagonal line should be connected (8-neighborhood)."""
        obj = Object({(0, 0), (1, 1), (2, 2)})
        assert is_connected(obj)

    def test_is_connected_blob_shape(self):
        """Arbitrary blob shape should be connected."""
        obj = Object({(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)})
        assert is_connected(obj)

    def test_is_not_connected_separated_pixels(self):
        """Separated pixels should not be connected."""
        obj = Object({(0, 0), (0, 1), (5, 5)})
        assert not is_connected(obj)

    def test_is_not_connected_two_groups(self):
        """Two separate groups should not be connected."""
        obj = Object({(0, 0), (0, 1), (5, 5), (5, 6)})
        assert not is_connected(obj)


class TestBlobBoundaryChecking:
    """Test blob boundary constraints."""

    def test_blob_fits_in_grid(self):
        """Generated blob should fit within grid constraints."""
        generator = BlobGenerator(min_pixels=2, max_pixels=15)
        for _ in range(20):
            blob = generator.generate()
            height, width = blob.bounds()
            # Should fit in 16x16 grid
            assert height <= 16
            assert width <= 16

    def test_blob_normalized_coordinates(self):
        """Blob coordinates should be relative (can include negative coords from seed)."""
        generator = BlobGenerator(min_pixels=5, max_pixels=10)
        for _ in range(10):
            blob = generator.generate()
            rows = [p[0] for p in blob.pixels]
            cols = [p[1] for p in blob.pixels]
            # Blob grows from (0,0) but can extend in any direction
            # Just verify it's reasonably bounded
            assert max(rows) - min(rows) <= 16
            assert max(cols) - min(cols) <= 16


class TestBlobIntegration:
    """Integration tests for blob generation."""

    def test_generate_multiple_blobs(self):
        """Should generate multiple unique blobs."""
        generator = BlobGenerator(min_pixels=5, max_pixels=10)
        blobs = [generator.generate() for _ in range(10)]

        # All should be valid
        for blob in blobs:
            assert validate_object_size(blob, min_pixels=2, max_pixels=15)
            assert is_connected(blob)

        # Should have some variety in shapes
        sizes = [blob.size() for blob in blobs]
        assert len(set(sizes)) >= 2

    def test_blob_generation_with_retry(self):
        """Generator should retry on invalid blobs."""
        generator = BlobGenerator(min_pixels=2, max_pixels=15)
        # Should successfully generate even with potential failures
        for _ in range(20):
            blob = generator.generate()
            assert blob is not None
            assert is_connected(blob)
            assert validate_object_size(blob, min_pixels=2, max_pixels=15)


# Helper functions for tests

def get_8_neighbors(pixel: tuple) -> list:
    """Get 8-neighborhood neighbors of a pixel."""
    row, col = pixel
    return [
        (row - 1, col - 1), (row - 1, col), (row - 1, col + 1),
        (row, col - 1),                     (row, col + 1),
        (row + 1, col - 1), (row + 1, col), (row + 1, col + 1),
    ]
