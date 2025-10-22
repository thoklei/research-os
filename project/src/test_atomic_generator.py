"""
Test suite for atomic image generator - Task 1: Core grid and placement infrastructure.

This module tests:
- Grid initialization and data structure
- Collision detection
- Placement engine with retry logic
- Size constraint validation
"""

import numpy as np
import pytest
from atomic_generator import (
    Grid,
    Object,
    PlacementEngine,
    validate_object_size,
)


class TestGridInitialization:
    """Test grid data structure and initialization."""

    def test_grid_creates_16x16_array(self):
        """Grid should initialize as 16x16 numpy array."""
        grid = Grid()
        assert grid.data.shape == (16, 16)

    def test_grid_initializes_with_background(self):
        """Grid should initialize all cells to background color (0)."""
        grid = Grid()
        assert np.all(grid.data == 0)

    def test_grid_uses_int_dtype(self):
        """Grid should use integer dtype for color values."""
        grid = Grid()
        assert grid.data.dtype == np.int64 or grid.data.dtype == np.int32

    def test_grid_accepts_valid_colors(self):
        """Grid should accept colors from palette {1-9}."""
        grid = Grid()
        grid.data[0, 0] = 5
        assert grid.data[0, 0] == 5

    def test_grid_copy(self):
        """Grid copy should create independent copy."""
        grid = Grid()
        grid.data[0, 0] = 3
        grid_copy = grid.copy()
        grid_copy.data[0, 0] = 7
        assert grid.data[0, 0] == 3
        assert grid_copy.data[0, 0] == 7


class TestObject:
    """Test object representation and properties."""

    def test_object_from_pixels(self):
        """Object should be created from set of pixel coordinates."""
        pixels = {(0, 0), (0, 1), (1, 0)}
        obj = Object(pixels)
        assert obj.pixels == pixels

    def test_object_size(self):
        """Object size should return pixel count."""
        pixels = {(0, 0), (0, 1), (1, 0), (1, 1)}
        obj = Object(pixels)
        assert obj.size() == 4

    def test_object_bounds(self):
        """Object bounds should return bounding box dimensions."""
        pixels = {(0, 0), (0, 2), (3, 1)}
        obj = Object(pixels)
        height, width = obj.bounds()
        # Max row is 3, min row is 0 -> height = 4
        # Max col is 2, min col is 0 -> width = 3
        assert height == 4
        assert width == 3

    def test_object_translate(self):
        """Object should translate to new position."""
        pixels = {(0, 0), (0, 1), (1, 0)}
        obj = Object(pixels)
        translated = obj.translate(5, 7)
        expected = {(5, 7), (5, 8), (6, 7)}
        assert translated.pixels == expected

    def test_empty_object(self):
        """Empty object should have size 0."""
        obj = Object(set())
        assert obj.size() == 0


class TestCollisionDetection:
    """Test collision detection algorithm."""

    def test_no_collision_on_empty_grid(self):
        """Object should not collide with empty grid."""
        grid = Grid()
        obj = Object({(0, 0), (0, 1)})
        engine = PlacementEngine(grid)
        assert not engine.has_collision(obj, 0, 0)

    def test_collision_with_existing_object(self):
        """Object should collide with non-zero cells."""
        grid = Grid()
        grid.data[5, 5] = 3  # Place existing object
        obj = Object({(0, 0)})
        engine = PlacementEngine(grid)
        assert engine.has_collision(obj, 5, 5)

    def test_no_collision_adjacent_cells(self):
        """Adjacent objects should not collide (no spacing requirement)."""
        grid = Grid()
        grid.data[5, 5] = 3
        obj = Object({(0, 0)})
        engine = PlacementEngine(grid)
        # Test all 8 neighbors
        assert not engine.has_collision(obj, 4, 4)  # Top-left
        assert not engine.has_collision(obj, 4, 5)  # Top
        assert not engine.has_collision(obj, 4, 6)  # Top-right
        assert not engine.has_collision(obj, 5, 4)  # Left
        assert not engine.has_collision(obj, 5, 6)  # Right
        assert not engine.has_collision(obj, 6, 4)  # Bottom-left
        assert not engine.has_collision(obj, 6, 5)  # Bottom
        assert not engine.has_collision(obj, 6, 6)  # Bottom-right

    def test_collision_with_multi_pixel_object(self):
        """Multi-pixel object should detect collision with any pixel."""
        grid = Grid()
        grid.data[5, 7] = 2
        obj = Object({(0, 0), (0, 1), (0, 2)})
        engine = PlacementEngine(grid)
        # Object at (5, 5) would occupy (5,5), (5,6), (5,7)
        assert engine.has_collision(obj, 5, 5)

    def test_out_of_bounds_collision(self):
        """Object should collide if it goes out of bounds."""
        grid = Grid()
        obj = Object({(0, 0), (0, 1), (0, 2)})
        engine = PlacementEngine(grid)
        # Object would extend beyond grid width at position (0, 14)
        assert engine.has_collision(obj, 0, 14)
        assert engine.has_collision(obj, 0, 15)
        assert engine.has_collision(obj, -1, 0)


class TestPlacementEngine:
    """Test placement engine with retry logic."""

    def test_successful_placement_on_empty_grid(self):
        """Object should be placed successfully on empty grid."""
        grid = Grid()
        obj = Object({(0, 0), (0, 1)})
        engine = PlacementEngine(grid)
        position = engine.find_position(obj, max_attempts=50)
        assert position is not None
        assert isinstance(position, tuple)
        assert len(position) == 2

    def test_placement_returns_valid_position(self):
        """Returned position should be within grid bounds."""
        grid = Grid()
        obj = Object({(0, 0), (1, 0), (2, 0)})
        engine = PlacementEngine(grid)
        position = engine.find_position(obj, max_attempts=50)
        if position:
            y, x = position
            assert 0 <= y < 16
            assert 0 <= x < 16

    def test_placement_fails_on_full_grid(self):
        """Placement should fail when grid is full."""
        grid = Grid()
        grid.data[:, :] = 5  # Fill entire grid
        obj = Object({(0, 0)})
        engine = PlacementEngine(grid)
        position = engine.find_position(obj, max_attempts=50)
        assert position is None

    def test_place_object_on_grid(self):
        """Object should be placed on grid with correct color."""
        grid = Grid()
        obj = Object({(0, 0), (0, 1), (1, 0)})
        engine = PlacementEngine(grid)
        engine.place_object(obj, 5, 7, color=4)
        assert grid.data[5, 7] == 4
        assert grid.data[5, 8] == 4
        assert grid.data[6, 7] == 4
        # Other cells should remain background
        assert grid.data[5, 6] == 0
        assert grid.data[4, 7] == 0

    def test_placement_respects_max_attempts(self):
        """Placement should stop after max_attempts."""
        grid = Grid()
        grid.data[:, :] = 5  # Make it very hard to place
        obj = Object({(0, 0)})
        engine = PlacementEngine(grid)
        # Should fail quickly with limited attempts
        position = engine.find_position(obj, max_attempts=5)
        assert position is None


class TestSizeConstraintValidation:
    """Test size constraint validation (2-15 pixels)."""

    def test_valid_object_size_2_pixels(self):
        """Object with 2 pixels should pass validation."""
        obj = Object({(0, 0), (0, 1)})
        assert validate_object_size(obj, min_pixels=2, max_pixels=15)

    def test_valid_object_size_15_pixels(self):
        """Object with 15 pixels should pass validation."""
        pixels = {(i, j) for i in range(3) for j in range(5)}
        obj = Object(pixels)
        assert obj.size() == 15
        assert validate_object_size(obj, min_pixels=2, max_pixels=15)

    def test_valid_object_size_middle_range(self):
        """Object with 8 pixels should pass validation."""
        pixels = {(i, j) for i in range(2) for j in range(4)}
        obj = Object(pixels)
        assert obj.size() == 8
        assert validate_object_size(obj, min_pixels=2, max_pixels=15)

    def test_invalid_object_size_too_small(self):
        """Object with 1 pixel should fail validation."""
        obj = Object({(0, 0)})
        assert not validate_object_size(obj, min_pixels=2, max_pixels=15)

    def test_invalid_object_size_too_large(self):
        """Object with 16 pixels should fail validation."""
        pixels = {(i, j) for i in range(4) for j in range(4)}
        obj = Object(pixels)
        assert obj.size() == 16
        assert not validate_object_size(obj, min_pixels=2, max_pixels=15)

    def test_empty_object_fails_validation(self):
        """Empty object should fail validation."""
        obj = Object(set())
        assert not validate_object_size(obj, min_pixels=2, max_pixels=15)

    def test_custom_size_constraints(self):
        """Validation should work with custom min/max values."""
        obj = Object({(0, 0), (0, 1), (1, 0)})
        assert obj.size() == 3
        assert validate_object_size(obj, min_pixels=3, max_pixels=10)
        assert not validate_object_size(obj, min_pixels=4, max_pixels=10)
        assert not validate_object_size(obj, min_pixels=1, max_pixels=2)


class TestIntegration:
    """Integration tests for core infrastructure."""

    def test_place_multiple_objects_without_collision(self):
        """Multiple objects should be placed without overlapping."""
        grid = Grid()
        engine = PlacementEngine(grid)

        # Place first object
        obj1 = Object({(0, 0), (0, 1), (1, 0)})
        pos1 = engine.find_position(obj1, max_attempts=50)
        assert pos1 is not None
        engine.place_object(obj1, pos1[0], pos1[1], color=3)

        # Place second object
        obj2 = Object({(0, 0), (1, 0), (2, 0)})
        pos2 = engine.find_position(obj2, max_attempts=50)
        assert pos2 is not None
        engine.place_object(obj2, pos2[0], pos2[1], color=5)

        # Verify no overlaps
        color_3_count = np.sum(grid.data == 3)
        color_5_count = np.sum(grid.data == 5)
        assert color_3_count == obj1.size()
        assert color_5_count == obj2.size()

    def test_full_pipeline_with_size_validation(self):
        """Test complete workflow with size validation."""
        grid = Grid()
        engine = PlacementEngine(grid)

        # Create object that passes size validation
        obj = Object({(0, 0), (0, 1), (1, 0), (1, 1)})
        assert validate_object_size(obj, min_pixels=2, max_pixels=15)

        # Place object
        position = engine.find_position(obj, max_attempts=50)
        assert position is not None
        engine.place_object(obj, position[0], position[1], color=7)

        # Verify placement
        assert np.sum(grid.data == 7) == 4
