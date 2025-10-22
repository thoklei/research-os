"""
Test suite for shape generators - Task 3: Rectangle, Line, and Pattern generators.

This module tests:
- Rectangle generator with random dimensions
- Line generator (horizontal, vertical, diagonal)
- Pattern generator with templates (checkerboard, L-shape, T-shape, plus, zigzag)
- Size constraint validation for all generators
"""

import numpy as np
import pytest
from atomic_generator import Object, validate_object_size
from shape_generators import (
    RectangleGenerator,
    LineGenerator,
    PatternGenerator,
)


class TestRectangleGenerator:
    """Test rectangle generator with random dimensions."""

    def test_rectangle_generator_creates_object(self):
        """RectangleGenerator should create an Object instance."""
        generator = RectangleGenerator()
        rect = generator.generate()
        assert isinstance(rect, Object)

    def test_rectangle_is_filled(self):
        """Generated rectangle should be completely filled."""
        generator = RectangleGenerator(min_height=3, max_height=4, min_width=3, max_width=4)
        rect = generator.generate()

        # Check that rectangle is solid (no holes)
        rows = [p[0] for p in rect.pixels]
        cols = [p[1] for p in rect.pixels]

        height = max(rows) - min(rows) + 1
        width = max(cols) - min(cols) + 1
        expected_size = height * width

        assert rect.size() == expected_size

    def test_rectangle_respects_size_constraints(self):
        """Rectangle should respect 2-15 pixel constraint."""
        generator = RectangleGenerator()
        for _ in range(20):
            rect = generator.generate()
            assert validate_object_size(rect, min_pixels=2, max_pixels=15)

    def test_rectangle_dimensions_within_range(self):
        """Rectangle dimensions should be within specified range."""
        generator = RectangleGenerator(min_height=2, max_height=3, min_width=2, max_width=4)
        for _ in range(20):
            rect = generator.generate()
            height, width = rect.bounds()
            assert 2 <= height <= 3
            assert 2 <= width <= 4

    def test_rectangle_variability(self):
        """Rectangles should have varying dimensions."""
        generator = RectangleGenerator(min_height=1, max_height=4, min_width=1, max_width=4)
        rectangles = [generator.generate() for _ in range(30)]
        sizes = [r.size() for r in rectangles]

        # Should have variety in sizes
        unique_sizes = len(set(sizes))
        assert unique_sizes >= 3

    def test_rectangle_default_constraints(self):
        """Default generator should produce valid rectangles."""
        generator = RectangleGenerator()
        for _ in range(10):
            rect = generator.generate()
            assert rect.size() >= 2
            assert rect.size() <= 15

    def test_small_rectangle(self):
        """Should generate 1x2 rectangle (2 pixels minimum)."""
        generator = RectangleGenerator(min_height=1, max_height=1, min_width=2, max_width=2)
        rect = generator.generate()
        assert rect.size() == 2

    def test_square_rectangle(self):
        """Should generate square rectangles."""
        generator = RectangleGenerator(min_height=3, max_height=3, min_width=3, max_width=3)
        rect = generator.generate()
        height, width = rect.bounds()
        assert height == 3
        assert width == 3
        assert rect.size() == 9


class TestLineGenerator:
    """Test line generator (horizontal, vertical, diagonal)."""

    def test_line_generator_creates_object(self):
        """LineGenerator should create an Object instance."""
        generator = LineGenerator()
        line = generator.generate()
        assert isinstance(line, Object)

    def test_horizontal_line(self):
        """Horizontal lines should have all pixels in same row."""
        generator = LineGenerator(direction='horizontal', min_length=3, max_length=5)
        line = generator.generate()

        rows = [p[0] for p in line.pixels]
        assert len(set(rows)) == 1  # All same row

    def test_vertical_line(self):
        """Vertical lines should have all pixels in same column."""
        generator = LineGenerator(direction='vertical', min_length=3, max_length=5)
        line = generator.generate()

        cols = [p[1] for p in line.pixels]
        assert len(set(cols)) == 1  # All same column

    def test_diagonal_line(self):
        """Diagonal lines should have slope of ±1."""
        generator = LineGenerator(direction='diagonal', min_length=4, max_length=6)
        line = generator.generate()

        pixels_list = sorted(line.pixels)
        # Check if it's a diagonal (either positive or negative slope)
        is_positive_diagonal = all(
            pixels_list[i+1][0] - pixels_list[i][0] == pixels_list[i+1][1] - pixels_list[i][1]
            for i in range(len(pixels_list) - 1)
        )
        is_negative_diagonal = all(
            pixels_list[i+1][0] - pixels_list[i][0] == -(pixels_list[i+1][1] - pixels_list[i][1])
            for i in range(len(pixels_list) - 1)
        )
        assert is_positive_diagonal or is_negative_diagonal

    def test_line_respects_length_constraints(self):
        """Line length should be within specified range."""
        generator = LineGenerator(min_length=3, max_length=7)
        for _ in range(20):
            line = generator.generate()
            assert 3 <= line.size() <= 7

    def test_line_respects_size_constraints(self):
        """Lines should respect 2-15 pixel constraint."""
        generator = LineGenerator()
        for _ in range(20):
            line = generator.generate()
            assert validate_object_size(line, min_pixels=2, max_pixels=15)

    def test_random_direction_selection(self):
        """Random direction should produce variety."""
        generator = LineGenerator(direction='random', min_length=4, max_length=4)
        lines = [generator.generate() for _ in range(30)]

        # Check that we get different line types
        bounds = [line.bounds() for line in lines]
        # Some should be horizontal (height=1), some vertical (width=1), some diagonal (height=width)
        has_horizontal = any(h == 1 and w > 1 for h, w in bounds)
        has_vertical = any(w == 1 and h > 1 for h, w in bounds)

        assert has_horizontal or has_vertical  # At least some variety

    def test_line_minimum_length(self):
        """Line should respect minimum length of 2."""
        generator = LineGenerator(min_length=2, max_length=2)
        line = generator.generate()
        assert line.size() == 2


class TestPatternGenerator:
    """Test pattern generator with templates."""

    def test_pattern_generator_creates_object(self):
        """PatternGenerator should create an Object instance."""
        generator = PatternGenerator()
        pattern = generator.generate()
        assert isinstance(pattern, Object)

    def test_checkerboard_pattern(self):
        """Checkerboard pattern should have specific structure."""
        generator = PatternGenerator(pattern_type='checkerboard')
        pattern = generator.generate()

        # Checkerboard should be alternating pattern
        assert pattern.size() >= 2
        assert validate_object_size(pattern, min_pixels=2, max_pixels=15)

    def test_l_shape_pattern(self):
        """L-shape pattern should have specific structure."""
        generator = PatternGenerator(pattern_type='l_shape')
        pattern = generator.generate()

        assert pattern.size() >= 3
        assert validate_object_size(pattern, min_pixels=2, max_pixels=15)

    def test_t_shape_pattern(self):
        """T-shape pattern should have specific structure."""
        generator = PatternGenerator(pattern_type='t_shape')
        pattern = generator.generate()

        assert pattern.size() >= 4
        assert validate_object_size(pattern, min_pixels=2, max_pixels=15)

    def test_plus_pattern(self):
        """Plus pattern should have specific structure."""
        generator = PatternGenerator(pattern_type='plus')
        pattern = generator.generate()

        assert pattern.size() >= 5
        assert validate_object_size(pattern, min_pixels=2, max_pixels=15)

    def test_zigzag_pattern(self):
        """Zigzag pattern should have specific structure."""
        generator = PatternGenerator(pattern_type='zigzag')
        pattern = generator.generate()

        assert pattern.size() >= 3
        assert validate_object_size(pattern, min_pixels=2, max_pixels=15)

    def test_random_pattern_selection(self):
        """Random pattern should produce variety."""
        generator = PatternGenerator(pattern_type='random')
        patterns = [generator.generate() for _ in range(50)]
        sizes = [p.size() for p in patterns]

        # Should have variety in pattern sizes
        unique_sizes = len(set(sizes))
        assert unique_sizes >= 2

    def test_all_patterns_valid_size(self):
        """All patterns should respect size constraints."""
        generator = PatternGenerator(pattern_type='random')
        for _ in range(30):
            pattern = generator.generate()
            assert validate_object_size(pattern, min_pixels=2, max_pixels=15)

    def test_pattern_types_available(self):
        """All pattern types should be generatable."""
        pattern_types = ['checkerboard', 'l_shape', 't_shape', 'plus', 'zigzag']
        for pattern_type in pattern_types:
            generator = PatternGenerator(pattern_type=pattern_type)
            pattern = generator.generate()
            assert pattern is not None
            assert isinstance(pattern, Object)


class TestSizeConstraintsAllGenerators:
    """Test size constraints across all generators."""

    def test_rectangle_meets_constraints(self):
        """All rectangles should meet 2-15 pixel constraint."""
        generator = RectangleGenerator()
        for _ in range(30):
            obj = generator.generate()
            assert validate_object_size(obj, min_pixels=2, max_pixels=15)

    def test_line_meets_constraints(self):
        """All lines should meet 2-15 pixel constraint."""
        generator = LineGenerator()
        for _ in range(30):
            obj = generator.generate()
            assert validate_object_size(obj, min_pixels=2, max_pixels=15)

    def test_pattern_meets_constraints(self):
        """All patterns should meet 2-15 pixel constraint."""
        generator = PatternGenerator()
        for _ in range(30):
            obj = generator.generate()
            assert validate_object_size(obj, min_pixels=2, max_pixels=15)

    def test_no_objects_too_small(self):
        """No generator should produce objects < 2 pixels."""
        generators = [
            RectangleGenerator(),
            LineGenerator(),
            PatternGenerator(),
        ]

        for generator in generators:
            for _ in range(20):
                obj = generator.generate()
                assert obj.size() >= 2

    def test_no_objects_too_large(self):
        """No generator should produce objects > 15 pixels."""
        generators = [
            RectangleGenerator(),
            LineGenerator(),
            PatternGenerator(),
        ]

        for generator in generators:
            for _ in range(20):
                obj = generator.generate()
                assert obj.size() <= 15


class TestIntegration:
    """Integration tests for all shape generators."""

    def test_all_generators_produce_objects(self):
        """All generators should produce valid Object instances."""
        generators = [
            RectangleGenerator(),
            LineGenerator(),
            PatternGenerator(),
        ]

        for generator in generators:
            obj = generator.generate()
            assert isinstance(obj, Object)
            assert obj.size() > 0

    def test_generators_produce_variety(self):
        """Different generators should produce different shapes."""
        rect_gen = RectangleGenerator()
        line_gen = LineGenerator()
        pattern_gen = PatternGenerator()

        rectangles = [rect_gen.generate() for _ in range(10)]
        lines = [line_gen.generate() for _ in range(10)]
        patterns = [pattern_gen.generate() for _ in range(10)]

        # All should be valid
        all_objects = rectangles + lines + patterns
        for obj in all_objects:
            assert validate_object_size(obj, min_pixels=2, max_pixels=15)
