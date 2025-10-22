"""
Shape Generators - Task 3

This module implements generators for rectangles, lines, and patterns.

Components:
- RectangleGenerator: Filled rectangles with random dimensions
- LineGenerator: Horizontal, vertical, and diagonal lines
- PatternGenerator: Template-based patterns (checkerboard, L-shape, T-shape, plus, zigzag)
"""

import numpy as np
from typing import Set, Tuple, Literal
from atomic_generator import Object


class RectangleGenerator:
    """
    Generator for filled rectangles with random dimensions.

    Rectangles are solid (completely filled) with configurable height and width ranges.
    """

    def __init__(self,
                 min_height: int = 1,
                 max_height: int = 5,
                 min_width: int = 1,
                 max_width: int = 5):
        """
        Initialize rectangle generator.

        Constraints are adjusted to ensure rectangles stay within 2-15 pixel limit.

        Args:
            min_height: Minimum rectangle height (default: 1)
            max_height: Maximum rectangle height (default: 5)
            min_width: Minimum rectangle width (default: 1)
            max_width: Maximum rectangle width (default: 5)
        """
        self.min_height = min_height
        self.max_height = max_height
        self.min_width = min_width
        self.max_width = max_width

    def generate(self) -> Object:
        """
        Generate a filled rectangle.

        Ensures rectangle size is within 2-15 pixel constraint by:
        1. Randomly selecting height and width
        2. Checking if product is within bounds
        3. Retrying if needed

        Returns:
            Object instance representing filled rectangle
        """
        max_attempts = 50

        for _ in range(max_attempts):
            height = np.random.randint(self.min_height, self.max_height + 1)
            width = np.random.randint(self.min_width, self.max_width + 1)
            size = height * width

            # Check size constraint
            if 2 <= size <= 15:
                pixels = {(row, col) for row in range(height) for col in range(width)}
                return Object(pixels)

        # Fallback: 2x1 rectangle
        return Object({(0, 0), (1, 0)})


class LineGenerator:
    """
    Generator for straight lines in multiple directions.

    Supports:
    - Horizontal lines (all pixels in same row)
    - Vertical lines (all pixels in same column)
    - Diagonal lines (slope = ±1)
    """

    def __init__(self,
                 direction: Literal['horizontal', 'vertical', 'diagonal', 'random'] = 'random',
                 min_length: int = 2,
                 max_length: int = 8):
        """
        Initialize line generator.

        Args:
            direction: Line direction ('horizontal', 'vertical', 'diagonal', 'random')
            min_length: Minimum line length (default: 2)
            max_length: Maximum line length (default: 8)
        """
        self.direction = direction
        self.min_length = min_length
        self.max_length = max_length

    def generate(self) -> Object:
        """
        Generate a line object.

        Returns:
            Object instance representing line
        """
        # Choose direction
        if self.direction == 'random':
            direction = np.random.choice(['horizontal', 'vertical', 'diagonal'])
        else:
            direction = self.direction

        # Choose length (ensure within 2-15 constraint)
        length = np.random.randint(self.min_length, min(self.max_length, 15) + 1)
        length = max(2, length)  # Ensure at least 2 pixels

        # Generate line
        if direction == 'horizontal':
            pixels = {(0, col) for col in range(length)}
        elif direction == 'vertical':
            pixels = {(row, 0) for row in range(length)}
        elif direction == 'diagonal':
            # Random choice between positive and negative diagonal
            if np.random.random() < 0.5:
                pixels = {(i, i) for i in range(length)}  # Positive diagonal
            else:
                pixels = {(i, -i) for i in range(length)}  # Negative diagonal
        else:
            # Fallback
            pixels = {(0, col) for col in range(length)}

        return Object(pixels)


class PatternGenerator:
    """
    Generator for template-based patterns.

    Supports patterns:
    - Checkerboard: Alternating 2x2 grid pattern
    - L-shape: L-shaped configuration
    - T-shape: T-shaped configuration
    - Plus: Plus/cross shape
    - Zigzag: Zigzag pattern
    """

    def __init__(self, pattern_type: Literal['checkerboard', 'l_shape', 't_shape', 'plus', 'zigzag', 'random'] = 'random'):
        """
        Initialize pattern generator.

        Args:
            pattern_type: Type of pattern to generate or 'random' for random selection
        """
        self.pattern_type = pattern_type

    def generate(self) -> Object:
        """
        Generate a pattern object.

        Returns:
            Object instance representing pattern
        """
        # Choose pattern type
        if self.pattern_type == 'random':
            pattern_type = np.random.choice(['checkerboard', 'l_shape', 't_shape', 'plus', 'zigzag'])
        else:
            pattern_type = self.pattern_type

        # Generate pattern
        if pattern_type == 'checkerboard':
            pixels = self._generate_checkerboard()
        elif pattern_type == 'l_shape':
            pixels = self._generate_l_shape()
        elif pattern_type == 't_shape':
            pixels = self._generate_t_shape()
        elif pattern_type == 'plus':
            pixels = self._generate_plus()
        elif pattern_type == 'zigzag':
            pixels = self._generate_zigzag()
        else:
            # Fallback
            pixels = self._generate_l_shape()

        return Object(pixels)

    def _generate_checkerboard(self) -> Set[Tuple[int, int]]:
        """
        Generate checkerboard pattern.

        Creates alternating pattern within size constraints.

        Returns:
            Set of pixel coordinates
        """
        # Random size (2x2 to 3x3 checkerboard = 4-6 pixels)
        size = np.random.choice([2, 3])

        pixels = set()
        for row in range(size):
            for col in range(size):
                # Checkerboard: include if (row + col) is even
                if (row + col) % 2 == 0:
                    pixels.add((row, col))

        return pixels

    def _generate_l_shape(self) -> Set[Tuple[int, int]]:
        """
        Generate L-shape pattern.

        Returns:
            Set of pixel coordinates
        """
        # Random L-shape size (3-7 pixels)
        arm_length = np.random.randint(2, 5)

        # L-shape: horizontal arm + vertical arm
        pixels = set()

        # Horizontal arm
        for col in range(arm_length):
            pixels.add((0, col))

        # Vertical arm (excluding corner to avoid duplicate)
        for row in range(1, arm_length):
            pixels.add((row, 0))

        # Ensure within size limit
        if len(pixels) > 15:
            pixels = {(0, 0), (0, 1), (1, 0)}

        return pixels

    def _generate_t_shape(self) -> Set[Tuple[int, int]]:
        """
        Generate T-shape pattern.

        Returns:
            Set of pixel coordinates
        """
        # T-shape: horizontal bar + vertical stem
        bar_width = np.random.randint(3, 5)
        stem_height = np.random.randint(2, 4)

        pixels = set()

        # Horizontal bar (top)
        center = bar_width // 2
        for col in range(bar_width):
            pixels.add((0, col))

        # Vertical stem (center)
        for row in range(1, stem_height):
            pixels.add((row, center))

        # Ensure within size limit
        if len(pixels) > 15:
            pixels = {(0, 0), (0, 1), (0, 2), (1, 1)}

        return pixels

    def _generate_plus(self) -> Set[Tuple[int, int]]:
        """
        Generate plus/cross pattern.

        Returns:
            Set of pixel coordinates
        """
        # Plus: vertical and horizontal lines crossing at center
        arm_length = np.random.randint(2, 4)

        pixels = set()

        # Center pixel
        center = arm_length
        pixels.add((center, center))

        # Horizontal arms
        for offset in range(1, arm_length + 1):
            pixels.add((center, center - offset))  # Left
            pixels.add((center, center + offset))  # Right

        # Vertical arms
        for offset in range(1, arm_length + 1):
            pixels.add((center - offset, center))  # Up
            pixels.add((center + offset, center))  # Down

        # Ensure within size limit
        if len(pixels) > 15:
            pixels = {(1, 0), (1, 1), (1, 2), (0, 1), (2, 1)}

        return pixels

    def _generate_zigzag(self) -> Set[Tuple[int, int]]:
        """
        Generate zigzag pattern.

        Returns:
            Set of pixel coordinates
        """
        # Zigzag: alternating direction
        length = np.random.randint(3, 8)

        pixels = set()
        row, col = 0, 0

        for i in range(length):
            pixels.add((row, col))

            # Alternate between right and down
            if i % 2 == 0:
                col += 1
            else:
                row += 1

        # Ensure within size limit
        if len(pixels) > 15:
            pixels = {(0, 0), (0, 1), (1, 1), (1, 2)}

        return pixels
