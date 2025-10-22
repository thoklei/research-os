"""
Atomic Image Generator - Core Infrastructure (Task 1)

This module implements the foundational grid and placement infrastructure for
generating ARC-like atomic images.

Components:
- Grid: 16x16 grid data structure with 9-color palette
- Object: Representation of objects as sets of pixel coordinates
- PlacementEngine: Collision detection and placement with retry logic
- validate_object_size: Size constraint validation (2-15 pixels)
"""

import numpy as np
from typing import Set, Tuple, Optional
from copy import deepcopy


class Grid:
    """
    16x16 grid for ARC-like atomic images.

    Uses:
    - Background color: 0
    - Object colors: 1-9 (9-color palette)
    """

    def __init__(self):
        """Initialize 16x16 grid with background color (0)."""
        self.data = np.zeros((16, 16), dtype=np.int64)

    def copy(self) -> 'Grid':
        """Create independent copy of the grid."""
        new_grid = Grid()
        new_grid.data = self.data.copy()
        return new_grid


class Object:
    """
    Representation of an object as a set of relative pixel coordinates.

    Coordinates are stored relative to (0, 0) origin, allowing for easy
    translation to any position on the grid.
    """

    def __init__(self, pixels: Set[Tuple[int, int]]):
        """
        Initialize object from set of pixel coordinates.

        Args:
            pixels: Set of (row, col) tuples representing object shape
        """
        self.pixels = pixels

    def size(self) -> int:
        """Return number of pixels in object."""
        return len(self.pixels)

    def bounds(self) -> Tuple[int, int]:
        """
        Calculate bounding box dimensions.

        Returns:
            (height, width) tuple of bounding box
        """
        if not self.pixels:
            return (0, 0)

        rows = [pixel[0] for pixel in self.pixels]
        cols = [pixel[1] for pixel in self.pixels]

        height = max(rows) - min(rows) + 1
        width = max(cols) - min(cols) + 1

        return (height, width)

    def translate(self, row_offset: int, col_offset: int) -> 'Object':
        """
        Create new object translated by given offset.

        Args:
            row_offset: Vertical translation
            col_offset: Horizontal translation

        Returns:
            New Object instance at translated position
        """
        translated_pixels = {
            (row + row_offset, col + col_offset)
            for row, col in self.pixels
        }
        return Object(translated_pixels)


class PlacementEngine:
    """
    Engine for placing objects on grid with collision detection.

    Handles:
    - Collision detection (overlap checking)
    - Boundary validation
    - Placement with retry logic (up to max_attempts)
    """

    def __init__(self, grid: Grid):
        """
        Initialize placement engine for given grid.

        Args:
            grid: Grid instance to place objects on
        """
        self.grid = grid

    def has_collision(self, obj: Object, row: int, col: int) -> bool:
        """
        Check if object at given position would collide.

        Collision occurs if:
        1. Object extends beyond grid boundaries
        2. Object overlaps with non-zero cells

        Note: Adjacent objects (touching but not overlapping) do NOT collide,
        as there is no spacing requirement.

        Args:
            obj: Object to check
            row: Row position to check
            col: Column position to check

        Returns:
            True if collision detected, False otherwise
        """
        for dr, dc in obj.pixels:
            target_row = row + dr
            target_col = col + dc

            # Check boundary collision
            if (target_row < 0 or target_row >= 16 or
                target_col < 0 or target_col >= 16):
                return True

            # Check overlap collision
            if self.grid.data[target_row, target_col] != 0:
                return True

        return False

    def find_position(self, obj: Object, max_attempts: int = 50) -> Optional[Tuple[int, int]]:
        """
        Find valid position for object using random search.

        Tries random positions up to max_attempts times.
        Returns first valid position found.

        Args:
            obj: Object to place
            max_attempts: Maximum number of random positions to try

        Returns:
            (row, col) tuple if valid position found, None otherwise
        """
        for _ in range(max_attempts):
            # Random position in grid
            row = np.random.randint(0, 16)
            col = np.random.randint(0, 16)

            if not self.has_collision(obj, row, col):
                return (row, col)

        return None

    def place_object(self, obj: Object, row: int, col: int, color: int) -> None:
        """
        Place object on grid at given position with specified color.

        WARNING: Does not check for collisions. Use find_position() first
        to ensure valid placement.

        Args:
            obj: Object to place
            row: Row position
            col: Column position
            color: Color value from palette {1-9}
        """
        for dr, dc in obj.pixels:
            self.grid.data[row + dr, col + dc] = color


def validate_object_size(obj: Object, min_pixels: int = 2, max_pixels: int = 15) -> bool:
    """
    Validate object size against constraints.

    Per spec requirements, objects must contain 2-15 pixels.
    This is enforced as a hard constraint with regeneration.

    Args:
        obj: Object to validate
        min_pixels: Minimum number of pixels (default: 2)
        max_pixels: Maximum number of pixels (default: 15)

    Returns:
        True if object size is within constraints, False otherwise
    """
    size = obj.size()
    return min_pixels <= size <= max_pixels
