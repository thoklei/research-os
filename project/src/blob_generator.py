"""
Blob Object Generator - Task 2

This module implements blob generation using connectivity-biased growth algorithm.

The algorithm creates cohesive, connected shapes by:
1. Starting with a seed pixel at (0, 0)
2. Iteratively growing by adding neighbor pixels
3. Biasing growth towards pixels with more existing neighbors (connectivity bias)
4. Continuing until target size is reached

This creates natural, blob-like shapes that are fully connected and cohesive.
"""

import numpy as np
from typing import Set, Tuple, List
from atomic_generator import Object


class BlobGenerator:
    """
    Generator for blob objects using connectivity-biased growth.

    Blobs are cohesive, connected shapes generated through an iterative
    growth process that favors adding pixels adjacent to multiple existing pixels.
    """

    def __init__(self,
                 min_pixels: int = 2,
                 max_pixels: int = 15,
                 color_palette: List[int] = None):
        """
        Initialize blob generator.

        Args:
            min_pixels: Minimum blob size (default: 2)
            max_pixels: Maximum blob size (default: 15)
            color_palette: Color palette to sample from (default: {1-9})
        """
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.color_palette = color_palette if color_palette is not None else list(range(1, 10))

    def generate(self, max_regeneration_attempts: int = 10) -> Object:
        """
        Generate a blob object using connectivity-biased growth.

        Implements retry logic in case of generation failures.

        Args:
            max_regeneration_attempts: Maximum attempts to generate valid blob

        Returns:
            Object instance representing the generated blob
        """
        for attempt in range(max_regeneration_attempts):
            try:
                blob = self._generate_blob()
                if blob.size() >= self.min_pixels:
                    return blob
            except Exception:
                continue

        # Fallback: create minimal valid blob
        return self._generate_minimal_blob()

    def _generate_blob(self) -> Object:
        """
        Internal method to generate a single blob.

        Uses connectivity-biased growth algorithm:
        1. Start with seed pixel at (0, 0)
        2. Build candidate list of neighboring pixels
        3. Weight candidates by number of existing neighbors
        4. Randomly select weighted candidate and add to blob
        5. Repeat until target size reached

        Returns:
            Object instance
        """
        # Determine target size
        target_size = np.random.randint(self.min_pixels, self.max_pixels + 1)

        # Start with seed pixel
        pixels = {(0, 0)}

        # 8-neighborhood offsets
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        # Grow blob
        for _ in range(target_size - 1):
            # Build weighted candidate list
            candidates = []

            for pixel in pixels:
                for dr, dc in neighbor_offsets:
                    candidate = (pixel[0] + dr, pixel[1] + dc)

                    # Skip if already in blob
                    if candidate in pixels:
                        continue

                    # Count how many neighbors this candidate has in the blob
                    neighbor_count = 0
                    for ndr, ndc in neighbor_offsets:
                        neighbor = (candidate[0] + ndr, candidate[1] + ndc)
                        if neighbor in pixels:
                            neighbor_count += 1

                    # Add candidate multiple times based on connectivity
                    # This creates the connectivity bias
                    for _ in range(neighbor_count + 1):
                        candidates.append(candidate)

            # If no candidates available, stop growth
            if not candidates:
                break

            # Select random candidate (weighted by connectivity)
            selected_pixel = candidates[np.random.randint(len(candidates))]
            pixels.add(selected_pixel)

        return Object(pixels)

    def _generate_minimal_blob(self) -> Object:
        """
        Generate minimal valid blob as fallback.

        Returns:
            Simple 2-pixel blob
        """
        return Object({(0, 0), (0, 1)})

    def sample_color(self) -> int:
        """
        Sample color uniformly from palette.

        Returns:
            Integer color value from palette
        """
        return np.random.choice(self.color_palette)


def is_connected(obj: Object) -> bool:
    """
    Check if object is fully connected using 8-neighborhood.

    Uses flood-fill algorithm to verify all pixels are reachable
    from a starting pixel.

    Args:
        obj: Object to check

    Returns:
        True if all pixels are connected, False otherwise
    """
    if obj.size() == 0:
        return True

    if obj.size() == 1:
        return True

    # Start flood fill from arbitrary pixel
    pixels = obj.pixels.copy()
    start_pixel = next(iter(pixels))

    visited = set()
    queue = [start_pixel]

    # 8-neighborhood offsets
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    # Flood fill
    while queue:
        current = queue.pop(0)

        if current in visited:
            continue

        visited.add(current)

        # Check all neighbors
        for dr, dc in neighbor_offsets:
            neighbor = (current[0] + dr, current[1] + dc)
            if neighbor in pixels and neighbor not in visited:
                queue.append(neighbor)

    # Connected if all pixels were visited
    return len(visited) == len(pixels)
