"""
Procedural Image Generation Pipeline - Task 4

This module implements the complete pipeline for generating ARC-like atomic images:
- ImageGenerator: Generates single images with 1-4 objects
- generate_corpus: Batch generation of multiple images
- split_corpus: Train/val/test splitting (80/10/10)

Object type distribution:
- Blobs: 40%
- Rectangles: 20%
- Lines: 20%
- Patterns: 20%
"""

import numpy as np
from typing import List, Tuple, Optional
from atomic_generator import Grid, PlacementEngine, validate_object_size
from blob_generator import BlobGenerator
from shape_generators import RectangleGenerator, LineGenerator, PatternGenerator


class ImageGenerator:
    """
    Generator for single ARC-like atomic images with multiple objects.

    Places 1-4 objects on a 16x16 grid with:
    - Non-overlapping placement
    - Uniform color sampling from palette {1-9}
    - Object type distribution: 40% blob, 20% rectangle, 20% line, 20% pattern
    """

    def __init__(self, num_objects: Optional[int] = None):
        """
        Initialize image generator.

        Args:
            num_objects: Number of objects to place (None = random 1-4)
        """
        self.num_objects = num_objects

        # Initialize object generators
        self.blob_generator = BlobGenerator()
        self.rectangle_generator = RectangleGenerator()
        self.line_generator = LineGenerator()
        self.pattern_generator = PatternGenerator()

    def generate(self) -> Grid:
        """
        Generate a single atomic image.

        Returns:
            Grid instance with 1-4 objects placed
        """
        # Create empty grid
        grid = Grid()
        placement_engine = PlacementEngine(grid)

        # Determine number of objects
        if self.num_objects is None:
            num_objects = np.random.randint(1, 5)  # 1-4 objects
        else:
            num_objects = self.num_objects

        # Place objects
        objects_placed = 0
        max_attempts = 100  # Total attempts to place all objects

        for attempt in range(max_attempts):
            if objects_placed >= num_objects:
                break

            # Select object type with specified distribution
            object_type = np.random.choice(
                ['blob', 'rectangle', 'line', 'pattern'],
                p=[0.4, 0.2, 0.2, 0.2]
            )

            # Generate object
            obj = self._generate_object(object_type)

            # Validate size
            if not validate_object_size(obj, min_pixels=2, max_pixels=15):
                continue

            # Try to place object
            position = placement_engine.find_position(obj, max_attempts=50)

            if position is not None:
                # Sample color
                color = self.blob_generator.sample_color()

                # Place object
                placement_engine.place_object(obj, position[0], position[1], color)
                objects_placed += 1

        return grid

    def _generate_object(self, object_type: str):
        """
        Generate object of specified type.

        Args:
            object_type: Type of object ('blob', 'rectangle', 'line', 'pattern')

        Returns:
            Object instance
        """
        if object_type == 'blob':
            return self.blob_generator.generate()
        elif object_type == 'rectangle':
            return self.rectangle_generator.generate()
        elif object_type == 'line':
            return self.line_generator.generate()
        elif object_type == 'pattern':
            return self.pattern_generator.generate()
        else:
            # Fallback to blob
            return self.blob_generator.generate()


def generate_corpus(corpus_size: int = 10) -> List[Grid]:
    """
    Generate corpus of atomic images.

    Args:
        corpus_size: Number of images to generate (default: 10)

    Returns:
        List of Grid instances
    """
    generator = ImageGenerator()
    corpus = []

    for _ in range(corpus_size):
        grid = generator.generate()
        corpus.append(grid)

    return corpus


def split_corpus(corpus: List[Grid],
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1) -> Tuple[List[Grid], List[Grid], List[Grid]]:
    """
    Split corpus into train/val/test sets.

    Uses random shuffling to ensure variety in each set.

    Args:
        corpus: List of Grid instances to split
        train_ratio: Proportion for training set (default: 0.8)
        val_ratio: Proportion for validation set (default: 0.1)
        test_ratio: Proportion for test set (default: 0.1)

    Returns:
        Tuple of (train, val, test) lists
    """
    # Shuffle corpus
    indices = np.arange(len(corpus))
    np.random.shuffle(indices)

    # Calculate split points
    n = len(corpus)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Split
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train = [corpus[i] for i in train_indices]
    val = [corpus[i] for i in val_indices]
    test = [corpus[i] for i in test_indices]

    return train, val, test
