#!/usr/bin/env python3
"""
Simplified Dataset Generation for Model Capacity Validation

This script generates a simplified 100k dataset containing only parameterized shapes
(no blob objects) for validating model capacity without KL regularization.

Usage:
    python generate_simplified_dataset.py --num-images 100000 --output data/simplified_dataset_100k.npz --seed 42

Features:
- Uses only shape generators: rectangles, lines, and patterns (checkerboard, l_shape, t_shape, plus, zigzag)
- Excludes blob objects to reduce variability
- 16x16 grids with 10-color palette (0=background, 1-9=objects)
- 80/10/10 train/val/test split
- Memory-efficient uint8 storage (~25MB for 100k images)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

from atomic_generator import Grid, PlacementEngine, validate_object_size
from shape_generators import RectangleGenerator, LineGenerator, PatternGenerator


class SimplifiedImageGenerator:
    """
    Generator for ARC-like images using ONLY parameterized shapes (no blobs).

    Places 1-6 objects on a 16x16 grid with:
    - Non-overlapping placement
    - Uniform color sampling from palette {1-9}
    - Object types: rectangles, lines, patterns (equal distribution)
    """

    def __init__(self, num_objects=None):
        """
        Initialize simplified image generator.

        Args:
            num_objects: Number of objects to place (None = random 1-6)
        """
        self.num_objects = num_objects

        # Initialize shape generators ONLY (no BlobGenerator)
        self.rectangle_generator = RectangleGenerator()
        self.line_generator = LineGenerator()
        self.pattern_generator = PatternGenerator()

    def generate(self):
        """
        Generate a single atomic image with shape objects only.

        Returns:
            Grid instance with 1-6 shape objects placed
        """
        # Create empty grid
        grid = Grid()
        placement_engine = PlacementEngine(grid)

        # Determine number of objects (1-6 for good coverage)
        if self.num_objects is None:
            num_objects = np.random.randint(1, 7)  # 1-6 objects
        else:
            num_objects = self.num_objects

        # Place objects
        objects_placed = 0
        max_attempts = 100  # Total attempts to place all objects

        for attempt in range(max_attempts):
            if objects_placed >= num_objects:
                break

            # Select object type - equal distribution across shapes
            object_type = np.random.choice(['rectangle', 'line', 'pattern'])

            # Generate object
            obj = self._generate_object(object_type)

            # Validate size
            if not validate_object_size(obj, min_pixels=2, max_pixels=15):
                continue

            # Try to place object
            position = placement_engine.find_position(obj, max_attempts=50)

            if position is not None:
                # Sample color uniformly from {1-9}
                color = np.random.randint(1, 10)

                # Place object
                placement_engine.place_object(obj, position[0], position[1], color)
                objects_placed += 1

        return grid

    def _generate_object(self, object_type):
        """
        Generate shape object of specified type.

        Args:
            object_type: Type of object ('rectangle', 'line', 'pattern')

        Returns:
            Object instance
        """
        if object_type == 'rectangle':
            return self.rectangle_generator.generate()
        elif object_type == 'line':
            return self.line_generator.generate()
        elif object_type == 'pattern':
            return self.pattern_generator.generate()
        else:
            # Fallback to rectangle
            return self.rectangle_generator.generate()


def generate_simplified_corpus(corpus_size, dtype=np.uint8, show_progress=True):
    """
    Generate corpus of simplified images (shapes only, no blobs).

    Args:
        corpus_size: Number of images to generate
        dtype: NumPy dtype for grid data (default: uint8 for memory efficiency)
        show_progress: Show tqdm progress bar

    Returns:
        NumPy array of shape (corpus_size, 16, 16) with dtype
    """
    generator = SimplifiedImageGenerator()

    # Pre-allocate array for efficiency
    images = np.zeros((corpus_size, 16, 16), dtype=dtype)

    # Create iterator with progress bar
    iterator = range(corpus_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating simplified images", unit="img")

    for i in iterator:
        grid = generator.generate()
        images[i] = grid.data.astype(dtype)

    return images


def split_dataset(images, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into train/val/test sets with shuffling.

    Args:
        images: NumPy array of images
        train_ratio: Proportion for training set (default: 0.8)
        val_ratio: Proportion for validation set (default: 0.1)
        test_ratio: Proportion for test set (default: 0.1)

    Returns:
        Tuple of (train, val, test) arrays
    """
    # Shuffle indices
    n = len(images)
    indices = np.arange(n)
    np.random.shuffle(indices)

    # Calculate split points
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Split
    train = images[indices[:train_end]]
    val = images[indices[train_end:val_end]]
    test = images[indices[val_end:]]

    return train, val, test


def estimate_memory(num_images, dtype=np.uint8):
    """
    Estimate memory required for dataset.

    Args:
        num_images: Number of images to generate
        dtype: NumPy dtype for storage

    Returns:
        Tuple of (bytes_needed, megabytes_estimate)
    """
    bytes_per_element = np.dtype(dtype).itemsize
    bytes_needed = num_images * 16 * 16 * bytes_per_element
    mb_estimate = bytes_needed / (1024 * 1024)
    return bytes_needed, mb_estimate


def format_size(bytes_count):
    """Format bytes as human-readable size."""
    if bytes_count < 1024:
        return f"{bytes_count} bytes"
    elif bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f} KB"
    else:
        return f"{bytes_count / (1024 * 1024):.2f} MB"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate simplified dataset for model capacity validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100k images (default configuration)
  python generate_simplified_dataset.py --num-images 100000

  # Generate with custom output path and seed
  python generate_simplified_dataset.py --num-images 100000 --output my_data.npz --seed 42
        """
    )

    parser.add_argument(
        '--num-images',
        type=int,
        default=100000,
        help='Number of images to generate (default: 100000)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/simplified_dataset_100k.npz',
        help='Output file path (default: data/simplified_dataset_100k.npz)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )

    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Skip confirmation prompt'
    )

    args = parser.parse_args()

    if args.num_images <= 0:
        parser.error('--num-images must be positive')

    return args


def main():
    """Main entry point."""
    args = parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"[SEED] Set random seed: {args.seed}")

    # Estimate memory
    bytes_needed, mb_estimate = estimate_memory(args.num_images, dtype=np.uint8)

    # Display configuration
    print("\n" + "=" * 70)
    print("SIMPLIFIED DATASET GENERATION")
    print("=" * 70)
    print(f"Number of images:  {args.num_images:,}")
    print(f"Grid size:         16x16")
    print(f"Color palette:     10 colors (0=background, 1-9=objects)")
    print(f"Object types:      Rectangles, Lines, Patterns (NO BLOBS)")
    print(f"Objects per grid:  1-6 (random)")
    print(f"Data type:         uint8 (memory efficient)")
    print(f"Estimated size:    {format_size(bytes_needed)} ({mb_estimate:.2f} MB)")
    print(f"Output file:       {args.output}")
    print(f"Train/Val/Test:    80% / 10% / 10%")
    print("=" * 70)

    # Confirm generation
    if not args.no_confirm:
        response = input("\nProceed with generation? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Generation cancelled.")
            return 1

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    print(f"\n[GENERATION] Generating {args.num_images:,} simplified images...")
    images = generate_simplified_corpus(
        corpus_size=args.num_images,
        dtype=np.uint8,
        show_progress=True
    )

    # Split dataset
    print("\n[SPLIT] Splitting into train/val/test (80/10/10)...")
    train, val, test = split_dataset(images)

    # Save to .npz file
    print(f"\n[SAVE] Saving to {args.output}...")
    np.savez_compressed(
        args.output,
        train=train,
        val=val,
        test=test
    )

    # Verify file size
    actual_size = output_path.stat().st_size

    # Display summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Output file:       {args.output}")
    print(f"Actual file size:  {format_size(actual_size)}")
    print(f"Train samples:     {len(train):,}")
    print(f"Val samples:       {len(val):,}")
    print(f"Test samples:      {len(test):,}")
    print(f"Total samples:     {len(train) + len(val) + len(test):,}")

    if args.seed is not None:
        print(f"Random seed:       {args.seed} (reproducible)")

    print("\n" + "=" * 70)
    print("DATASET CHARACTERISTICS")
    print("=" * 70)
    print("✓ Contains ONLY parameterized shapes (rectangles, lines, patterns)")
    print("✓ NO blob objects (excluded for reduced variability)")
    print("✓ Suitable for model capacity validation with beta=0")
    print("=" * 70)

    print("\nNext steps:")
    print(f"  1. Train model: python train_vae.py --data-path {args.output} --beta-schedule constant --beta-max 0.0")
    print("  2. Target accuracy: >95% (above 93% collapse threshold)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
