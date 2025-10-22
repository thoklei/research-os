#!/usr/bin/env python3
"""
Large-Scale Dataset Generation CLI

This script generates ARC-like atomic image datasets at scale (1K → 100K images).

Usage:
    python generate_dataset.py --num-images 1000 --output-dir datasets/run-001 --seed 42

Features:
- Hierarchical directory structure (datasets/YYYY-MM-DD-vN/)
- Memory estimation with user confirmation
- Progress tracking with tqdm
- Metadata generation
- uint8 dtype for memory efficiency (87.5% savings)
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

from pipeline import generate_corpus, split_corpus, estimate_memory
from visualization import save_corpus, save_metadata, create_metadata, validate_dataset


def parse_args(args=None):
    """
    Parse command-line arguments.

    Args:
        args: List of argument strings (for testing)

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Generate large-scale ARC-like atomic image datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1,000 images with automatic directory naming (saved to ../datasets/)
  python generate_dataset.py --num-images 1000

  # Generate 100,000 images with custom directory and seed
  python generate_dataset.py --num-images 100000 --output-dir ../datasets/my-run --seed 42

  # Generate with specific version
  python generate_dataset.py --num-images 1000 --seed 123
        """
    )

    parser.add_argument(
        '--num-images',
        type=int,
        required=True,
        help='Number of images to generate (e.g., 1000, 100000)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: auto-generated ../datasets/YYYY-MM-DD-vN/)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )

    parsed_args = parser.parse_args(args)

    # Validate num_images
    if parsed_args.num_images <= 0:
        parser.error('--num-images must be positive')

    return parsed_args


def create_dataset_directory(base_dir='.', version='v1', create_datasets_subdir=False):
    """
    Create hierarchical directory structure.

    Format: datasets/YYYY-MM-DD-vN/

    Args:
        base_dir: Base directory (default: current directory)
        version: Version string (default: 'v1')
        create_datasets_subdir: Create datasets/ subdirectory (default: False)

    Returns:
        Path to created directory
    """
    # Get current date
    date_str = datetime.now().strftime('%Y-%m-%d')

    # Create directory name
    dirname = f"{date_str}-{version}"

    # Build full path
    if create_datasets_subdir:
        output_dir = os.path.join(base_dir, 'datasets', dirname)
    else:
        output_dir = os.path.join(base_dir, dirname)

    # Create directory
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def format_memory_estimate(bytes_needed):
    """
    Format memory estimate in human-readable form.

    Args:
        bytes_needed: Number of bytes

    Returns:
        Formatted string (e.g., "25.6 MB", "256 KB")
    """
    if bytes_needed < 1024:
        return f"{bytes_needed} bytes"
    elif bytes_needed < 1024 * 1024:
        kb = bytes_needed / 1024
        return f"{kb:.1f} KB"
    else:
        mb = bytes_needed / (1024 * 1024)
        return f"{mb:.2f} MB"


def confirm_generation(num_images, bytes_needed, mb_estimate):
    """
    Display memory estimate and ask user to confirm.

    Args:
        num_images: Number of images to generate
        bytes_needed: Bytes required
        mb_estimate: MB estimate

    Returns:
        True if user confirms, False otherwise
    """
    print("\n" + "=" * 60)
    print("DATASET GENERATION")
    print("=" * 60)
    print(f"Number of images:  {num_images:,}")
    print(f"Grid size:         16x16")
    print(f"Data type:         uint8 (memory efficient)")
    print(f"Estimated memory:  {format_memory_estimate(bytes_needed)} ({mb_estimate:.2f} MB)")
    print("=" * 60)

    response = input("\nProceed with generation? [y/N]: ").strip().lower()
    return response in ['y', 'yes']


def generate_and_save_dataset(num_images, output_dir, seed=None, version='v1', show_progress=True):
    """
    Generate dataset and save to disk.

    Args:
        num_images: Number of images to generate
        output_dir: Output directory path
        seed: Random seed (optional)
        version: Dataset version
        show_progress: Show progress bar

    Returns:
        Path to saved corpus file
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Generate corpus with progress bar
    print(f"\nGenerating {num_images:,} images...")
    corpus = generate_corpus(
        corpus_size=num_images,
        dtype=np.uint8,
        show_progress=show_progress
    )

    # Split corpus
    print("Splitting into train/val/test (80/10/10)...")
    train, val, test = split_corpus(corpus)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save corpus
    corpus_path = os.path.join(output_dir, 'corpus.npz')
    print(f"Saving corpus to {corpus_path}...")
    save_corpus(corpus, corpus_path, train=train, val=val, test=test)

    # Create and save metadata
    metadata = create_metadata(
        num_images=num_images,
        dtype='uint8',
        seed=seed,
        version=version,
        train_size=len(train),
        val_size=len(val),
        test_size=len(test),
        object_types=['blob', 'rectangle', 'line', 'pattern'],
        object_distribution=[0.4, 0.2, 0.2, 0.2]
    )

    metadata_path = os.path.join(output_dir, 'metadata.json')
    print(f"Saving metadata to {metadata_path}...")
    save_metadata(metadata, metadata_path)

    print(f"\n✓ Dataset saved to: {output_dir}")
    print(f"  - Corpus:   corpus.npz")
    print(f"  - Metadata: metadata.json")
    print(f"  - Train:    {len(train)} images")
    print(f"  - Val:      {len(val)} images")
    print(f"  - Test:     {len(test)} images")

    return corpus_path


def main():
    """Main entry point for CLI."""
    # Parse arguments
    args = parse_args()

    # Estimate memory
    bytes_needed, mb_estimate = estimate_memory(args.num_images, dtype=np.uint8)

    # Confirm with user
    if not confirm_generation(args.num_images, bytes_needed, mb_estimate):
        print("\nGeneration cancelled.")
        return 1

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Auto-generate directory name in ../datasets/
        output_dir = create_dataset_directory(
            base_dir='../datasets',
            version='v1',
            create_datasets_subdir=False
        )

    # Generate and save dataset
    try:
        generate_and_save_dataset(
            num_images=args.num_images,
            output_dir=output_dir,
            seed=args.seed,
            version='v1',
            show_progress=True
        )

        # Validate generated dataset
        corpus_path = os.path.join(output_dir, 'corpus.npz')
        print("\nValidating generated dataset...")
        is_valid, message = validate_dataset(corpus_path)

        if is_valid:
            print(f"✓ {message}")
        else:
            print(f"✗ Validation failed: {message}")
            return 1

        print("\n" + "=" * 60)
        print("GENERATION COMPLETE")
        print("=" * 60)
        print(f"Dataset location: {output_dir}")
        print(f"Total images:     {args.num_images:,}")
        print(f"Memory used:      {format_memory_estimate(bytes_needed)}")

        if args.seed is not None:
            print(f"Random seed:      {args.seed} (reproducible)")

        print("\nNext steps:")
        print("  1. Inspect samples: python -c \"from visualization import *; from pipeline import *; corpus=load_corpus('{}/corpus.npz'); visualize_gallery(corpus['images'][:12], rows=3, cols=4)\"".format(output_dir))
        print("  2. Use for training: load with np.load('{}/corpus.npz')".format(output_dir))

        return 0

    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
