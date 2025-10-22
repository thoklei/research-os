#!/usr/bin/env python3
"""
Visual Validation Tool for Dataset Quality Assurance

This script loads a generated dataset and displays 100 random samples in a 10x10 grid
for visual inspection and quality assurance.

Usage:
    python validate_visual.py --corpus-path datasets/2025-10-22-v1/corpus.npz \
                              --metadata-path datasets/2025-10-22-v1/metadata.json \
                              --output validation.png \
                              --seed 42

Features:
- Random sample selection (100 samples by default)
- 10x10 grid layout with ARC color scheme
- Metadata overlay (version, timestamp, sample indices)
- Visual statistics (mean, std, min, max pixel values)
- PNG export for documentation
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from visualization import load_corpus, ARC_COLORMAP


def select_random_samples(corpus: List[np.ndarray],
                          num_samples: int = 100,
                          seed: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Select random samples from corpus.

    Args:
        corpus: List of grid arrays or numpy array (N, 16, 16)
        num_samples: Number of samples to select (default: 100)
        seed: Random seed for reproducibility (optional)

    Returns:
        Tuple of (selected samples, indices)
    """
    # Convert to numpy array if needed
    if isinstance(corpus, list):
        corpus_array = np.array([g.data if hasattr(g, 'data') else g for g in corpus])
    else:
        corpus_array = corpus

    total_samples = len(corpus_array)

    # Limit to available samples
    num_samples = min(num_samples, total_samples)

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Select random indices
    indices = np.random.choice(total_samples, size=num_samples, replace=False)
    indices = np.sort(indices)  # Sort for easier reference

    # Select samples
    samples = corpus_array[indices]

    return samples, indices


def compute_visual_statistics(samples: np.ndarray) -> Dict[str, float]:
    """
    Compute visual statistics for samples.

    Args:
        samples: Array of samples (N, 16, 16)

    Returns:
        Dictionary with mean, std, min, max
    """
    stats = {
        'mean': float(np.mean(samples)),
        'std': float(np.std(samples)),
        'min': int(np.min(samples)),
        'max': int(np.max(samples))
    }

    return stats


def create_validation_grid(samples: np.ndarray,
                          indices: np.ndarray,
                          metadata: Optional[Dict[str, Any]] = None,
                          statistics: Optional[Dict[str, float]] = None,
                          show: bool = True,
                          figsize: Tuple[int, int] = (20, 20)) -> plt.Figure:
    """
    Create 10x10 grid visualization of samples.

    Args:
        samples: Array of samples (N, 16, 16)
        indices: Array of sample indices
        metadata: Optional metadata dictionary
        statistics: Optional statistics dictionary
        show: Whether to display the plot
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure instance
    """
    # Create colormap
    cmap = ListedColormap(ARC_COLORMAP)

    # Create figure with subplots
    rows, cols = 10, 10
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Plot each sample
    num_samples = min(len(samples), rows * cols)

    for idx in range(rows * cols):
        ax = axes_flat[idx]

        if idx < num_samples:
            # Display sample
            ax.imshow(samples[idx], cmap=cmap, vmin=0, vmax=9, interpolation='nearest')

            # Add sample index as title
            ax.set_title(f"#{indices[idx]}", fontsize=8, pad=2)

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Add subtle gridlines
            ax.set_xticks(np.arange(-0.5, 16, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.2)
        else:
            # Hide empty subplots
            ax.axis('off')

    # Create title with metadata
    title_parts = ["Visual Validation Grid"]

    if metadata:
        version = metadata.get('version', 'unknown')
        timestamp = metadata.get('timestamp', 'unknown')
        num_images = metadata.get('num_images', 'unknown')

        title_parts.append(f"Version: {version}")
        title_parts.append(f"Dataset: {num_images} images")
        title_parts.append(f"Generated: {timestamp[:10]}")  # Just the date

    if statistics:
        stats_str = f"Stats: μ={statistics['mean']:.2f}, σ={statistics['std']:.2f}, range=[{statistics['min']}, {statistics['max']}]"
        title_parts.append(stats_str)

    # Add sample range
    if len(indices) > 0:
        title_parts.append(f"Samples: {len(indices)} (indices {indices[0]}-{indices[-1]})")

    title = " | ".join(title_parts)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def save_validation_grid(fig: plt.Figure,
                         output_path: str,
                         dpi: int = 150):
    """
    Save validation grid as PNG.

    Args:
        fig: matplotlib Figure instance
        output_path: Path to save PNG
        dpi: Resolution (default: 150)
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Save with high quality
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved validation grid to: {output_path}")


def validate_visual_dataset(corpus_path: str,
                            metadata_path: str,
                            output_path: str,
                            num_samples: int = 100,
                            seed: Optional[int] = None,
                            show: bool = False):
    """
    Complete visual validation workflow.

    Args:
        corpus_path: Path to corpus.npz file
        metadata_path: Path to metadata.json file
        output_path: Path to save validation PNG
        num_samples: Number of samples to display (default: 100)
        seed: Random seed for reproducibility (optional)
        show: Whether to display the plot (default: False)
    """
    print("=" * 60)
    print("VISUAL VALIDATION")
    print("=" * 60)

    # Load corpus
    print(f"Loading corpus from: {corpus_path}")
    data = load_corpus(corpus_path)
    images = data['images']
    print(f"  - Loaded {len(images)} images")
    print(f"  - Shape: {images.shape}")
    print(f"  - Dtype: {images.dtype}")

    # Load metadata
    print(f"\nLoading metadata from: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"  - Version: {metadata.get('version', 'unknown')}")
    print(f"  - Timestamp: {metadata.get('timestamp', 'unknown')}")

    # Select random samples
    print(f"\nSelecting {num_samples} random samples...")
    samples, indices = select_random_samples(
        images,
        num_samples=num_samples,
        seed=seed
    )
    print(f"  - Selected {len(samples)} samples")
    print(f"  - Index range: {indices[0]} to {indices[-1]}")

    # Compute statistics
    print(f"\nComputing visual statistics...")
    stats = compute_visual_statistics(samples)
    print(f"  - Mean: {stats['mean']:.2f}")
    print(f"  - Std:  {stats['std']:.2f}")
    print(f"  - Min:  {stats['min']}")
    print(f"  - Max:  {stats['max']}")

    # Create validation grid
    print(f"\nCreating 10x10 validation grid...")
    fig = create_validation_grid(
        samples,
        indices,
        metadata=metadata,
        statistics=stats,
        show=show
    )

    # Save to PNG
    print(f"\nSaving validation grid...")
    save_validation_grid(fig, output_path)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Output saved to: {output_path}")
    print(f"Samples validated: {len(samples)}/{len(images)}")

    plt.close(fig)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Visual validation tool for dataset quality assurance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate dataset with default settings
  python validate_visual.py --corpus-path datasets/2025-10-22-v1/corpus.npz \\
                            --metadata-path datasets/2025-10-22-v1/metadata.json \\
                            --output validation.png

  # Validate with custom seed for reproducibility
  python validate_visual.py --corpus-path datasets/2025-10-22-v1/corpus.npz \\
                            --metadata-path datasets/2025-10-22-v1/metadata.json \\
                            --output validation.png \\
                            --seed 42

  # Validate with fewer samples
  python validate_visual.py --corpus-path datasets/2025-10-22-v1/corpus.npz \\
                            --metadata-path datasets/2025-10-22-v1/metadata.json \\
                            --output validation.png \\
                            --num-samples 50
        """
    )

    parser.add_argument(
        '--corpus-path',
        type=str,
        required=True,
        help='Path to corpus.npz file'
    )

    parser.add_argument(
        '--metadata-path',
        type=str,
        required=True,
        help='Path to metadata.json file'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save validation PNG'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples to display (default: 100)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the plot (default: False)'
    )

    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = parse_args()

    # Validate inputs
    if not os.path.exists(args.corpus_path):
        print(f"Error: Corpus file not found: {args.corpus_path}")
        return 1

    if not os.path.exists(args.metadata_path):
        print(f"Error: Metadata file not found: {args.metadata_path}")
        return 1

    # Run validation
    try:
        validate_visual_dataset(
            corpus_path=args.corpus_path,
            metadata_path=args.metadata_path,
            output_path=args.output,
            num_samples=args.num_samples,
            seed=args.seed,
            show=args.show
        )

        return 0

    except Exception as e:
        print(f"\n✗ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
