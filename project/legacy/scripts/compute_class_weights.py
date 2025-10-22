"""
Compute and save class weights for the training dataset.

This computes weights based on class frequency to handle the severe
class imbalance in ARC grids (~93% black pixels).
"""

import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent))

from data import ARCDataset
from models import compute_class_weights, save_class_weights


def main():
    parser = argparse.ArgumentParser(description="Compute class weights for ARC dataset")
    parser.add_argument(
        '--data-path',
        type=str,
        default='../datasets/test-100k/corpus.npz',
        help='Path to dataset .npz file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='class_weights.pth',
        help='Output path for class weights'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='inverse',
        choices=['inverse', 'sqrt_inverse', 'balanced'],
        help='Weighting method'
    )
    parser.add_argument(
        '--smooth',
        type=float,
        default=1.0,
        help='Smoothing factor'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Computing Class Weights for ARC Dataset")
    print("=" * 70)
    print(f"  Data path: {args.data_path}")
    print(f"  Method: {args.method}")
    print(f"  Smooth: {args.smooth}")
    print()

    # Load training dataset
    print("Loading training dataset...")
    dataset = ARCDataset(npz_path=args.data_path, split='train', transform=None)
    print(f"  Loaded {len(dataset)} training samples\n")

    # Compute weights
    weights = compute_class_weights(
        dataset=dataset,
        num_classes=10,
        method=args.method,
        smooth=args.smooth,
        normalize=True
    )

    # Save weights
    print()
    save_class_weights(weights, args.output)

    # Summary
    print("\n" + "=" * 70)
    print("Class Weights Computed Successfully")
    print("=" * 70)
    print(f"Weights saved to: {args.output}")
    print("\nTo use these weights in training:")
    print("  python train_vae.py --use-class-weights --class-weights-path class_weights.pth")
    print("=" * 70)


if __name__ == "__main__":
    main()
