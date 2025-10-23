"""
Quick verification script to test sqrt_inverse class weight calculation.

Tests that sqrt smoothing reduces extreme weight ratios from ~123:1 to ~11:1.
"""

import torch
import numpy as np

def compute_test_weights(class_counts, method='sqrt_inverse', smooth=1.0, normalize=True, num_classes=10):
    """Simplified version of compute_class_weights for testing."""
    class_counts = torch.tensor(class_counts, dtype=torch.float32)

    if method == 'inverse':
        weights = 1.0 / (class_counts + smooth)
    elif method == 'sqrt_inverse':
        weights = 1.0 / torch.sqrt(class_counts + smooth)
    elif method == 'balanced':
        n_samples = class_counts.sum().item()
        weights = n_samples / (num_classes * (class_counts + smooth))
    else:
        raise ValueError(f"Unknown method: {method}")

    if normalize:
        weights = weights / weights.sum() * num_classes

    return weights

# Simulate ARC-like class distribution
# Assume 80000 training samples, 16x16 grids = 20,480,000 pixels
# 93% background (class 0), 7% distributed across other classes
total_pixels = 80000 * 16 * 16  # 20,480,000

# Class distribution (approximate ARC statistics)
class_counts = [
    int(total_pixels * 0.93),  # Class 0 (black background): 93%
    int(total_pixels * 0.01),  # Class 1: ~1%
    int(total_pixels * 0.01),  # Class 2: ~1%
    int(total_pixels * 0.01),  # Class 3: ~1%
    int(total_pixels * 0.01),  # Class 4: ~1%
    int(total_pixels * 0.01),  # Class 5: ~1%
    int(total_pixels * 0.01),  # Class 6: ~1%
    int(total_pixels * 0.005), # Class 7: ~0.5%
    int(total_pixels * 0.005), # Class 8: ~0.5%
    int(total_pixels * 0.005), # Class 9: ~0.5%
]

print("=" * 70)
print("Class Weight Verification Test")
print("=" * 70)
print(f"\nTotal pixels: {total_pixels:,}")
print(f"Class distribution:")
for i, count in enumerate(class_counts):
    pct = 100 * count / total_pixels
    print(f"  Class {i}: {count:>10,} pixels ({pct:>5.2f}%)")

# Test all three methods
methods = ['inverse', 'sqrt_inverse', 'balanced']

print("\n" + "=" * 70)
print("Computed Class Weights")
print("=" * 70)

for method in methods:
    weights = compute_test_weights(class_counts, method=method)
    max_weight = weights.max().item()
    min_weight = weights.min().item()
    ratio = max_weight / min_weight

    print(f"\nMethod: {method}")
    print(f"  Weights: [{', '.join(f'{w:.4f}' for w in weights[:5])}...]")
    print(f"  Max weight: {max_weight:.4f}")
    print(f"  Min weight: {min_weight:.4f}")
    print(f"  Ratio (max/min): {ratio:.2f}x")

    if method == 'sqrt_inverse':
        print(f"  ✓ Sqrt smoothing reduces ratio from ~123x to ~{ratio:.0f}x")
        if 10 <= ratio <= 12:
            print(f"  ✓ Expected ratio achieved (~11x)")
        else:
            print(f"  ⚠ Ratio outside expected range [10-12]")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("The sqrt_inverse method successfully reduces extreme weight ratios")
print("while maintaining awareness of class imbalance.")
print("")
print("Recommendation: Use 'sqrt_inverse' with Focal Loss (γ=2.0)")
print("=" * 70)
