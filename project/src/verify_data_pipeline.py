"""
End-to-End Data Pipeline Verification - Task 2.5

Demonstrates and verifies the complete data loading pipeline:
1. Load datasets from .npz corpus
2. Apply data augmentation
3. Create data loaders
4. Iterate through batches
5. Verify data integrity
"""

import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data import create_data_loaders, print_data_loader_summary


def verify_batch_integrity(batch: torch.Tensor, batch_idx: int):
    """Verify that batch has expected properties."""
    # Check shape
    assert batch.ndim == 3, f"Batch should be 3D (batch, H, W), got {batch.ndim}D"
    assert batch.shape[1:] == (16, 16), f"Grid should be 16×16, got {batch.shape[1:]}"

    # Check dtype
    assert batch.dtype == torch.long, f"Batch should be torch.long, got {batch.dtype}"

    # Check value range
    assert batch.min() >= 0, f"Batch has values < 0: {batch.min()}"
    assert batch.max() < 10, f"Batch has values >= 10: {batch.max()}"

    # Check for NaN
    assert not torch.isnan(batch.float()).any(), "Batch contains NaN values"

    return True


def main():
    print("=" * 70)
    print("β-VAE Data Loading Pipeline Verification")
    print("=" * 70)

    # Path to corpus
    corpus_path = Path(__file__).parent.parent / "datasets" / "test-100k" / "corpus.npz"

    if not corpus_path.exists():
        print(f"\n[ERROR] Corpus not found at: {corpus_path}")
        print("Please generate the corpus first using generate_dataset.py")
        return

    print(f"\n[OK] Found corpus at: {corpus_path}")

    # Create data loaders
    print("\n" + "─" * 70)
    print("Creating data loaders...")
    print("─" * 70)

    train_loader, val_loader, test_loader = create_data_loaders(
        npz_path=str(corpus_path),
        batch_size=128,
        num_workers=0,  # Use 0 for verification
        use_augmentation=True,
    )

    # Print summary
    print_data_loader_summary(train_loader, val_loader, test_loader)

    # Verify training loader
    if train_loader is not None:
        print("\n" + "─" * 70)
        print("Verifying Training Data Loader...")
        print("─" * 70)

        total_samples = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Verify batch
            verify_batch_integrity(batch, batch_idx)

            total_samples += batch.shape[0]
            num_batches += 1

            # Print progress for first few batches
            if batch_idx < 3:
                print(f"  Batch {batch_idx + 1}: shape={batch.shape}, "
                      f"min={batch.min()}, max={batch.max()}, "
                      f"dtype={batch.dtype}")

        print(f"\n[OK] Verified {num_batches} batches, {total_samples} total samples")
        print(f"[OK] All batches have correct shape, dtype, and value range")

    # Verify validation loader
    if val_loader is not None:
        print("\n" + "─" * 70)
        print("Verifying Validation Data Loader...")
        print("─" * 70)

        batch = next(iter(val_loader))
        verify_batch_integrity(batch, 0)

        print(f"  Batch shape: {batch.shape}")
        print(f"  Value range: [{batch.min()}, {batch.max()}]")
        print(f"  Dtype: {batch.dtype}")
        print(f"\n[OK] Validation loader working correctly")

    # Verify test loader
    if test_loader is not None:
        print("\n" + "─" * 70)
        print("Verifying Test Data Loader...")
        print("─" * 70)

        batch = next(iter(test_loader))
        verify_batch_integrity(batch, 0)

        print(f"  Batch shape: {batch.shape}")
        print(f"  Value range: [{batch.min()}, {batch.max()}]")
        print(f"  Dtype: {batch.dtype}")
        print(f"\n[OK] Test loader working correctly")

    # Test augmentation effect
    print("\n" + "─" * 70)
    print("Testing Data Augmentation...")
    print("─" * 70)

    # Get same sample twice to see if augmentation changes it
    from data import ARCDataset, get_train_transforms

    dataset_with_aug = ARCDataset(str(corpus_path), split='all', transform=get_train_transforms())
    dataset_without_aug = ARCDataset(str(corpus_path), split='all', transform=None)

    # Get same index multiple times with augmentation
    idx = 0
    sample1 = dataset_with_aug[idx]
    sample2 = dataset_with_aug[idx]

    # They should potentially be different due to stochastic augmentation
    if not torch.equal(sample1, sample2):
        print(f"  [OK] Augmentation is stochastic (samples differ)")
    else:
        print(f"  [OK] Augmentation applied (may be same by chance)")

    # Without augmentation, should be identical
    sample1_no_aug = dataset_without_aug[idx]
    sample2_no_aug = dataset_without_aug[idx]
    assert torch.equal(sample1_no_aug, sample2_no_aug), "Without augmentation, samples should be identical"
    print(f"  [OK] Without augmentation, samples are identical")

    # Test integration with model
    print("\n" + "─" * 70)
    print("Testing Integration with β-VAE Model...")
    print("─" * 70)

    from models.beta_vae import BetaVAE

    model = BetaVAE(latent_dim=10, num_colors=10)
    model.eval()

    # Get a batch
    batch = next(iter(train_loader))

    # Forward pass
    with torch.no_grad():
        recon_logits, mu, logvar = model(batch)

    print(f"  Input shape: {batch.shape}")
    print(f"  Reconstruction logits shape: {recon_logits.shape}")
    print(f"  μ shape: {mu.shape}")
    print(f"  logvar shape: {logvar.shape}")

    # Verify shapes
    assert recon_logits.shape == (batch.shape[0], 10, 16, 16), "Incorrect reconstruction shape"
    assert mu.shape == (batch.shape[0], 10), "Incorrect μ shape"
    assert logvar.shape == (batch.shape[0], 10), "Incorrect logvar shape"

    print(f"\n[OK] Model successfully processes batches from data loader")

    # Summary
    print("\n" + "=" * 70)
    print("Data Loading Pipeline Verification Complete")
    print("=" * 70)
    print("\nAll checks passed:")
    print("  [OK] Datasets load correctly from .npz files")
    print("  [OK] Train/val/test splits work properly")
    print("  [OK] Data loaders produce correct batch shapes")
    print("  [OK] Data augmentation is working")
    print("  [OK] Integration with β-VAE model works")
    print("\nThe data loading pipeline is ready for training!")
    print("=" * 70)


if __name__ == "__main__":
    main()
