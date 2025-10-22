"""
Debug reconstruction issue - check actual values.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

from models.beta_vae import BetaVAE
from data import create_data_loaders
from training import load_checkpoint


def main():
    print("Debugging reconstruction issue...")

    # Load model
    print("\n[1] Loading model...")
    model = BetaVAE(latent_dim=10, num_colors=10)
    checkpoint = load_checkpoint(
        filepath='experiments/0.2-beta-vae/checkpoints/best_model.pth',
        model=model
    )
    model.eval()
    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Load data
    print("\n[2] Loading data...")
    _, _, test_loader = create_data_loaders(
        npz_path='../datasets/test-100k/corpus.npz',
        batch_size=8,
        num_workers=0,
        use_augmentation=False
    )

    # Get a batch
    batch = next(iter(test_loader))
    print(f"  Batch shape: {batch.shape}")
    print(f"  Batch dtype: {batch.dtype}")
    print(f"  Batch min/max: {batch.min()}/{batch.max()}")
    print(f"  Batch unique values: {torch.unique(batch)}")

    # Forward pass
    print("\n[3] Running forward pass...")
    with torch.no_grad():
        recon_logits, mu, logvar = model(batch)
        recon = recon_logits.argmax(dim=1)

    print(f"  Recon logits shape: {recon_logits.shape}")
    print(f"  Recon shape: {recon.shape}")
    print(f"  Recon dtype: {recon.dtype}")
    print(f"  Recon min/max: {recon.min()}/{recon.max()}")
    print(f"  Recon unique values: {torch.unique(recon)}")

    # Check first sample in detail
    print("\n[4] First sample details:")
    original = batch[0].numpy()
    reconstructed = recon[0].numpy()

    print(f"  Original shape: {original.shape}")
    print(f"  Original:\n{original}")
    print(f"\n  Reconstructed shape: {reconstructed.shape}")
    print(f"  Reconstructed:\n{reconstructed}")

    # Check accuracy
    matches = (original == reconstructed).sum()
    total = original.size
    accuracy = matches / total
    print(f"\n  Matches: {matches}/{total}")
    print(f"  Accuracy: {accuracy:.1%}")

    # Check latent codes
    print("\n[5] Latent space:")
    print(f"  mu shape: {mu.shape}")
    print(f"  mu mean: {mu.mean().item():.6f}")
    print(f"  mu std: {mu.std().item():.6f}")
    print(f"  mu min/max: {mu.min().item():.6f}/{mu.max().item():.6f}")
    print(f"  First sample mu: {mu[0].numpy()}")

    print(f"\n  logvar shape: {logvar.shape}")
    print(f"  logvar mean: {logvar.mean().item():.6f}")
    print(f"  logvar std: {logvar.std().item():.6f}")

    # Check if it's predicting all same class
    print("\n[6] Prediction distribution:")
    for i in range(10):
        count = (recon == i).sum().item()
        percentage = count / recon.numel() * 100
        print(f"  Class {i}: {count} pixels ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
