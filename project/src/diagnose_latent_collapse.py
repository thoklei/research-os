"""
Diagnostic script to investigate posterior collapse in β-VAE.

This script will:
1. Check encoder outputs (mu, logvar) statistics
2. Check decoder inputs (z) statistics
3. Visualize latent space diversity
4. Check gradient flow
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.append(str(Path(__file__).parent))

from models import BetaVAE
from data import create_data_loaders
from training import load_checkpoint


def diagnose_model(checkpoint_path, data_loader, device='cpu'):
    """Comprehensive diagnosis of VAE training issues."""

    print("=" * 80)
    print("VAE DIAGNOSTIC REPORT")
    print("=" * 80)

    # Load model
    print("\n[1] Loading model...")
    model = BetaVAE(latent_dim=16, num_colors=10, use_focal_loss=True, focal_gamma=2.0)
    checkpoint = load_checkpoint(
        filepath=str(checkpoint_path),
        model=model,
        optimizer=None,
        scheduler=None
    )
    model = model.to(device)
    model.eval()
    print(f"    Loaded from epoch {checkpoint['epoch']}")

    # Get a batch
    batch = next(iter(data_loader))[:32].to(device)
    print(f"\n[2] Testing on batch of {batch.shape[0]} samples")
    print(f"    Input shape: {batch.shape}")
    print(f"    Input value range: [{batch.min().item()}, {batch.max().item()}]")

    # Forward pass with diagnostics
    print("\n[3] Encoder Output Analysis")
    print("-" * 80)

    with torch.no_grad():
        # Encode
        mu, logvar = model.encoder(batch)

        print(f"    Latent dimension: {mu.shape[1]}")
        print(f"\n    μ (mean) statistics:")
        print(f"      Range: [{mu.min().item():.6f}, {mu.max().item():.6f}]")
        print(f"      Mean: {mu.mean().item():.6f}")
        print(f"      Std: {mu.std().item():.6f}")
        print(f"      Per-dimension mean: {mu.mean(dim=0)[:8].numpy()}")  # First 8 dims
        print(f"      Per-dimension std: {mu.std(dim=0)[:8].numpy()}")

        print(f"\n    logvar (log variance) statistics:")
        print(f"      Range: [{logvar.min().item():.6f}, {logvar.max().item():.6f}]")
        print(f"      Mean: {logvar.mean().item():.6f}")
        print(f"      Std: {logvar.std().item():.6f}")

        # Compute actual variance
        var = torch.exp(logvar)
        print(f"\n    var = exp(logvar) statistics:")
        print(f"      Range: [{var.min().item():.10f}, {var.max().item():.10f}]")
        print(f"      Mean: {var.mean().item():.10f}")

        # Check if variance is collapsed
        if var.max().item() < 1e-6:
            print(f"\n    ⚠️  WARNING: Variance is COLLAPSED (max var < 1e-6)")
            print(f"    ⚠️  Encoder is outputting deterministic latents!")
        elif var.max().item() < 0.01:
            print(f"\n    ⚠️  WARNING: Variance is very small (max var < 0.01)")

        # Sample from latent
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        print(f"\n    z (sampled latent) statistics:")
        print(f"      Range: [{z.min().item():.6f}, {z.max().item():.6f}]")
        print(f"      Mean: {z.mean().item():.6f}")
        print(f"      Std: {z.std().item():.6f}")

        # Check diversity
        print(f"\n    Latent diversity check:")
        z_diff = torch.pdist(z)  # Pairwise distances
        print(f"      Pairwise L2 distances between latents:")
        print(f"        Min: {z_diff.min().item():.6f}")
        print(f"        Max: {z_diff.max().item():.6f}")
        print(f"        Mean: {z_diff.mean().item():.6f}")

        if z_diff.mean().item() < 0.1:
            print(f"    ⚠️  WARNING: Latent vectors are very similar (mean dist < 0.1)")
            print(f"    ⚠️  Model may be encoding all images to the same latent!")

    # Decoder output analysis
    print("\n[4] Decoder Output Analysis")
    print("-" * 80)

    with torch.no_grad():
        recon_logits = model.decoder(z)
        recon = recon_logits.argmax(dim=1)

        print(f"    Reconstruction logits shape: {recon_logits.shape}")
        print(f"    Logits range: [{recon_logits.min().item():.4f}, {recon_logits.max().item():.4f}]")

        # Check per-class logit bias
        print(f"\n    Per-class average logits:")
        for c in range(10):
            avg_logit = recon_logits[:, c, :, :].mean().item()
            print(f"      Class {c}: {avg_logit:+.4f}")

        # Check reconstruction distribution
        print(f"\n    Reconstruction class distribution:")
        unique, counts = torch.unique(recon, return_counts=True)
        total = recon.numel()
        for cls, count in zip(unique.tolist(), counts.tolist()):
            pct = 100 * count / total
            print(f"      Class {cls}: {count:6d} pixels ({pct:5.2f}%)")

    # KL divergence check
    print("\n[5] KL Divergence Analysis")
    print("-" * 80)

    with torch.no_grad():
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_sample = kl_div / batch.shape[0]
        kl_per_dim = kl_per_sample / mu.shape[1]

        print(f"    Total KL: {kl_div.item():.6f}")
        print(f"    KL per sample: {kl_per_sample.item():.6f}")
        print(f"    KL per dimension: {kl_per_dim.item():.6f}")

        if kl_per_dim.item() < 0.01:
            print(f"\n    ⚠️  CRITICAL: KL/dim < 0.01 indicates posterior collapse!")
            print(f"    ⚠️  The latent space is not learning useful representations")

    # Reconstruction quality
    print("\n[6] Reconstruction Quality")
    print("-" * 80)

    with torch.no_grad():
        correct = (recon == batch).sum().item()
        total = batch.numel()
        acc = correct / total

        print(f"    Pixel accuracy: {acc:.4f} ({correct}/{total})")

        # Per-class accuracy
        print(f"\n    Per-class accuracy:")
        for c in range(10):
            mask = (batch == c)
            if mask.sum() > 0:
                class_correct = ((recon == batch) & mask).sum().item()
                class_total = mask.sum().item()
                class_acc = class_correct / class_total
                print(f"      Class {c}: {class_acc:.4f} ({class_correct}/{class_total} pixels)")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

    # Summary and recommendations
    print("\n[SUMMARY]")
    if var.max().item() < 1e-6:
        print("  ⚠️  CRITICAL ISSUE: Complete posterior collapse detected")
        print("  ⚠️  Encoder variance → 0, latent space is deterministic")
        print("\n[RECOMMENDATIONS]")
        print("  1. Try β=0 (standard autoencoder) to verify reconstruction works")
        print("  2. Implement free-bits constraint to prevent KL collapse")
        print("  3. Use cyclical β-annealing or start from β=0")
        print("  4. Check if encoder/decoder architecture has sufficient capacity")
    elif z_diff.mean().item() < 0.1:
        print("  ⚠️  WARNING: Low latent diversity")
        print("\n[RECOMMENDATIONS]")
        print("  1. Reduce β to allow more latent variance")
        print("  2. Increase latent dimension")
        print("  3. Check if reconstruction loss is providing enough signal")


def main():
    checkpoint_path = Path("experiments/0.2-beta-vae/checkpoints/best_model.pth")

    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return

    # Load data
    print("Loading test data...")
    _, _, test_loader = create_data_loaders(
        npz_path='../datasets/test-100k/corpus.npz',
        batch_size=32,
        num_workers=0,
        use_augmentation=False
    )

    # Run diagnostics
    diagnose_model(checkpoint_path, test_loader, device='cpu')


if __name__ == "__main__":
    main()
