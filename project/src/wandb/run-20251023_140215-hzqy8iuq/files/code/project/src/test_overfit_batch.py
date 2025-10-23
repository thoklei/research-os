"""
Test script to verify the model can overfit a single batch.

This is a sanity check to ensure:
1. Model has sufficient capacity to represent the data
2. Loss function is working correctly
3. Gradients are flowing properly

Usage:
    python test_overfit_batch.py [--no-wandb]
"""

import argparse
import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

from models.beta_vae import BetaVAE
from data import create_data_loaders
from training import compute_pixel_accuracy, compute_kl_divergence_per_dim
from evaluation.visualization import plot_reconstructions

def create_single_batch_dataloader(single_batch):
    """Create a DataLoader-like object that returns the same batch repeatedly."""
    class SingleBatchLoader:
        def __init__(self, batch):
            self.batch = batch
            self.dataset = type('Dataset', (), {'__len__': lambda self: batch.size(0)})()

        def __iter__(self):
            yield self.batch

        def __len__(self):
            return 1

    return SingleBatchLoader(single_batch)


def test_overfit_single_batch(use_wandb=True):
    """Test if model can overfit a single batch."""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Single batch overfitting test")
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    args = parser.parse_args()

    use_wandb = use_wandb and not args.no_wandb

    print("="*70)
    print("Single Batch Overfitting Test")
    print("="*70)

    # Configuration
    latent_dim = 32
    num_colors = 10
    batch_size = 32
    num_epochs = 10  # Changed from iterations to epochs for consistency
    iters_per_epoch = 50
    learning_rate = 1e-3

    # Initialize wandb
    wandb_run = None
    run_id = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="arc-beta-vae",
                name="overfit-test",
                tags=["overfit-test", "sanity-check"],
                config={
                    "latent_dim": latent_dim,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "iters_per_epoch": iters_per_epoch,
                    "learning_rate": learning_rate,
                }
            )
            run_id = wandb_run.id
            print(f"\n[W&B] Initialized: {run_id}")
        except ImportError:
            print("\n[W&B] Not installed, skipping")
            use_wandb = False

    # Create output directory
    output_dir = Path(f"../experiments/overfit-tests/{run_id or 'no-wandb'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[OUTPUT] Results will be saved to: {output_dir}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[SETUP] Device: {device}")

    # Load data
    print("\n[DATA] Loading dataset...")
    train_loader, _, _ = create_data_loaders(
        npz_path="../datasets/test-100k/corpus.npz",
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        use_augmentation=False,
    )

    # Get single batch
    single_batch = next(iter(train_loader)).to(device)
    print(f"[DATA] Single batch shape: {single_batch.shape}")
    print(f"[DATA] Batch size: {single_batch.size(0)}")

    # Create DataLoader wrapper for visualization
    single_batch_loader = create_single_batch_dataloader(single_batch)

    # Create model
    print(f"\n[MODEL] Creating BetaVAE (latent_dim={latent_dim})")
    model = BetaVAE(
        latent_dim=latent_dim,
        num_colors=num_colors,
        use_focal_loss=True,
        focal_gamma=2.0,
        class_weights=None  # No class weights for overfitting test
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"\n[TRAIN] Starting overfitting test ({num_epochs} epochs, {iters_per_epoch} iters/epoch)...")
    print("="*70)

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-"*70)

        model.train()
        epoch_metrics = {'loss': 0, 'recon_loss': 0, 'kl_loss': 0, 'pixel_acc': 0, 'kl_per_dim': 0}

        for iteration in range(iters_per_epoch):
            # Forward pass
            recon_logits, mu, logvar = model(single_batch)

            # Compute loss with beta=1.0 (standard VAE)
            loss_dict = model.loss_function(recon_logits, single_batch, mu, logvar, beta=1.0)

            # Backward pass
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()

            # Compute metrics
            pixel_acc = compute_pixel_accuracy(recon_logits, single_batch)
            kl_per_dim = compute_kl_divergence_per_dim(mu, logvar)

            # Accumulate metrics
            epoch_metrics['loss'] += loss_dict['loss'].item()
            epoch_metrics['recon_loss'] += loss_dict['recon_loss'].item()
            epoch_metrics['kl_loss'] += loss_dict['kl_loss'].item()
            epoch_metrics['pixel_acc'] += pixel_acc
            epoch_metrics['kl_per_dim'] += kl_per_dim

            # Log to wandb
            if use_wandb and iteration % 10 == 0:
                wandb.log({
                    'train/loss': loss_dict['loss'].item(),
                    'train/recon_loss': loss_dict['recon_loss'].item(),
                    'train/kl_loss': loss_dict['kl_loss'].item(),
                    'train/pixel_acc': pixel_acc,
                    'train/kl_per_dim': kl_per_dim,
                    'epoch': epoch,
                }, step=global_step)

            global_step += 1

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= iters_per_epoch

        # Print epoch summary
        print(f"  Loss: {epoch_metrics['loss']:.4f} | "
              f"Recon: {epoch_metrics['recon_loss']:.4f} | "
              f"KL: {epoch_metrics['kl_loss']:.4f} | "
              f"Acc: {epoch_metrics['pixel_acc']:.4f} | "
              f"KL/dim: {epoch_metrics['kl_per_dim']:.4f}")

        # Generate visualization
        model.eval()
        viz_path = output_dir / f"overfit_epoch_{epoch}.png"
        try:
            plot_reconstructions(
                model=model,
                data_loader=single_batch_loader,
                device=device,
                num_samples=min(10, batch_size),
                save_path=str(viz_path),
                show=False
            )
            print(f"  [VIZ] Saved: {viz_path.name}")

            # Log to wandb
            if use_wandb:
                wandb.log({"reconstruction": wandb.Image(str(viz_path))}, step=global_step)
        except Exception as e:
            print(f"  [VIZ] Warning: Failed to generate visualization: {e}")

    print("="*70)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        recon_logits, mu, logvar = model(single_batch)
        loss_dict = model.loss_function(recon_logits, single_batch, mu, logvar, beta=1.0)
        pixel_acc = compute_pixel_accuracy(recon_logits, single_batch)
        kl_per_dim = compute_kl_divergence_per_dim(mu, logvar)

        # Get predictions
        predictions = torch.argmax(recon_logits, dim=1)

        # Compute per-sample accuracy
        per_sample_acc = (predictions == single_batch).float().mean(dim=(1,2))

    print("\n[RESULTS]")
    print(f"  Final Loss:         {loss_dict['loss'].item():.4f}")
    print(f"  Reconstruction Loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"  KL Loss:            {loss_dict['kl_loss'].item():.4f}")
    print(f"  Pixel Accuracy:     {pixel_acc:.4f} ({pixel_acc*100:.2f}%)")
    print(f"  KL per dimension:   {kl_per_dim:.4f}")
    print(f"\n  Per-sample accuracy stats:")
    print(f"    Min:  {per_sample_acc.min().item():.4f}")
    print(f"    Max:  {per_sample_acc.max().item():.4f}")
    print(f"    Mean: {per_sample_acc.mean().item():.4f}")
    print(f"    Std:  {per_sample_acc.std().item():.4f}")

    # Log final results to wandb
    if use_wandb:
        wandb.log({
            'final/loss': loss_dict['loss'].item(),
            'final/recon_loss': loss_dict['recon_loss'].item(),
            'final/kl_loss': loss_dict['kl_loss'].item(),
            'final/pixel_acc': pixel_acc,
            'final/kl_per_dim': kl_per_dim,
        })

        # Mark run as finished
        wandb.finish()

    # Success criteria
    print("\n[SANITY CHECK]")
    if pixel_acc > 0.95:
        print("  ✓ SUCCESS: Model can overfit single batch (>95% accuracy)")
        print("  → Model capacity is sufficient")
        print("  → Loss function is working correctly")
        status = "SUCCESS"
    elif pixel_acc > 0.80:
        print("  ⚠ PARTIAL: Model achieved >80% but <95% accuracy")
        print("  → May need more iterations or higher learning rate")
        status = "PARTIAL"
    else:
        print("  ✗ FAILURE: Model cannot overfit single batch (<80% accuracy)")
        print("  → Check model architecture or loss function")
        print("  → Possible issues: gradient flow, loss calculation, data format")
        status = "FAILURE"

    print(f"\n[OUTPUT] Visualizations saved to: {output_dir}")
    print("="*70)

    return pixel_acc, status


if __name__ == "__main__":
    accuracy, status = test_overfit_single_batch()
