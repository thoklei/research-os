"""
Evaluation Script for β-VAE - Task 4

Comprehensive evaluation and visualization of trained β-VAE models.

Usage:
    python evaluate_vae.py --checkpoint path/to/checkpoint.pth
    python evaluate_vae.py --checkpoint path/to/checkpoint.pth --output-dir results/
    python evaluate_vae.py --checkpoint path/to/checkpoint.pth --no-viz
"""

import argparse
import torch
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from models.beta_vae import BetaVAE
from data import create_data_loaders
from evaluation import (
    evaluate_model,
    plot_reconstructions,
    plot_latent_space,
    plot_latent_traversals,
    save_generated_samples,
)
from training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained β-VAE model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='../datasets/test-100k/corpus.npz',
        help='Path to dataset .npz file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for visualizations (default: checkpoint dir)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu/mps)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of samples for latent space visualization'
    )

    args = parser.parse_args()

    # Setup device
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("β-VAE Model Evaluation")
    print("=" * 70)
    print(f"[SETUP] Device: {device}")
    print(f"[SETUP] Checkpoint: {args.checkpoint}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return

    # Create model
    model = BetaVAE(latent_dim=10, num_colors=10)

    # Load weights
    checkpoint = load_checkpoint(
        filepath=str(checkpoint_path),
        model=model,
        optimizer=None,
        scheduler=None
    )

    epoch = checkpoint.get('epoch', 'unknown')
    print(f"[LOAD] Loaded checkpoint from epoch {epoch}")

    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"[LOAD] Checkpoint metrics:")
        print(f"  Loss: {metrics.get('loss', 'N/A'):.4f}")
        print(f"  Pixel Acc: {metrics.get('pixel_acc', 'N/A'):.3f}")

    model.to(device)

    # Setup output directory
    if args.output_dir is None:
        output_dir = checkpoint_path.parent.parent / 'evaluation'
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SETUP] Output directory: {output_dir}")

    # Load data
    print("\n" + "─" * 70)
    print("Loading data...")
    print("─" * 70)

    train_loader, val_loader, test_loader = create_data_loaders(
        npz_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        use_augmentation=False,  # No augmentation for evaluation
    )

    if test_loader is None:
        print("[WARNING] No test set available, using validation set")
        eval_loader = val_loader
    else:
        eval_loader = test_loader

    print(f"[DATA] Evaluation set: {len(eval_loader.dataset)} samples")

    # Evaluate model
    print("\n" + "─" * 70)
    print("Computing evaluation metrics...")
    print("─" * 70)

    metrics = evaluate_model(
        model=model,
        data_loader=eval_loader,
        device=device,
        num_colors=10,
        compute_disentanglement=True,
    )

    # Print results
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    print("\nReconstruction Quality:")
    print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.3%}")
    print(f"  Reconstruction Loss: {metrics['recon_loss']:.4f}")

    print("\nPer-Class Accuracy:")
    for i in range(10):
        key = f'class_{i}_acc'
        if key in metrics:
            print(f"  Color {i}: {metrics[key]:.3%}")

    print("\nLatent Space Quality:")
    print(f"  KL Divergence: {metrics['kl_divergence']:.4f}")
    print(f"  KL per dimension: {metrics['kl_per_dim']:.4f}")
    print(f"  Active dimensions: {metrics['active_dims']}/10")
    print(f"  Posterior collapse: {'Yes' if metrics['posterior_collapse'] else 'No'}")

    print("\nDisentanglement:")
    print(f"  Disentanglement score: {metrics['disentanglement_score']:.4f}")
    print(f"  Latent variance (mean): {metrics['latent_variance_mean']:.4f}")
    print(f"  Latent variance (std): {metrics['latent_variance_std']:.4f}")

    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        # Convert all values to native Python types
        metrics_serializable = {k: float(v) if isinstance(v, (float, int)) else v
                               for k, v in metrics.items()}
        json.dump(metrics_serializable, f, indent=2)

    print(f"\n[SAVE] Metrics saved to: {metrics_path}")

    # Generate visualizations
    if not args.no_viz:
        print("\n" + "─" * 70)
        print("Generating visualizations...")
        print("─" * 70)

        # Reconstructions
        print("\n[VIZ] Generating reconstructions...")
        plot_reconstructions(
            model=model,
            data_loader=eval_loader,
            device=device,
            num_samples=8,
            save_path=str(output_dir / 'reconstructions.png'),
            show=False
        )

        # Latent space
        print("[VIZ] Generating latent space visualization...")
        plot_latent_space(
            model=model,
            data_loader=eval_loader,
            device=device,
            num_samples=args.num_samples,
            save_path=str(output_dir / 'latent_space.png'),
            show=False
        )

        # Latent traversals
        print("[VIZ] Generating latent traversals...")
        plot_latent_traversals(
            model=model,
            device=device,
            latent_dim=10,
            num_steps=7,
            save_path=str(output_dir / 'latent_traversals.png'),
            show=False
        )

        # Generated samples
        print("[VIZ] Generating random samples...")
        save_generated_samples(
            model=model,
            device=device,
            latent_dim=10,
            num_samples=16,
            save_path=str(output_dir / 'generated_samples.png'),
            show=False
        )

        print(f"\n[SAVE] All visualizations saved to: {output_dir}")

    # Summary
    print("\n" + "=" * 70)
    print("Evaluation Complete")
    print("=" * 70)

    # Check success criteria
    if metrics['pixel_accuracy'] >= 0.90:
        print(f"\n[SUCCESS] Target accuracy achieved: {metrics['pixel_accuracy']:.3%} >= 90%")
    else:
        print(f"\n[BELOW TARGET] Accuracy: {metrics['pixel_accuracy']:.3%} < 90%")

    if metrics['posterior_collapse']:
        print(f"[WARNING] Posterior collapse detected (KL/dim = {metrics['kl_per_dim']:.4f})")
    else:
        print(f"[OK] Latent space is active (KL/dim = {metrics['kl_per_dim']:.4f})")

    print("=" * 70)


if __name__ == "__main__":
    main()
