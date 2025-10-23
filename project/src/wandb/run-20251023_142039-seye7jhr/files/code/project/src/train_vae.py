"""
Main Training Script for β-VAE

Usage:
    python train_vae.py                    # Train with default config
    python train_vae.py --quick-test       # Quick test run
    python train_vae.py --config path.json # Train with custom config
"""

import argparse
import torch
import random
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from models.beta_vae import BetaVAE
from data import create_data_loaders, create_overfit_data_loaders
from training import BetaVAETrainer, TrainingConfig, get_default_config, get_quick_test_config


class TeeLogger:
    """
    Logger that writes to file only (no console output).

    Redirects stdout/stderr to capture all print statements and errors
    to the run's log directory.
    """
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = open(log_path, 'w', buffering=1)  # Line buffered
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, message):
        self.log_file.write(message)

    def flush(self):
        self.log_file.flush()

    def close(self):
        self.log_file.close()

    def isatty(self):
        return False  # Log file is not a TTY

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.close()

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train β-VAE on ARC grids")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config JSON file'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with reduced epochs'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu/mps)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to dataset .npz file'
    )
    parser.add_argument(
        '--no-augmentation',
        action='store_true',
        help='Disable data augmentation'
    )
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=None,
        help='Latent space dimension'
    )
    parser.add_argument(
        '--overfit-batch',
        action='store_true',
        help='Run overfitting test on a single batch (sanity check)'
    )

    args = parser.parse_args()

    # Initialize wandb early to get run ID for folder naming
    wandb_run_id = None
    use_wandb = not args.quick_test or not args.no_wandb  # Default to True unless --no-wandb

    if use_wandb and not args.no_wandb:
        try:
            import wandb
            # Prepare tags based on mode
            wandb_tags = ["beta-vae", "arc", "experiment-0.2"]
            if args.overfit_batch:
                wandb_tags.append("overfit-test")
                wandb_tags.append("sanity-check")

            # Initialize wandb to get run ID
            temp_run = wandb.init(
                project="arc-beta-vae",
                tags=wandb_tags,
                mode="online"
            )
            wandb_run_id = temp_run.id
            print(f"[W&B] Initialized with run ID: {wandb_run_id}")
        except ImportError:
            print("[W&B] wandb not installed, using random run ID")
            use_wandb = False

    # Load config
    if args.config is not None:
        config = TrainingConfig.load(args.config)
        print(f"[CONFIG] Loaded from: {args.config}")
    elif args.quick_test:
        config = get_quick_test_config(run_id=wandb_run_id)
        print("[CONFIG] Using quick test config")
    else:
        config = get_default_config(run_id=wandb_run_id)
        print("[CONFIG] Using default config")

    # Override config from command line
    if args.no_wandb:
        config.use_wandb = False
        if wandb_run_id:
            # Finish the temp wandb run
            import wandb
            wandb.finish()

    if args.device is not None:
        config.device = args.device

    if args.data_path is not None:
        config.data_path = args.data_path

    if args.no_augmentation:
        config.use_augmentation = False

    if args.latent_dim is not None:
        config.latent_dim = args.latent_dim

    # Set up logging to file
    log_path = config.run_manager.get_log_path("training.log")
    print(f"[LOGGING] Output will be saved to: {log_path}")

    with TeeLogger(log_path):
        # Set seed
        set_seed(config.seed)
        print(f"[SETUP] Random seed: {config.seed}")

        # Print run information
        print("\n" + "=" * 70)
        print(f"RUN ID: {config.run_id}")
        print(f"RUN DIR: {config.run_manager.run_dir}")
        print("=" * 70)

        # Create data loaders
        print("\n" + "─" * 70)
        print("Loading data...")
        print("─" * 70)

        if args.overfit_batch:
            # Overfitting mode: use single batch for all loaders
            train_loader, val_loader, test_loader = create_overfit_data_loaders(
                npz_path=config.data_path,
                batch_size=config.batch_size,
                num_workers=0,  # Single batch doesn't need multiple workers
                pin_memory=config.pin_memory,
                num_repeats=50,  # 50 batches per epoch
            )
        else:
            # Normal training mode
            train_loader, val_loader, test_loader = create_data_loaders(
                npz_path=config.data_path,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                use_augmentation=config.use_augmentation,
            )

        if train_loader is None or val_loader is None:
            print("[ERROR] Could not load train/val data")
            return

        print(f"[DATA] Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
        print(f"[DATA] Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")
        if test_loader is not None:
            print(f"[DATA] Test:  {len(test_loader.dataset)} samples, {len(test_loader)} batches")

        # Compute or load class weights
        class_weights = None
        if config.use_class_weights:
            print("\n" + "─" * 70)
            print("Computing class weights...")
            print("─" * 70)

            from models import compute_class_weights

            # Compute weights from training data
            class_weights = compute_class_weights(
                dataset=train_loader.dataset,
                num_classes=config.num_colors,
                method=config.class_weight_method,
                smooth=config.class_weight_smooth,
                normalize=True
            )

            # Save weights to run directory
            weights_path = config.run_manager.run_dir / "class_weights.pth"
            from models import save_class_weights
            save_class_weights(class_weights, str(weights_path))
            print(f"[WEIGHTS] Saved to: {weights_path}")

        # Create model
        print("\n" + "─" * 70)
        print("Creating model...")
        print("─" * 70)

        model = BetaVAE(
            latent_dim=config.latent_dim,
            num_colors=config.num_colors,
            use_focal_loss=config.use_focal_loss,
            focal_gamma=config.focal_gamma,
            class_weights=class_weights
        )

        num_params = sum(p.numel() for p in model.parameters())
        print(f"[MODEL] Created: {num_params:,} parameters")
        if config.use_focal_loss:
            print(f"[MODEL] Using Focal Loss (γ={config.focal_gamma})")
        if class_weights is not None:
            print(f"[MODEL] Using class weights (method={config.class_weight_method})")

        # Create trainer
        trainer = BetaVAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
        )

        # Save config to run directory
        config.save()  # Uses run_manager by default
        print(f"[CONFIG] Saved to: {config.run_manager.run_dir / 'config.json'}")

        # Save run metadata
        config.run_manager.save_metadata({
            'model_params': num_params,
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset) if test_loader is not None else 0,
        })
        print(f"[METADATA] Saved to: {config.run_manager.run_dir / 'metadata.json'}")

        # Train
        best_val_loss = trainer.train()

        # Evaluate on test set if available
        if test_loader is not None:
            print("\n" + "=" * 70)
            print("Evaluating on test set...")
            print("=" * 70)

            model.eval()
            test_metrics = trainer.validate(epoch=config.max_epochs)

            print(f"\nTest Results:")
            print(f"  Loss: {test_metrics['loss']:.4f}")
            print(f"  Pixel Accuracy: {test_metrics['pixel_acc']:.3f}")
            print(f"  KL/dim: {test_metrics['kl_per_dim']:.4f}")

            # Check success criteria
            if test_metrics['pixel_acc'] >= config.target_pixel_accuracy:
                print(f"\n[SUCCESS] Target accuracy achieved!")
                print(f"   {test_metrics['pixel_acc']:.1%} >= {config.target_pixel_accuracy:.1%}")
            else:
                print(f"\n[BELOW TARGET] Did not reach target accuracy")
                print(f"   {test_metrics['pixel_acc']:.1%} < {config.target_pixel_accuracy:.1%}")

            # Check for posterior collapse
            if test_metrics['kl_per_dim'] < config.min_kl_per_dim:
                print(f"\n[WARNING] Posterior collapse detected")
                print(f"   KL/dim = {test_metrics['kl_per_dim']:.4f} < {config.min_kl_per_dim}")
            else:
                print(f"\n[OK] Latent space is active (KL/dim = {test_metrics['kl_per_dim']:.4f})")

        print("\n" + "=" * 70)
        print("Training complete!")
        print(f"Run ID: {config.run_id}")
        print(f"Results saved to: {config.run_manager.run_dir}")
        print("=" * 70)


if __name__ == "__main__":
    main()
