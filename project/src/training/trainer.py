"""
Training Loop for β-VAE - Task 3.3

Main trainer class that handles:
- Training and validation loops
- β-annealing
- Checkpointing
- Logging (console and W&B)
- Early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Tuple
import time
from tqdm import tqdm

from .config import TrainingConfig
from .utils import (
    BetaScheduler,
    MetricTracker,
    EarlyStopping,
    compute_pixel_accuracy,
    compute_kl_divergence_per_dim,
    save_checkpoint,
    load_checkpoint,
)


class BetaVAETrainer:
    """
    Trainer for β-VAE model.

    Args:
        model: BetaVAE model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        test_loader: Optional test data loader
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        test_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config

        # Device setup
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )

        # Learning rate scheduler
        self.lr_scheduler = None
        if config.use_lr_schedule:
            if config.lr_schedule_type == "cosine":
                self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.max_epochs,
                    eta_min=config.lr_min
                )
            elif config.lr_schedule_type == "step":
                self.lr_scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=config.lr_decay_epochs,
                    gamma=config.lr_decay_rate
                )
            elif config.lr_schedule_type == "exponential":
                self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    gamma=config.lr_decay_rate
                )

        # β-annealing scheduler
        self.beta_scheduler = BetaScheduler(schedule_type=config.beta_schedule)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode="min"  # Monitor validation loss
        )

        # Metric tracking
        self.train_metrics = MetricTracker(["loss", "recon_loss", "kl_loss", "pixel_acc", "kl_per_dim"])
        self.val_metrics = MetricTracker(["loss", "recon_loss", "kl_loss", "pixel_acc", "kl_per_dim"])

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0

        # W&B logging
        self.use_wandb = config.use_wandb
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb_run = None
            except ImportError:
                print("⚠️  wandb not installed, disabling W&B logging")
                self.use_wandb = False

    def init_wandb(self):
        """Initialize Weights & Biases logging."""
        if not self.use_wandb:
            return

        self.wandb_run = self.wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.config.wandb_run_name,
            tags=self.config.wandb_tags,
            config=self.config.to_dict(),
        )

        # Watch model
        self.wandb.watch(self.model, log="all", log_freq=100)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number (1-indexed)

        Returns:
            Dictionary of average metrics for the epoch
        """
        self.model.train()
        self.train_metrics.reset()

        # Get β for this epoch
        beta = self.beta_scheduler.step(epoch)

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.max_epochs} [Train]",
            leave=False
        )

        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            batch_size = batch.size(0)

            # Forward pass
            recon_logits, mu, logvar = self.model(batch)

            # Compute loss
            loss_dict = self.model.loss_function(recon_logits, batch, mu, logvar, beta=beta)

            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()

            # Gradient clipping for training stability
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.gradient_clip_norm
                )

            self.optimizer.step()

            # Compute additional metrics
            pixel_acc = compute_pixel_accuracy(recon_logits, batch)
            kl_per_dim = compute_kl_divergence_per_dim(mu, logvar)

            # Update metrics
            metrics = {
                'loss': loss_dict['loss'].item(),
                'recon_loss': loss_dict['recon_loss'].item(),
                'kl_loss': loss_dict['kl_loss'].item(),
                'pixel_acc': pixel_acc,
                'kl_per_dim': kl_per_dim,
            }
            self.train_metrics.update(metrics, n=batch_size)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['pixel_acc']:.3f}",
                'β': f"{beta:.2f}",
            })

            # Log to W&B
            if self.use_wandb and self.global_step % self.config.log_every_n_steps == 0:
                log_dict = {
                    'train/loss': metrics['loss'],
                    'train/recon_loss': metrics['recon_loss'],
                    'train/kl_loss': metrics['kl_loss'],
                    'train/pixel_acc': metrics['pixel_acc'],
                    'train/kl_per_dim': metrics['kl_per_dim'],
                    'train/beta': beta,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                }
                self.wandb.log(log_dict, step=self.global_step)

            self.global_step += 1

        # Return epoch averages
        return self.train_metrics.get_averages()

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate on validation set.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of average metrics
        """
        self.model.eval()
        self.val_metrics.reset()

        # Get β for this epoch (for consistent logging)
        beta = self.beta_scheduler.get_current_beta()

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch}/{self.config.max_epochs} [Val]",
            leave=False
        )

        for batch in pbar:
            batch = batch.to(self.device)
            batch_size = batch.size(0)

            # Forward pass
            recon_logits, mu, logvar = self.model(batch)

            # Compute loss
            loss_dict = self.model.loss_function(recon_logits, batch, mu, logvar, beta=beta)

            # Compute additional metrics
            pixel_acc = compute_pixel_accuracy(recon_logits, batch)
            kl_per_dim = compute_kl_divergence_per_dim(mu, logvar)

            # Update metrics
            metrics = {
                'loss': loss_dict['loss'].item(),
                'recon_loss': loss_dict['recon_loss'].item(),
                'kl_loss': loss_dict['kl_loss'].item(),
                'pixel_acc': pixel_acc,
                'kl_per_dim': kl_per_dim,
            }
            self.val_metrics.update(metrics, n=batch_size)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['pixel_acc']:.3f}",
            })

        # Get epoch averages
        val_metrics = self.val_metrics.get_averages()

        # Log to W&B
        if self.use_wandb:
            log_dict = {
                'val/loss': val_metrics['loss'],
                'val/recon_loss': val_metrics['recon_loss'],
                'val/kl_loss': val_metrics['kl_loss'],
                'val/pixel_acc': val_metrics['pixel_acc'],
                'val/kl_per_dim': val_metrics['kl_per_dim'],
                'epoch': epoch,
            }
            self.wandb.log(log_dict, step=self.global_step)

        return val_metrics

    def save_model(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        # Save regular checkpoint
        if epoch % self.config.save_every_n_epochs == 0 or is_best:
            filename = f"checkpoint_epoch_{epoch}.pth" if not is_best else "best_model.pth"
            filepath = self.config.checkpoint_dir / filename

            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
                epoch=epoch,
                metrics=metrics,
                filepath=str(filepath),
            )

            print(f"  [CHECKPOINT] Saved: {filepath.name}")

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 70)
        print("Starting β-VAE Training")
        print("=" * 70)
        print(self.config)

        # Initialize W&B
        if self.use_wandb:
            self.init_wandb()
            print("\n[W&B] Weights & Biases logging initialized")

        print(f"\n[SETUP] Device: {self.device}")
        print(f"[SETUP] Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"[SETUP] Training samples: {len(self.train_loader.dataset):,}")
        print(f"[SETUP] Validation samples: {len(self.val_loader.dataset):,}")
        print(f"[SETUP] Batches per epoch: {len(self.train_loader)}")

        start_time = time.time()

        try:
            for epoch in range(1, self.config.max_epochs + 1):
                self.current_epoch = epoch
                epoch_start = time.time()

                # Train
                train_metrics = self.train_epoch(epoch)

                # Validate
                if epoch % self.config.validate_every_n_epochs == 0:
                    val_metrics = self.validate(epoch)
                else:
                    val_metrics = None

                # Learning rate step
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # Print epoch summary
                epoch_time = time.time() - epoch_start
                print(f"\nEpoch {epoch}/{self.config.max_epochs} ({epoch_time:.1f}s)")
                print(f"  Train - Loss: {train_metrics['loss']:.4f} | "
                      f"Acc: {train_metrics['pixel_acc']:.3f} | "
                      f"KL/dim: {train_metrics['kl_per_dim']:.4f}")

                if val_metrics is not None:
                    print(f"  Val   - Loss: {val_metrics['loss']:.4f} | "
                          f"Acc: {val_metrics['pixel_acc']:.3f} | "
                          f"KL/dim: {val_metrics['kl_per_dim']:.4f}")

                    # Check for best model
                    is_best = val_metrics['loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics['loss']
                        print(f"  [BEST] New best validation loss: {self.best_val_loss:.4f}")

                    # Save checkpoint
                    self.save_model(epoch, val_metrics, is_best=is_best)

                    # Early stopping check
                    if self.early_stopping(val_metrics['loss']):
                        print(f"\n[EARLY STOP] Triggered after {epoch} epochs")
                        print(f"  No improvement for {self.config.early_stopping_patience} epochs")
                        break

                    # Check success criteria
                    if val_metrics['pixel_acc'] >= self.config.target_pixel_accuracy:
                        print(f"\n[SUCCESS] Target accuracy reached: {val_metrics['pixel_acc']:.3f} >= {self.config.target_pixel_accuracy}")

                    # Check for posterior collapse
                    if val_metrics['kl_per_dim'] < self.config.min_kl_per_dim:
                        print(f"\n[WARNING] Possible posterior collapse (KL/dim = {val_metrics['kl_per_dim']:.4f})")

        except KeyboardInterrupt:
            print("\n\n[INTERRUPT] Training interrupted by user")

        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"Training Complete")
        print(f"  Total time: {total_time / 3600:.2f} hours")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print("=" * 70)

        # Finish W&B
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.finish()

        return self.best_val_loss
