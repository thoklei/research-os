"""
Training Utilities - Task 3.1

Utility functions for training including:
- β-annealing schedule
- Learning rate scheduling
- Metric tracking
"""

import torch
import numpy as np
from typing import Dict, List, Optional


def get_beta_schedule(
    epoch: int,
    schedule_type: str = "linear_warmup",
    max_beta: float = 0.5,
    warmup_epochs: int = 10,
    ramp_epochs: int = 50,
    cycle_length: int = 20
) -> float:
    """
    Get β value for current epoch based on annealing schedule.

    Supports multiple schedule types with configurable parameters:
    - linear_warmup: Configurable gradual warmup (default)
    - ultra_conservative: Same as linear_warmup (kept for config clarity)
    - cyclical: Alternates between reconstruction and regularization
    - constant: Fixed beta=1.0 (for ablation studies)
    - aggressive: Faster warmup (for experiments)

    Args:
        epoch (int): Current epoch (1-indexed)
        schedule_type (str): Type of schedule
        max_beta (float): Maximum beta value to reach (default: 0.5)
        warmup_epochs (int): Epochs to stay at beta=0 (default: 10)
        ramp_epochs (int): Epochs to ramp from 0 to max_beta (default: 50)
        cycle_length (int): For cyclical: epochs per cycle (default: 20)

    Returns:
        float: β value for current epoch
    """
    if schedule_type == "linear_warmup" or schedule_type == "ultra_conservative":
        if epoch <= warmup_epochs:
            # Free reconstruction for warmup epochs (β=0)
            # Prevents premature posterior collapse to prior N(0,1)
            return 0.0
        elif epoch <= warmup_epochs + ramp_epochs:
            # Gradual warm-up from 0.0 to max_beta
            # Slow ramp prevents KL penalty from overwhelming reconstruction
            progress = (epoch - warmup_epochs) / ramp_epochs
            return progress * max_beta
        else:
            # Cap at max_beta
            return max_beta

    elif schedule_type == "cyclical":
        # Cyclical annealing: alternates between reconstruction and regularization
        # Gives model repeated opportunities to recover from collapse
        cycle_position = epoch % cycle_length

        if cycle_position < cycle_length / 2:
            # Ascending phase: 0 -> max_beta
            progress = cycle_position / (cycle_length / 2)
            return progress * max_beta
        else:
            # Descending phase: max_beta -> 0
            progress = (cycle_length - cycle_position) / (cycle_length / 2)
            return progress * max_beta

    elif schedule_type == "constant":
        # For ablation studies - fixed beta
        return 1.0

    elif schedule_type == "aggressive":
        # Faster warm-up (for experiments)
        # Still conservative to avoid collapse, just faster than default
        if epoch <= 5:
            return 0.0
        elif epoch <= 30:
            return (epoch - 5) / 50.0  # Reach 0.5 at epoch 30
        else:
            return 0.5

    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


class BetaScheduler:
    """
    β-annealing scheduler that tracks β values over training.

    Args:
        schedule_type (str): Type of schedule (default: "linear_warmup")
        max_beta (float): Maximum beta value (default: 0.5)
        warmup_epochs (int): Epochs at beta=0 (default: 10)
        ramp_epochs (int): Epochs to ramp from 0 to max_beta (default: 50)
        cycle_length (int): For cyclical schedule (default: 20)
    """

    def __init__(
        self,
        schedule_type: str = "linear_warmup",
        max_beta: float = 0.5,
        warmup_epochs: int = 10,
        ramp_epochs: int = 50,
        cycle_length: int = 20
    ):
        self.schedule_type = schedule_type
        self.max_beta = max_beta
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.cycle_length = cycle_length
        self.current_epoch = 0
        self.beta_history: List[float] = []

    def step(self, epoch: int) -> float:
        """
        Get β value for current epoch and record it.

        Args:
            epoch (int): Current epoch (1-indexed)

        Returns:
            float: β value
        """
        self.current_epoch = epoch
        beta = get_beta_schedule(
            epoch,
            self.schedule_type,
            max_beta=self.max_beta,
            warmup_epochs=self.warmup_epochs,
            ramp_epochs=self.ramp_epochs,
            cycle_length=self.cycle_length
        )
        self.beta_history.append(beta)
        return beta

    def get_current_beta(self) -> float:
        """Get β value for current epoch."""
        return get_beta_schedule(
            self.current_epoch,
            self.schedule_type,
            max_beta=self.max_beta,
            warmup_epochs=self.warmup_epochs,
            ramp_epochs=self.ramp_epochs,
            cycle_length=self.cycle_length
        )

    def get_history(self) -> List[float]:
        """Get history of β values."""
        return self.beta_history


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update statistics with new value.

        Args:
            val: New value to add
            n: Number of samples this value represents (default: 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __str__(self):
        return f"{self.name}: {self.avg:.4f} (latest: {self.val:.4f})"


class MetricTracker:
    """
    Tracks multiple metrics during training/validation.

    Args:
        metrics (list): List of metric names to track
    """

    def __init__(self, metrics: List[str]):
        self.metrics = {name: AverageMeter(name) for name in metrics}

    def update(self, metric_dict: Dict[str, float], n: int = 1):
        """
        Update all metrics from dictionary.

        Args:
            metric_dict: Dictionary of {metric_name: value}
            n: Number of samples these values represent
        """
        for name, value in metric_dict.items():
            if name in self.metrics:
                self.metrics[name].update(value, n)

    def reset(self):
        """Reset all metrics."""
        for meter in self.metrics.values():
            meter.reset()

    def get_averages(self) -> Dict[str, float]:
        """Get average values for all metrics."""
        return {name: meter.avg for name, meter in self.metrics.items()}

    def __str__(self):
        return " | ".join([str(meter) for meter in self.metrics.values()])


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.

    Args:
        patience (int): How many epochs to wait after last improvement
        min_delta (float): Minimum change to qualify as improvement
        mode (str): 'min' for loss, 'max' for accuracy
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == "min":
            self.is_better = lambda score, best: score < best - min_delta
        else:
            self.is_better = lambda score, best: score > best + min_delta

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score (loss or accuracy)

        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.is_better(score, self.best_score):
            # Improvement
            self.best_score = score
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def compute_pixel_accuracy(pred_logits: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute pixel-wise accuracy for reconstruction.

    Args:
        pred_logits: Predicted logits with shape (batch, num_colors, H, W)
        target: Ground truth labels with shape (batch, H, W)

    Returns:
        float: Pixel-wise accuracy in [0, 1]
    """
    # Get predicted labels
    pred_labels = torch.argmax(pred_logits, dim=1)  # (batch, H, W)

    # Compute accuracy
    correct = (pred_labels == target).float()
    accuracy = correct.mean().item()

    return accuracy


def compute_per_class_accuracy(pred_logits: torch.Tensor, target: torch.Tensor, num_classes: int = 10) -> List[float]:
    """
    Compute per-class pixel-wise accuracy for reconstruction.

    Args:
        pred_logits: Predicted logits with shape (batch, num_colors, H, W)
        target: Ground truth labels with shape (batch, H, W)
        num_classes: Number of color classes (default: 10)

    Returns:
        List[float]: Per-class accuracy in [0, 1] for each class
    """
    # Get predicted labels
    pred_labels = torch.argmax(pred_logits, dim=1)  # (batch, H, W)

    per_class_acc = []
    for c in range(num_classes):
        # Find all pixels of class c in the target
        mask = (target == c)
        if mask.sum() > 0:
            # Compute accuracy for this class
            correct = (pred_labels[mask] == target[mask]).float()
            acc = correct.mean().item()
        else:
            # No pixels of this class in batch
            acc = 0.0
        per_class_acc.append(acc)

    return per_class_acc


def compute_kl_divergence_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> float:
    """
    Compute KL divergence per latent dimension.

    Args:
        mu: Mean of latent distribution (batch, latent_dim)
        logvar: Log variance of latent distribution (batch, latent_dim)

    Returns:
        float: KL divergence per dimension
    """
    # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Average over batch and dimensions
    kl_per_dim = kl_loss / (mu.size(0) * mu.size(1))

    return kl_per_dim.item()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    filepath: str,
):
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state (optional)
        epoch: Current epoch
        metrics: Dictionary of metrics to save
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Dict:
    """
    Load training checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: LR scheduler to load state into (optional)

    Returns:
        Dictionary with epoch and metrics
    """
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint.get('metrics', {}),
    }
