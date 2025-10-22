"""
Training Package for β-VAE

This package provides:
- Training configuration
- Training loop with β-annealing
- W&B logging integration
- Experiment tracking with unique run IDs
- Checkpointing and model saving
- Early stopping
- Training utilities
"""

from .config import TrainingConfig, get_default_config, get_quick_test_config, get_ablation_config
from .trainer import BetaVAETrainer
from .run_manager import RunManager, generate_run_id
from .utils import (
    get_beta_schedule,
    BetaScheduler,
    AverageMeter,
    MetricTracker,
    EarlyStopping,
    compute_pixel_accuracy,
    compute_kl_divergence_per_dim,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    # Config
    'TrainingConfig',
    'get_default_config',
    'get_quick_test_config',
    'get_ablation_config',
    # Trainer
    'BetaVAETrainer',
    # Run Manager
    'RunManager',
    'generate_run_id',
    # Utils
    'get_beta_schedule',
    'BetaScheduler',
    'AverageMeter',
    'MetricTracker',
    'EarlyStopping',
    'compute_pixel_accuracy',
    'compute_kl_divergence_per_dim',
    'save_checkpoint',
    'load_checkpoint',
]
