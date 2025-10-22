"""
Training Configuration - Task 3.2

Configuration class for β-VAE training with all hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import json
from .run_manager import RunManager


@dataclass
class TrainingConfig:
    """
    Configuration for β-VAE training.

    All hyperparameters from the specification are defined here.
    """

    # Model architecture
    latent_dim: int = 10
    num_colors: int = 10

    # Loss function configuration
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    use_class_weights: bool = True
    class_weight_method: str = 'inverse'  # Options: inverse, sqrt_inverse, balanced
    class_weight_smooth: float = 1.0

    # Training hyperparameters
    batch_size: int = 128
    learning_rate: float = 5e-4  # Reduced from 1e-3 for stability with larger batches
    max_epochs: int = 50
    num_workers: int = 4
    gradient_clip_norm: float = 1.0  # Gradient clipping for training stability

    # β-annealing schedule
    beta_schedule: str = "linear_warmup"  # Options: linear_warmup, constant, aggressive

    # Learning rate schedule
    use_lr_schedule: bool = True
    lr_schedule_type: str = "cosine"  # Options: cosine, step, exponential
    lr_min: float = 1e-5  # For cosine annealing
    lr_decay_epochs: int = 10  # For step decay
    lr_decay_rate: float = 0.5  # For step/exponential decay

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0

    # Data augmentation
    use_augmentation: bool = True

    # Checkpointing
    save_every_n_epochs: int = 5
    save_best_only: bool = True

    # Weights & Biases logging
    use_wandb: bool = True
    wandb_project: str = "arc-beta-vae"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: list = field(default_factory=lambda: ["beta-vae", "arc", "experiment-0.2"])

    # Logging
    log_every_n_steps: int = 10
    validate_every_n_epochs: int = 1

    # Hardware
    device: str = "cuda"  # Options: cuda, cpu, mps
    pin_memory: bool = True

    # Reproducibility
    seed: int = 42

    # Paths
    data_path: str = "../datasets/test-100k/corpus.npz"
    runs_base_dir: str = "../experiments/runs"  # Base directory for all runs (project/experiments/runs/)

    # Run tracking (WandB-style)
    run_id: Optional[str] = None  # Auto-generated if None

    # Success criteria
    target_pixel_accuracy: float = 0.90
    min_kl_per_dim: float = 0.05  # To detect posterior collapse

    def __post_init__(self):
        """Validate configuration and create run manager."""
        # Validate values
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.max_epochs > 0, "max_epochs must be positive"
        assert 0 < self.target_pixel_accuracy <= 1.0, "target_pixel_accuracy must be in (0, 1]"

        # Initialize run manager (generates unique run_id if not provided)
        self.run_manager = RunManager(
            base_dir=self.runs_base_dir,
            run_id=self.run_id
        )

        # Update run_id and set checkpoint_dir
        self.run_id = self.run_manager.run_id
        self.checkpoint_dir = self.run_manager.checkpoints_dir

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = {}
        for key, value in self.__dict__.items():
            # Skip run_manager object (not serializable)
            if key == 'run_manager':
                continue
            # Convert Path objects to strings
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict

    def save(self, filepath: Optional[str] = None):
        """
        Save configuration to JSON file.

        Args:
            filepath: Optional custom path. If None, saves to run folder.
        """
        if filepath is None:
            # Save to run manager's directory
            if self.run_manager is not None:
                self.run_manager.save_config(self)
            else:
                raise ValueError("No run_manager available and no filepath provided")
        else:
            # Save to custom path
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = ["Training Configuration:", "=" * 60]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        lines.append("=" * 60)
        return "\n".join(lines)


def get_default_config() -> TrainingConfig:
    """Get default training configuration from spec."""
    return TrainingConfig()


def get_quick_test_config() -> TrainingConfig:
    """Get configuration for quick testing (reduced epochs, optimized batch size)."""
    return TrainingConfig(
        batch_size=128,  # Increased from 32 for more stable gradients
        learning_rate=5e-4,  # Reduced from 1e-3 for stability
        max_epochs=5,
        num_workers=0,
        use_wandb=False,
        save_every_n_epochs=1,
        early_stopping_patience=3,
    )


def get_ablation_config(beta_schedule: str = "constant") -> TrainingConfig:
    """Get configuration for ablation studies."""
    return TrainingConfig(
        beta_schedule=beta_schedule,
        wandb_tags=["beta-vae", "arc", "ablation", f"beta-{beta_schedule}"],
    )
