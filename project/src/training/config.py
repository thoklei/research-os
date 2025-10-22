"""
Training Configuration - Task 3.2

Configuration class for β-VAE training with all hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import json


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
    checkpoint_dir: str = "checkpoints"

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
    output_dir: str = "experiments/0.2-beta-vae"

    # Success criteria
    target_pixel_accuracy: float = 0.90
    min_kl_per_dim: float = 0.05  # To detect posterior collapse

    def __post_init__(self):
        """Validate configuration and create directories."""
        # Create output directories
        self.output_dir = Path(self.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Validate values
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.max_epochs > 0, "max_epochs must be positive"
        assert 0 < self.target_pixel_accuracy <= 1.0, "target_pixel_accuracy must be in (0, 1]"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict

    def save(self, filepath: str):
        """Save configuration to JSON file."""
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
