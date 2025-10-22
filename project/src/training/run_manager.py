"""
Run Manager - Experiment Tracking System

WandB-style experiment tracking with unique run IDs and organized folder structure.
Each training run gets a unique hash ID and dedicated folder for config/results.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def generate_run_id(length: int = 8) -> str:
    """
    Generate a unique run ID hash (similar to WandB).

    Uses timestamp and random component to ensure uniqueness.

    Args:
        length: Length of the hash (default: 8)

    Returns:
        Unique hash string (e.g., "a3f2b9c1")
    """
    # Combine timestamp with nanosecond precision for uniqueness
    timestamp = str(time.time_ns())
    hash_input = timestamp.encode('utf-8')

    # Generate hash
    hash_obj = hashlib.sha256(hash_input)
    run_id = hash_obj.hexdigest()[:length]

    return run_id


class RunManager:
    """
    Manages experiment runs with unique IDs and folder structure.

    Directory structure:
        experiments/
        └── runs/
            ├── a3f2b9c1/
            │   ├── config.json
            │   ├── metadata.json
            │   ├── checkpoints/
            │   │   ├── best_model.pth
            │   │   └── checkpoint_epoch_*.pth
            │   ├── logs/
            │   │   └── training.log
            │   └── results/
            │       └── metrics.json
            └── latest -> a3f2b9c1 (symlink)
    """

    def __init__(self, base_dir: str = "experiments/runs", run_id: Optional[str] = None):
        """
        Initialize run manager.

        Args:
            base_dir: Base directory for all runs
            run_id: Use existing run ID, or generate new one if None
        """
        self.base_dir = Path(base_dir)
        self.run_id = run_id or generate_run_id()
        self.run_dir = self.base_dir / self.run_id

        # Create directory structure
        self._setup_directories()

    def _setup_directories(self):
        """Create run directory structure."""
        # Main run directory
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.logs_dir = self.run_dir / "logs"
        self.results_dir = self.run_dir / "results"

        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        # Update 'latest' symlink
        self._update_latest_symlink()

    def _update_latest_symlink(self):
        """Update 'latest' symlink to point to this run."""
        latest_link = self.base_dir / "latest"

        # Remove existing symlink if it exists
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()

        # Create new symlink (relative path for portability)
        latest_link.symlink_to(self.run_id)

    def save_config(self, config: Any):
        """
        Save training configuration to run folder.

        Args:
            config: TrainingConfig object or dict
        """
        config_path = self.run_dir / "config.json"

        # Convert to dict if needed
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise ValueError("Config must be a dict or have a to_dict() method")

        # Save config
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def save_metadata(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Save run metadata (timestamp, user info, git hash, etc.).

        Args:
            metadata: Additional metadata to save (optional)
        """
        metadata_path = self.run_dir / "metadata.json"

        run_metadata = {
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(),
            "run_dir": str(self.run_dir),
        }

        # Add user-provided metadata
        if metadata:
            run_metadata.update(metadata)

        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(run_metadata, f, indent=2)

    def get_checkpoint_path(self, name: str = "best_model.pth") -> Path:
        """
        Get path to checkpoint file.

        Args:
            name: Checkpoint filename

        Returns:
            Path to checkpoint
        """
        return self.checkpoints_dir / name

    def get_log_path(self, name: str = "training.log") -> Path:
        """
        Get path to log file.

        Args:
            name: Log filename

        Returns:
            Path to log file
        """
        return self.logs_dir / name

    def get_results_path(self, name: str = "metrics.json") -> Path:
        """
        Get path to results file.

        Args:
            name: Results filename

        Returns:
            Path to results file
        """
        return self.results_dir / name

    def save_results(self, results: Dict[str, Any], filename: str = "metrics.json"):
        """
        Save training results to run folder.

        Args:
            results: Dictionary of results/metrics
            filename: Results filename
        """
        results_path = self.get_results_path(filename)

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    @classmethod
    def load_run(cls, run_id: str, base_dir: str = "experiments/runs") -> 'RunManager':
        """
        Load existing run by ID.

        Args:
            run_id: Run ID or "latest"
            base_dir: Base directory for runs

        Returns:
            RunManager for the specified run
        """
        base_path = Path(base_dir)

        # Handle "latest" symlink
        if run_id == "latest":
            latest_link = base_path / "latest"
            if not latest_link.exists():
                raise ValueError("No 'latest' run found")
            run_id = latest_link.readlink().name

        # Verify run exists
        run_dir = base_path / run_id
        if not run_dir.exists():
            raise ValueError(f"Run {run_id} not found in {base_dir}")

        return cls(base_dir=base_dir, run_id=run_id)

    @classmethod
    def list_runs(cls, base_dir: str = "experiments/runs") -> list[str]:
        """
        List all run IDs in chronological order.

        Args:
            base_dir: Base directory for runs

        Returns:
            List of run IDs
        """
        base_path = Path(base_dir)

        if not base_path.exists():
            return []

        # Get all run directories (exclude 'latest' symlink)
        runs = []
        for item in base_path.iterdir():
            if item.is_dir() and item.name != "latest":
                runs.append(item.name)

        # Sort by creation time
        runs.sort(key=lambda r: (base_path / r).stat().st_ctime)

        return runs

    def __str__(self) -> str:
        """String representation."""
        return f"RunManager(run_id='{self.run_id}', run_dir='{self.run_dir}')"

    def __repr__(self) -> str:
        """String representation."""
        return self.__str__()
