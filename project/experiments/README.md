# Experiments

This directory contains all training run outputs managed by the RunManager system.

## Directory Structure

```
experiments/
└── runs/
    ├── <run-id-1>/
    │   ├── config.json          # Training configuration
    │   ├── metadata.json        # Run metadata (timestamp, samples, etc.)
    │   ├── checkpoints/         # Model checkpoints
    │   │   ├── best_model.pth
    │   │   └── checkpoint_epoch_*.pth
    │   ├── logs/                # Training logs
    │   └── results/             # Evaluation results
    ├── <run-id-2>/
    └── latest -> <run-id-N>     # Symlink to most recent run
```

## Run Management

Each training run gets a unique 8-character run ID (e.g., `4c8d6f1b`) and a dedicated folder.

The `latest` symlink always points to the most recent training run for easy access.

## Accessing Run Data

```python
from training import RunManager

# Load latest run
rm = RunManager.load_run("latest")
print(f"Run directory: {rm.run_dir}")

# List all runs
runs = RunManager.list_runs()
print(f"All runs: {runs}")
```

## Configuration

Run storage location is configured in `src/training/config.py`:
```python
runs_base_dir: str = "../experiments/runs"  # Relative to src/
```
