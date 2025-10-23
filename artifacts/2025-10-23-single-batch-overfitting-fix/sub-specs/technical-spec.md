# Technical Specification

This is the technical specification for the artifact detailed in research-os/artifacts/2025-10-23-single-batch-overfitting-fix/spec.md

## Root Cause Analysis

**Current Implementation** (train_vae.py:227-266):
When `--overfit-batch` flag is used, class weights ARE computed from the single batch (32 samples = 8,192 pixels total). However, this sample is too small to produce statistically meaningful weights.

**Example**: In a single batch:
- Class 0 (black): ~7,373 pixels (90%)
- Other colors: ~819 pixels total, ~91 pixels per color on average
- Some colors may have < 10 pixels

With `sqrt_inverse` method and `smooth=1.0`:
- Weights are computed but lack statistical significance
- The model still learns to predict all black despite the weights

**Successful Approach** (normal training with 80K samples):
- Class 0: 19,081,814 pixels → weight: 0.099
- Other classes: ~154,000-157,000 pixels each → weights: ~1.09-1.10
- Weight ratio: 11.13x (effective at preventing mode collapse)

## Technical Requirements

### 1. Modify Overfit Mode to Use Global Class Weights

**Objective**: When `--overfit-batch` is used, compute class weights from the full training dataset instead of the single batch

**Current Code** (train_vae.py:227-266):
```python
if args.overfit_batch and config.use_class_weights:
    # Currently computes from single batch
    if isinstance(train_loader.dataset, SingleBatchDataset):
        unique_batch = train_loader.dataset.batch
        # ... creates TempDataset wrapper ...
        class_weights = compute_class_weights(
            dataset=temp_dataset,  # ← Single batch only!
            num_classes=config.num_colors,
            method=config.class_weight_method,
            smooth=config.class_weight_smooth,
            normalize=True
        )
```

**Required Change**:
```python
if args.overfit_batch and config.use_class_weights:
    print("\n" + "─" * 70)
    print("[OVERFIT MODE] Computing class weights from FULL dataset...")
    print("─" * 70)

    from models import compute_class_weights
    from data import ARC_Dataset  # Import the full dataset class
    import numpy as np

    # Load full dataset to compute global weights
    data = np.load(config.data_path)
    full_train_images = data['train']

    # Create temporary dataset for weight computation
    class TempDataset:
        def __init__(self, images):
            self.images = images
        def __len__(self):
            return len(self.images)
        def __getitem__(self, idx):
            return self.images[idx]

    full_dataset = TempDataset(full_train_images)

    # Compute weights from FULL dataset
    class_weights = compute_class_weights(
        dataset=full_dataset,  # ← Full dataset!
        num_classes=config.num_colors,
        method=config.class_weight_method,
        smooth=config.class_weight_smooth,
        normalize=True
    )

    # Also log single batch statistics for comparison
    print("\n[OVERFIT MODE] Single batch statistics:")
    if isinstance(train_loader.dataset, SingleBatchDataset):
        unique_batch = train_loader.dataset.batch
        batch_counts = torch.zeros(config.num_colors, dtype=torch.long)
        for c in range(config.num_colors):
            batch_counts[c] = (unique_batch == c).sum().item()
        print(f"  Batch class counts: {batch_counts.tolist()}")
        print(f"  Batch total pixels: {batch_counts.sum().item()}")
```

**Code Location**: `project/src/train_vae.py:227-266`

### 2. Add Class Weight Caching (Optional Enhancement)

**Objective**: Cache computed class weights to avoid recomputation on subsequent runs

**Implementation**:
- Add `load_class_weights()` function in `models/losses.py`
- Check for cached weights file before computing
- Allow `--use-cached-weights` flag to skip computation

**Code Location**: `project/src/models/losses.py` (new function after line 202)

```python
def load_class_weights(filepath: str) -> torch.Tensor:
    """Load class weights from file."""
    data = torch.load(filepath)
    return data['class_weights']
```

**Update to train_vae.py**:
```python
# Check for cached weights first
weights_cache_path = Path(config.data_path).parent / "class_weights.pth"
if weights_cache_path.exists():
    print(f"[WEIGHTS] Loading cached weights from: {weights_cache_path}")
    from models import load_class_weights
    class_weights = load_class_weights(str(weights_cache_path))
else:
    # Compute from full dataset as shown above
    # ... computation code ...
    # Save for future use
    torch.save({'class_weights': class_weights}, weights_cache_path)
    print(f"[WEIGHTS] Cached for future use: {weights_cache_path}")
```

### 3. Enhanced Logging for Debugging

**Objective**: Add detailed logging to compare global vs batch statistics

**Implementation**:

Add after class weight computation:
```python
print("\n" + "─" * 70)
print("Class Weight Analysis")
print("─" * 70)
print(f"  Method: {config.class_weight_method}")
print(f"  Source: Full dataset ({len(full_dataset)} samples)")
print(f"  Weight range: {class_weights.min().item():.4f} - {class_weights.max().item():.4f}")
print(f"  Weight ratio (max/min): {(class_weights.max() / class_weights.min()).item():.2f}x")
print(f"\nClass weights:")
for i, w in enumerate(class_weights):
    print(f"  Color {i}: {w.item():.4f}")
```

**Code Location**: `project/src/train_vae.py` (after line 266)

### 4. Per-Class Accuracy Tracking

**Objective**: Track reconstruction accuracy per color class during training

**Implementation**:

Add to trainer.py validate() function:
```python
def compute_per_class_accuracy(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 10):
    """Compute accuracy for each class."""
    per_class_acc = []
    for c in range(num_classes):
        mask = (targets == c)
        if mask.sum() > 0:
            acc = (preds[mask] == targets[mask]).float().mean().item()
        else:
            acc = 0.0
        per_class_acc.append(acc)
    return per_class_acc
```

Add logging in training loop:
```python
# After computing predictions
per_class_acc = compute_per_class_accuracy(preds, targets, config.num_colors)
if batch_idx % config.log_every_n_steps == 0:
    print(f"  Per-class accuracy: {[f'{a:.2f}' for a in per_class_acc]}")
```

**Code Location**: `project/src/training/trainer.py` (add new function and update train_epoch)

### 5. Validation Metrics

**Objective**: Add early stopping criterion for overfitting mode

**Implementation**:

Add to trainer config:
```python
# In TrainingConfig
overfit_target_accuracy: float = 0.95  # Target for overfitting mode
```

Add early stopping check in training loop:
```python
if args.overfit_batch and pixel_acc >= config.overfit_target_accuracy:
    print(f"\n✓ Overfitting target achieved! ({pixel_acc:.1%} >= {config.overfit_target_accuracy:.1%})")
    print("Stopping early as model has successfully memorized the batch.")
    break
```

**Code Location**: `project/src/training/trainer.py:train_epoch()`

## Command-Line Interface

**Usage**:
```bash
# Overfit mode with global class weights
python project/src/train_vae.py --overfit-batch

# Quick test with overfitting
python project/src/train_vae.py --overfit-batch --quick-test

# Disable class weights (for comparison)
# Note: Requires adding a --no-class-weights flag
python project/src/train_vae.py --overfit-batch --no-class-weights
```

## Expected Output

```
[W&B] Initialized with run ID: abc123

──────────────────────────────────────────────────────────────────────
Loading data...
──────────────────────────────────────────────────────────────────────
[DATA] Train: 1600 samples (32 unique × 50 repeats), 50 batches
[DATA] Val:   1600 samples (32 unique × 50 repeats), 50 batches

──────────────────────────────────────────────────────────────────────
[OVERFIT MODE] Computing class weights from FULL dataset...
──────────────────────────────────────────────────────────────────────
Loading full dataset for class weight computation...
Computing class weights from 80000 samples...
  Processed 10000/80000 samples...
  ...
  Processed 80000/80000 samples...
  Class counts: [19081814, 154289, 154457, 153994, 155609, 155472, 155012, 155371, 157799, 156183]
  Class weights: [0.099, 1.104, 1.103, 1.105, 1.099, 1.100, 1.101, 1.100, 1.092, 1.097]
  Weight ratio (max/min): 11.13x

[OVERFIT MODE] Single batch statistics:
  Batch class counts: [7373, 91, 87, 93, 89, 95, 82, 97, 103, 82]
  Batch total pixels: 8192
  Note: Using global weights (not batch weights) for better statistical significance

──────────────────────────────────────────────────────────────────────
Class Weight Analysis
──────────────────────────────────────────────────────────────────────
  Method: sqrt_inverse
  Source: Full dataset (80000 samples)
  Weight range: 0.0993 - 1.1049
  Weight ratio (max/min): 11.13x

Class weights:
  Color 0: 0.0993
  Color 1: 1.1039
  Color 2: 1.1033
  ...
  Color 9: 1.0971

[WEIGHTS] Saved to: ../experiments/runs/xyz789/class_weights.pth

──────────────────────────────────────────────────────────────────────
Creating model...
──────────────────────────────────────────────────────────────────────
[MODEL] Created: 801,034 parameters
[MODEL] Using Focal Loss (γ=2.0)
[MODEL] Using class weights (method=sqrt_inverse, source=full_dataset)

Epoch 1/50 [Train]:  20%|██ | 10/50 [00:05<00:20, 2.00it/s, loss=0.245, acc=0.923, β=0.00]
  Per-class accuracy: ['0.92', '0.15', '0.23', '0.31', '0.18', '0.28', '0.12', '0.34', '0.41', '0.19']

Epoch 1/50 [Train]:  40%|████ | 20/50 [00:10<00:15, 2.00it/s, loss=0.123, acc=0.957, β=0.00]
  Per-class accuracy: ['0.96', '0.87', '0.91', '0.94', '0.89', '0.93', '0.84', '0.96', '0.97', '0.88']

Epoch 2/50 [Train]:  20%|██ | 10/50 [00:05<00:20, 2.00it/s, loss=0.045, acc=0.984, β=0.01]
  Per-class accuracy: ['0.98', '0.96', '0.97', '0.99', '0.96', '0.98', '0.95', '0.99', '0.99', '0.97']

✓ Overfitting target achieved! (98.4% >= 95.0%)
Stopping early as model has successfully memorized the batch.
```

## Integration Requirements

- Must not break normal training mode (without --overfit-batch)
- Must maintain compatibility with existing focal loss implementation
- Should work with existing BetaVAE architecture
- Must not require changes to data loaders (only how they're used for weight computation)

## Performance Criteria

- Single batch overfitting should achieve >95% pixel accuracy within 10 epochs
- Class weight computation from full dataset should complete in < 30 seconds
- Memory overhead: negligible (class weights are only 10 floats = 40 bytes)

## Error Handling

- Gracefully handle missing dataset file
- Validate class weights tensor shape (must be [num_classes])
- Check for NaN or infinite values in computed weights
- Warn if --overfit-batch is used without class weights enabled
