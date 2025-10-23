# Technical Specification

This is the technical specification for the artifact detailed in research-os/artifacts/2025-10-23-checkpoint-viz-loss-testing/spec.md

> Created: 2025-10-23
> Version: 1.0.0

## Technical Requirements

### 1. Checkpoint Visualization Integration

**File to Modify:** `src/training/trainer.py` (lines 292-308, `save_model()` method)

**Implementation:**
- After calling `save_checkpoint()`, generate visualization using existing `plot_reconstructions()` function
- Sample 10 random batches from `self.train_loader` (not val_loader to see what model is learning from)
- Pass model, data, device to visualization function
- Save to: `{checkpoint_dir}/reconstructions_epoch_{epoch}.png` or `reconstructions_best.png`
- Handle model.eval() mode and torch.no_grad() context
- Log visualization save to console: `[VISUALIZATION] Saved: {filename}`

**Code Location:**
```python
# In trainer.py, save_model() method, after save_checkpoint() call
# Around line 305-308, add visualization generation
```

### 2. Visualization Function Enhancement

**File to Modify:** `src/evaluation/visualization.py` (existing `plot_reconstructions()` function)

**Current Signature:**
```python
def plot_reconstructions(
    model,
    dataloader,
    device,
    num_samples=6,
    save_path=None,
    show=True
)
```

**Required Changes:**
- Function already supports `save_path` parameter ✓
- Function already generates side-by-side grids ✓
- Function already computes per-sample metrics ✓
- **No changes needed** - existing implementation is sufficient

**Usage Pattern:**
```python
from evaluation.visualization import plot_reconstructions

# In trainer.py after checkpoint save
plot_reconstructions(
    model=self.model,
    dataloader=self.train_loader,
    device=self.device,
    num_samples=10,
    save_path=checkpoint_dir / f"reconstructions_epoch_{epoch}.png",
    show=False
)
```

### 3. Square Root Class Weight Smoothing

**File to Modify:** `src/models/beta_vae.py` (class weight calculation)

**Current Implementation:**
- Class weights computed in `__init__()` method
- Uses `class_weight_method` from config: 'inverse', 'sqrt_inverse', 'balanced'
- Currently only 'inverse' is implemented

**Required Implementation:**
```python
# In beta_vae.py, __init__ method, class weight calculation section
if class_weight_method == 'inverse':
    # Existing: weights = total / (num_classes * counts)
    # Results in 123:1 ratio (extreme)
    class_weights = total_samples / (num_classes * class_counts)
elif class_weight_method == 'sqrt_inverse':
    # NEW: Square root smoothing
    # Formula: sqrt(total / (num_classes * counts))
    # Results in ~11:1 ratio (smoothed from 123:1)
    class_weights = torch.sqrt(total_samples / (num_classes * class_counts))
elif class_weight_method == 'balanced':
    # Existing implementation
    pass
```

**Mathematical Rationale:**
- Original weights: `w_obj = 123 × w_bg` (assuming 93% background, 7% objects)
- Sqrt smoothed: `w_obj = √123 × w_bg ≈ 11 × w_bg`
- Reduces extreme ratio while maintaining object emphasis

### 4. Loss Configuration Testing Framework

**Experiment Configurations:**

**Option A** (Baseline - Already Tested):
```python
# config.py
use_focal_loss = True
focal_gamma = 2.0
use_class_weights = False
# Result: 93% accuracy = mode collapse
```

**Option B** (Proposed Solution):
```python
# config.py
use_focal_loss = True
focal_gamma = 2.0
use_class_weights = True
class_weight_method = 'sqrt_inverse'  # NEW
# Expected: 70-80% accuracy, stable training
```

**Option C** (Fallback):
```python
# config.py
use_focal_loss = True
focal_gamma = 1.0  # REDUCED from 2.0
use_class_weights = True
class_weight_method = 'inverse'  # Original 123:1
# Expected: Less aggressive focal loss allows weights to work
```

**Test Execution:**
- Run each option with `--quick-test` flag (5 epochs)
- Save results to separate log files: `option_b_test.log`, `option_c_test.log`
- Use `--no-wandb` flag for local testing
- Generate checkpoint visualizations for each option

### 5. Result Comparison

**Metrics to Compare:**
- Final training accuracy (should be 70-80%, NOT 93%)
- Training curve stability (no wild fluctuations)
- Reconstruction quality from checkpoint visualizations
- KL divergence per dimension (should be > 0.05 to avoid posterior collapse)

**Success Criteria:**
- Accuracy converges to 70-85% range (not 93%)
- Checkpoint visualizations show recognizable object reconstructions (not all-black grids)
- Loss decreases monotonically without pathological behavior
- Training stable across all 5 epochs

## File Paths Summary

**Files to Modify:**
1. `src/training/trainer.py` - Add visualization call after checkpoint save
2. `src/models/beta_vae.py` - Add sqrt_inverse class weight method
3. `src/training/config.py` - Update configs for Options B and C

**Files to Use (No Changes):**
- `src/evaluation/visualization.py` - Already has required functionality

**Output Paths:**
- Checkpoints: `../experiments/runs/{run_id}/checkpoints/`
- Visualizations: `../experiments/runs/{run_id}/checkpoints/reconstructions_epoch_{N}.png`
- Logs: `../option_b_test.log`, `../option_c_test.log`
