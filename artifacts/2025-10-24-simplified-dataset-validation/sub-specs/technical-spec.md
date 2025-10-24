# Technical Specification

This is the technical specification for the artifact detailed in research-os/artifacts/2025-10-24-simplified-dataset-validation/spec.md

> Created: 2025-10-24
> Version: 1.0.0

## Technical Requirements

### 1. Simplified Dataset Generation

**Implementation Location**: New script `project/src/generate_simplified_dataset.py`

**Functionality**:
- Generate exactly 100,000 samples using atomic_generator.py infrastructure
- Use only shape generators from shape_generators.py:
  - `RectangleGenerator` (filled rectangles)
  - `LineGenerator` (horizontal, vertical, diagonal lines)
  - `PatternGenerator` with patterns: checkerboard, l_shape, t_shape, plus, zigzag
- Explicitly exclude blob generation (do not use BlobGenerator)
- Maintain 16x16 grid format with 10-color palette (ARC format)
- Follow existing dataset split convention: 80% train, 10% val, 10% test
- Output as .npz file format consistent with current data loading pipeline

**Technical Details**:
- Import and instantiate generators from shape_generators.py
- Use atomic_generator.AtomicImageGenerator to compose shapes on grids
- Set appropriate generator parameters (object count per grid: 1-6)
- Apply standard ARC constraints (grid size, color palette)
- Save as `data/simplified_dataset_100k.npz` with keys: 'train', 'val', 'test'

### 2. Beta Schedule Configuration

**Implementation Location**: Modify existing `training/config.py` or use command-line arguments in `train_vae.py`

**Functionality**:
- Set beta schedule to 'linear_warmup' or 'constant'
- Set `beta_max` to 0.0 or very small value (0.001) to effectively disable KL regularization
- If using linear schedule: set warmup to 0 epochs and maintain beta at ~0 throughout training
- Ensure focal loss and class weighting remain active (these prevent collapse)

**Command-line Usage** (existing support in train_vae.py):
```bash
python train_vae.py --beta-schedule constant --beta-max 0.0 --data-path data/simplified_dataset_100k.npz
```

**Technical Details**:
- Beta schedule is already implemented in training/config.py and beta_schedule.py
- Use existing `--beta-schedule` and `--beta-max` CLI arguments
- No code changes needed for beta control, just configuration
- Focal loss (focal_gamma) and class weighting remain enabled by default

### 3. Training Configuration for Capacity Validation

**Implementation Location**: Command-line arguments or custom config JSON

**Training Parameters**:
- Data path: `data/simplified_dataset_100k.npz`
- Beta schedule: constant or linear with max_beta=0.0
- Latent dimension: Keep default (likely 8 or 10 based on existing config)
- Batch size: Keep default (likely 32 or 64)
- Max epochs: 100-200 epochs (sufficient for convergence without beta)
- Early stopping: Enable with patience=20 (stop if no improvement)
- Focal loss: gamma=2.0 (keep existing value)
- Class weighting: method='inverse_sqrt' (keep existing)
- Augmentation: Enable (standard rotations/flips)
- Weights & Biases: Enable for tracking
- Tags: ["simplified-dataset", "capacity-validation", "beta-disabled"]

**Example Command**:
```bash
pyenv activate zeus && python project/src/train_vae.py \
  --data-path data/simplified_dataset_100k.npz \
  --beta-schedule constant \
  --beta-max 0.0 \
  --no-early-stopping \
  --device cuda
```

### 4. Evaluation Metrics and Success Criteria

**Metrics to Track** (already implemented in trainer.py):
- **Pixel Accuracy**: Primary metric, should exceed 95%
- **Per-class Accuracy**: Verify no collapse to single color (especially black)
- **Reconstruction Loss**: Cross-entropy loss should decrease steadily
- **KL Divergence**: Should be near zero (beta=0 means no KL pressure)
- **Color Distribution**: Histogram of predicted vs ground truth colors

**Success Criteria**:
1. Pixel accuracy > 95% (significantly above 93% collapse threshold)
2. Per-class accuracy reasonable across all 10 colors (no single color dominance)
3. Visual inspection: reconstructed samples show shape structure, not all black
4. Stable training: loss decreases without oscillation or divergence

**Validation Approach**:
- Monitor training via W&B dashboard
- Save sample reconstructions every N epochs (existing functionality in trainer.py)
- Generate final evaluation report with accuracy breakdown
- Compare to baseline: 93% accuracy = collapsed model (all black predictions)

### 5. Code Modifications Required

**New Files**:
1. `project/src/generate_simplified_dataset.py` - Dataset generation script

**Modified Files**:
None required - existing training infrastructure supports all needed configurations via CLI arguments

**Integration Points**:
- Dataset generation uses existing `atomic_generator.py` and `shape_generators.py`
- Training uses existing `train_vae.py` with appropriate CLI flags
- Evaluation uses existing `trainer.py` metrics and visualization

## External Dependencies

No new external dependencies required. All functionality can be implemented using existing codebase:
- PyTorch (already installed)
- NumPy (already installed)
- Existing shape generators (shape_generators.py)
- Existing atomic generator (atomic_generator.py)
- Existing training pipeline (train_vae.py, trainer.py)
- Weights & Biases (already configured)
