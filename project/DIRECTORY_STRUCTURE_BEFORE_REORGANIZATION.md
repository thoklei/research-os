# Directory Structure Before Reorganization

**Date:** 2025-10-22
**Branch:** refactor/project-structure-reorganization
**Purpose:** Documentation of src/ directory structure before cleanup

---

## Core Module Structure (KEEP)

### models/
- `__init__.py` - Package init
- `beta_vae.py` - Main β-VAE model
- `decoder.py` - Decoder architecture
- `encoder.py` - Encoder architecture
- `losses.py` - Loss functions (focal loss, etc.)

### data/
- `__init__.py` - Package init
- `arc_dataset.py` - PyTorch Dataset for ARC grids
- `data_loaders.py` - DataLoader utilities
- `transforms.py` - Data augmentation

### training/
- `__init__.py` - Package init
- `config.py` - Training configuration
- `run_manager.py` - Experiment run management (NEW)
- `trainer.py` - BetaVAETrainer class
- `utils.py` - Training utilities

### evaluation/
- `__init__.py` - Package init
- `metrics.py` - Evaluation metrics
- `visualization.py` - Reconstruction visualizations

---

## Main Scripts (KEEP)

- `train_vae.py` - Main training script
- `evaluate_vae.py` - VAE evaluation script
- `generate_dataset.py` - Dataset generation CLI
- `validate_visual.py` - Visual validation script
- `visualization.py` - Core ARC grid visualization utilities

---

## Generator Code (KEEP)

- `atomic_generator.py` - Atomic image generator
- `blob_generator.py` - Blob shape generation
- `shape_generators.py` - Rectangle, line, pattern generators
- `pipeline.py` - Generation pipeline

---

## Test Files (MOVE to tests/)

### Model Tests → tests/models/
- `test_encoder.py`
- `test_decoder.py`
- `test_beta_vae.py`
- `test_focal_loss.py`
- `test_loss_computation.py`

### Data Tests → tests/data/
- `test_data_loading.py`

### Evaluation Tests → tests/evaluation/
- None currently

### Integration Tests → tests/integration/
- `test_integration.py`
- `test_large_scale_pipeline.py`

### Generator Tests → tests/ (root)
- `test_atomic_generator.py`
- `test_blob_generator.py`
- `test_shape_generators.py`
- `test_pipeline.py`

### CLI Tests → tests/
- `test_cli.py`

### Visualization Tests → tests/
- `test_large_scale_visualization.py`
- `test_visual_validation.py`
- `test_visualization.py`

**Total Test Files:** 17

---

## Debug Scripts (MOVE to legacy/scripts/)

- `debug_logits.py` - Debug logit outputs
- `debug_reconstructions.py` - Debug reconstructions
- `diagnose_latent_collapse.py` - Diagnose latent collapse
- `generate_reconstructions_debug.py` - Generate debug reconstructions
- `regenerate_reconstructions.py` - Regenerate visualizations (one-off)

---

## One-Off Utilities (MOVE to legacy/scripts/)

- `simple_gen.py` - Simple generator script
- `compute_class_weights.py` - Standalone class weight computation
- `show_images.py` - Image display utility
- `demo.py` - Demo script
- `verify_data_pipeline.py` - Pipeline verification

---

## Issue Documentation (MOVE to project/issues/)

- `ISSUE_MODE_COLLAPSE.md` - Mode collapse documentation

---

## Experiment Outputs (MOVE to legacy/experiments/)

### experiments/0.2-beta-vae/
- `config.json` - Training config
- `class_weights.pth` - Class weights
- `checkpoints/` - Model checkpoints (4 files)
  - `best_model.pth`
  - `checkpoint_epoch_3.pth`
  - `checkpoint_epoch_4.pth`
  - `checkpoint_epoch_5.pth`
- `evaluation/` - Evaluation outputs
  - `metrics.json`

---

## Generated Data (KEEP - but may need reorganization)

- `atomic_corpus.npz` - Generated atomic dataset

---

## Pytest Cache (IGNORE - .gitignore)

- `.pytest_cache/` - Pytest cache directory

---

## Summary Statistics

**Total Python Files:** 60+
**Core Modules:** 4 directories (models/, data/, training/, evaluation/)
**Main Scripts:** 5
**Test Files:** 17 (to be moved)
**Debug Scripts:** 5 (to be moved to legacy)
**One-Off Scripts:** 5 (to be moved to legacy)
**Experiment Folders:** 1 (to be moved to legacy)

---

## Migration Plan

### CREATE New Directories:
1. `project/legacy/scripts/` - Debug and one-off scripts
2. `project/legacy/experiments/` - Old experiment outputs
3. `project/legacy/configs/` - Legacy config files
4. `project/issues/` - Issue documentation
5. `project/tests/` - Test suite with mirrored structure
6. `project/experiments/runs/` - New run manager output

### MOVE Files:
- 17 test files → `tests/` with proper subdirectories
- 5 debug scripts → `legacy/scripts/`
- 5 one-off scripts → `legacy/scripts/`
- 1 issue doc → `issues/`
- 1 experiment folder → `legacy/experiments/`

### UPDATE Paths:
- All test imports to use sys.path
- Run manager `runs_base_dir` to `../experiments/runs/`
- pytest.ini configuration

### VERIFY:
- All 321+ tests still pass
- Run manager creates dirs at correct location
- No broken imports
