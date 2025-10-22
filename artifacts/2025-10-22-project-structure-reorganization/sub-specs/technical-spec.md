# Technical Specification

This is the technical specification for the artifact detailed in research-os/artifacts/2025-10-22-project-structure-reorganization/spec.md

## Technical Requirements

### 1. Directory Structure Changes

**New Directories to Create:**
```
project/
├── legacy/                          # NEW: Historical/experimental code
│   ├── scripts/                     # Debug and one-off scripts
│   ├── experiments/                 # Old experiment outputs
│   └── configs/                     # Legacy configuration files
├── issues/                          # NEW: Issue documentation
├── tests/                           # NEW: Test suite (outside src/)
│   ├── __init__.py
│   ├── conftest.py                  # Root-level fixtures
│   ├── models/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_encoder.py
│   │   ├── test_decoder.py
│   │   ├── test_beta_vae.py
│   │   └── test_losses.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_arc_dataset.py
│   │   ├── test_transforms.py
│   │   └── test_data_loaders.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   └── test_trainer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   └── test_metrics.py
│   └── integration/
│       ├── __init__.py
│       └── test_integration.py
└── experiments/
    └── runs/                        # UPDATED PATH: Run manager output
```

**Files to Move:**

**To `project/legacy/scripts/`:**
- `src/debug_dataloader.py`
- `src/debug_generation.py`
- `src/debug_latent.py`
- `src/debug_loss.py`
- `src/debug_vae_training.py`
- `src/diagnose_*.py` (if any exist)
- `src/simple_gen.py`
- `src/compute_class_weights.py`
- `src/show_images.py`
- `src/demo.py` (if considered experimental)

**To `project/legacy/experiments/`:**
- `src/experiments/0.2-beta-vae/` (entire directory)

**To `project/legacy/configs/`:**
- Any standalone `config.json` files in src/

**To `project/issues/`:**
- `src/ISSUE_MODE_COLLAPSE.md`

**Special Case - Verify Before Moving:**
- `src/regenerate_reconstruction.py` - Check if this contains current visualization logic; if so, keep in src/, otherwise move to legacy

### 2. Test Migration Requirements

**Test Files to Move:**
All `test_*.py` files from `src/` to `tests/` with proper subdirectory placement:

**To `tests/models/`:**
- `test_encoder.py`
- `test_decoder.py`
- `test_beta_vae.py`
- `test_losses.py` (if exists)

**To `tests/data/`:**
- `test_arc_dataset.py`
- `test_data_loaders.py` (or `test_data_loading.py`)
- `test_transforms.py` (if exists)

**To `tests/training/`:**
- `test_trainer.py`
- `test_run_manager.py` (if exists)

**To `tests/evaluation/`:**
- `test_metrics.py`
- `test_visualization.py` (if exists)

**To `tests/integration/`:**
- `test_integration.py`
- `test_large_scale_pipeline.py` (if integration-level)

**To `tests/` (root level for generator tests):**
- `test_atomic_generator.py`
- `test_blob_generator.py`
- `test_shape_generators.py`
- `test_pipeline.py`
- `test_visual_validation.py`

### 3. Import Path Updates

**Files Requiring Import Changes:**

**All test files must update their imports from:**
```python
# OLD (when tests were in src/)
from models.beta_vae import BetaVAE
from data import create_data_loaders
```

**To:**
```python
# NEW (tests in project/tests/)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.beta_vae import BetaVAE
from data import create_data_loaders
```

**Alternative approach using pytest and sys.path configuration in `tests/conftest.py`:**
```python
# tests/conftest.py
import sys
from pathlib import Path

# Add src to Python path for all tests
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
```

### 4. Run Manager Configuration Updates

**Files to Modify:**

**`src/training/config.py`:**
- Update `runs_base_dir` default from `"experiments/runs"` to `"../experiments/runs"`
- This makes paths relative to src/ point to project/experiments/runs/

**Before:**
```python
runs_base_dir: str = "experiments/runs"  # Base directory for all runs
```

**After:**
```python
runs_base_dir: str = "../experiments/runs"  # Base directory for all runs (project/experiments/runs/)
```

**Verification Required:**
- Test that RunManager correctly creates directories at `project/experiments/runs/`
- Verify run_id generation and folder creation works
- Confirm checkpoints save to correct location

### 5. pytest Configuration

**Create `project/pytest.ini`:**
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --tb=short
    --strict-markers
```

**Create `project/tests/conftest.py`:**
```python
"""
Root conftest.py for test suite.
Configures Python path and shared fixtures.
"""
import sys
from pathlib import Path

# Add src to path for all tests
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

# Import pytest for fixtures
import pytest
```

### 6. Verification and Testing Requirements

**Step-by-Step Verification Process:**

1. **Pre-Migration Test Run:**
   ```bash
   cd /Users/thomas/Projects/hackathon/research-os/project
   pyenv activate zeus
   pytest src/ -v
   ```
   Record all passing tests as baseline.

2. **File Migration:**
   - Execute all file moves as specified
   - Do NOT delete any files (move to legacy instead)

3. **Import Path Updates:**
   - Update all test imports as specified
   - Add conftest.py files with sys.path configuration

4. **Post-Migration Test Run:**
   ```bash
   cd /Users/thomas/Projects/hackathon/research-os/project
   pyenv activate zeus
   pytest tests/ -v
   ```
   Verify all tests that passed in step 1 still pass.

5. **Run Manager Verification:**
   ```bash
   pyenv activate zeus
   python -c "
   from training import RunManager
   rm = RunManager()
   print(f'Run dir: {rm.run_dir}')
   assert 'project/experiments/runs' in str(rm.run_dir.absolute())
   print('✓ Run manager path correct')
   "
   ```

6. **Integration Test:**
   ```bash
   # Quick training test to verify end-to-end functionality
   python src/train_vae.py --quick-test --no-wandb
   # Verify run created at project/experiments/runs/<run_id>/
   ```

### 7. Safety Requirements

**Critical Safety Checks:**

- **No File Deletion:** All files must be moved, not deleted. If uncertain about a file's purpose, move to legacy.
- **Import Verification:** Every test file must have working imports after migration.
- **Functionality Preservation:** Complete test suite must pass with same results pre and post migration.
- **Run Manager Path Testing:** Explicit verification that experiments save to correct location outside src/.
- **Git Safety:** Create a git branch before starting migration for easy rollback if needed.

### 8. Performance Considerations

**None.** This is a pure file reorganization with no performance implications. Test execution time should remain identical.

### 9. Documentation Updates

**Files to Update After Migration:**

- `project/README.md` (if exists) - Update directory structure documentation
- `project/src/README.md` (if exists) - Update to reflect new test location
- Any developer documentation referencing test file locations

**No External Dependencies Required** - This refactoring uses only existing tools and libraries.
