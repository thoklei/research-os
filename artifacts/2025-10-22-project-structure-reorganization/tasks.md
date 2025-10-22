# Spec Tasks

## Tasks

- [x] 1. Pre-Migration Safety Setup and Verification
  - [x] 1.1 Create git branch for reorganization (safety rollback)
  - [x] 1.2 Run baseline test suite and record all passing tests
  - [x] 1.3 Verify regenerate_reconstruction.py contains current visualization logic
  - [x] 1.4 Document current directory structure for reference

- [ ] 2. Create New Directory Structure
  - [ ] 2.1 Create project/legacy/ with subdirectories (scripts/, experiments/, configs/)
  - [ ] 2.2 Create project/issues/ directory
  - [ ] 2.3 Create project/tests/ with mirrored structure (models/, data/, training/, evaluation/, integration/)
  - [ ] 2.4 Create project/experiments/runs/ directory for run manager output
  - [ ] 2.5 Add __init__.py files to all test subdirectories

- [ ] 3. Migrate Legacy and Issue Files
  - [ ] 3.1 Move debug scripts (debug_*.py, diagnose_*.py) to legacy/scripts/
  - [ ] 3.2 Move one-off utilities (simple_gen.py, compute_class_weights.py, show_images.py) to legacy/scripts/
  - [ ] 3.3 Move experiments/0.2-beta-vae/ to legacy/experiments/
  - [ ] 3.4 Move standalone config.json files to legacy/configs/
  - [ ] 3.5 Move ISSUE_MODE_COLLAPSE.md to project/issues/
  - [ ] 3.6 Verify all intended files moved correctly

- [ ] 4. Migrate and Configure Test Suite
  - [ ] 4.1 Create pytest.ini in project root
  - [ ] 4.2 Create root conftest.py with sys.path configuration
  - [ ] 4.3 Move model tests to tests/models/ and update imports
  - [ ] 4.4 Move data tests to tests/data/ and update imports
  - [ ] 4.5 Move training tests to tests/training/ and update imports
  - [ ] 4.6 Move evaluation tests to tests/evaluation/ and update imports
  - [ ] 4.7 Move integration tests to tests/integration/ and update imports
  - [ ] 4.8 Move generator tests to tests/ root and update imports

- [ ] 5. Update Run Manager Configuration
  - [ ] 5.1 Update runs_base_dir in src/training/config.py to "../experiments/runs"
  - [ ] 5.2 Test RunManager path resolution with verification script
  - [ ] 5.3 Verify run directory creation at correct location (project/experiments/runs/)
  - [ ] 5.4 Test checkpoint saving to correct path

- [ ] 6. Post-Migration Verification and Testing
  - [ ] 6.1 Run complete test suite from project root (pytest tests/ -v)
  - [ ] 6.2 Compare test results with baseline from task 1.2
  - [ ] 6.3 Run integration test with quick training run (train_vae.py --quick-test)
  - [ ] 6.4 Verify experiment outputs saved to project/experiments/runs/
  - [ ] 6.5 Document any import path issues and fixes applied
  - [ ] 6.6 Confirm all tests passing and functionality preserved
