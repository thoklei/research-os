# Spec Requirements Document

> Spec: Project Structure Reorganization
> Created: 2025-10-22

## Overview

Reorganize the project repository to improve maintainability by consolidating legacy experimental files, establishing proper test directory structure following Python best practices, and relocating experiment run storage outside the src directory. This cleanup will preserve all core functionality while eliminating technical debt and improving developer experience.

## User Stories

### Clean Codebase for Active Development

As a researcher/developer, I want a clean and organized codebase with clearly separated core functionality, tests, and legacy files, so that I can quickly navigate the project, understand what code is actively maintained, and reduce cognitive overhead when implementing new features.

The current scattered structure makes it difficult to distinguish between production code, experimental scripts, and test files. After reorganization, developers will immediately understand the project layout and know where to find or add new code.

### Proper Test Organization

As a developer, I want tests organized in a mirror structure of the src directory, so that I can easily locate tests for specific modules and maintain test coverage as the codebase grows.

Currently, all test files (20+ files) are scattered in the src root directory, making it hard to find relevant tests and maintain test organization. The new structure will have tests/models/, tests/data/, tests/training/ mirroring the src structure.

### Safe Experiment Storage

As a researcher, I want experiment runs stored outside the src directory, so that version control, deployment, and code distribution are cleaner and experiment artifacts don't interfere with source code management.

The run manager currently stores experiments relative to src/, which mixes generated artifacts with source code. Moving to project/experiments/runs/ provides clear separation between code and experimental outputs.

## Spec Scope

1. **Legacy File Migration** - Move debug scripts (debug_*.py, diagnose_*.py), one-off utilities (simple_gen.py, compute_class_weights.py, show_images.py), old experiment folders (experiments/0.2-beta-vae/), and existing config.json files to a dedicated legacy/ directory for historical reference.

2. **Test Directory Restructuring** - Migrate all test_*.py files from src/ to a new tests/ directory with structure mirroring src/ (tests/models/, tests/data/, tests/training/, tests/evaluation/, tests/integration/), with distributed conftest.py files for module-specific fixtures.

3. **Issue Documentation Organization** - Create project/issues/ directory and move ISSUE_MODE_COLLAPSE.md and any other issue documentation out of src/ to maintain clean source tree.

4. **Run Manager Path Update** - Update training/run_manager.py and training/config.py to point experiment storage to project/experiments/runs/ (outside src/) instead of the current experiments/runs/ relative path.

5. **Functionality Verification** - Run complete test suite after reorganization to ensure no import paths are broken and all functionality remains intact.

## Out of Scope

- Refactoring or rewriting any core functionality
- Adding new features or capabilities
- Modifying the structure of existing modules (models/, data/, training/, evaluation/)
- Deleting any code (everything moves to legacy if not actively used)
- Changing test implementations (only moving files, not modifying test logic)
- Updating .gitignore or other repository configuration files (can be done separately)

## Expected Deliverable

1. **Clean Source Directory** - src/ contains only actively maintained code organized in clear module directories (models/, data/, training/, evaluation/) with main scripts (train_vae.py, evaluate_vae.py, generate_dataset.py) and no scattered test or debug files.

2. **Proper Test Structure** - All tests located in project/tests/ with mirrored directory structure, making it easy to find and run tests for specific modules.

3. **All Tests Passing** - Complete test suite runs successfully with no broken imports or functionality after reorganization, verified by running pytest from project root.

4. **Updated Run Storage** - Experiment runs saved to project/experiments/runs/ with run manager configuration updated and tested to confirm correct path usage.
