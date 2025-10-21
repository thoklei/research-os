# Spec Requirements: Project Structure Setup

## Initial Description
create the overall structure for the project, to get started on experiment 0.1

## Requirements Discussion

### First Round Questions

**Q1:** For the ML framework, I assume we'll use PyTorch as specified in the tech stack documentation (PyTorch 2.0+) with Torch for neural encoder training and data loading. Is that correct, or would you prefer JAX/Flax?
**Answer:** Definitely PyTorch

**Q2:** I'm thinking the project structure should follow Python best practices with directories for src/, experiments/, data/, notebooks/, configs/, tests/, and scripts/. Should we include additional directories specific to ARC task generation like grids/, encoders/, or generators/?
**Answer:** Sounds good (src/, experiments/, data/, notebooks/, configs/, tests/, scripts/)

**Q3:** For the atomic image generator module (Experiment 0.1), I assume we'll organize it as src/generators/atomic_image.py with classes for generating blobs, rectangles, lines, and patterns. Is that the right structure, or would you prefer a different organization?
**Answer:** Sounds good for now (src/generators/atomic_image.py with classes for blobs, rectangles, lines, patterns)

**Q4:** I'm thinking we should set up configuration management using YAML files (as shown in the tech stack) with configs for encoder training, task generation, and experiment parameters. Should we use Hydra for config management or stick with basic YAML loading?
**Answer:** YAML sounds good

**Q5:** For data storage, I assume we'll use NumPy arrays for grids and JSON for metadata initially, then potentially move to HDF5 for the larger benchmark dataset. Should we set up HDF5 structure from the start?
**Answer:** No, all good, no need for HDF5 (stick with NumPy arrays + JSON metadata)

**Q6:** For experiment tracking, the tech stack mentions Weights & Biases (wandb). Should we set that up immediately or start with local logging and add wandb later?
**Answer:** wandb is great

**Q7:** I assume we should include basic development tools like pre-commit hooks for code formatting (black), linting (flake8), and testing (pytest) from the beginning. Should we set these up now?
**Answer:** Yes please set up pre-commit hooks for code formatting with black

**Q8:** Is there anything specific you want to exclude from this initial setup, or any particular area you want me to focus on most?
**Answer:** Focus on writing well-documented code with tests. No existing code to reuse.

### Existing Code to Reference
No similar existing features identified for reference.

### Follow-up Questions
None needed - all requirements are clear.

## Visual Assets

### Files Provided:
No visual assets provided.

### Visual Insights:
N/A

## Requirements Summary

### Functional Requirements
- Set up complete Python project structure for ARC task generation research
- Create directory structure: src/, experiments/, data/, notebooks/, configs/, tests/, scripts/
- Implement atomic image generator module at src/generators/atomic_image.py
- Support generating 4 object types: blobs, rectangles, lines, and patterns
- Use PyTorch 2.0+ as the ML framework
- Implement YAML-based configuration management
- Store data using NumPy arrays for grids and JSON for metadata
- Integrate Weights & Biases (wandb) for experiment tracking
- Set up pre-commit hooks for black code formatting

### Non-Functional Requirements
- Write well-documented code with comprehensive docstrings
- Include tests for all implemented functionality
- Follow Python best practices and PEP 8 standards
- Ensure reproducibility with proper seed management
- Design for extensibility to support future encoder and transformation development

### Reusability Opportunities
- No existing code to reuse (starting from scratch)

### Scope Boundaries
**In Scope:**
- Complete project directory structure
- Atomic image generator implementation (Experiment 0.1 foundation)
- PyTorch setup and environment configuration
- YAML configuration system
- NumPy/JSON data storage setup
- Wandb integration for tracking
- Pre-commit hooks with black formatting
- Basic testing infrastructure with pytest

**Out of Scope:**
- HDF5 data storage (will stay with NumPy + JSON)
- Other linting tools beyond black (no flake8 setup requested)
- Hydra config management (using basic YAML loading)
- Full encoder implementation (future work)
- Complete sub-function library (future work)
- Human study infrastructure (future work)

### Technical Considerations
- Project aligns with research mission: controllable visual reasoning task generation
- Following roadmap's Experiment 0.1: Atomic Image Generator Implementation
- Using tech stack specifications: PyTorch 2.0+, matplotlib for visualization
- Grid specifications: 10x10 or 16x16, 10-color ARC palette
- Object generation via connectivity-biased growth algorithm
- Target corpus: 50,000-100,000 atomic images for encoder training