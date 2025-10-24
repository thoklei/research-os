# Artifact Tasks

These are the tasks to be completed for the artifact detailed in research-os/artifacts/2025-10-24-simplified-dataset-validation/spec.md

> Created: 2025-10-24
> Status: Ready for Implementation

## Tasks

### 1. Dataset Generation Script

Create script to generate 100k simplified dataset using only parameterized shapes (no blobs), targeting ~25MB output size.

- [x] 1.1 Create `project/src/generate_simplified_dataset.py` script
  - Import shape generators: RectangleGenerator, LineGenerator, PatternGenerator
  - Import AtomicImageGenerator for composing shapes on grids
  - Set up argument parsing for output path and sample count
  - Configure 16x16 grid size with 10-color ARC palette

- [x] 1.2 Implement shape-only generation logic
  - Instantiate generators excluding BlobGenerator
  - Configure PatternGenerator with: checkerboard, l_shape, t_shape, plus, zigzag
  - Set object count per grid: 1-6 objects
  - Randomize shape selection across samples

- [x] 1.3 Implement dataset splitting and output
  - Generate exactly 100,000 samples total
  - Split: 80% train (80k), 10% val (10k), 10% test (10k)
  - Save as `data/simplified_dataset_100k.npz` with keys: 'train', 'val', 'test'
  - Add shape metadata to verify no blobs present

- [x] 1.4 Write tests for dataset generation
  - Test shape generator instantiation (no blobs)
  - Test output file format (.npz with correct keys)
  - Test sample count (100k total, correct split ratios)
  - Test grid dimensions (16x16) and color palette (10 colors)
  - Test file size is approximately 25MB (±5MB tolerance)

- [x] 1.5 Run dataset generation and verify output
  - Execute: `python project/src/generate_simplified_dataset.py`
  - Verify file created at `data/simplified_dataset_100k.npz`
  - Check file size is ~25MB (compressed to 1.6MB)
  - Visually inspect sample images to confirm only shapes (no blobs)
  - Verify all tests pass

### 2. Training Configuration and Execution

Train model on simplified dataset with beta disabled (beta_max=0.0) to validate model capacity.

- [ ] 2.1 Verify existing CLI supports required configuration
  - Confirm `--beta-schedule` and `--beta-max` arguments exist in train_vae.py
  - Confirm `--data-path` argument works with custom datasets
  - Verify focal loss and class weighting remain enabled by default
  - Document command-line invocation for capacity validation run

- [ ] 2.2 Configure Weights & Biases tracking
  - Set up W&B run with tags: ["simplified-dataset", "capacity-validation", "beta-disabled"]
  - Configure metrics logging: pixel_accuracy, per_class_accuracy, reconstruction_loss, kl_divergence
  - Enable sample reconstruction visualization (every 10 epochs)
  - Set run name: "simplified-dataset-beta0-capacity-validation"

- [ ] 2.3 Execute training run
  - Run command: `python project/src/train_vae.py --data-path data/simplified_dataset_100k.npz --beta-schedule constant --beta-max 0.0`
  - Use default latent dimension, batch size, and other hyperparameters
  - Train for 100-200 epochs with early stopping (patience=20)
  - Monitor training progress via W&B dashboard

- [ ] 2.4 Verify training behavior during execution
  - Confirm KL divergence stays near zero (beta=0 disables KL regularization)
  - Monitor reconstruction loss decreases steadily
  - Check sample visualizations show shape structure (not collapsed to black)
  - Verify no training instability or divergence

### 3. Evaluation and Validation

Verify model achieves >95% pixel accuracy without collapse to trivial solutions.

- [ ] 3.1 Analyze training metrics
  - Extract final pixel accuracy from W&B logs
  - Calculate per-class accuracy across all 10 colors
  - Generate color distribution histogram (predicted vs ground truth)
  - Compare reconstruction loss curve to baseline runs

- [ ] 3.2 Validate success criteria
  - Confirm pixel accuracy > 95% (above 93% collapse threshold)
  - Verify per-class accuracy shows reasonable distribution (no single color dominance)
  - Confirm black pixel class accuracy < 95% (indicates not collapsed to all-black)
  - Visual inspection: sample reconstructions preserve shape structure

- [ ] 3.3 Document results and findings
  - Create summary report with final metrics table
  - Include sample visualizations (input vs reconstruction)
  - Document training configuration and hyperparameters used
  - Note any unexpected behaviors or observations
  - Confirm model capacity is sufficient for simplified dataset

- [ ] 3.4 Verify all tests pass
  - Run dataset generation tests
  - Verify output dataset meets all specifications
  - Confirm training completed successfully without errors
  - Validate final accuracy exceeds 95% threshold
