# Artifact Tasks

These are the tasks to be completed for the artifact detailed in research-os/artifacts/2025-10-23-checkpoint-viz-loss-testing/spec.md

> Created: 2025-10-23
> Status: Ready for Implementation

## Tasks

- [x] 1. Implement Checkpoint Visualization Integration
  - [x] 1.1 Modify `src/training/trainer.py` to import visualization function
  - [x] 1.2 Add visualization generation call in `save_model()` method after checkpoint save
  - [x] 1.3 Sample 10 random batches from train_loader for visualization
  - [x] 1.4 Save visualization to checkpoint directory with epoch-specific filename
  - [x] 1.5 Add console logging for visualization saves
  - [x] 1.6 Test visualization generation with quick training run

- [x] 2. Implement Square Root Class Weight Smoothing
  - [x] 2.1 Modify `src/models/beta_vae.py` to add `sqrt_inverse` weight calculation method
  - [x] 2.2 Update class weight computation logic to support three methods: inverse, sqrt_inverse, balanced
  - [x] 2.3 Add inline documentation explaining sqrt smoothing rationale (123:1 → 11:1)
  - [x] 2.4 Verify weight calculation produces expected ~11:1 ratio for typical ARC data distribution

- [x] 3. Test Option B: Focal Loss + Sqrt Smoothed Weights
  - [x] 3.1 Update `src/training/config.py` to set Option B configuration (use_focal_loss=True, focal_gamma=2.0, use_class_weights=True, class_weight_method='sqrt_inverse')
  - [x] 3.2 Run quick test: `cd src && python train_vae.py --quick-test --no-wandb 2>&1 | tee ../experiments/runs/{run_id}/training.log` (log saved in run folder)
  - [x] 3.3 Review checkpoint visualizations at `../experiments/runs/{run_id}/checkpoints/reconstructions_epoch_*.png`
  - [x] 3.4 Verify training accuracy converges to 70-85% range (not 93%)
  - [x] 3.5 Check training curve stability (no pathological fluctuations)
  - [x] 3.6 Document Option B results: run_id, final accuracy, reconstruction quality notes

- [ ] 4. Test Option C: Reduced Focal Loss + Original Weights (If Option B Fails)
  - [ ] 4.1 Update `src/training/config.py` to set Option C configuration (use_focal_loss=True, focal_gamma=1.0, use_class_weights=True, class_weight_method='inverse')
  - [ ] 4.2 Run quick test: `cd src && python train_vae.py --quick-test --no-wandb 2>&1 | tee ../experiments/runs/{run_id}/training.log` (log saved in run folder)
  - [ ] 4.3 Review checkpoint visualizations at `../experiments/runs/{run_id}/checkpoints/reconstructions_epoch_*.png`
  - [ ] 4.4 Verify training accuracy converges to 70-85% range (not 93%)
  - [ ] 4.5 Check training curve stability
  - [ ] 4.6 Document Option C results: run_id, final accuracy, reconstruction quality notes

- [ ] 5. Generate Comparison Report and Select Best Configuration
  - [ ] 5.1 Compare training curves (loss, accuracy, KL divergence) across Options A, B, C using saved metrics.json in each run folder
  - [ ] 5.2 Compare final reconstruction quality from checkpoint visualizations in each run's checkpoints directory
  - [ ] 5.3 Identify recommended configuration based on stability + reconstruction quality
  - [ ] 5.4 Document findings in `artifacts/2025-10-23-checkpoint-viz-loss-testing/results.md` with run_ids for each option
  - [ ] 5.5 Update `src/training/config.py` with recommended default configuration for future full training runs
