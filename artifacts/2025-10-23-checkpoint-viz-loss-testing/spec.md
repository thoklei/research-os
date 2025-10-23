# Spec Requirements Document

> Spec: Checkpoint Visualization & Loss Configuration Testing
> Created: 2025-10-23

## Overview

Implement automatic visualization generation during training checkpoints to enable qualitative assessment of reconstruction quality, and systematically test three loss configuration options to resolve the training instability issue where Focal Loss combined with extreme class weights creates pathological loss landscapes.

## User Stories

### As a Researcher Training the β-VAE Model

As a researcher training the β-VAE model, I want to automatically visualize reconstructions whenever a checkpoint is saved, so that I can qualitatively assess whether the model is learning meaningful object reconstructions or just predicting background pixels throughout the training process.

**Workflow:** During training, whenever a checkpoint is saved (every N epochs or when achieving best validation loss), the system should automatically:
1. Sample 10 random images from the training set
2. Generate reconstructions using the current model state
3. Create a side-by-side visualization showing original vs reconstructed grids
4. Save the visualization with the checkpoint epoch/name in the filename
5. Display per-sample metrics (pixel accuracy, loss) on the visualization

This allows me to quickly scan through checkpoint visualizations and identify when/if the model transitions from learning objects to mode collapse (predicting all backgrounds).

### As a Researcher Debugging Loss Configuration Issues

As a researcher debugging the training instability caused by Focal Loss + extreme class weights, I want to systematically test three different loss configurations in sequence, so that I can identify which configuration enables stable training with proper object learning (not mode collapse to backgrounds).

**Workflow:** Execute three experiments sequentially, each with different loss configurations:
1. **Option A** (Baseline - Already Tested): Focal Loss (γ=2.0) without class weights → Result: 93% accuracy = mode collapse to backgrounds
2. **Option B** (Proposed Solution): Focal Loss (γ=2.0) with square-root smoothed class weights (11:1 ratio instead of 123:1) → Expected: Stable training with 70-80% accuracy
3. **Option C** (Fallback): Reduced Focal Loss (γ=1.0) with original class weights (123:1) → Expected: Less aggressive focal weighting allows class weights to work

Each experiment should run for 5 epochs (quick test), generate checkpoint visualizations, and produce a summary report comparing training curves and final reconstruction quality.

## Spec Scope

1. **Checkpoint Visualization Integration** - Modify `trainer.py` to automatically call visualization function after every checkpoint save, generating side-by-side original vs reconstruction plots for 10 random training samples
2. **Visualization Enhancement** - Extend existing `plot_reconstructions()` function to support saving to checkpoint-specific paths with epoch numbers in filenames
3. **Loss Configuration Testing Framework** - Implement systematic testing of three loss configurations (Option A/B/C) with automated experiment execution and result comparison
4. **Square Root Class Weight Smoothing** - Add `sqrt_inverse` method to class weight calculation in `beta_vae.py` to smooth extreme 123:1 ratio down to 11:1
5. **Experiment Result Comparison** - Generate comparison report showing training curves (loss, accuracy, KL divergence) and final reconstructions across all three options

## Out of Scope

- Real-time visualization during training (only at checkpoint saves)
- Interactive visualization tools or dashboards
- Automated hyperparameter tuning beyond the three specified options
- Full 50-epoch training runs (experiments use 5-epoch quick tests)
- Weights & Biases integration for this debugging phase (using local logging only)

## Expected Deliverable

1. **Automatic Checkpoint Visualizations** - Every checkpoint save produces a PNG file showing 10 original vs reconstructed grid pairs with per-sample metrics, saved in the run's checkpoint directory
2. **Working Option B or C** - At least one of the loss configurations (B or C) produces stable training with accuracy in 70-80% range (not 93% mode collapse) and qualitative reconstruction visualizations showing learned objects
3. **Experiment Comparison Report** - Document summarizing results of all three options with training curve plots and final reconstruction examples, identifying the recommended loss configuration for full training
