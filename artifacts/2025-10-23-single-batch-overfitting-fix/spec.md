# Spec Requirements Document

> Spec: Fix Single Batch Overfitting Mode Collapse
> Created: 2025-10-23

## Overview

Fix the mode collapse issue in single batch overfitting where the model produces completely black reconstructed images instead of memorizing the batch. This sanity check is critical for validating model capacity before full training runs.

## User Stories

### Researcher Validating Model Capacity

As a machine learning researcher, I want to overfit a single batch successfully so that I can validate my model has sufficient capacity and my training pipeline is working correctly before committing to expensive full training runs.

**Current Problem**: When running `python train_vae.py --overfit-batch`, the model produces completely black reconstructed images. The model learns a trivial solution by predicting the most common pixel value (black/0) for all pixels instead of memorizing the specific images in the batch. This defeats the purpose of the sanity check.

**Root Cause**: While class weights ARE being computed from the single batch (32 samples = 8,192 pixels), this sample is too small to produce statistically meaningful weights. The sparse class distribution in such a limited sample makes the weights ineffective at preventing mode collapse.

**Expected Workflow**: The researcher runs `python train_vae.py --overfit-batch`, and after a few hundred iterations, the model should achieve near-perfect reconstruction of the training batch images, demonstrating that the architecture has sufficient capacity to learn.

### Developer Debugging Training Issues

As a developer debugging training failures, I want class weighting to work consistently between normal training and overfitting modes so that I can isolate whether issues are architectural versus optimization-related.

**Current Problem**: Class weighting successfully prevents mode collapse during normal training (80K samples, weight ratio 11.13x) but is ineffective during single batch overfitting (32 samples, 8,192 pixels total). The weights computed from such a small sample lack statistical significance.

**Expected Workflow**: The developer can use the same class weighting configuration (focal loss + class weights) in both training modes and see consistent behavior, allowing proper isolation of issues.

## Spec Scope

1. **Use global class weights for overfitting mode** - Modify train_vae.py to compute class weights from the full dataset instead of the single batch when --overfit-batch flag is used
2. **Add class weight caching** - Save computed class weights to disk and allow loading them to avoid recomputation on subsequent overfitting runs
3. **Improve batch statistics logging** - Log both global class distribution (from full dataset) and single-batch distribution for comparison
4. **Validate overfitting success** - Ensure the model can achieve near-perfect reconstruction (e.g., >95% pixel accuracy) within 500 iterations
5. **Add per-class accuracy tracking** - Track and log reconstruction accuracy per color class to identify if specific colors are still being ignored

## Out of Scope

- Changing the core model architecture (encoder/decoder design)
- Modifying the normal training pipeline or full dataset training behavior
- Implementing new loss functions beyond the existing focal loss
- Addressing performance optimization or training speed improvements
- Handling multi-GPU or distributed training scenarios

## Expected Deliverable

1. **Successful single batch overfitting**: Running `python train_vae.py --overfit-batch` produces reconstructed images that visually match the input images (not all black) and achieve >95% pixel accuracy within 500 iterations
2. **Consistent class weighting behavior**: The global class weights computed from the full dataset (weight ratio ~11x) are used during overfitting mode, matching the approach that works in normal training
3. **Clear logging output**: The script prints both global class distribution and single-batch distribution, class weight statistics, and per-class reconstruction accuracy to help diagnose issues
