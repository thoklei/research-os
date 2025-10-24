# Spec Requirements Document

> Spec: Simplified Dataset for Model Capacity Validation
> Created: 2025-10-24

## Overview

Validate that the β-VAE model can represent the dataset by training on a simplified 100k sample dataset containing only parameterized shapes (no blob objects), with beta effectively disabled to isolate model capacity evaluation.

## User Stories

### Dataset Simplification

As a researcher, I want to generate a simplified dataset with only deterministic parameterized shapes, so that I can eliminate variability from blob objects and focus on evaluating whether the model architecture has sufficient capacity to represent structured patterns.

The simplified dataset will contain 100k samples generated using only the shape generators from shape_generators.py: Lines, Rectangles, Checkerboards, L-shapes, T-shapes, Plus patterns, and Zigzag patterns. This removes the high variability introduced by blob objects while maintaining diverse geometric patterns across 10 color classes.

### Model Capacity Validation

As a researcher, I want to train the model with beta effectively disabled (using a linear schedule with max_beta ≈ 0), so that I can verify the model architecture has sufficient capacity to learn the dataset without the confounding factor of KL divergence regularization.

This training will focus purely on reconstruction quality, using the existing focal loss and class weighting mechanisms to prevent collapse to black pixels. Success means achieving high pixel accuracy (>95%) without the model collapsing to trivial solutions.

## Spec Scope

1. **Dataset Generation Script** - Create a script to generate 100k simplified dataset samples using only shape_generators.py (Lines, Rectangles, Checkerboards, L-shapes, T-shapes, Plus, Zigzag), excluding blob objects
2. **Beta Schedule Modification** - Implement or use existing linear schedule with max_beta set to 0 or near-0 (e.g., 0.001) to effectively disable KL regularization
3. **Training Configuration** - Set up training config for capacity validation run with simplified dataset and disabled beta
4. **Evaluation Metrics** - Track reconstruction accuracy and verify no collapse to black pixels (accuracy should exceed 95%, not stay at 93%)
5. **Documentation** - Document results showing model can represent simplified dataset before moving to full dataset with beta-VAE training

## Out of Scope

- Blob object generation (explicitly excluded due to high variability)
- Beta-VAE training with actual KL regularization (beta > 0)
- Compositional transformation framework (Phase 1 work)
- Latent space interpretability analysis
- Human difficulty correlation studies
- Production dataset generation beyond 100k validation samples

## Expected Deliverable

1. Successfully generate 100k sample .npz dataset file containing only parameterized shapes (verifiable by inspecting generated samples)
2. Train model to >95% pixel accuracy on simplified dataset with beta ≈ 0, demonstrating model capacity without posterior collapse
3. Training results showing color distribution across all 10 classes (not collapsed to black/93% accuracy)
