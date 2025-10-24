# Spec Requirements Document

> Spec: Conservative Beta Scheduling and Free Bits for VAE Training
> Created: 2025-10-23

## Overview

Implement conservative beta-annealing schedules and free bits mechanism to prevent posterior collapse in beta-VAE training when dealing with severe class imbalance (93% black pixels in ARC grids). The current implementation collapses to predicting all black pixels when beta increases above zero, despite having focal loss and class weighting implemented.

## User Stories

### Machine Learning Researcher - Preventing Mode Collapse

As a machine learning researcher training a beta-VAE on class-imbalanced data, I want to use a conservative beta schedule and free bits mechanism, so that I can increase beta above zero without the model collapsing to predicting only the majority class.

**Current Problem:**
- Training works well at beta=0, achieving diverse reconstructions
- When beta increases (even to 0.01), model collapses to predicting all black pixels
- Achieves 93% accuracy by exploiting class imbalance (93% of pixels are black)
- KL divergence drops to near-zero (posterior collapse)

**Desired Outcome:**
- Model maintains diverse reconstructions as beta increases
- KL/dim stays above minimum threshold (e.g., 0.05 nats/dim)
- Non-black pixel accuracy remains high throughout training
- Smooth transition from reconstruction focus (beta=0) to regularized learning (beta>0)

### ML Engineer - Configurable Training Strategies

As an ML engineer, I want to easily configure different beta scheduling strategies and free bits parameters, so that I can experiment with different approaches to find the optimal training regime for my specific data distribution.

**Workflow:**
1. Set beta schedule type in config (e.g., "ultra_conservative", "cyclical", "adaptive")
2. Configure free bits threshold per dimension (e.g., 0.3 nats/dim)
3. Run training and monitor KL/dim and per-class accuracy metrics
4. Compare results across different schedule configurations

## Spec Scope

1. **Ultra-Conservative Beta Schedule** - Implement new beta schedule with much longer warm-up (20 epochs), slower ramp rate (60 epochs to reach max), and lower maximum beta (0.1 instead of 0.5)

2. **Free Bits Mechanism** - Add per-dimension KL clamping to prevent any latent dimension from collapsing below a minimum information threshold, ensuring each dimension carries meaningful information

3. **Cyclical Beta Schedule** - Implement cyclical annealing where beta alternates between reconstruction focus and regularization, giving the model repeated opportunities to recover from partial collapse

4. **Configuration System** - Add config parameters for schedule type selection, free bits threshold, and schedule hyperparameters (max_beta, warmup_epochs, etc.)

5. **Enhanced Monitoring** - Extend logging to track per-dimension KL divergence and non-black pixel accuracy to detect collapse early

## Out of Scope

- Architectural changes to the encoder/decoder networks
- Alternative loss functions beyond focal loss and class weighting (already implemented)
- Adaptive beta scheduling based on metrics (deferred to future iteration)
- Changes to data augmentation or preprocessing
- Multi-GPU training optimizations

## Expected Deliverable

1. **Training succeeds with beta > 0**: Model trains successfully with final beta values of 0.1-0.5 without collapsing to all-black predictions

2. **KL/dim remains healthy**: Per-dimension KL divergence stays above 0.05 nats/dim throughout training, indicating active latent space usage

3. **Non-black accuracy maintained**: Per-class accuracy for non-black colors (classes 1-9) remains above 30% even as beta increases

4. **Configurable schedules**: Users can select between "ultra_conservative", "cyclical", and standard "linear_warmup" schedules via config file

5. **Observable in logs**: Training logs show per-dimension KL values and per-class accuracies, making collapse immediately visible
