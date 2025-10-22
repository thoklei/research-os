# Spec Requirements Document

> Spec: β-VAE Training Pipeline for ARC Grid Encoding
> Created: 2025-10-22

## Overview

Implement a β-VAE (Beta Variational Autoencoder) training pipeline that learns to encode and decode 16×16 ARC-like grids into a low-dimensional latent space (d=10), achieving ≥90% pixel-wise reconstruction accuracy to enable compositional transformation operations in later experiments.

## User Stories

### Story 1: Researcher Training Encoder Model

As a researcher, I want to train a β-VAE on the atomic image corpus, so that I have a trained encoder/decoder pair that can map ARC grids to/from a structured latent space for compositional operations.

**Workflow:**
1. Load the existing 100K atomic image corpus from `.npz` files
2. Configure β-VAE architecture (d=10 latent dimensions) and training hyperparameters
3. Run training with β-annealing schedule (0→1→2 over 30 epochs)
4. Monitor training progress via Weights & Biases dashboard
5. Evaluate final model on test set to verify ≥90% pixel-wise accuracy
6. Save trained encoder/decoder checkpoints for use in Experiment 0.3

### Story 2: Researcher Evaluating Model Quality

As a researcher, I want to evaluate the trained β-VAE's reconstruction quality and latent space structure, so that I can verify it meets requirements before proceeding to compositional transformations.

**Workflow:**
1. Load trained model checkpoints
2. Run evaluation script on 10K test images
3. Review pixel-wise accuracy metric (target: ≥90%)
4. Inspect reconstructed samples visually to identify failure modes
5. Generate random samples from prior z ~ N(0,I) to verify generative capability
6. Visualize latent space structure (PCA/t-SNE) to confirm smooth manifold

### Story 3: Researcher Generating Samples

As a researcher, I want to sample new ARC-like grids from the trained decoder, so that I can verify the model has learned a valid generative distribution over grid space.

**Workflow:**
1. Load trained decoder checkpoint
2. Sample random latent codes z ~ N(0,I)
3. Decode to 16×16 grids
4. Visually inspect generated grids for validity (proper ARC-like structure)
5. Save generated samples for documentation

## Spec Scope

1. **β-VAE Architecture Implementation** - Encoder (16×16×10 → d=10 latent), Decoder (d=10 → 16×16×10 logits), with reparameterization trick
2. **Training Pipeline** - Data loading from `.npz` corpus, β-annealing schedule (0→1→2), Adam optimizer with cosine learning rate decay, early stopping
3. **Loss Function** - Cross-entropy reconstruction loss + β-weighted KL divergence with proper β scheduling
4. **Evaluation Framework** - Pixel-wise accuracy computation, visual reconstruction inspection, KL monitoring for posterior collapse detection
5. **Weights & Biases Integration** - Training metrics logging, hyperparameter tracking, model checkpoint artifacts
6. **Sample Generation Scripts** - Decode random latent samples, save outputs for qualitative assessment

## Out of Scope

- Latent space transformations (translate, rotate, scale) - reserved for Experiment 0.3
- Disentanglement metrics (MIG, SAP) - optional, not required for success
- Alternative architectures (Slot Attention, VQ-VAE) - only if β-VAE fails
- Hyperparameter search - use specified defaults from encoder-spec-summary.md
- Production deployment - research prototype only

## Expected Deliverable

1. **Trained model achieving ≥90% pixel-wise accuracy** on test set (10K images), verified through evaluation script
2. **Saved checkpoints** (encoder.pth, decoder.pth, vae_best.pth) loadable via PyTorch for downstream experiments
3. **Training artifacts** logged to Weights & Biases: loss curves, accuracy over epochs, sample reconstructions, generated samples
4. **Sample generation script** that loads decoder and produces valid ARC-like grids from random latent codes
5. **Evaluation report** (metrics.json) documenting final pixel-wise accuracy, KL divergence, active latent dimensions
