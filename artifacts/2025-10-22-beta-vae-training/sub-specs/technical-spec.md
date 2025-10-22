# Technical Specification

This is the technical specification for the artifact detailed in research-os/artifacts/2025-10-22-beta-vae-training/spec.md

## Architecture Specifications

### Encoder: Grid → Latent (φ: ℝ^(16×16) → ℝ^10)

**Input Format:**
- Shape: (batch, 16, 16) integer labels [0-9]
- Preprocessing: Convert to one-hot encoding (batch, 10, 16, 16)

**Network Architecture:**
```
Conv2D(in=10, out=32, kernel=3, stride=1, padding=1) + ReLU
Conv2D(in=32, out=64, kernel=3, stride=2, padding=1) + ReLU  → (batch, 64, 8, 8)
Conv2D(in=64, out=128, kernel=3, stride=2, padding=1) + ReLU → (batch, 128, 4, 4)
Flatten → (batch, 2048)
Dense(2048 → 128) + ReLU
```

**Latent Distribution Heads:**
```
μ_head: Dense(128 → 10)           # Mean vector
σ_head: Dense(128 → 10) + Softplus # Standard deviation (ensure positive)
```

**Output:** μ(z), σ(z) where z ~ N(μ, σ²) with d=10 dimensions

### Reparameterization Trick

```python
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

### Decoder: Latent → Grid (ψ: ℝ^10 → ℝ^(16×16×10))

**Input:** z ~ N(μ, σ²) with shape (batch, 10)

**Network Architecture:**
```
Dense(10 → 128) + ReLU
Dense(128 → 2048) + ReLU
Reshape → (batch, 128, 4, 4)
ConvTranspose2D(in=128, out=64, kernel=4, stride=2, padding=1) + ReLU → (batch, 64, 8, 8)
ConvTranspose2D(in=64, out=32, kernel=4, stride=2, padding=1) + ReLU  → (batch, 32, 16, 16)
Conv2D(in=32, out=10, kernel=3, stride=1, padding=1)                  → (batch, 10, 16, 16)
```

**Output:** Logits (batch, 10, 16, 16) - unnormalized scores for 10 color classes

---

## Loss Function

### Total Loss
```python
L = L_reconstruction + β(epoch) * L_KL
```

### Reconstruction Loss (Cross-Entropy)
```python
L_reconstruction = CrossEntropyLoss(decoder_logits, ground_truth_labels)
# decoder_logits: (batch, 10, 16, 16)
# ground_truth: (batch, 16, 16) integers [0-9]
```

### KL Divergence Loss
```python
# KL(q(z|x) || p(z)) where p(z) = N(0, I)
L_KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
L_KL = L_KL.mean()  # Average over batch and latent dimensions
```

### β-Annealing Schedule
```python
def get_beta(epoch):
    if epoch <= 10:
        # Phase 1: Warm-up (prevent posterior collapse)
        return epoch / 10.0  # Linear: 0.0 → 1.0
    elif epoch <= 30:
        # Phase 2: Encourage disentanglement
        return 1.0 + (epoch - 10) / 20.0  # Linear: 1.0 → 2.0
    else:
        # Phase 3: Refinement
        return 2.0  # Fixed
```

---

## Training Configuration

### Dataset Specifications
- **Source:** Load from `.npz` files in `../datasets/` using `visualization.load_corpus()`
- **Total images:** 100,000 (from Experiment 0.1)
- **Train split:** 80,000 images (80%)
- **Validation split:** 10,000 images (10%)
- **Test split:** 10,000 images (10%)
- **Format:** (N, 16, 16) uint8 arrays, values [0-9]

### Hyperparameters
```yaml
model:
  latent_dim: 10
  encoder_channels: [10, 32, 64, 128]
  decoder_channels: [128, 64, 32, 10]

optimizer:
  type: Adam
  learning_rate: 1e-3
  betas: [0.9, 0.999]
  weight_decay: 0.0

lr_scheduler:
  type: CosineAnnealingLR
  T_max: 50
  eta_min: 1e-5

training:
  batch_size: 128
  max_epochs: 50
  gradient_clip: 1.0  # Prevent exploding gradients

early_stopping:
  patience: 10
  monitor: val_loss
  mode: min

beta_schedule:
  phase1_epochs: 10   # Warm-up: β = 0→1
  phase2_epochs: 20   # Disentanglement: β = 1→2
  phase3_epochs: 20   # Refinement: β = 2 (fixed)
```

### Data Augmentation
```python
transforms:
  - RandomRotation: choices=[0°, 90°, 180°, 270°]
  - RandomHorizontalFlip: p=0.5
  - RandomVerticalFlip: p=0.5
```

---

## Evaluation Metrics

### Primary Metric: Pixel-wise Accuracy

**Definition:**
```python
pred_labels = torch.argmax(decoder_logits, dim=1)  # (batch, 16, 16)
pixel_accuracy = (pred_labels == ground_truth).float().mean()
```

**Target:** ≥ 90%

**Interpretation:** Fraction of pixels with correct color prediction (e.g., 92% = 236/256 pixels correct on average)

### Secondary Monitoring (for debugging)

**KL Divergence per Dimension:**
```python
kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
mean_kl_per_dim = kl_per_dim.mean(dim=0)  # (latent_dim,)
```
**Target:** > 0.05 per dimension (avoid posterior collapse)

**Active Latent Dimensions:**
Count dimensions where KL > 0.05

**Latent Space Visualization:**
- PCA/t-SNE on test set embeddings
- Verify similar images cluster together

---

## Weights & Biases Integration

### Project Configuration
```yaml
wandb:
  project: arc-controllable-generation
  entity: [your-username]
  name: beta-vae-d10-b2.0-run-{run_id}
  tags: [experiment-0.2, beta-vae, encoder-training]
```

### Logged Metrics (Every Training Step)
- `train/loss_total`
- `train/loss_reconstruction`
- `train/loss_kl`
- `train/beta`
- `train/pixel_accuracy`
- `train/active_dims`

### Logged Metrics (Every Validation Epoch)
- `val/loss_total`
- `val/pixel_accuracy`
- `val/kl_per_dim` (histogram)

### Logged Artifacts
- **Checkpoints:** Every 10 epochs + best validation loss
  - `encoder_epoch_{n}.pth`
  - `decoder_epoch_{n}.pth`
  - `vae_best.pth`
- **Images:** Every 5 epochs
  - `reconstructions_epoch_{n}.png` (grid of original vs reconstructed)
  - `samples_epoch_{n}.png` (random samples from z ~ N(0,I))
- **Final Artifacts:**
  - `latent_space_pca.png`
  - `latent_space_tsne.png`
  - `metrics.json`

---

## Model Checkpointing

### Checkpoint Format
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'pixel_accuracy': pixel_accuracy,
    'config': config_dict,
}
```

### Checkpoint Strategy
- Save every 10 epochs: `checkpoints/vae_epoch_{n}.pth`
- Save best validation loss: `checkpoints/vae_best.pth`
- Save final model: `checkpoints/vae_final.pth`
- Keep separate encoder/decoder: `checkpoints/encoder.pth`, `checkpoints/decoder.pth`

---

## Sample Generation

### Sampling from Prior
```python
def generate_samples(decoder, num_samples=64, device='cuda'):
    """Generate samples from standard normal prior."""
    z = torch.randn(num_samples, 10).to(device)
    with torch.no_grad():
        logits = decoder(z)
        pred = torch.argmax(logits, dim=1)  # (num_samples, 16, 16)
    return pred
```

### Reconstruction Visualization
```python
def visualize_reconstructions(model, test_loader, num_pairs=16):
    """Show original vs reconstructed grids."""
    # Sample batch from test set
    # Forward pass through VAE
    # Display original and reconstructed side-by-side
    # Save as grid image
```

---

## Implementation Requirements

### Code Structure
```
project/src/
├── models/
│   ├── __init__.py
│   ├── encoder.py          # Encoder class
│   ├── decoder.py          # Decoder class
│   └── beta_vae.py         # Complete VAE model
├── training/
│   ├── __init__.py
│   ├── train.py            # Main training script
│   ├── evaluate.py         # Evaluation on test set
│   └── generate_samples.py # Sample generation from prior
├── utils/
│   ├── __init__.py
│   ├── data_loader.py      # Dataset class for .npz loading
│   ├── metrics.py          # Pixel accuracy, KL divergence
│   └── visualization.py    # Plot reconstructions, samples
├── config/
│   └── config.yaml         # Training configuration
└── notebooks/
    └── inspect_results.ipynb # Interactive result exploration
```

### Key Dependencies
```python
# Core ML
torch >= 2.0.0
torchvision >= 0.15.0

# Experiment tracking
wandb >= 0.15.0

# Data & visualization
numpy >= 1.24.0
matplotlib >= 3.7.0
seaborn >= 0.12.0

# Utilities
tqdm >= 4.65.0
PyYAML >= 6.0
scikit-learn >= 1.3.0  # For PCA/t-SNE
```

---

## Hardware Requirements

### GPU Specifications
- **Target:** NVIDIA RTX 2080 Ti (11 GB VRAM)
- **Batch size:** 128 fits comfortably in 11 GB
- **Training time:** ~2-3 hours for 50 epochs

### Memory Estimates
- Model parameters: ~2M parameters (~8 MB)
- Batch (128, 16, 16): ~0.5 MB
- Optimizer states: ~16 MB (Adam with momentum)
- Total VRAM usage: ~2-3 GB (plenty of headroom)

### CPU Alternative (Fallback)
- Reduce batch size to 32
- Estimated training time: ~8-12 hours

---

## Success Criteria

### Must Achieve
1. **Pixel-wise accuracy ≥ 90%** on test set
2. **No posterior collapse:** KL > 0.05 per dimension
3. **Valid samples:** Random z ~ N(0,I) decode to recognizable ARC-like grids
4. **Smooth interpolation:** Linear paths in latent space produce valid intermediate grids

### Deliverables
1. Trained checkpoints: `encoder.pth`, `decoder.pth`, `vae_best.pth`
2. W&B logs: Training curves, sample images, metrics
3. Evaluation report: `metrics.json` with final accuracy
4. Sample generation script: Functional and documented

---

## Fallback Actions

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Low pixel accuracy | < 90% | Increase latent_dim to 12-16, retrain |
| Posterior collapse | KL < 0.05 | Reduce β to 1.0-1.5, extend warm-up |
| Training instability | Loss spikes | Add gradient clipping, reduce LR |
| Poor sample quality | Samples invalid | Increase β gradually, inspect decoder |

---

## External Dependencies

This spec requires the following external libraries beyond the existing tech stack:

- **wandb (0.15+)** - Experiment tracking and model checkpointing
  - **Justification:** Required per user specification for tracking training on GPU cluster, enables collaborative experiment monitoring

- **scikit-learn (1.3+)** - For PCA/t-SNE latent space visualization
  - **Justification:** Standard dimensionality reduction for qualitative latent space inspection (already in tech-stack.md)

All other dependencies (PyTorch 2.0+, NumPy, Matplotlib) are already confirmed in the tech stack.
