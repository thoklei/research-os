# Experiment 0.2: β-VAE Training - Specification Summary

**Status**: Ready for implementation
**Objective**: Train a β-VAE to encode/decode 16×16 ARC grids with maximum pixel-wise reconstruction accuracy

---

## Model Choice: β-VAE

**Rationale**:
- Regularized latent space for compositional operations
- Generative capability (sample new grids from z ~ N(0,I))
- Continuous latent space supports smooth transformations
- Well-established training techniques

See `encoder-architecture-analysis.md` for detailed comparison vs. alternatives.

---

## Architecture

### Encoder: 16×16 Grid → d=10 latent
```
Input: (batch, 16, 16) integers [0-9]
→ Embed to one-hot: (batch, 10, 16, 16)
→ Conv2D(10→32, k=3, s=1) + ReLU
→ Conv2D(32→64, k=3, s=2) + ReLU  # 8×8
→ Conv2D(64→128, k=3, s=2) + ReLU # 4×4
→ Flatten → Dense(2048→128) + ReLU
→ μ: Dense(128→10), σ: Dense(128→10) + Softplus
```

### Latent: z ~ N(μ, σ²), d=10 dimensions

### Decoder: d=10 latent → 16×16 Grid
```
Input: z (batch, 10)
→ Dense(10→128) + ReLU
→ Dense(128→2048) + ReLU
→ Reshape(128, 4, 4)
→ ConvTranspose2D(128→64, k=4, s=2) + ReLU # 8×8
→ ConvTranspose2D(64→32, k=4, s=2) + ReLU  # 16×16
→ Conv2D(32→10, k=3, s=1)
Output: (batch, 10, 16, 16) logits
```

### Loss
```python
L = CrossEntropyLoss(logits, labels) + β(epoch) * KL(q(z|x) || N(0,I))
```

---

## Training Configuration

**Dataset**:
- Train: 80,000 images (from existing 100K corpus)
- Val: 10,000 images
- Test: 10,000 images
- Format: `.npz` files, load with `visualization.load_corpus()`

**Hyperparameters**:
```yaml
latent_dim: 10
batch_size: 128
learning_rate: 1e-3
optimizer: Adam
lr_schedule: CosineAnnealingLR(T_max=50, eta_min=1e-5)
max_epochs: 50
early_stopping_patience: 10
```

**β-Annealing Schedule**:
```python
Epochs 1-10:  β = 0.0 → 1.0  (linear warm-up)
Epochs 11-30: β = 1.0 → 2.0  (encourage disentanglement)
Epochs 31-50: β = 2.0         (fixed)
```

**Data Augmentation**:
- Random rotation: {0°, 90°, 180°, 270°}
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)

---

## Evaluation Metric: Pixel-wise Accuracy

**Definition**:
```python
pred_labels = torch.argmax(decoder_logits, dim=1)  # (batch, 16, 16)
pixel_accuracy = (pred_labels == ground_truth).float().mean()
```

**Interpretation**: Fraction of correctly predicted pixels
- Example: 92.3% = on average, 236 out of 256 pixels correct per image

**Target**: **≥ 90%** pixel-wise accuracy on test set

---

## Success Criteria

**Primary Goal**: Maximize pixel-wise accuracy on test set (target ≥90%)

**Secondary Monitoring** (for debugging, not pass/fail):
- KL divergence > 0.05 per dimension (detect posterior collapse)
- Visual inspection: random samples from z ~ N(0,I) look valid

---

## Fallback Strategy

| Pixel-wise Accuracy | Action |
|---------------------|--------|
| ≥ 90% | ✅ Success - proceed to Experiment 0.3 |
| 85-90% | Increase latent dim to d=12-16, retrain |
| 80-85% | Reduce β to 1.0 (standard VAE), retrain |
| < 80% | Consider Slot Attention (fallback architecture) |

---

## Implementation Plan

**Recommended**: Adapt PyTorch VAE reference implementation

**Steps**:
1. Start with: https://github.com/pytorch/examples/tree/main/vae
2. Modify for 16×16×10 grids (not 28×28 grayscale)
3. Change loss: BCE → CrossEntropy
4. Add β parameter and annealing schedule
5. Integrate with existing `load_corpus()` data loading
6. Train for 50 epochs (~2-3 hours on GPU)

---

## Code Structure

```
project/src/
├── models/
│   └── beta_vae.py              # Encoder, Decoder, VAE classes
├── training/
│   ├── train_encoder.py         # Training loop
│   └── evaluate_encoder.py      # Test set evaluation
├── config/
│   └── encoder_config.yaml      # Hyperparameters
└── utils/
    └── data_loader.py           # Dataset wrapper for .npz
```

---

## Expected Outputs

**Checkpoints**:
- `checkpoints/encoder.pth` - trained encoder weights
- `checkpoints/decoder.pth` - trained decoder weights
- `checkpoints/vae_best.pth` - full model (best val loss)

**Results**:
- `results/training_curves.png` - loss and accuracy over epochs
- `results/reconstructions.png` - test set: original vs reconstructed
- `results/samples.png` - random samples from z ~ N(0,I)
- `results/metrics.json` - final test accuracy

**Metric Report**:
```
Test Set Evaluation (10,000 images):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pixel-wise Accuracy:  92.3%  ✓
Mean KL Divergence:   0.18
Active Dimensions:    9.8/10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Architecture + data loading | Working model code |
| 2 | Training loop + β-annealing | Training script running |
| 3 | Train (50 epochs, ~2-3h) | Trained checkpoints |
| 4 | Evaluate + document | Metrics, plots, analysis |

---

## Ready for Engineer Agent

This specification contains everything needed for implementation:
- ✅ Exact architecture (layer-by-layer specs)
- ✅ Precise loss function and β-schedule
- ✅ Dataset location and format
- ✅ Clear success criterion (≥90% pixel accuracy)
- ✅ Fallback thresholds and actions
- ✅ Implementation strategy
- ✅ Code structure

Next step: Use `/new-artifact` or similar to generate implementation spec.
