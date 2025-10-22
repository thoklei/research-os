# Mode Collapse Issue - Training Analysis

## Problem Summary

The trained β-VAE model is predicting class 0 (black) for 100% of pixels, resulting in completely black reconstructions despite reporting 93% accuracy.

## Root Cause Analysis

### 1. Class Imbalance
- ARC grids are ~93% black (class 0) pixels
- Model learned to always predict class 0 to minimize loss
- This gives high accuracy (93%) but completely wrong reconstructions

### 2. Logit Analysis
From debug output:
```
Average logits per class:
  Class 0:  3.30   ← MUCH higher
  Class 1: -1.69
  Class 2: -1.70
  Class 3: -1.67
  ...
  Class 9: -1.67
```

Example pixel that should be green (class 3):
- Class 0 logit:  3.27 → 94.8% probability
- Class 3 logit: -1.83 →  0.6% probability

### 3. Posterior Collapse
- KL divergence per dimension: 0.0000
- Latent μ values: ~0.0003 (near zero)
- No active dimensions (0/10)
- Latent space has collapsed

### 4. Training Issues
- Only 5 epochs (quick test)
- β-annealing started at 0.1 (epochs 1-5 use β=0.1-0.5)
- With low β, model optimized reconstruction loss only
- Found trivial solution: predict most common class

## Why Metrics Were Misleading

Per-class accuracy showed:
```
Color 0: 100.000%  ← Predicts everything as black
Color 1:   0.000%  ← Never predicts other colors
Color 2:   0.000%
...
```

Overall accuracy: 93.1% (matches percentage of black pixels in dataset)

## Solutions

### Short-term (Fix current training):
1. **Weighted Cross-Entropy**: Add class weights inversely proportional to frequency
2. **Focal Loss**: Down-weight easy examples (black pixels)
3. **Longer Training**: Run full 50 epochs with proper β-annealing (β=0→1→2)

### Recommended Fix:
```python
# Calculate class weights from training data
class_counts = ...  # Count occurrences of each color
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * 10  # Normalize

# Use in loss
recon_loss = F.cross_entropy(
    recon_logits,
    x.long(),
    weight=torch.tensor(class_weights),
    reduction='mean'
)
```

### Long-term:
- Monitor per-class accuracy during training (not just overall)
- Add class balance metrics to W&B logging
- Consider data augmentation that changes color distributions
- Use stratified sampling to ensure color balance in batches

## Status

- Training code: ✓ Implemented
- Evaluation code: ✓ Implemented
- Issue identified: ✓ Mode collapse due to class imbalance
- Fix needed: Weighted loss or focal loss
