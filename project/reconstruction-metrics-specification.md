# Reconstruction Accuracy Metrics for Discrete Grid VAE

**Date**: 2025-10-22
**Context**: Experiment 0.2 - β-VAE evaluation metrics

---

## Problem: Ambiguity in "Reconstruction Accuracy"

For discrete 16×16 grids with 10-color palette, we need precise metric definitions.

---

## Recommended Metrics

### 1. **Pixel-Wise Accuracy** (Primary Metric)

**Definition**: Fraction of pixels where predicted color matches ground truth

```python
pixel_accuracy = (num_correct_pixels) / (total_pixels)
                = (num_correct_pixels) / (16 × 16)
                = (num_correct_pixels) / 256
```

**Computation**:
```python
# Decoder outputs: (batch, 10, 16, 16) logits
# Ground truth: (batch, 16, 16) integer labels [0-9]

pred_labels = torch.argmax(decoder_output, dim=1)  # (batch, 16, 16)
correct = (pred_labels == ground_truth).float()    # (batch, 16, 16)
pixel_accuracy = correct.mean()                    # scalar in [0, 1]
```

**Example**:
- Grid has 256 pixels
- 245 pixels correctly predicted → accuracy = 245/256 = 95.7%

**Target**: ≥90% pixel-wise accuracy on test set

---

### 2. **Perfect Reconstruction Rate** (Secondary Metric)

**Definition**: Fraction of images where ALL pixels are correctly predicted

```python
perfect_recon_rate = (num_perfectly_reconstructed_images) / (total_images)
```

**Computation**:
```python
pred_labels = torch.argmax(decoder_output, dim=1)  # (batch, 16, 16)
perfect_match = (pred_labels == ground_truth).all(dim=[1,2])  # (batch,) bool
perfect_recon_rate = perfect_match.float().mean()  # scalar in [0, 1]
```

**Example**:
- 1000 test images
- 720 images perfectly reconstructed (all 256 pixels correct)
- Perfect reconstruction rate = 720/1000 = 72%

**Target**: ≥70% perfect reconstruction rate on test set

---

### 3. **Color-Weighted Accuracy** (Optional)

**Definition**: Pixel accuracy weighted by color frequency (addresses class imbalance)

**Rationale**: Background (color 0) dominates (~70-80% of pixels), so pixel accuracy can be high even if object colors are wrong.

```python
# Compute per-color accuracy
for color in range(10):
    mask = (ground_truth == color)
    if mask.sum() > 0:
        color_accuracy[color] = (pred_labels[mask] == color).float().mean()

# Weighted average (balanced across colors)
balanced_accuracy = color_accuracy.mean()
```

**Target**: ≥85% balanced accuracy (if background dominates standard accuracy)

---

## Recommended Success Criteria (Precise)

### For Experiment 0.2 β-VAE:

| Metric | Target | Interpretation |
|--------|--------|----------------|
| **Pixel-wise accuracy** | ≥90% | On average, 230+ out of 256 pixels correct |
| **Perfect reconstruction rate** | ≥70% | 7 out of 10 test images perfectly reconstructed |
| **Per-object accuracy** | ≥85% | Object pixels (non-background) correctly predicted |

### Failure Thresholds:

| Metric | Failure Threshold | Action |
|--------|-------------------|--------|
| Pixel-wise accuracy | <85% | Try increasing latent dim to d=12-16 |
| Perfect reconstruction rate | <50% | Check for systematic errors (e.g., color confusion) |
| KL divergence | <0.05 per dim | Posterior collapse - reduce β or adjust warm-up |

---

## Implementation: Evaluation Loop

```python
def evaluate_reconstruction(model, test_loader, device):
    """
    Evaluate reconstruction metrics on test set.

    Returns:
        dict with:
            - pixel_accuracy: float [0, 1]
            - perfect_recon_rate: float [0, 1]
            - per_color_accuracy: dict {color: accuracy}
            - mean_kl_divergence: float
    """
    model.eval()

    total_pixels = 0
    correct_pixels = 0
    perfect_images = 0
    total_images = 0

    color_correct = {i: 0 for i in range(10)}
    color_total = {i: 0 for i in range(10)}

    kl_divs = []

    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)  # (batch, 16, 16) or (batch, 1, 16, 16)

            # Forward pass
            recon_logits, mu, logvar = model(x)  # recon: (batch, 10, 16, 16)

            # Predicted colors
            pred = torch.argmax(recon_logits, dim=1)  # (batch, 16, 16)

            # Pixel-wise accuracy
            correct = (pred == x.squeeze(1))  # (batch, 16, 16) bool
            correct_pixels += correct.sum().item()
            total_pixels += x.numel()

            # Perfect reconstruction rate
            perfect = correct.all(dim=[1, 2])  # (batch,) bool
            perfect_images += perfect.sum().item()
            total_images += x.size(0)

            # Per-color accuracy
            for color in range(10):
                mask = (x.squeeze(1) == color)
                if mask.sum() > 0:
                    color_correct[color] += (pred[mask] == color).sum().item()
                    color_total[color] += mask.sum().item()

            # KL divergence
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_divs.append(kl.mean().item())

    # Aggregate metrics
    pixel_accuracy = correct_pixels / total_pixels
    perfect_recon_rate = perfect_images / total_images

    per_color_accuracy = {
        color: color_correct[color] / color_total[color]
        if color_total[color] > 0 else 0.0
        for color in range(10)
    }

    mean_kl = sum(kl_divs) / len(kl_divs)

    return {
        'pixel_accuracy': pixel_accuracy,
        'perfect_recon_rate': perfect_recon_rate,
        'per_color_accuracy': per_color_accuracy,
        'mean_kl_divergence': mean_kl,
    }
```

---

## Reporting Format

```
Evaluation on Test Set (10,000 images):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pixel-wise Accuracy:       92.3%  ✓ (target: ≥90%)
Perfect Reconstruction:    74.1%  ✓ (target: ≥70%)
Mean KL Divergence:        0.18   ✓ (target: >0.1)

Per-Color Accuracy:
  Color 0 (background):    95.2%
  Color 1 (blue):          88.7%
  Color 2 (red):           89.3%
  Color 3 (green):         87.1%
  Color 4 (yellow):        86.9%
  Color 5 (gray):          88.4%
  Color 6 (magenta):       87.8%
  Color 7 (orange):        88.2%
  Color 8 (cyan):          89.0%
  Color 9 (maroon):        87.5%

Mean (balanced):           88.8%  ✓ (target: ≥85%)

Latent Space:
  Active dimensions:       9.8/10
  Latent std deviation:    0.94 (close to N(0,1))
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Edge Cases and Clarifications

### 1. Input Format
**Question**: Are inputs one-hot encoded or integer labels?

**Answer**:
- **During training**: One-hot encode for loss computation
  - Input: (batch, 16, 16) integers → (batch, 10, 16, 16) one-hot
  - Decoder output: (batch, 10, 16, 16) logits
  - Loss: CrossEntropyLoss expects (batch, 10, 16, 16) logits and (batch, 16, 16) labels

- **During evaluation**: Use integer labels directly
  - Ground truth: (batch, 16, 16) integers [0-9]
  - Predictions: argmax(logits, dim=1) → (batch, 16, 16) integers
  - Accuracy: (pred == ground_truth).float().mean()

### 2. Background vs. Object Accuracy
**Question**: Should we weight background and objects equally?

**Answer**: Report both:
- **Pixel-wise accuracy**: Includes all pixels (background-heavy)
- **Object-only accuracy**: Mask out background (color 0), compute accuracy on remaining pixels
- **Balanced accuracy**: Average per-color accuracy (each color weighted equally)

### 3. Partial Credit
**Question**: Do we give partial credit for close colors?

**Answer**: **No** - use exact match only
- Rationale: Colors are discrete, semantic categories (not continuous)
- Color 1 (blue) vs Color 2 (red) are equally wrong
- No meaningful distance metric between ARC colors

---

## Summary for Spec

**Use these exact targets in Experiment 0.2 specification:**

```yaml
success_criteria:
  pixel_wise_accuracy: ">= 90%"  # 230+ / 256 pixels correct on average
  perfect_reconstruction_rate: ">= 70%"  # 7/10 images fully correct
  balanced_color_accuracy: ">= 85%"  # Avg accuracy across 10 colors
  mean_kl_divergence: "> 0.1"  # No posterior collapse (per dimension)

fallback_thresholds:
  pixel_wise_accuracy: "< 85%"  # Trigger: increase latent dim to d=12-16
  perfect_reconstruction_rate: "< 50%"  # Trigger: investigate systematic errors
  mean_kl_divergence: "< 0.05"  # Trigger: adjust β schedule or reduce β
```

These metrics are:
- **Precise**: Clear computational definition
- **Interpretable**: Maps to real quality (e.g., "74% of images perfect")
- **Actionable**: Thresholds trigger specific responses
- **Standard**: Aligns with VAE literature for discrete data
