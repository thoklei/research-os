"""
Loss Functions for β-VAE Training

Includes Focal Loss to handle severe class imbalance in ARC grids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and example difficulty imbalance.

    From: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    https://arxiv.org/abs/1708.02002

    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Where:
    - p_t: Predicted probability of true class
    - γ (gamma): Focusing parameter (γ=0 → standard CE, γ=2 typical)
    - α_t: Class-specific weighting factor (optional)

    The (1 - p_t)^γ term down-weights easy examples:
    - Easy examples (p_t=0.99): Reduced by ~1000x with γ=2
    - Hard examples (p_t=0.60): Reduced by ~6x with γ=2

    This forces the model to focus on hard examples and minority classes.

    Args:
        gamma: Focusing parameter (default: 2.0 from original paper)
        alpha: Class weights as tensor of shape (num_classes,) or None
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                raise TypeError(f"alpha must be a torch.Tensor, got {type(alpha)}")
            if alpha.ndim != 1:
                raise ValueError(f"alpha must be 1D, got shape {alpha.shape}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Predicted logits, shape (N, C) or (N, C, H, W) for images
            targets: Ground truth labels, shape (N,) or (N, H, W) for images

        Returns:
            Focal loss (scalar if reduction='mean'/'sum', tensor if reduction='none')
        """
        # Get probabilities
        # For image classification (N, C, H, W), we need to handle spatially
        if logits.ndim == 4:  # (N, C, H, W)
            # Reshape to (N*H*W, C) for easier processing
            N, C, H, W = logits.shape
            logits_flat = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
            targets_flat = targets.view(-1)

            # Compute probabilities
            probs = F.softmax(logits_flat, dim=1)

            # Get probability of true class for each pixel
            # Shape: (N*H*W,)
            p_t = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

        elif logits.ndim == 2:  # (N, C)
            probs = F.softmax(logits, dim=1)
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            targets_flat = targets
        else:
            raise ValueError(f"logits must be 2D or 4D, got shape {logits.shape}")

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Cross-entropy loss (without reduction)
        if logits.ndim == 4:
            # Use original 4D logits for cross_entropy (more efficient)
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            ce_loss = ce_loss.view(-1)  # Flatten to (N*H*W,)
        else:
            ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Focal loss = focal_weight * ce_loss
        focal_loss = focal_weight * ce_loss

        # Apply alpha balancing if provided
        if self.alpha is not None:
            # Move alpha to same device as targets
            if self.alpha.device != targets_flat.device:
                self.alpha = self.alpha.to(targets_flat.device)

            alpha_t = self.alpha[targets_flat]
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


def compute_class_weights(
    dataset: torch.utils.data.Dataset,
    num_classes: int = 10,
    method: str = 'inverse',
    smooth: float = 1.0,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced datasets.

    Args:
        dataset: Dataset to compute statistics from
        num_classes: Number of classes
        method: Weighting method:
            - 'inverse': weight = 1 / (count + smooth)
              Full inverse weighting - can create extreme ratios (e.g., 123:1)

            - 'sqrt_inverse': weight = 1 / sqrt(count + smooth)
              Square root smoothing - reduces extreme weight ratios while maintaining
              class imbalance awareness. Transforms extreme ratios (123:1 → ~11:1).
              Recommended for use with Focal Loss to avoid pathological loss landscapes.

            - 'balanced': weight = n_samples / (n_classes * count)
              Sklearn-style balanced weights - equivalent to inverse but with
              different normalization factor.

        smooth: Smoothing factor to avoid division by zero (default: 1.0)
        normalize: Whether to normalize weights to sum to num_classes

    Returns:
        Class weights tensor of shape (num_classes,)

    Example:
        For ARC grids with ~93% background (class 0) and 7% objects:
        - inverse method: weight_ratio ≈ 123:1 (extreme)
        - sqrt_inverse method: weight_ratio ≈ 11:1 (moderate)
        - balanced method: weight_ratio ≈ 123:1 (same as inverse)
    """
    # Count occurrences of each class
    class_counts = torch.zeros(num_classes, dtype=torch.long)

    print(f"Computing class weights from {len(dataset)} samples...")
    for i, sample in enumerate(dataset):
        if isinstance(sample, (tuple, list)):
            sample = sample[0]  # Handle (data, label) tuples

        # Count each color
        for c in range(num_classes):
            class_counts[c] += (sample == c).sum().item()

        # Progress indicator
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{len(dataset)} samples...")

    print(f"  Class counts: {class_counts.tolist()}")

    # Compute weights based on method
    if method == 'inverse':
        weights = 1.0 / (class_counts.float() + smooth)
    elif method == 'sqrt_inverse':
        weights = 1.0 / torch.sqrt(class_counts.float() + smooth)
    elif method == 'balanced':
        n_samples = class_counts.sum().item()
        weights = n_samples / (num_classes * (class_counts.float() + smooth))
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize weights to sum to num_classes
    if normalize:
        weights = weights / weights.sum() * num_classes

    print(f"  Class weights: {weights.tolist()}")
    print(f"  Weight ratio (max/min): {weights.max().item() / weights.min().item():.2f}x")

    return weights


def save_class_weights(weights: torch.Tensor, filepath: str):
    """Save class weights to file."""
    torch.save({'class_weights': weights}, filepath)
    print(f"[SAVE] Class weights saved to: {filepath}")


def load_class_weights(filepath: str) -> torch.Tensor:
    """Load class weights from file."""
    data = torch.load(filepath)
    return data['class_weights']
