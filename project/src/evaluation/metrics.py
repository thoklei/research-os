"""
Evaluation Metrics - Task 4.1

Comprehensive metrics for β-VAE evaluation:
- Reconstruction quality (pixel accuracy, per-class accuracy)
- Latent space quality (KL divergence, posterior collapse detection)
- Disentanglement metrics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_reconstruction_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_colors: int = 10
) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics.

    Args:
        model: Trained β-VAE model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        num_colors: Number of color classes

    Returns:
        Dictionary with reconstruction metrics:
        - pixel_accuracy: Overall pixel-wise accuracy
        - per_class_accuracy: Accuracy for each color class
        - recon_loss: Average reconstruction loss
    """
    model.eval()

    total_correct = 0
    total_pixels = 0
    total_recon_loss = 0.0
    num_batches = 0

    # Per-class accuracy tracking
    class_correct = torch.zeros(num_colors)
    class_total = torch.zeros(num_colors)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing reconstruction metrics", leave=False):
            batch = batch.to(device)
            batch_size = batch.size(0)

            # Forward pass
            recon_logits, mu, logvar = model(batch)

            # Reconstruction loss
            recon_loss = nn.functional.cross_entropy(
                recon_logits.view(-1, num_colors),
                batch.view(-1),
                reduction='mean'
            )
            total_recon_loss += recon_loss.item()

            # Predictions
            pred = recon_logits.argmax(dim=1)

            # Overall accuracy
            correct = (pred == batch).sum().item()
            total_correct += correct
            total_pixels += batch.numel()

            # Per-class accuracy
            for c in range(num_colors):
                mask = (batch == c)
                class_total[c] += mask.sum().item()
                class_correct[c] += ((pred == batch) & mask).sum().item()

            num_batches += 1

    # Compute final metrics
    pixel_accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0
    avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0.0

    # Per-class accuracy
    per_class_acc = {}
    for c in range(num_colors):
        if class_total[c] > 0:
            per_class_acc[f"class_{c}_acc"] = (class_correct[c] / class_total[c]).item()
        else:
            per_class_acc[f"class_{c}_acc"] = 0.0

    return {
        'pixel_accuracy': pixel_accuracy,
        'recon_loss': avg_recon_loss,
        **per_class_acc,
    }


def compute_latent_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute latent space quality metrics.

    Args:
        model: Trained β-VAE model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with latent metrics:
        - kl_divergence: Average KL divergence
        - kl_per_dim: KL divergence per latent dimension
        - active_dims: Number of active latent dimensions
        - posterior_collapse: Boolean indicating posterior collapse
    """
    model.eval()

    total_kl = 0.0
    kl_per_dim_sum = torch.zeros(model.latent_dim).to(device)
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing latent metrics", leave=False):
            batch = batch.to(device)

            # Forward pass
            _, mu, logvar = model(batch)

            # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            total_kl += kl_div.mean().item()

            # KL per dimension
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)
            kl_per_dim_sum += kl_per_dim

            num_batches += 1

    # Average metrics
    avg_kl = total_kl / num_batches if num_batches > 0 else 0.0
    avg_kl_per_dim = kl_per_dim_sum / num_batches if num_batches > 0 else torch.zeros(model.latent_dim)

    # Active dimensions (KL > threshold)
    active_threshold = 0.05
    active_dims = (avg_kl_per_dim > active_threshold).sum().item()

    # Posterior collapse detection
    posterior_collapse = avg_kl_per_dim.mean().item() < 0.05

    return {
        'kl_divergence': avg_kl,
        'kl_per_dim': avg_kl_per_dim.mean().item(),
        'active_dims': active_dims,
        'posterior_collapse': posterior_collapse,
        'kl_per_dim_std': avg_kl_per_dim.std().item(),
    }


def compute_disentanglement_score(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_samples: int = 1000
) -> Dict[str, float]:
    """
    Compute disentanglement metrics using variance-based approach.

    This measures how well individual latent dimensions correspond to
    independent factors of variation in the data.

    Args:
        model: Trained β-VAE model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        num_samples: Number of samples to use for evaluation

    Returns:
        Dictionary with disentanglement metrics:
        - latent_variance: Variance of each latent dimension
        - disentanglement_score: Overall disentanglement score
    """
    model.eval()

    # Collect latent codes
    latent_codes = []
    samples_collected = 0

    with torch.no_grad():
        for batch in data_loader:
            if samples_collected >= num_samples:
                break

            batch = batch.to(device)
            _, mu, _ = model(batch)
            latent_codes.append(mu.cpu())
            samples_collected += mu.size(0)

    # Concatenate all latent codes
    latent_codes = torch.cat(latent_codes, dim=0)[:num_samples]

    # Compute variance per dimension
    latent_variance = latent_codes.var(dim=0)

    # Disentanglement score: normalized variance
    # Higher variance in individual dimensions suggests better disentanglement
    disentanglement = latent_variance.std() / (latent_variance.mean() + 1e-8)

    return {
        'disentanglement_score': disentanglement.item(),
        'latent_variance_mean': latent_variance.mean().item(),
        'latent_variance_std': latent_variance.std().item(),
    }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_colors: int = 10,
    compute_disentanglement: bool = True,
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.

    Computes all evaluation metrics:
    - Reconstruction quality
    - Latent space quality
    - Disentanglement (optional)

    Args:
        model: Trained β-VAE model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        num_colors: Number of color classes
        compute_disentanglement: Whether to compute disentanglement metrics

    Returns:
        Dictionary with all evaluation metrics
    """
    print("Evaluating model...")

    # Reconstruction metrics
    recon_metrics = compute_reconstruction_metrics(model, data_loader, device, num_colors)
    print(f"  Reconstruction - Pixel Acc: {recon_metrics['pixel_accuracy']:.3f}, "
          f"Loss: {recon_metrics['recon_loss']:.4f}")

    # Latent metrics
    latent_metrics = compute_latent_metrics(model, data_loader, device)
    print(f"  Latent Space - KL/dim: {latent_metrics['kl_per_dim']:.4f}, "
          f"Active dims: {latent_metrics['active_dims']}/{model.latent_dim}")

    # Combine metrics
    all_metrics = {**recon_metrics, **latent_metrics}

    # Disentanglement metrics (optional, more expensive)
    if compute_disentanglement:
        disentangle_metrics = compute_disentanglement_score(model, data_loader, device)
        print(f"  Disentanglement - Score: {disentangle_metrics['disentanglement_score']:.4f}")
        all_metrics.update(disentangle_metrics)

    return all_metrics
