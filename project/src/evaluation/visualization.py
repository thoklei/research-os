"""
Visualization Utilities - Task 4.2

Visualization tools for β-VAE training and evaluation:
- Training curves
- Reconstruction comparisons
- Latent space visualization
- Latent traversals
- Sample generation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


# ARC color palette (0-9)
ARC_COLORS = [
    '#000000',  # 0: Black
    '#0074D9',  # 1: Blue
    '#FF4136',  # 2: Red
    '#2ECC40',  # 3: Green
    '#FFDC00',  # 4: Yellow
    '#AAAAAA',  # 5: Grey
    '#F012BE',  # 6: Magenta
    '#FF851B',  # 7: Orange
    '#7FDBFF',  # 8: Light Blue
    '#870C25',  # 9: Dark Red
]


def grid_to_color_array(grid: np.ndarray) -> np.ndarray:
    """
    Convert integer grid to RGB color array.

    Args:
        grid: (H, W) array of integers 0-9

    Returns:
        (H, W, 3) RGB array
    """
    from matplotlib.colors import hex2color

    h, w = grid.shape
    rgb = np.zeros((h, w, 3))

    for i in range(h):
        for j in range(w):
            color_idx = int(grid[i, j])
            rgb[i, j] = hex2color(ARC_COLORS[color_idx])

    return rgb


def plot_grid(ax, grid: np.ndarray, title: str = "", show_grid: bool = True):
    """
    Plot a single ARC grid.

    Args:
        ax: Matplotlib axis
        grid: (H, W) integer array
        title: Title for the plot
        show_grid: Whether to show grid lines
    """
    rgb = grid_to_color_array(grid)
    ax.imshow(rgb, interpolation='nearest')

    if show_grid:
        # Add grid lines
        h, w = grid.shape
        for i in range(h + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
        for j in range(w + 1):
            ax.axvline(j - 0.5, color='black', linewidth=0.5, alpha=0.3)

    ax.set_title(title)
    ax.axis('off')


def plot_training_curves(
    log_file: str,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training curves from log file or W&B history.

    Args:
        log_file: Path to training log JSON file
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    # Load training history
    with open(log_file, 'r') as f:
        history = json.load(f)

    # Extract metrics
    epochs = history.get('epochs', [])
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    train_acc = history.get('train_pixel_acc', [])
    val_acc = history.get('val_pixel_acc', [])
    kl_per_dim = history.get('kl_per_dim', [])
    beta = history.get('beta', [])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('β-VAE Training Curves', fontsize=16, fontweight='bold')

    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label='Train', linewidth=2)
    ax.plot(epochs, val_loss, label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, train_acc, label='Train', linewidth=2)
    ax.plot(epochs, val_acc, label='Validation', linewidth=2)
    ax.axhline(0.90, color='red', linestyle='--', label='Target (90%)', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pixel Accuracy')
    ax.set_title('Pixel-wise Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # KL divergence per dimension
    ax = axes[1, 0]
    ax.plot(epochs, kl_per_dim, linewidth=2, color='purple')
    ax.axhline(0.05, color='red', linestyle='--', label='Collapse threshold', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL / dim')
    ax.set_title('KL Divergence per Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Beta annealing schedule
    ax = axes[1, 1]
    ax.plot(epochs, beta, linewidth=2, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('β')
    ax.set_title('β Annealing Schedule')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVE] Training curves saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_reconstructions(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 8,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot original vs reconstructed grids side-by-side with metrics.

    Args:
        model: Trained β-VAE model
        data_loader: Data loader
        device: Device to run on
        num_samples: Number of samples to visualize
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    model.eval()

    # Get a batch
    batch = next(iter(data_loader))[:num_samples]
    batch = batch.to(device)

    # Forward pass
    with torch.no_grad():
        recon_logits, mu, logvar = model(batch)
        recon = recon_logits.argmax(dim=1)

    # Move to CPU
    original = batch.cpu().numpy()
    reconstructed = recon.cpu().numpy()
    recon_logits_cpu = recon_logits.cpu()

    # Create figure: 2 columns (original, reconstructed) x num_samples rows
    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 2.5 * num_samples))
    fig.suptitle('Reconstruction Validation: Input vs Output',
                 fontsize=16, fontweight='bold', y=0.995)

    # Handle single sample case
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Compute per-sample metrics
        sample_correct = (reconstructed[i] == original[i]).sum()
        sample_total = original[i].size
        sample_accuracy = sample_correct / sample_total

        # Compute per-sample loss
        sample_recon_logits = recon_logits_cpu[i:i+1]
        sample_target = torch.from_numpy(original[i:i+1])
        sample_loss = torch.nn.functional.cross_entropy(
            sample_recon_logits.view(-1, 10),
            sample_target.view(-1)
        ).item()

        # Left column: Original
        plot_grid(axes[i, 0], original[i],
                 title=f"Sample {i+1}: Input",
                 show_grid=True)

        # Right column: Reconstructed with metrics
        plot_grid(axes[i, 1], reconstructed[i],
                 title=f"Reconstruction (Acc: {sample_accuracy:.1%}, Loss: {sample_loss:.3f})",
                 show_grid=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVE] Reconstructions saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_latent_space(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 1000,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize latent space using PCA or t-SNE.

    Args:
        model: Trained β-VAE model
        data_loader: Data loader
        device: Device to run on
        num_samples: Number of samples to encode
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    from sklearn.decomposition import PCA

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

    # Concatenate
    latent_codes = torch.cat(latent_codes, dim=0)[:num_samples].numpy()

    # Apply PCA to 2D
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_codes)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1],
                        alpha=0.5, s=10, c=range(num_samples),
                        cmap='viridis')

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title(f'Latent Space Visualization (PCA)\n'
                 f'Explained variance: {pca.explained_variance_ratio_.sum():.2%}')
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label='Sample index')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVE] Latent space plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_latent_traversals(
    model: nn.Module,
    device: torch.device,
    latent_dim: int,
    num_steps: int = 7,
    range_scale: float = 2.0,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot latent traversals for each dimension.

    Shows how the output changes when traversing each latent dimension
    while keeping others fixed at 0.

    Args:
        model: Trained β-VAE model
        device: Device to run on
        latent_dim: Number of latent dimensions
        num_steps: Number of steps for traversal
        range_scale: Scale of traversal range (± range_scale)
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    model.eval()

    # Create traversal values
    traversal_values = np.linspace(-range_scale, range_scale, num_steps)

    # Show only first few dimensions to keep plot manageable
    dims_to_show = min(latent_dim, 5)

    fig, axes = plt.subplots(dims_to_show, num_steps,
                            figsize=(2 * num_steps, 2 * dims_to_show))
    fig.suptitle('Latent Dimension Traversals', fontsize=14, fontweight='bold')

    with torch.no_grad():
        for dim_idx in range(dims_to_show):
            for step_idx, value in enumerate(traversal_values):
                # Create latent code (all zeros except one dimension)
                z = torch.zeros(1, latent_dim).to(device)
                z[0, dim_idx] = value

                # Decode
                recon_logits = model.decoder(z)
                recon = recon_logits.argmax(dim=1).cpu().numpy()[0]

                # Plot
                if dims_to_show == 1:
                    ax = axes[step_idx]
                else:
                    ax = axes[dim_idx, step_idx]

                plot_grid(ax, recon,
                         title=f"z{dim_idx}={value:.1f}" if dim_idx == 0 else "",
                         show_grid=False)

    # Add dimension labels
    for dim_idx in range(dims_to_show):
        if dims_to_show == 1:
            axes[0].set_ylabel(f"Dim 0", fontsize=10, fontweight='bold')
        else:
            axes[dim_idx, 0].set_ylabel(f"Dim {dim_idx}", fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVE] Latent traversals saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def save_generated_samples(
    model: nn.Module,
    device: torch.device,
    latent_dim: int,
    num_samples: int = 16,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Generate and save random samples from the model.

    Args:
        model: Trained β-VAE model
        device: Device to run on
        latent_dim: Number of latent dimensions
        num_samples: Number of samples to generate
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    model.eval()

    # Sample from prior N(0, I)
    z = torch.randn(num_samples, latent_dim).to(device)

    # Decode
    with torch.no_grad():
        recon_logits = model.decoder(z)
        samples = recon_logits.argmax(dim=1).cpu().numpy()

    # Plot in grid
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size,
                            figsize=(2 * grid_size, 2 * grid_size))
    fig.suptitle(f'Generated Samples from N(0, I)', fontsize=14, fontweight='bold')

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j

            if grid_size == 1:
                ax = axes
            elif grid_size == 2 and i == 0 and j == 0:
                ax = axes if num_samples == 1 else axes[i, j]
            else:
                ax = axes[i, j] if grid_size > 1 else axes[idx]

            if idx < num_samples:
                plot_grid(ax, samples[idx], title="", show_grid=False)
            else:
                ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVE] Generated samples saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
