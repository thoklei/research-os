"""
Encoder for β-VAE - Task 1.2

Encoder architecture: 16×16 Grid → d=10 latent (μ, logvar)

Architecture:
    Input: (batch, 16, 16) integer labels [0-9]
    → One-hot encoding: (batch, 10, 16, 16)
    → Conv2D(10→32, k=3, s=1, p=1) + ReLU
    → Conv2D(32→64, k=3, s=2, p=1) + ReLU  # 8×8
    → Conv2D(64→128, k=3, s=2, p=1) + ReLU # 4×4
    → Flatten: (batch, 128*4*4=2048)
    → Dense(2048→128) + ReLU
    → μ: Dense(128→latent_dim)
    → logvar: Dense(128→latent_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder for β-VAE that maps 16×16 ARC grids to latent distribution parameters.

    Args:
        latent_dim (int): Dimension of latent space (default: 10)
        num_colors (int): Number of color classes (default: 10 for ARC palette)
    """

    def __init__(self, latent_dim=10, num_colors=10):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.num_colors = num_colors

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=num_colors,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Fully connected layers
        # After conv3: (batch, 128, 4, 4) → flatten → (batch, 2048)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)

        # Latent distribution parameter heads
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        """
        Forward pass through encoder.

        Args:
            x (torch.Tensor): Input grid with shape (batch, 16, 16)
                             Integer labels in range [0, num_colors-1]

        Returns:
            tuple: (mu, logvar) where:
                - mu: Mean of latent distribution (batch, latent_dim)
                - logvar: Log variance of latent distribution (batch, latent_dim)
        """
        # Input validation and one-hot encoding
        batch_size = x.size(0)

        # Convert integer labels to one-hot encoding
        # x: (batch, 16, 16) → x_onehot: (batch, num_colors, 16, 16)
        x_onehot = F.one_hot(x.long(), num_classes=self.num_colors)  # (batch, 16, 16, num_colors)
        x_onehot = x_onehot.permute(0, 3, 1, 2).float()  # (batch, num_colors, 16, 16)

        # Convolutional layers
        h = F.relu(self.conv1(x_onehot))  # (batch, 32, 16, 16)
        h = F.relu(self.conv2(h))          # (batch, 64, 8, 8)
        h = F.relu(self.conv3(h))          # (batch, 128, 4, 4)

        # Flatten
        h = h.reshape(batch_size, -1)  # (batch, 128*4*4=2048)

        # Fully connected layer
        h = F.relu(self.fc1(h))  # (batch, 128)

        # Latent distribution parameters
        mu = self.fc_mu(h)  # (batch, latent_dim)
        logvar = self.fc_logvar(h)  # (batch, latent_dim)

        # Clamp mu and logvar to prevent numerical instability and KL explosion
        # μ ∈ [-5, 5] - reasonable range for latent space means
        # logvar ∈ [-20, 2] → variance ∈ [2e-9, 7.4]
        mu = torch.clamp(mu, min=-5.0, max=5.0)
        logvar = torch.clamp(logvar, min=-20.0, max=2.0)

        return mu, logvar

    def encode(self, x):
        """
        Convenience method for encoding (alias for forward).

        Args:
            x (torch.Tensor): Input grid (batch, 16, 16)

        Returns:
            tuple: (mu, logvar)
        """
        return self.forward(x)
