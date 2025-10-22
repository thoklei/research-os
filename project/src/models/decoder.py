"""
Decoder for β-VAE - Task 1.4

Decoder architecture: d=10 latent → 16×16 Grid

Architecture:
    Input: (batch, latent_dim) continuous latent vectors
    → Dense(latent_dim→128) + ReLU
    → Dense(128→2048) + ReLU
    → Reshape: (batch, 128, 4, 4)
    → ConvTranspose2D(128→64, k=4, s=2, p=1) + ReLU  # 8×8
    → ConvTranspose2D(64→32, k=4, s=2, p=1) + ReLU   # 16×16
    → Conv2D(32→10, k=3, s=1, p=1)
    Output: (batch, 10, 16, 16) logits for 10 color classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    Decoder for β-VAE that maps latent vectors to 16×16 ARC grid logits.

    Args:
        latent_dim (int): Dimension of latent space (default: 10)
        num_colors (int): Number of color classes (default: 10 for ARC palette)
    """

    def __init__(self, latent_dim=10, num_colors=10):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.num_colors = num_colors

        # Fully connected layers
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 128 * 4 * 4)

        # Transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1
        )

        # Final convolution for color logits
        self.conv_out = nn.Conv2d(
            in_channels=32,
            out_channels=num_colors,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, z):
        """
        Forward pass through decoder.

        Args:
            z (torch.Tensor): Latent vector with shape (batch, latent_dim)

        Returns:
            torch.Tensor: Logits for color classes with shape (batch, num_colors, 16, 16)
        """
        batch_size = z.size(0)

        # Fully connected layers
        h = F.relu(self.fc1(z))  # (batch, 128)
        h = F.relu(self.fc2(h))  # (batch, 2048)

        # Reshape to spatial dimensions
        h = h.reshape(batch_size, 128, 4, 4)  # (batch, 128, 4, 4)

        # Transposed convolutions
        h = F.relu(self.deconv1(h))  # (batch, 64, 8, 8)
        h = F.relu(self.deconv2(h))  # (batch, 32, 16, 16)

        # Output logits
        logits = self.conv_out(h)  # (batch, num_colors, 16, 16)

        return logits

    def decode(self, z):
        """
        Convenience method for decoding (alias for forward).

        Args:
            z (torch.Tensor): Latent vector (batch, latent_dim)

        Returns:
            torch.Tensor: Logits (batch, num_colors, 16, 16)
        """
        return self.forward(z)
