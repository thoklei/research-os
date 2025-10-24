"""
β-VAE Implementation - Task 1.6

Complete β-VAE model combining Encoder and Decoder with reparameterization trick.

Loss function:
    L = FocalLoss(logits, labels) + β * KL(q(z|x) || N(0,I))

where:
    - FocalLoss: Reconstruction loss with class imbalance handling
    - KL divergence: Regularization term for latent distribution
    - β: Annealing parameter (0.0 → 1.0 → 2.0 over training)

NOTE: Focal Loss is CRITICAL for ARC grids due to severe class imbalance
      (~93% black pixels). Standard cross-entropy causes mode collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .encoder import Encoder
from .decoder import Decoder
from .losses import FocalLoss


class BetaVAE(nn.Module):
    """
    β-VAE model for encoding/decoding 16×16 ARC grids.

    Args:
        latent_dim (int): Dimension of latent space (default: 10)
        num_colors (int): Number of color classes (default: 10 for ARC palette)
    """

    def __init__(
        self,
        latent_dim: int = 10,
        num_colors: int = 10,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_colors = num_colors
        self.use_focal_loss = use_focal_loss

        self.encoder = Encoder(latent_dim=latent_dim, num_colors=num_colors)
        self.decoder = Decoder(latent_dim=latent_dim, num_colors=num_colors)

        # Loss function for reconstruction
        if use_focal_loss:
            self.recon_loss_fn = FocalLoss(
                gamma=focal_gamma,
                alpha=class_weights,
                reduction='mean'
            )
        else:
            # Standard cross-entropy (not recommended for ARC grids)
            self.recon_loss_fn = None
            self.class_weights = class_weights

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1).

        Args:
            mu (torch.Tensor): Mean of latent distribution (batch, latent_dim)
            logvar (torch.Tensor): Log variance of latent distribution (batch, latent_dim)

        Returns:
            torch.Tensor: Sampled latent vector (batch, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        """
        Forward pass through β-VAE.

        Args:
            x (torch.Tensor): Input grid with shape (batch, 16, 16)
                             Integer labels in range [0, num_colors-1]

        Returns:
            tuple: (recon_logits, mu, logvar) where:
                - recon_logits: Reconstruction logits (batch, num_colors, 16, 16)
                - mu: Mean of latent distribution (batch, latent_dim)
                - logvar: Log variance of latent distribution (batch, latent_dim)
        """
        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        recon_logits = self.decoder(z)

        return recon_logits, mu, logvar

    def loss_function(self, recon_logits, x, mu, logvar, beta=1.0, free_bits=0.0):
        """
        Compute β-VAE loss with optional free bits mechanism.

        Args:
            recon_logits (torch.Tensor): Reconstruction logits (batch, num_colors, 16, 16)
            x (torch.Tensor): Ground truth grid (batch, 16, 16)
            mu (torch.Tensor): Latent mean (batch, latent_dim)
            logvar (torch.Tensor): Latent log variance (batch, latent_dim)
            beta (float): β parameter for KL weighting (default: 1.0)
            free_bits (float): Minimum KL per dimension in nats (default: 0.0).
                              If > 0, prevents posterior collapse by ensuring each
                              dimension carries at least this much information.

        Returns:
            dict: Dictionary containing:
                - 'loss': Total loss (scalar)
                - 'recon_loss': Reconstruction loss (scalar)
                - 'kl_loss': KL divergence loss (scalar)
                - 'kl_per_dim': Unclamped KL per dimension (latent_dim,) for monitoring
        """
        # Reconstruction loss (Focal Loss or CrossEntropy)
        if self.use_focal_loss:
            recon_loss = self.recon_loss_fn(recon_logits, x.long())
        else:
            # Fallback to standard cross-entropy with optional class weights
            if self.class_weights is not None:
                recon_loss = F.cross_entropy(
                    recon_logits,
                    x.long(),
                    weight=self.class_weights,
                    reduction='mean'
                )
            else:
                recon_loss = F.cross_entropy(
                    recon_logits,
                    x.long(),
                    reduction='mean'
                )

        # KL divergence per dimension: -0.5 * (1 + log(σ²) - μ² - σ²)
        # Shape: (batch, latent_dim)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        # Store unclamped KL per dimension for monitoring (average over batch)
        kl_per_dim_actual = kl_per_dim.mean(dim=0)  # (latent_dim,)

        # Apply free bits if specified
        if free_bits > 0:
            # Clamp each dimension to minimum free_bits
            # This prevents any dimension from collapsing below the threshold
            kl_per_dim_clamped = torch.clamp(kl_per_dim, min=free_bits)
            # Sum over dimensions, mean over batch
            kl_loss = kl_per_dim_clamped.sum(dim=1).mean()
        else:
            # Standard VAE: no clamping
            kl_loss = kl_per_dim.sum(dim=1).mean()

        # Total loss
        total_loss = recon_loss + beta * kl_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'kl_per_dim': kl_per_dim_actual  # For monitoring (latent_dim,)
        }

    def sample(self, num_samples, device='cpu'):
        """
        Generate samples by sampling from N(0, I) and decoding.

        Args:
            num_samples (int): Number of samples to generate
            device (str): Device to generate samples on

        Returns:
            torch.Tensor: Generated samples (num_samples, num_colors, 16, 16)
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decoder(z)
        return samples

    def reconstruct(self, x):
        """
        Reconstruct input by encoding and decoding (using mean, no sampling).

        Args:
            x (torch.Tensor): Input grid (batch, 16, 16)

        Returns:
            torch.Tensor: Reconstructed logits (batch, num_colors, 16, 16)
        """
        mu, _ = self.encoder(x)
        recon_logits = self.decoder(mu)
        return recon_logits
