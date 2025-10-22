"""
β-VAE Models for ARC Grid Encoding

This package contains the encoder, decoder, and complete β-VAE implementation
for learning latent representations of 16×16 ARC grids.
"""

from .encoder import Encoder
from .decoder import Decoder
from .beta_vae import BetaVAE
from .losses import FocalLoss, compute_class_weights, save_class_weights, load_class_weights

__all__ = [
    'Encoder',
    'Decoder',
    'BetaVAE',
    'FocalLoss',
    'compute_class_weights',
    'save_class_weights',
    'load_class_weights',
]
