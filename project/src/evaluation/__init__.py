"""
Evaluation Package for β-VAE

This package provides:
- Evaluation metrics computation
- Visualization utilities
- Sample generation tools
- Reconstruction quality assessment
"""

from .metrics import (
    evaluate_model,
    compute_reconstruction_metrics,
    compute_latent_metrics,
    compute_disentanglement_score,
)

from .visualization import (
    plot_training_curves,
    plot_reconstructions,
    plot_latent_space,
    plot_latent_traversals,
    save_generated_samples,
)

__all__ = [
    # Metrics
    'evaluate_model',
    'compute_reconstruction_metrics',
    'compute_latent_metrics',
    'compute_disentanglement_score',
    # Visualization
    'plot_training_curves',
    'plot_reconstructions',
    'plot_latent_space',
    'plot_latent_traversals',
    'save_generated_samples',
]
