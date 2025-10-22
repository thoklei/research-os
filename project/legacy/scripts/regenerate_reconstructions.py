"""
Regenerate reconstruction visualizations with improved layout.
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent))

from models.beta_vae import BetaVAE
from data import create_data_loaders
from evaluation import plot_reconstructions
from training import load_checkpoint


def main():
    print("Regenerating reconstruction visualizations...")

    # Load model
    print("[1/3] Loading model...")
    model = BetaVAE(latent_dim=10, num_colors=10)
    checkpoint = load_checkpoint(
        filepath='experiments/0.2-beta-vae/checkpoints/best_model.pth',
        model=model
    )
    model.eval()
    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Load data
    print("[2/3] Loading data...")
    _, _, test_loader = create_data_loaders(
        npz_path='../datasets/test-100k/corpus.npz',
        batch_size=128,
        num_workers=0,
        use_augmentation=False
    )
    print(f"  Loaded {len(test_loader.dataset)} test samples")

    # Generate visualization
    print("[3/3] Generating visualizations...")
    plot_reconstructions(
        model=model,
        data_loader=test_loader,
        device=torch.device('cpu'),
        num_samples=8,
        save_path='experiments/0.2-beta-vae/evaluation/reconstructions_v2.png',
        show=False
    )

    print("\nDone! New visualization saved to:")
    print("  experiments/0.2-beta-vae/evaluation/reconstructions_v2.png")


if __name__ == "__main__":
    main()
