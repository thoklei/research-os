"""
Generate reconstruction visualizations for debugging.
"""

import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent))

from models import BetaVAE
from data import create_data_loaders
from evaluation import plot_reconstructions
from training import load_checkpoint

def main():
    print("=" * 70)
    print("Generating Reconstruction Visualizations (No Augmentation)")
    print("=" * 70)

    # Load data WITHOUT augmentation
    print("\nLoading test data (no augmentation)...")
    _, _, test_loader = create_data_loaders(
        npz_path='../datasets/test-100k/corpus.npz',
        batch_size=32,
        num_workers=0,
        use_augmentation=False  # CRITICAL: No augmentation
    )

    if test_loader is None:
        print("[ERROR] Could not load test data")
        return

    print(f"  Loaded {len(test_loader.dataset)} test samples")

    # Create model
    print("\nCreating model...")
    model = BetaVAE(
        latent_dim=10,
        num_colors=10,
        use_focal_loss=True,
        focal_gamma=2.0,
        class_weights=None  # Will be loaded with checkpoint
    )

    # Load checkpoint
    checkpoint_path = Path("experiments/0.2-beta-vae/checkpoints/best_model.pth")
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(
        filepath=str(checkpoint_path),
        model=model,
        optimizer=None,
        scheduler=None
    )

    print(f"  Loaded model from epoch {checkpoint['epoch']}")

    # Set device
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Generate visualizations
    print("\nGenerating visualizations...")
    output_path = Path("experiments/0.2-beta-vae/evaluation/reconstructions_no_aug.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_reconstructions(
        model=model,
        data_loader=test_loader,
        device=device,
        num_samples=8,
        save_path=str(output_path),
        show=False
    )

    print("\n" + "=" * 70)
    print("Visualizations Generated Successfully")
    print("=" * 70)
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
