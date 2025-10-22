"""
Debug the actual logits values from the model.
"""

import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent))

from models.beta_vae import BetaVAE
from data import create_data_loaders
from training import load_checkpoint


def main():
    print("Debugging model logits...")

    # Load model
    model = BetaVAE(latent_dim=10, num_colors=10)
    checkpoint = load_checkpoint(
        filepath='experiments/0.2-beta-vae/checkpoints/best_model.pth',
        model=model
    )
    model.eval()

    # Load data
    _, _, test_loader = create_data_loaders(
        npz_path='../datasets/test-100k/corpus.npz',
        batch_size=2,
        num_workers=0,
        use_augmentation=False
    )

    # Get a batch
    batch = next(iter(test_loader))

    print(f"\nInput batch:")
    print(f"  Shape: {batch.shape}")
    print(f"  First sample:\n{batch[0]}")

    # Forward pass
    with torch.no_grad():
        recon_logits, mu, logvar = model(batch)

    print(f"\nReconstruction logits:")
    print(f"  Shape: {recon_logits.shape}")
    print(f"  Mean: {recon_logits.mean().item():.6f}")
    print(f"  Std: {recon_logits.std().item():.6f}")
    print(f"  Min: {recon_logits.min().item():.6f}")
    print(f"  Max: {recon_logits.max().item():.6f}")

    # Check logits for one pixel
    print(f"\nLogits for pixel [0, 5, 1] (should have color 3):")
    pixel_logits = recon_logits[0, :, 5, 1]
    print(f"  Logits: {pixel_logits.numpy()}")
    print(f"  Softmax probs: {torch.softmax(pixel_logits, dim=0).numpy()}")
    print(f"  Predicted class: {pixel_logits.argmax().item()}")
    print(f"  True class: {batch[0, 5, 1].item()}")

    # Check average logits per class
    print(f"\nAverage logits per class across all pixels:")
    for c in range(10):
        avg_logit = recon_logits[:, c, :, :].mean().item()
        print(f"  Class {c}: {avg_logit:.6f}")

    # Check if decoder weights are reasonable
    print(f"\nDecoder final layer weights:")
    final_conv = model.decoder.decoder[-1]  # Last conv layer
    print(f"  Weight shape: {final_conv.weight.shape}")
    print(f"  Weight mean: {final_conv.weight.mean().item():.6f}")
    print(f"  Weight std: {final_conv.weight.std().item():.6f}")
    print(f"  Bias mean: {final_conv.bias.mean().item():.6f}")
    print(f"  Bias std: {final_conv.bias.std().item():.6f}")
    print(f"  Bias values: {final_conv.bias.detach().numpy()}")


if __name__ == "__main__":
    main()
