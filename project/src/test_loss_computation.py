"""
Test loss computation to understand the issue.
"""

import torch
import torch.nn.functional as F

# Simulate model output
batch_size = 2
recon_logits = torch.randn(batch_size, 10, 16, 16)  # (batch, num_colors, H, W)
x = torch.randint(0, 10, (batch_size, 16, 16))  # (batch, H, W)

print("Testing loss computation...")
print(f"recon_logits shape: {recon_logits.shape}")
print(f"x shape: {x.shape}")

# Method 1: Current implementation (potentially wrong)
print("\n[Method 1] Current implementation:")
try:
    loss1 = F.cross_entropy(recon_logits, x.long(), reduction='mean')
    print(f"  Loss: {loss1.item():.4f}")
    print("  SUCCESS")
except Exception as e:
    print(f"  ERROR: {e}")

# Method 2: Proper way - reshape logits
print("\n[Method 2] Reshape logits to (N, C):")
try:
    # Reshape: (batch, num_colors, H, W) -> (batch*H*W, num_colors)
    recon_logits_reshaped = recon_logits.permute(0, 2, 3, 1).contiguous().view(-1, 10)
    x_reshaped = x.view(-1)

    loss2 = F.cross_entropy(recon_logits_reshaped, x_reshaped.long(), reduction='mean')
    print(f"  recon_logits_reshaped shape: {recon_logits_reshaped.shape}")
    print(f"  x_reshaped shape: {x_reshaped.shape}")
    print(f"  Loss: {loss2.item():.4f}")
    print("  SUCCESS")
except Exception as e:
    print(f"  ERROR: {e}")

# Test what the model is actually computing
print("\n[Checking what cross_entropy does with (B, C, H, W) input]:")
print("  PyTorch cross_entropy with 4D input treats it as 2D image classification")
print("  Expected input: (N, C) or (N, C, d1, d2, ...)")
print("  In our case: (batch, num_colors=10, H=16, W=16)")
print("  This should work for 2D spatial data")

# Verify predictions
print("\n[Verify predictions]:")
pred1 = recon_logits.argmax(dim=1)  # (batch, H, W)
print(f"  Predictions shape (argmax dim=1): {pred1.shape}")
print(f"  Target shape: {x.shape}")
print(f"  Shapes match: {pred1.shape == x.shape}")
