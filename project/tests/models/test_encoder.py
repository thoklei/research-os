"""
Test suite for Encoder class - Task 1.1

Tests for the Encoder component of the β-VAE:
- Input/output shape validation
- Latent distribution parameters (μ, σ)
- Proper one-hot encoding handling
- Batch processing
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


class TestEncoderShapes:
    """Test Encoder input/output shapes."""

    def test_encoder_output_shapes(self):
        """Encoder should output μ and logvar with shape (batch, latent_dim)."""
        from models.encoder import Encoder

        batch_size = 16
        latent_dim = 10

        encoder = Encoder(latent_dim=latent_dim)

        # Input: (batch, 16, 16) integer labels [0-9]
        x = torch.randint(0, 10, (batch_size, 16, 16))

        mu, logvar = encoder(x)

        assert mu.shape == (batch_size, latent_dim), f"Expected μ shape {(batch_size, latent_dim)}, got {mu.shape}"
        assert logvar.shape == (batch_size, latent_dim), f"Expected logvar shape {(batch_size, latent_dim)}, got {logvar.shape}"

    def test_encoder_single_sample(self):
        """Encoder should handle single sample (batch_size=1)."""
        from models.encoder import Encoder

        encoder = Encoder(latent_dim=10)
        x = torch.randint(0, 10, (1, 16, 16))

        mu, logvar = encoder(x)

        assert mu.shape == (1, 10)
        assert logvar.shape == (1, 10)

    def test_encoder_large_batch(self):
        """Encoder should handle large batches."""
        from models.encoder import Encoder

        batch_size = 128
        encoder = Encoder(latent_dim=10)
        x = torch.randint(0, 10, (batch_size, 16, 16))

        mu, logvar = encoder(x)

        assert mu.shape == (batch_size, 10)
        assert logvar.shape == (batch_size, 10)

    def test_encoder_different_latent_dims(self):
        """Encoder should support different latent dimensions."""
        from models.encoder import Encoder

        for latent_dim in [8, 10, 12, 16]:
            encoder = Encoder(latent_dim=latent_dim)
            x = torch.randint(0, 10, (4, 16, 16))

            mu, logvar = encoder(x)

            assert mu.shape == (4, latent_dim)
            assert logvar.shape == (4, latent_dim)


class TestEncoderLatentDistribution:
    """Test Encoder latent distribution parameters."""

    def test_mu_is_continuous(self):
        """μ should be continuous values (not discrete)."""
        from models.encoder import Encoder

        encoder = Encoder(latent_dim=10)
        x = torch.randint(0, 10, (8, 16, 16))

        mu, _ = encoder(x)

        # μ should have floating point values
        assert mu.dtype == torch.float32 or mu.dtype == torch.float64

    def test_logvar_is_continuous(self):
        """logvar should be continuous values."""
        from models.encoder import Encoder

        encoder = Encoder(latent_dim=10)
        x = torch.randint(0, 10, (8, 16, 16))

        _, logvar = encoder(x)

        # logvar should have floating point values
        assert logvar.dtype == torch.float32 or logvar.dtype == torch.float64

    def test_mu_varies_with_input(self):
        """μ should vary based on input (not constant)."""
        from models.encoder import Encoder

        encoder = Encoder(latent_dim=10)

        # Two different inputs
        x1 = torch.zeros((1, 16, 16), dtype=torch.long)
        x2 = torch.ones((1, 16, 16), dtype=torch.long) * 5

        mu1, _ = encoder(x1)
        mu2, _ = encoder(x2)

        # μ values should be different for different inputs
        assert not torch.allclose(mu1, mu2, atol=1e-6), "μ should vary with input"

    def test_logvar_varies_with_input(self):
        """logvar should vary based on input."""
        from models.encoder import Encoder

        encoder = Encoder(latent_dim=10)

        x1 = torch.zeros((1, 16, 16), dtype=torch.long)
        x2 = torch.ones((1, 16, 16), dtype=torch.long) * 5

        _, logvar1 = encoder(x1)
        _, logvar2 = encoder(x2)

        # logvar values should be different for different inputs
        assert not torch.allclose(logvar1, logvar2, atol=1e-6), "logvar should vary with input"


class TestEncoderInputValidation:
    """Test Encoder input validation and edge cases."""

    def test_encoder_accepts_valid_colors(self):
        """Encoder should accept integer labels in range [0, 9]."""
        from models.encoder import Encoder

        encoder = Encoder(latent_dim=10)

        # Valid input: all colors from 0-9
        x = torch.randint(0, 10, (4, 16, 16))

        mu, logvar = encoder(x)

        # Should not raise, and should produce valid outputs
        assert mu.shape == (4, 10)
        assert logvar.shape == (4, 10)

    def test_encoder_with_zeros(self):
        """Encoder should handle all-zero (background) grids."""
        from models.encoder import Encoder

        encoder = Encoder(latent_dim=10)
        x = torch.zeros((2, 16, 16), dtype=torch.long)

        mu, logvar = encoder(x)

        assert mu.shape == (2, 10)
        assert logvar.shape == (2, 10)
        # Should not produce NaN
        assert not torch.isnan(mu).any()
        assert not torch.isnan(logvar).any()

    def test_encoder_with_max_colors(self):
        """Encoder should handle all color 9 grids."""
        from models.encoder import Encoder

        encoder = Encoder(latent_dim=10)
        x = torch.ones((2, 16, 16), dtype=torch.long) * 9

        mu, logvar = encoder(x)

        assert mu.shape == (2, 10)
        assert logvar.shape == (2, 10)
        assert not torch.isnan(mu).any()
        assert not torch.isnan(logvar).any()


class TestEncoderGradients:
    """Test Encoder gradient flow."""

    def test_encoder_gradients_flow(self):
        """Gradients should flow through encoder."""
        from models.encoder import Encoder

        encoder = Encoder(latent_dim=10)
        x = torch.randint(0, 10, (4, 16, 16))

        mu, logvar = encoder(x)

        # Create dummy loss
        loss = mu.sum() + logvar.sum()
        loss.backward()

        # Check that some parameters have gradients
        has_gradients = False
        for param in encoder.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break

        assert has_gradients, "No gradients found in encoder parameters"

    def test_encoder_trainable_parameters(self):
        """Encoder should have trainable parameters."""
        from models.encoder import Encoder

        encoder = Encoder(latent_dim=10)

        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

        assert trainable_params > 0, "Encoder should have trainable parameters"
        # Rough estimate: should have at least 100K parameters
        assert trainable_params > 100000, f"Encoder has suspiciously few parameters: {trainable_params}"


class TestEncoderDeterminism:
    """Test Encoder deterministic behavior."""

    def test_encoder_deterministic_in_eval(self):
        """Encoder should be deterministic in eval mode."""
        from models.encoder import Encoder

        encoder = Encoder(latent_dim=10)
        encoder.eval()

        x = torch.randint(0, 10, (4, 16, 16))

        # Two forward passes with same input
        mu1, logvar1 = encoder(x)
        mu2, logvar2 = encoder(x)

        # Should produce identical outputs
        assert torch.allclose(mu1, mu2), "Encoder should be deterministic in eval mode"
        assert torch.allclose(logvar1, logvar2), "Encoder should be deterministic in eval mode"

    def test_encoder_reproducible_with_seed(self):
        """Encoder with same initialization should produce same outputs."""
        from models.encoder import Encoder

        # Set seed
        torch.manual_seed(42)
        encoder1 = Encoder(latent_dim=10)

        # Reset seed
        torch.manual_seed(42)
        encoder2 = Encoder(latent_dim=10)

        x = torch.randint(0, 10, (2, 16, 16))

        mu1, logvar1 = encoder1(x)
        mu2, logvar2 = encoder2(x)

        # Should produce same outputs due to same initialization
        assert torch.allclose(mu1, mu2), "Encoders with same seed should produce same outputs"
        assert torch.allclose(logvar1, logvar2), "Encoders with same seed should produce same outputs"


class TestEncoderArchitecture:
    """Test Encoder architecture details."""

    def test_encoder_has_conv_layers(self):
        """Encoder should contain convolutional layers."""
        from models.encoder import Encoder

        encoder = Encoder(latent_dim=10)

        # Check for conv layers in the architecture
        has_conv = any(isinstance(m, nn.Conv2d) for m in encoder.modules())

        assert has_conv, "Encoder should contain Conv2d layers"

    def test_encoder_has_linear_layers(self):
        """Encoder should contain linear layers for μ and logvar heads."""
        from models.encoder import Encoder

        encoder = Encoder(latent_dim=10)

        # Check for linear layers
        has_linear = any(isinstance(m, nn.Linear) for m in encoder.modules())

        assert has_linear, "Encoder should contain Linear layers"

    def test_encoder_output_reasonable_range(self):
        """Encoder outputs should be in reasonable range (not exploding)."""
        from models.encoder import Encoder

        encoder = Encoder(latent_dim=10)
        x = torch.randint(0, 10, (8, 16, 16))

        mu, logvar = encoder(x)

        # μ should be in reasonable range (e.g., -10 to 10 for normalized data)
        assert mu.abs().max() < 100, f"μ values too large: {mu.abs().max()}"

        # logvar should be in reasonable range (e.g., -10 to 10)
        assert logvar.abs().max() < 50, f"logvar values too large: {logvar.abs().max()}"
