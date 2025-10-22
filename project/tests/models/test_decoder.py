"""
Test suite for Decoder class - Task 1.3

Tests for the Decoder component of the β-VAE:
- Input/output shape validation
- Latent-to-grid decoding
- Proper logits output for color prediction
- Batch processing
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


class TestDecoderShapes:
    """Test Decoder input/output shapes."""

    def test_decoder_output_shapes(self):
        """Decoder should output logits with shape (batch, num_colors, 16, 16)."""
        from models.decoder import Decoder

        batch_size = 16
        latent_dim = 10
        num_colors = 10

        decoder = Decoder(latent_dim=latent_dim, num_colors=num_colors)

        # Input: (batch, latent_dim) continuous latent vectors
        z = torch.randn(batch_size, latent_dim)

        logits = decoder(z)

        assert logits.shape == (batch_size, num_colors, 16, 16), \
            f"Expected logits shape {(batch_size, num_colors, 16, 16)}, got {logits.shape}"

    def test_decoder_single_sample(self):
        """Decoder should handle single latent vector (batch_size=1)."""
        from models.decoder import Decoder

        decoder = Decoder(latent_dim=10)
        z = torch.randn(1, 10)

        logits = decoder(z)

        assert logits.shape == (1, 10, 16, 16)

    def test_decoder_large_batch(self):
        """Decoder should handle large batches."""
        from models.decoder import Decoder

        batch_size = 128
        decoder = Decoder(latent_dim=10)
        z = torch.randn(batch_size, 10)

        logits = decoder(z)

        assert logits.shape == (batch_size, 10, 16, 16)

    def test_decoder_different_latent_dims(self):
        """Decoder should support different latent dimensions."""
        from models.decoder import Decoder

        for latent_dim in [8, 10, 12, 16]:
            decoder = Decoder(latent_dim=latent_dim)
            z = torch.randn(4, latent_dim)

            logits = decoder(z)

            assert logits.shape == (4, 10, 16, 16)


class TestDecoderOutputType:
    """Test Decoder output type and format."""

    def test_decoder_outputs_logits(self):
        """Decoder should output logits (not probabilities or labels)."""
        from models.decoder import Decoder

        decoder = Decoder(latent_dim=10)
        z = torch.randn(8, 10)

        logits = decoder(z)

        # Logits should be continuous values (not [0, 1] probabilities)
        assert logits.dtype == torch.float32 or logits.dtype == torch.float64

        # Logits can be any real number (not bounded to [0, 1])
        # Check that at least some values are outside [0, 1]
        assert (logits < 0).any() or (logits > 1).any(), \
            "Decoder should output logits (unbounded), not probabilities"

    def test_decoder_logits_vary_with_input(self):
        """Decoder logits should vary based on latent input."""
        from models.decoder import Decoder

        decoder = Decoder(latent_dim=10)

        # Two different latent vectors
        z1 = torch.randn(1, 10)
        z2 = torch.randn(1, 10)

        logits1 = decoder(z1)
        logits2 = decoder(z2)

        # Logits should be different for different inputs
        assert not torch.allclose(logits1, logits2, atol=1e-6), \
            "Decoder should produce different outputs for different latent vectors"

    def test_decoder_logits_to_labels(self):
        """Decoder logits should convert to valid color labels via argmax."""
        from models.decoder import Decoder

        decoder = Decoder(latent_dim=10)
        z = torch.randn(4, 10)

        logits = decoder(z)
        labels = torch.argmax(logits, dim=1)  # (batch, 16, 16)

        # Labels should be in range [0, 9]
        assert labels.shape == (4, 16, 16)
        assert labels.min() >= 0
        assert labels.max() < 10


class TestDecoderInputValidation:
    """Test Decoder input validation and edge cases."""

    def test_decoder_accepts_zero_latent(self):
        """Decoder should handle all-zero latent vectors."""
        from models.decoder import Decoder

        decoder = Decoder(latent_dim=10)
        z = torch.zeros(2, 10)

        logits = decoder(z)

        assert logits.shape == (2, 10, 16, 16)
        # Should not produce NaN
        assert not torch.isnan(logits).any()

    def test_decoder_accepts_standard_normal(self):
        """Decoder should handle standard normal z ~ N(0,1)."""
        from models.decoder import Decoder

        decoder = Decoder(latent_dim=10)
        z = torch.randn(8, 10)

        logits = decoder(z)

        assert logits.shape == (8, 10, 16, 16)
        assert not torch.isnan(logits).any()

    def test_decoder_accepts_extreme_values(self):
        """Decoder should handle extreme latent values without NaN."""
        from models.decoder import Decoder

        decoder = Decoder(latent_dim=10)

        # Large positive values
        z_large = torch.ones(2, 10) * 10.0
        logits_large = decoder(z_large)
        assert not torch.isnan(logits_large).any()

        # Large negative values
        z_small = torch.ones(2, 10) * -10.0
        logits_small = decoder(z_small)
        assert not torch.isnan(logits_small).any()


class TestDecoderGradients:
    """Test Decoder gradient flow."""

    def test_decoder_gradients_flow(self):
        """Gradients should flow through decoder."""
        from models.decoder import Decoder

        decoder = Decoder(latent_dim=10)
        z = torch.randn(4, 10, requires_grad=True)

        logits = decoder(z)

        # Create dummy loss
        loss = logits.sum()
        loss.backward()

        # Check that latent input has gradients
        assert z.grad is not None
        assert z.grad.abs().sum() > 0, "Gradients should flow back to latent input"

        # Check that decoder parameters have gradients
        has_gradients = False
        for param in decoder.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break

        assert has_gradients, "Decoder parameters should have gradients"

    def test_decoder_trainable_parameters(self):
        """Decoder should have trainable parameters."""
        from models.decoder import Decoder

        decoder = Decoder(latent_dim=10)

        trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

        assert trainable_params > 0, "Decoder should have trainable parameters"
        # Rough estimate: should have at least 100K parameters
        assert trainable_params > 100000, \
            f"Decoder has suspiciously few parameters: {trainable_params}"


class TestDecoderDeterminism:
    """Test Decoder deterministic behavior."""

    def test_decoder_deterministic_in_eval(self):
        """Decoder should be deterministic in eval mode."""
        from models.decoder import Decoder

        decoder = Decoder(latent_dim=10)
        decoder.eval()

        z = torch.randn(4, 10)

        # Two forward passes with same input
        logits1 = decoder(z)
        logits2 = decoder(z)

        # Should produce identical outputs
        assert torch.allclose(logits1, logits2), \
            "Decoder should be deterministic in eval mode"

    def test_decoder_reproducible_with_seed(self):
        """Decoder with same initialization should produce same outputs."""
        from models.decoder import Decoder

        # Set seed
        torch.manual_seed(42)
        decoder1 = Decoder(latent_dim=10)

        # Reset seed
        torch.manual_seed(42)
        decoder2 = Decoder(latent_dim=10)

        z = torch.randn(2, 10)

        logits1 = decoder1(z)
        logits2 = decoder2(z)

        # Should produce same outputs due to same initialization
        assert torch.allclose(logits1, logits2), \
            "Decoders with same seed should produce same outputs"


class TestDecoderArchitecture:
    """Test Decoder architecture details."""

    def test_decoder_has_linear_layers(self):
        """Decoder should contain linear layers."""
        from models.decoder import Decoder

        decoder = Decoder(latent_dim=10)

        # Check for linear layers in the architecture
        has_linear = any(isinstance(m, nn.Linear) for m in decoder.modules())

        assert has_linear, "Decoder should contain Linear layers"

    def test_decoder_has_deconv_layers(self):
        """Decoder should contain transposed convolutional layers."""
        from models.decoder import Decoder

        decoder = Decoder(latent_dim=10)

        # Check for ConvTranspose2d layers
        has_deconv = any(isinstance(m, nn.ConvTranspose2d) for m in decoder.modules())

        assert has_deconv, "Decoder should contain ConvTranspose2d layers"

    def test_decoder_has_conv_layers(self):
        """Decoder should contain final Conv2d layer for output."""
        from models.decoder import Decoder

        decoder = Decoder(latent_dim=10)

        # Check for Conv2d layers
        has_conv = any(isinstance(m, nn.Conv2d) for m in decoder.modules())

        assert has_conv, "Decoder should contain Conv2d layer for output"

    def test_decoder_output_reasonable_range(self):
        """Decoder logits should be in reasonable range (not exploding)."""
        from models.decoder import Decoder

        decoder = Decoder(latent_dim=10)
        z = torch.randn(8, 10)

        logits = decoder(z)

        # Logits should not explode (reasonable range for untrained model)
        assert logits.abs().max() < 100, \
            f"Logits values too large: {logits.abs().max()}"


class TestDecoderEncodeDecodeConsistency:
    """Test consistency between Encoder and Decoder."""

    def test_encoder_decoder_compatible_shapes(self):
        """Encoder output should be compatible with Decoder input."""
        from models.encoder import Encoder
        from models.decoder import Decoder

        latent_dim = 10
        encoder = Encoder(latent_dim=latent_dim)
        decoder = Decoder(latent_dim=latent_dim)

        # Encode
        x = torch.randint(0, 10, (4, 16, 16))
        mu, logvar = encoder(x)

        # Decode using mu
        logits = decoder(mu)

        # Should produce valid output
        assert logits.shape == (4, 10, 16, 16)
        assert not torch.isnan(logits).any()

    def test_encode_decode_round_trip(self):
        """Encode-decode round trip should produce valid grid."""
        from models.encoder import Encoder
        from models.decoder import Decoder

        encoder = Encoder(latent_dim=10)
        decoder = Decoder(latent_dim=10)

        # Original grid
        x = torch.randint(0, 10, (2, 16, 16))

        # Encode
        mu, _ = encoder(x)

        # Decode
        logits = decoder(mu)

        # Get predicted labels
        x_pred = torch.argmax(logits, dim=1)

        # Should produce valid grid
        assert x_pred.shape == x.shape
        assert x_pred.min() >= 0
        assert x_pred.max() < 10
