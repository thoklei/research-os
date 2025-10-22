"""
Test suite for BetaVAE class - Task 1.5

Tests for the complete β-VAE model:
- End-to-end forward pass
- Reparameterization trick
- Loss function components
- Sampling and reconstruction
- β-parameter effects
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


class TestBetaVAEForward:
    """Test BetaVAE forward pass."""

    def test_forward_pass_shapes(self):
        """Forward pass should return (recon_logits, mu, logvar) with correct shapes."""
        from models.beta_vae import BetaVAE

        batch_size = 8
        model = BetaVAE(latent_dim=10, num_colors=10)

        x = torch.randint(0, 10, (batch_size, 16, 16))

        recon_logits, mu, logvar = model(x)

        assert recon_logits.shape == (batch_size, 10, 16, 16), \
            f"Expected recon shape {(batch_size, 10, 16, 16)}, got {recon_logits.shape}"
        assert mu.shape == (batch_size, 10), \
            f"Expected mu shape {(batch_size, 10)}, got {mu.shape}"
        assert logvar.shape == (batch_size, 10), \
            f"Expected logvar shape {(batch_size, 10)}, got {logvar.shape}"

    def test_forward_pass_no_nan(self):
        """Forward pass should not produce NaN values."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)
        x = torch.randint(0, 10, (4, 16, 16))

        recon_logits, mu, logvar = model(x)

        assert not torch.isnan(recon_logits).any(), "Reconstruction logits contain NaN"
        assert not torch.isnan(mu).any(), "μ contains NaN"
        assert not torch.isnan(logvar).any(), "logvar contains NaN"

    def test_forward_pass_deterministic_in_eval(self):
        """Forward pass should be deterministic in eval mode."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)
        model.eval()

        x = torch.randint(0, 10, (2, 16, 16))

        # Set seed for reparameterization sampling
        torch.manual_seed(42)
        recon1, mu1, logvar1 = model(x)

        torch.manual_seed(42)
        recon2, mu2, logvar2 = model(x)

        # μ and logvar should be deterministic
        assert torch.allclose(mu1, mu2), "μ should be deterministic"
        assert torch.allclose(logvar1, logvar2), "logvar should be deterministic"

        # Reconstruction with same seed should be identical
        assert torch.allclose(recon1, recon2), \
            "Reconstruction should be deterministic with same seed"


class TestReparameterization:
    """Test reparameterization trick."""

    def test_reparameterize_shape(self):
        """Reparameterization should maintain shape."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)

        mu = torch.randn(8, 10)
        logvar = torch.randn(8, 10)

        z = model.reparameterize(mu, logvar)

        assert z.shape == mu.shape, f"Expected shape {mu.shape}, got {z.shape}"

    def test_reparameterize_stochastic(self):
        """Reparameterization should be stochastic (different samples)."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)

        mu = torch.zeros(4, 10)
        logvar = torch.zeros(4, 10)  # σ = 1

        # Two samples should be different
        z1 = model.reparameterize(mu, logvar)
        z2 = model.reparameterize(mu, logvar)

        assert not torch.allclose(z1, z2, atol=1e-6), \
            "Reparameterization should produce different samples"

    def test_reparameterize_respects_mu(self):
        """Reparameterization should center around μ."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)

        # Large μ, small σ
        mu = torch.ones(1000, 10) * 5.0
        logvar = torch.ones(1000, 10) * -4.0  # σ ≈ 0.135

        z = model.reparameterize(mu, logvar)

        # Mean of samples should be close to μ
        z_mean = z.mean(dim=0)
        assert torch.allclose(z_mean, mu[0], atol=0.5), \
            f"Sample mean {z_mean.mean()} should be close to μ {mu[0].mean()}"

    def test_reparameterize_gradients_flow(self):
        """Gradients should flow through reparameterization."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)

        mu = torch.randn(4, 10, requires_grad=True)
        logvar = torch.randn(4, 10, requires_grad=True)

        z = model.reparameterize(mu, logvar)
        loss = z.sum()
        loss.backward()

        # Gradients should flow to μ and logvar
        assert mu.grad is not None, "Gradients should flow to μ"
        assert logvar.grad is not None, "Gradients should flow to logvar"
        assert mu.grad.abs().sum() > 0, "μ gradients should be non-zero"
        assert logvar.grad.abs().sum() > 0, "logvar gradients should be non-zero"


class TestLossFunction:
    """Test β-VAE loss function."""

    def test_loss_function_components(self):
        """Loss function should return total loss, recon loss, and KL loss."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)

        x = torch.randint(0, 10, (4, 16, 16))
        recon_logits, mu, logvar = model(x)

        loss_dict = model.loss_function(recon_logits, x, mu, logvar, beta=1.0)

        # Should return dictionary with 3 components
        assert 'loss' in loss_dict, "Should return 'loss'"
        assert 'recon_loss' in loss_dict, "Should return 'recon_loss'"
        assert 'kl_loss' in loss_dict, "Should return 'kl_loss'"

        # All should be scalars
        assert loss_dict['loss'].dim() == 0, "Total loss should be scalar"
        assert loss_dict['recon_loss'].dim() == 0, "Recon loss should be scalar"
        assert loss_dict['kl_loss'].dim() == 0, "KL loss should be scalar"

    def test_loss_function_positive(self):
        """Loss components should be positive."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)

        x = torch.randint(0, 10, (8, 16, 16))
        recon_logits, mu, logvar = model(x)

        loss_dict = model.loss_function(recon_logits, x, mu, logvar, beta=1.0)

        # Losses should be positive
        assert loss_dict['loss'] > 0, "Total loss should be positive"
        assert loss_dict['recon_loss'] >= 0, "Recon loss should be non-negative"
        assert loss_dict['kl_loss'] >= 0, "KL loss should be non-negative"

    def test_loss_function_beta_effect(self):
        """β parameter should affect total loss but not recon loss."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)

        x = torch.randint(0, 10, (4, 16, 16))
        recon_logits, mu, logvar = model(x)

        # Compute loss with different β values
        loss_beta1 = model.loss_function(recon_logits, x, mu, logvar, beta=1.0)
        loss_beta2 = model.loss_function(recon_logits, x, mu, logvar, beta=2.0)

        # Recon loss should be identical
        assert torch.allclose(loss_beta1['recon_loss'], loss_beta2['recon_loss']), \
            "Recon loss should not depend on β"

        # KL loss should be identical
        assert torch.allclose(loss_beta1['kl_loss'], loss_beta2['kl_loss']), \
            "KL loss value should not depend on β"

        # Total loss should be different (β=2 should be larger if KL > 0)
        if loss_beta1['kl_loss'] > 0:
            assert loss_beta2['loss'] > loss_beta1['loss'], \
                "Total loss should increase with β (if KL > 0)"

    def test_loss_function_perfect_reconstruction(self):
        """Recon loss should be ~0 for perfect reconstruction."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)

        # Create a simple input
        x = torch.zeros((2, 16, 16), dtype=torch.long)

        # Create perfect reconstruction logits (very high confidence for class 0)
        recon_logits = torch.ones(2, 10, 16, 16) * -10.0
        recon_logits[:, 0, :, :] = 10.0  # Class 0 has high logits

        # Dummy mu and logvar (not used for recon loss)
        mu = torch.zeros(2, 10)
        logvar = torch.zeros(2, 10)

        loss_dict = model.loss_function(recon_logits, x, mu, logvar, beta=1.0)

        # Recon loss should be very small
        assert loss_dict['recon_loss'] < 0.1, \
            f"Perfect reconstruction should have low loss, got {loss_dict['recon_loss']}"

    def test_loss_function_kl_divergence(self):
        """KL loss should be ~0 when distribution is N(0, 1)."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)

        x = torch.randint(0, 10, (4, 16, 16))

        # Create distribution close to N(0, 1)
        mu = torch.zeros(4, 10)
        logvar = torch.zeros(4, 10)  # log(1) = 0

        # Dummy reconstruction
        recon_logits = torch.randn(4, 10, 16, 16)

        loss_dict = model.loss_function(recon_logits, x, mu, logvar, beta=1.0)

        # KL loss should be very small (close to 0)
        assert loss_dict['kl_loss'] < 0.1, \
            f"KL should be ~0 for N(0,1), got {loss_dict['kl_loss']}"


class TestSampling:
    """Test sampling from prior."""

    def test_sample_shape(self):
        """Sample should generate correct shape."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)

        num_samples = 8
        samples = model.sample(num_samples, device='cpu')

        assert samples.shape == (num_samples, 10, 16, 16), \
            f"Expected shape {(num_samples, 10, 16, 16)}, got {samples.shape}"

    def test_sample_produces_valid_logits(self):
        """Samples should be valid logits (can convert to labels)."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)

        samples = model.sample(4, device='cpu')

        # Convert to labels
        labels = torch.argmax(samples, dim=1)

        assert labels.shape == (4, 16, 16)
        assert labels.min() >= 0
        assert labels.max() < 10

    def test_sample_stochastic(self):
        """Samples should be different each time."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)

        samples1 = model.sample(2, device='cpu')
        samples2 = model.sample(2, device='cpu')

        # Samples should be different
        assert not torch.allclose(samples1, samples2, atol=1e-6), \
            "Samples should be different each time"

    def test_sample_no_nan(self):
        """Samples should not contain NaN."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)

        samples = model.sample(8, device='cpu')

        assert not torch.isnan(samples).any(), "Samples should not contain NaN"


class TestReconstruction:
    """Test reconstruction method."""

    def test_reconstruct_shape(self):
        """Reconstruct should return correct shape."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)

        x = torch.randint(0, 10, (4, 16, 16))
        recon = model.reconstruct(x)

        assert recon.shape == (4, 10, 16, 16), \
            f"Expected shape {(4, 10, 16, 16)}, got {recon.shape}"

    def test_reconstruct_uses_mean(self):
        """Reconstruct should use mean (not sampling) for determinism."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)
        model.eval()

        x = torch.randint(0, 10, (2, 16, 16))

        # Two reconstructions should be identical (using mean)
        recon1 = model.reconstruct(x)
        recon2 = model.reconstruct(x)

        assert torch.allclose(recon1, recon2), \
            "Reconstruct should be deterministic (use mean, not sampling)"

    def test_reconstruct_no_nan(self):
        """Reconstruct should not produce NaN."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)

        x = torch.randint(0, 10, (4, 16, 16))
        recon = model.reconstruct(x)

        assert not torch.isnan(recon).any(), "Reconstruction should not contain NaN"


class TestEndToEnd:
    """Test end-to-end VAE functionality."""

    def test_train_step(self):
        """Simulate a training step."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randint(0, 10, (8, 16, 16))

        # Forward pass
        recon_logits, mu, logvar = model(x)

        # Compute loss
        loss_dict = model.loss_function(recon_logits, x, mu, logvar, beta=1.0)

        # Backward pass
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        optimizer.step()

        # Should complete without errors
        assert True

    def test_multiple_train_steps(self):
        """Run multiple training steps."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(5):
            x = torch.randint(0, 10, (4, 16, 16))

            recon_logits, mu, logvar = model(x)
            loss_dict = model.loss_function(recon_logits, x, mu, logvar, beta=1.0)

            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()

        # Should complete without errors
        assert True

    def test_eval_mode(self):
        """Test model in eval mode."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=10)
        model.eval()

        x = torch.randint(0, 10, (4, 16, 16))

        with torch.no_grad():
            recon_logits, mu, logvar = model(x)
            loss_dict = model.loss_function(recon_logits, x, mu, logvar, beta=1.0)

        # Should complete without errors
        assert True

    def test_save_load_model(self):
        """Test saving and loading model state."""
        from models.beta_vae import BetaVAE
        import tempfile
        import os

        model = BetaVAE(latent_dim=10)
        model.eval()

        x = torch.randint(0, 10, (2, 16, 16))

        # Get encoder outputs (these should be deterministic)
        mu1, logvar1 = model.encoder(x)

        # Get reconstruction using mean (deterministic)
        recon1 = model.reconstruct(x)

        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            torch.save(model.state_dict(), f.name)
            temp_path = f.name

        # Create new model and load weights
        model2 = BetaVAE(latent_dim=10)
        model2.load_state_dict(torch.load(temp_path))
        model2.eval()

        # Verify encoder produces same μ and logvar
        mu2, logvar2 = model2.encoder(x)
        assert torch.allclose(mu1, mu2), "Loaded model should produce same μ"
        assert torch.allclose(logvar1, logvar2), "Loaded model should produce same logvar"

        # Verify reconstruction is identical
        recon2 = model2.reconstruct(x)
        assert torch.allclose(recon1, recon2), "Loaded model should produce same reconstruction"

        # Clean up
        os.unlink(temp_path)


class TestDifferentLatentDims:
    """Test VAE with different latent dimensions."""

    def test_latent_dim_8(self):
        """Test with latent_dim=8."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=8)
        x = torch.randint(0, 10, (4, 16, 16))

        recon, mu, logvar = model(x)

        assert mu.shape == (4, 8)
        assert logvar.shape == (4, 8)
        assert recon.shape == (4, 10, 16, 16)

    def test_latent_dim_16(self):
        """Test with latent_dim=16."""
        from models.beta_vae import BetaVAE

        model = BetaVAE(latent_dim=16)
        x = torch.randint(0, 10, (4, 16, 16))

        recon, mu, logvar = model(x)

        assert mu.shape == (4, 16)
        assert logvar.shape == (4, 16)
        assert recon.shape == (4, 10, 16, 16)
