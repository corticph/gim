"""Tests for individual GIM components."""
import pytest
import torch
import torch.nn.functional as F


class TestSoftmaxTemperature:
    """Tests for softmax temperature scaling."""

    def test_softmax_forward_unchanged(self):
        """Test that forward pass is unchanged with temperature."""
        from gim.context.softmax import _softmax_bwT, stable_softmax

        x = torch.randn(2, 10)

        # Standard softmax
        y_standard = stable_softmax(x, dim=-1)

        # Temperature-modified (forward should be same)
        y_temp = _softmax_bwT(x, dim=-1, T=2.0)

        assert torch.allclose(y_standard, y_temp)

    def test_softmax_backward_differs(self):
        """Test that backward pass differs with temperature."""
        from gim.context.softmax import _softmax_bwT

        x1 = torch.randn(2, 10, requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)

        # T=1.0 (standard)
        y1 = _softmax_bwT(x1, dim=-1, T=1.0)
        y1.sum().backward()

        # T=2.0 (temperature scaled)
        y2 = _softmax_bwT(x2, dim=-1, T=2.0)
        y2.sum().backward()

        # Gradients should differ
        assert not torch.allclose(x1.grad, x2.grad)


class TestGradientScaling:
    """Tests for gradient scaling utilities."""

    def test_scale_grad_forward_unchanged(self):
        """Test that scale_grad doesn't change forward pass."""
        from gim.context.attention import scale_grad

        x = torch.randn(2, 10)
        y = scale_grad(x, 0.5)

        assert torch.equal(x, y)

    def test_scale_grad_backward_scaled(self):
        """Test that scale_grad scales gradients correctly."""
        from gim.context.attention import scale_grad

        x = torch.randn(2, 10, requires_grad=True)
        y = scale_grad(x, 0.5)
        y.sum().backward()

        # Gradient should be scaled by 0.5
        expected_grad = torch.ones_like(x) * 0.5
        assert torch.allclose(x.grad, expected_grad)


class TestNormDetach:
    """Tests for norm detaching."""

    def test_layernorm_detach_forward(self):
        """Test that LayerNormDetach produces same forward output."""
        from gim.context.norm import LayerNormDetach
        import torch.nn as nn

        d = 64
        ln_standard = nn.LayerNorm(d)
        ln_detach = LayerNormDetach(d)

        # Copy weights
        with torch.no_grad():
            ln_detach.weight.copy_(ln_standard.weight)
            ln_detach.bias.copy_(ln_standard.bias)

        x = torch.randn(2, 10, d)
        y_standard = ln_standard(x)
        y_detach = ln_detach(x)

        assert torch.allclose(y_standard, y_detach, atol=1e-5)

    def test_rmsnorm_detach_forward(self):
        """Test that RMSNormDetach produces same forward output."""
        from gim.context.norm import RMSNormDetach
        import torch.nn as nn

        d = 64
        rms_standard = nn.RMSNorm(d)
        rms_detach = RMSNormDetach(d, eps=1e-5)

        # Copy weights
        with torch.no_grad():
            rms_detach.weight.copy_(rms_standard.weight)

        x = torch.randn(2, 10, d)
        y_standard = rms_standard(x)
        y_detach = rms_detach(x)

        assert torch.allclose(y_standard, y_detach, atol=1e-5)
