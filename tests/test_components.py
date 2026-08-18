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


class _GateProjMLP(torch.nn.Module):
    """Minimal SwiGLU-style gated MLP using the standard gate_proj/up_proj convention."""

    def __init__(self, d_model=8, d_hidden=12):
        super().__init__()
        self.gate_proj = torch.nn.Linear(d_model, d_hidden, bias=False)
        self.up_proj = torch.nn.Linear(d_model, d_hidden, bias=False)
        self.down_proj = torch.nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Phi3MLP(torch.nn.Module):
    """Minimal fused gate_up_proj MLP, matching the registered Phi3MLP convention."""

    def __init__(self, d_model=8, d_hidden=12):
        super().__init__()
        self.gate_up_proj = torch.nn.Linear(d_model, 2 * d_hidden, bias=False)
        self.down_proj = torch.nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class WeirdGatedMLP(torch.nn.Module):
    """An MLP with an unrecognized gating convention (not in any known registry)."""

    def __init__(self, d_model=8, d_hidden=12):
        super().__init__()
        self.my_gate = torch.nn.Linear(d_model, d_hidden, bias=False)
        self.my_up = torch.nn.Linear(d_model, d_hidden, bias=False)
        self.down_proj = torch.nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.my_gate(x)) * self.my_up(x))


class PlainMLP(torch.nn.Module):
    """An ordinary, non-gated MLP (no "gate" anywhere)."""

    def __init__(self, d_model=8, d_hidden=12):
        super().__init__()
        self.fc1 = torch.nn.Linear(d_model, d_hidden)
        self.fc2 = torch.nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TestMLPGateScaling:
    """Tests for gated-MLP gate/up gradient scaling."""

    def test_gate_up_pair_forward_unchanged(self):
        from gim.context.mlp import _patch_gate_up_grad_scales

        torch.manual_seed(0)
        mlp = _GateProjMLP()
        x = torch.randn(2, 5, 8)
        y_before = mlp(x)
        with _patch_gate_up_grad_scales(mlp, gate_scale=0.5, up_scale=0.5):
            y_after = mlp(x)
        assert torch.equal(y_before, y_after)

    def test_gate_up_pair_backward_scaled(self):
        from gim.context.mlp import _patch_gate_up_grad_scales

        torch.manual_seed(0)
        mlp = _GateProjMLP()
        x = torch.randn(2, 5, 8)

        mlp.zero_grad(set_to_none=True)
        mlp(x).sum().backward()
        gate_grad_base = mlp.gate_proj.weight.grad.clone()
        up_grad_base = mlp.up_proj.weight.grad.clone()

        mlp.zero_grad(set_to_none=True)
        with _patch_gate_up_grad_scales(mlp, gate_scale=0.5, up_scale=0.25):
            mlp(x).sum().backward()

        assert torch.allclose(mlp.gate_proj.weight.grad, 0.5 * gate_grad_base, atol=1e-6)
        assert torch.allclose(mlp.up_proj.weight.grad, 0.25 * up_grad_base, atol=1e-6)

    def test_gate_up_pair_hooks_removed_on_exit(self):
        from gim.context.mlp import _patch_gate_up_grad_scales

        torch.manual_seed(0)
        mlp = _GateProjMLP()
        x = torch.randn(2, 5, 8)

        with _patch_gate_up_grad_scales(mlp, gate_scale=0.5, up_scale=0.5):
            pass

        assert len(mlp.gate_proj._forward_hooks) == 0
        assert len(mlp.up_proj._forward_hooks) == 0

        mlp.zero_grad(set_to_none=True)
        mlp(x).sum().backward()
        grad_no_patch_1 = mlp.gate_proj.weight.grad.clone()

        mlp.zero_grad(set_to_none=True)
        mlp(x).sum().backward()
        grad_no_patch_2 = mlp.gate_proj.weight.grad.clone()

        assert torch.allclose(grad_no_patch_1, grad_no_patch_2, atol=1e-6)

    def test_fused_gate_up_backward_scaled(self):
        from gim.context.mlp import _patch_gate_up_grad_scales

        torch.manual_seed(0)
        mlp = Phi3MLP()
        x = torch.randn(2, 5, 8)

        mlp.zero_grad(set_to_none=True)
        mlp(x).sum().backward()
        fused_grad_base = mlp.gate_up_proj.weight.grad.clone()
        d_hidden = fused_grad_base.shape[0] // 2
        gate_grad_base, up_grad_base = fused_grad_base[:d_hidden], fused_grad_base[d_hidden:]

        mlp.zero_grad(set_to_none=True)
        with _patch_gate_up_grad_scales(mlp, gate_scale=0.1, up_scale=0.9):
            mlp(x).sum().backward()

        fused_grad_patched = mlp.gate_up_proj.weight.grad.clone()
        gate_grad_patched = fused_grad_patched[:d_hidden]
        up_grad_patched = fused_grad_patched[d_hidden:]

        assert torch.allclose(gate_grad_patched, 0.1 * gate_grad_base, atol=1e-6)
        assert torch.allclose(up_grad_patched, 0.9 * up_grad_base, atol=1e-6)

    def test_unrecognized_gate_raises(self):
        from gim.context.mlp import _patch_gate_up_grad_scales

        mlp = WeirdGatedMLP()
        with pytest.raises(RuntimeError, match="gate-like"):
            with _patch_gate_up_grad_scales(mlp, gate_scale=0.5, up_scale=0.5):
                pass

    def test_unrecognized_gate_bypassed_via_gim_none(self):
        from gim import GIM

        mlp = WeirdGatedMLP()
        x = torch.randn(2, 5, 8)
        with GIM(mlp, freeze_norm=False, softmax_temperature=None,
                 q_scale=None, k_scale=None, v_scale=None,
                 gate_scale=None, up_scale=None):
            mlp(x).sum().backward()
        assert mlp.my_gate.weight.grad is not None

    def test_plain_mlp_is_noop(self):
        from gim.context.mlp import _patch_gate_up_grad_scales

        torch.manual_seed(0)
        mlp = PlainMLP()
        x = torch.randn(2, 5, 8)

        mlp.zero_grad(set_to_none=True)
        mlp(x).sum().backward()
        grad_base = mlp.fc1.weight.grad.clone()

        mlp.zero_grad(set_to_none=True)
        with _patch_gate_up_grad_scales(mlp, gate_scale=0.5, up_scale=0.5):
            mlp(x).sum().backward()

        assert torch.allclose(mlp.fc1.weight.grad, grad_base, atol=1e-6)


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
