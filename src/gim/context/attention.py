"""
Attention gradient scaling for GIM.

Provides utilities to scale gradients flowing through Q, K, V tensors
in attention mechanisms during backpropagation.
"""
import contextlib
import torch
import torch.nn.functional as F


class _ScaleGrad(torch.autograd.Function):
    """Autograd function that scales gradients by a constant factor during backward pass."""

    @staticmethod
    def forward(ctx, x, scale: float):
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, g):
        return g * ctx.scale, None


def scale_grad(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale gradients of a tensor by a constant factor during backward pass.

    Args:
        x: Input tensor.
        scale: Factor to multiply gradients by during backpropagation.

    Returns:
        Tensor with same values but scaled gradients.
    """
    return _ScaleGrad.apply(x, float(scale))


@contextlib.contextmanager
def _patch_sdpa_qkv_scales(q_scale: float, k_scale: float, v_scale: float):
    """Context manager that patches F.scaled_dot_product_attention with Q/K/V gradient scaling.

    Args:
        q_scale: Gradient scale for query tensors.
        k_scale: Gradient scale for key tensors.
        v_scale: Gradient scale for value tensors.
    """
    if not hasattr(F, "scaled_dot_product_attention"):
        raise RuntimeError(
            "Attention scale was requested and Pytorch model detected, "
            "but torch.nn.functional.scaled_dot_product_attention not found."
        )
    orig = F.scaled_dot_product_attention

    def sdpa(q, k, v, *args, **kw):
        q = scale_grad(q, q_scale)
        k = scale_grad(k, k_scale)
        v = scale_grad(v, v_scale)
        return orig(q, k, v, *args, **kw)

    F.scaled_dot_product_attention = sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = orig


def _tlens_qkv_scales(model, q_scale: float, k_scale: float, v_scale: float):
    """Register TransformerLens hooks to scale Q/K/V gradients.

    Args:
        model: TransformerLens HookedTransformer model.
        q_scale: Gradient scale for query tensors.
        k_scale: Gradient scale for key tensors.
        v_scale: Gradient scale for value tensors.

    Returns:
        Context manager that registers and removes hooks.
    """
    def hq(q, hook):
        return scale_grad(q, q_scale)

    def hk(k, hook):
        return scale_grad(k, k_scale)

    def hv(v, hook):
        return scale_grad(v, v_scale)

    fwd_hooks = [
        (lambda n: n.endswith(".attn.hook_q"), hq),
        (lambda n: n.endswith(".attn.hook_k"), hk),
        (lambda n: n.endswith(".attn.hook_v"), hv),
    ]
    if len(fwd_hooks) == 0:
        raise RuntimeError(
            "Attention scale was requested and TransformerLens model detected, "
            "but no Q/K/V hooks were found in the model."
        )
    return model.hooks(fwd_hooks=fwd_hooks, reset_hooks_end=True)
