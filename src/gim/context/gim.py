"""
Main GIM context manager.

Applies gradient modifications for better feature attribution during
the forward/backward pass.
"""
from typing import Any, Optional
from contextlib import ExitStack, contextmanager
from torch import nn

from gim.context.attention import _patch_sdpa_qkv_scales, _tlens_qkv_scales
from gim.context.mlp import _patch_gate_up_grad_scales, _tlens_gate_up_scales
from gim.context.norm import _swap_norms_with_detach, _tlens_detach_norm_scales
from gim.context.softmax import _patch_softmax_backward_T_only


def _is_tlens_model(obj: Any) -> bool:
    """Check if an object is a TransformerLens HookedTransformer model."""
    return hasattr(obj, "hook_dict") and hasattr(obj, "hooks")


@contextmanager
def GIM(model: Any,
        *,
        freeze_norm: bool = True,
        softmax_temperature: Optional[float] = 2,
        q_scale: Optional[float] = 0.25,
        k_scale: Optional[float] = 0.25,
        v_scale: Optional[float] = 0.5,
        gate_scale: Optional[float] = 0.5,
        up_scale: Optional[float] = 0.5):
    """
    Apply GIM tweaks for the duration of the `with` block and restore on exit.

    Args:
      model: PyTorch nn.Module or TransformerLens HookedTransformer.
      freeze_norm: if True, detach LN/RMSNorm statistics (std/rms).
      softmax_temperature: if set (T != 1), compute softmax backward at x/T.
      q_scale, k_scale, v_scale: gradient multipliers for attention Q/K/V respectively.
                                 e.g., q_scale=0.25, k_scale=0.25, v_scale=0.5  # ÷4, ÷4, ÷2
      gate_scale, up_scale: gradient multipliers for the gate/up branches of gated
                            MLP blocks (SwiGLU/GeGLU), e.g. gate_scale=0.5, up_scale=0.5  # ÷2, ÷2
    """
    with ExitStack() as stack:
        # 1) Freeze norm stats
        if freeze_norm:
            if _is_tlens_model(model):
                stack.enter_context(_tlens_detach_norm_scales(model))
            elif isinstance(model, nn.Module):
                stack.enter_context(_swap_norms_with_detach(model))
        # 2) Softmax backward temperature
        if softmax_temperature is not None and float(softmax_temperature) != 1.0:
            stack.enter_context(_patch_softmax_backward_T_only(float(softmax_temperature)))
        # 3) Attention Q/K/V gradient scales
        if any(s is not None for s in (q_scale, k_scale, v_scale)):
            qs = 1.0 if q_scale is None else float(q_scale)
            ks = 1.0 if k_scale is None else float(k_scale)
            vs = 1.0 if v_scale is None else float(v_scale)
            if _is_tlens_model(model):
                stack.enter_context(_tlens_qkv_scales(model, qs, ks, vs))
            else:
                stack.enter_context(_patch_sdpa_qkv_scales(qs, ks, vs))
        # 4) MLP gate/up gradient scales
        if any(s is not None for s in (gate_scale, up_scale)):
            gs = 1.0 if gate_scale is None else float(gate_scale)
            us = 1.0 if up_scale is None else float(up_scale)
            if _is_tlens_model(model):
                stack.enter_context(_tlens_gate_up_scales(model, gs, us))
            elif isinstance(model, nn.Module):
                stack.enter_context(_patch_gate_up_grad_scales(model, gs, us))
        yield
