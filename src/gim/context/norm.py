"""
Normalization layer modifications for GIM.

Provides LayerNorm and RMSNorm variants that detach normalization statistics
(std/rms) from the computation graph during backpropagation.
"""
import contextlib
import torch
from torch import nn


class LayerNormDetach(nn.LayerNorm):
    """LayerNorm that detaches the standard deviation from the backward pass.

    This prevents gradients from flowing through the normalization statistics,
    which can improve gradient-based feature attribution quality.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = (x - mean).pow(2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt().detach()
        y = (x - mean) / std
        if self.elementwise_affine:
            view = (1,) * (x.ndim - len(self.normalized_shape)) + tuple(self.normalized_shape)
            y = y * self.weight.view(view) + self.bias.view(view)
        return y


class RMSNormDetach(nn.Module):
    """RMSNorm that detaches the RMS value from the backward pass.

    This prevents gradients from flowing through the normalization statistics,
    which can improve gradient-based feature attribution quality.
    """

    def __init__(self, normalized_shape, eps: float | None, elementwise_affine: bool = True, device=None, dtype=None):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps or 1e-5
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        rms = (x.pow(2).mean(dim=dims, keepdim=True) + self.eps).sqrt().detach()
        y = x / rms
        if self.elementwise_affine:
            view = (1,) * (x.ndim - len(self.normalized_shape)) + tuple(self.normalized_shape)
            y = y * self.weight.view(view)
        return y


@contextlib.contextmanager
def _swap_norms_with_detach(root: nn.Module):
    """Context manager that swaps LayerNorm/RMSNorm modules with detached variants.

    Recursively finds all LayerNorm and RMSNorm modules in the model and replaces
    them with LayerNormDetach and RMSNormDetach respectively. Restores original
    modules on exit.

    Args:
        root: Root nn.Module to search for normalization layers.
    """
    swaps = []
    try:
        for parent in list(root.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.LayerNorm):
                    new = LayerNormDetach(child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine)
                    if child.elementwise_affine:
                        with torch.no_grad():
                            new.weight.copy_(child.weight)
                            new.bias.copy_(child.bias)
                    setattr(parent, name, new)
                    swaps.append((parent, name, child))
                elif hasattr(nn, "RMSNorm") and isinstance(child, nn.RMSNorm):
                    new = RMSNormDetach(
                        child.normalized_shape,
                        eps=child.eps,
                        elementwise_affine=child.elementwise_affine,
                        device=(child.weight.device if child.elementwise_affine else None),
                        dtype=(child.weight.dtype if child.elementwise_affine else None),
                    )
                    if child.elementwise_affine:
                        with torch.no_grad():
                            new.weight.copy_(child.weight)
                    setattr(parent, name, new)
                    swaps.append((parent, name, child))
        yield
    finally:
        for parent, name, old in reversed(swaps):
            setattr(parent, name, old)


def _tlens_detach_norm_scales(model) -> contextlib.AbstractContextManager:
    """Register TransformerLens hooks to detach normalization scale from backward pass.

    In TransformerLens, normalization layers expose hook_scale which contains the
    computed scale factor. This function hooks into those to detach them.

    Args:
        model: TransformerLens HookedTransformer model.

    Returns:
        Context manager that registers and removes hooks.
    """
    def detach_scale(scale: torch.Tensor, hook):
        return scale.detach()

    return model.hooks(fwd_hooks=[(lambda n: n.endswith("hook_scale"), detach_scale)], reset_hooks_end=True)
