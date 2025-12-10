"""
Softmax temperature scaling for GIM.

Provides utilities to modify the softmax backward pass to use a different
temperature than the forward pass, enabling temperature-scaled gradients.
"""
import contextlib
import torch
import torch.nn.functional as F


def stable_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute numerically stable softmax using log-sum-exp trick."""
    lse = torch.logsumexp(x, dim, keepdim=True)
    return torch.exp(x - lse)


class _SoftmaxBackwardTOnly(torch.autograd.Function):
    """Autograd function that uses temperature-scaled softmax only in backward pass.

    Forward: computes standard softmax(x)
    Backward: computes gradients as if softmax(x/T) was computed
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int, T: float):
        ctx.dim = int(dim)
        ctx.T = float(T)
        ctx.save_for_backward(x)
        return stable_softmax(x, dim=dim)

    @staticmethod
    def backward(ctx, gout: torch.Tensor):
        (x,) = ctx.saved_tensors
        dim, T = ctx.dim, ctx.T
        sT = stable_softmax(x / T, dim=dim)
        dot = (gout * sT).sum(dim=dim, keepdim=True)
        gin = sT * (gout - dot)
        return gin, None, None, None


def _softmax_bwT(x: torch.Tensor, *, dim=None, T=1.0):
    """Softmax with temperature-scaled backward pass.

    Args:
        x: Input tensor.
        dim: Dimension to apply softmax over (default: last dimension).
        T: Temperature for backward pass (forward always uses T=1).

    Returns:
        Softmax output with modified backward behavior.
    """
    if dim is None:
        dim = x.dim() - 1
    return _SoftmaxBackwardTOnly.apply(x, dim, float(T))


@contextlib.contextmanager
def _patch_softmax_backward_T_only(T: float):
    """Context manager that patches all softmax functions with temperature-scaled backward.

    Patches F.softmax, torch.softmax, and Tensor.softmax to use temperature T
    in the backward pass while keeping forward pass unchanged.

    Args:
        T: Temperature for backward pass (T > 1 produces softer attention gradients).
    """
    orig_F, orig_torch = F.softmax, torch.softmax
    orig_tensor_method = torch.Tensor.softmax

    def F_patched(input, dim=None):
        return _softmax_bwT(input, dim=dim, T=T)

    def torch_patched(input, dim):
        return _softmax_bwT(input, dim=dim, T=T)

    def tensor_method(self, dim):
        return F_patched(self, dim=dim)

    F.softmax = F_patched
    torch.softmax = torch_patched
    torch.Tensor.softmax = tensor_method
    try:
        yield
    finally:
        F.softmax = orig_F
        torch.softmax = orig_torch
        torch.Tensor.softmax = orig_tensor_method
