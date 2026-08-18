"""
Gated-MLP gradient scaling for GIM.

Applies the same gradient-normalization rule used for attention Q/K/V to the
gate/up multiplication inside gated MLP blocks (SwiGLU/GeGLU-style:
``down_proj(act_fn(gate_proj(x)) * up_proj(x))``): whenever two tensors are
multiplied, halve the gradient flowing back through each operand.

Covers several known conventions:
  - The standard ``gate_proj``/``up_proj`` pair (Llama, Mistral, Gemma, Qwen2,
    and most modern open decoder LLMs) - detected generically by attribute
    name and ``nn.Linear`` type (no class-name allowlist), so new models
    reusing this convention need no code changes here.
  - Fused ``gate_up_proj`` + ``chunk(2)`` (Phi3-style).
  - MoE expert gate/up/down triples with non-standard names (Mixtral-style).

If a model contains a gate-like Linear layer that doesn't match any known
convention, this raises instead of silently skipping it - applying half of
the correction (or none) would look identical to correct behavior while
being wrong.
"""
import contextlib
import re
from typing import Any

import torch
from torch import nn

from gim.context.attention import scale_grad

# Class name -> {"attr": fused Linear attribute name, "dim": split dim,
# "order": which half is which after `.chunk(2, dim=dim)`}.
_FUSED_GATE_UP_CLASSES = {
    "Phi3MLP": {"attr": "gate_up_proj", "dim": -1, "order": ("gate", "up")},
}

# Class name -> {"gate": attr, "up": attr, "down": attr} for MoE experts that
# don't use the gate_proj/up_proj naming convention.
_MOE_EXPERT_TRIPLE_CLASSES = {
    "MixtralBlockSparseTop2MLP": {"gate": "w1", "up": "w3", "down": "w2"},
}

# TransformerLens MLP classes this module knows how to handle (or knows to
# leave alone because they aren't gated). Sourced from
# transformer_lens/factories/mlp_factory.py, which constructs exactly one of
# these based on model config - a closed set.
_TLENS_KNOWN_PLAIN_MLP_CLASSES = {"MLP"}
_TLENS_KNOWN_GATED_MLP_CLASSES = {"GatedMLP", "GatedMLP4Bit"}
_TLENS_KNOWN_MOE_CLASSES = {"MoE"}
_TLENS_KNOWN_MOE_EXPERT_CLASSES = {"MoEGatedMLP"}


def _find_gate_up_targets(root: nn.Module):
    """Find gate/up multiplication targets in a PyTorch model.

    Returns:
        (targets, claimed_param_ids) where targets is a list of either
        ("pair", gate_linear, up_linear) or ("fused", fused_linear, dim, order),
        and claimed_param_ids is the set of id() of every Linear module already
        accounted for by a target (used by the unrecognized-gate safety net).
    """
    targets = []
    claimed_param_ids = set()

    for _, module in root.named_modules():
        cls_name = type(module).__name__
        gate_proj = getattr(module, "gate_proj", None)
        up_proj = getattr(module, "up_proj", None)
        if isinstance(gate_proj, nn.Linear) and isinstance(up_proj, nn.Linear):
            targets.append(("pair", gate_proj, up_proj))
            claimed_param_ids.update((id(gate_proj), id(up_proj)))
            continue

        fused_spec = _FUSED_GATE_UP_CLASSES.get(cls_name)
        if fused_spec is not None:
            fused = getattr(module, fused_spec["attr"], None)
            if isinstance(fused, nn.Linear):
                targets.append(("fused", fused, fused_spec["dim"], fused_spec["order"]))
                claimed_param_ids.add(id(fused))
                continue

        triple_spec = _MOE_EXPERT_TRIPLE_CLASSES.get(cls_name)
        if triple_spec is not None:
            gate = getattr(module, triple_spec["gate"], None)
            up = getattr(module, triple_spec["up"], None)
            if isinstance(gate, nn.Linear) and isinstance(up, nn.Linear):
                targets.append(("pair", gate, up))
                claimed_param_ids.update((id(gate), id(up)))
                continue

    return targets, claimed_param_ids


def _scan_for_unrecognized_gates(root: nn.Module, claimed_param_ids: set) -> None:
    """Raise if the model has gate-like Linear layers not covered by a known pattern."""
    unrecognized = []
    for name, module in root.named_modules():
        looks_mlp_ish = any(
            token in (name.lower() + type(module).__name__.lower())
            for token in ("mlp", "ffn", "feedforward", "feed_forward")
        )
        if not looks_mlp_ish:
            continue
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and "gate" in child_name.lower() and id(child) not in claimed_param_ids:
                qualified = f"{name}.{child_name}" if name else child_name
                unrecognized.append(f"{qualified} ({type(module).__name__})")

    if unrecognized:
        raise RuntimeError(
            "GIM found gate-like Linear layer(s) it doesn't know how to apply the "
            "gate/up gradient-scaling rule to: " + ", ".join(unrecognized) + ". "
            "This looks like a gated-MLP architecture GIM doesn't yet support "
            "(known conventions: gate_proj/up_proj, Phi3-style fused gate_up_proj, "
            "and Mixtral-style w1/w2/w3 experts). Pass gate_scale=None, up_scale=None "
            "to GIM(...) to skip this check if you want to proceed without the "
            "correction, or open an issue with the model architecture."
        )


@contextlib.contextmanager
def _patch_gate_up_grad_scales(root: nn.Module, gate_scale: float, up_scale: float):
    """Context manager that scales gate/up gradients in gated MLP blocks.

    Args:
        root: Root nn.Module to search for gated MLP blocks.
        gate_scale: Gradient scale for the gate-branch tensor.
        up_scale: Gradient scale for the up-branch (linear) tensor.
    """
    targets, claimed_param_ids = _find_gate_up_targets(root)
    _scan_for_unrecognized_gates(root, claimed_param_ids)

    handles = []
    try:
        for target in targets:
            if target[0] == "pair":
                _, gate_linear, up_linear = target
                handles.append(gate_linear.register_forward_hook(
                    lambda m, i, o, s=gate_scale: scale_grad(o, s)))
                handles.append(up_linear.register_forward_hook(
                    lambda m, i, o, s=up_scale: scale_grad(o, s)))
            else:  # "fused"
                _, fused_linear, dim, order = target
                scales = {"gate": gate_scale, "up": up_scale}

                def fused_hook(m, i, o, dim=dim, order=order, scales=scales):
                    first, second = o.chunk(2, dim=dim)
                    first = scale_grad(first, scales[order[0]])
                    second = scale_grad(second, scales[order[1]])
                    return torch.cat([first, second], dim=dim)

                handles.append(fused_linear.register_forward_hook(fused_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def _tlens_gate_up_scales(model: Any, gate_scale: float, up_scale: float):
    """Register TransformerLens hooks to scale gate/up gradients in gated MLPs.

    Validates that every MLP-like submodule in the model is a class this
    module knows about, raising instead of silently skipping unrecognized ones.

    Args:
        model: TransformerLens HookedTransformer model.
        gate_scale: Gradient scale for the gate-branch tensor.
        up_scale: Gradient scale for the up-branch (linear) tensor.

    Returns:
        Context manager that registers and removes hooks.
    """
    known_classes = (
        _TLENS_KNOWN_PLAIN_MLP_CLASSES
        | _TLENS_KNOWN_GATED_MLP_CLASSES
        | _TLENS_KNOWN_MOE_CLASSES
        | _TLENS_KNOWN_MOE_EXPERT_CLASSES
    )
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in ("mlp",) or re.search(r"\.experts\.\d+$", name):
            cls_name = type(module).__name__
            if cls_name not in known_classes:
                raise RuntimeError(
                    f"GIM found an MLP module '{name}' of type '{cls_name}' that it "
                    "doesn't recognize. TransformerLens's known MLP types are "
                    f"{sorted(known_classes)}. Pass gate_scale=None, up_scale=None to "
                    "GIM(...) to skip this check, or open an issue with the model "
                    "architecture."
                )

    def hook_gate(x, hook):
        return scale_grad(x, gate_scale)

    def hook_up(x, hook):
        return scale_grad(x, up_scale)

    fwd_hooks = [
        (lambda n: n.endswith(".mlp.hook_pre"), hook_gate),
        (lambda n: n.endswith(".mlp.hook_pre_linear"), hook_up),
        (lambda n: bool(re.search(r"\.experts\.\d+\.hook_gate$", n)), hook_gate),
        (lambda n: bool(re.search(r"\.experts\.\d+\.hook_pre$", n)), hook_up),
    ]
    return model.hooks(fwd_hooks=fwd_hooks, reset_hooks_end=True)
