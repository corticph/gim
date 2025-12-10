"""
Feature attribution using GIM gradient modifications.
"""
from typing import Any, Optional

import torch
from torch import nn

from gim.context import GIM
from gim.context.gim import _is_tlens_model


def _infer_baseline_token_id(tokenizer: Any) -> int:
    """Infer a baseline token ID from a tokenizer.

    Priority: space token -> pad_token_id -> unk_token_id
    """
    # Strategy 1: Use space token (most common approach)
    try:
        if hasattr(tokenizer, "encode"):
            space_ids = tokenizer.encode(" ", add_special_tokens=False)
            if space_ids:
                return space_ids[0]
    except Exception:
        pass

    # Strategy 2: Use pad_token_id if available
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id

    # Strategy 3: Use unk_token_id
    if hasattr(tokenizer, "unk_token_id") and tokenizer.unk_token_id is not None:
        return tokenizer.unk_token_id

    raise ValueError(
        "Could not infer baseline_token_id from tokenizer. "
        "Please provide baseline_token_id explicitly."
    )


def _explain_pytorch(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_token_id: Optional[int],
    target_position: int,
    baseline_token_id: int,
    **gim_kwargs,
) -> torch.Tensor:
    """Compute attributions for PyTorch nn.Module."""
    device = input_ids.device

    # Get embedding layer
    embed_fn = model.get_input_embeddings()

    # Create clean and corrupt embeddings
    with torch.no_grad():
        corrupt_ids = torch.full_like(input_ids, baseline_token_id)
        corrupt_embedding = embed_fn(corrupt_ids)

    clean_embedding = embed_fn(input_ids).clone().requires_grad_(True)
    clean_embedding.retain_grad()

    with GIM(model, **gim_kwargs):
        # Forward pass with embeddings instead of token IDs
        output = model(inputs_embeds=clean_embedding, use_cache=False)
        logits = output.logits if hasattr(output, "logits") else output

        # Get target logit
        target_logits = logits[:, target_position, :]
        if target_token_id is None:
            target_token_id = target_logits.argmax(dim=-1).item()

        # Select target and compute gradient
        target = target_logits[0, target_token_id]
        target.backward()

    # Compute attributions: (clean - corrupt) * grad, summed over d_model
    attributions = (
        (clean_embedding.grad * (clean_embedding.detach() - corrupt_embedding))
        .sum(dim=-1)
        .detach()
    )

    return attributions.squeeze(0)


def _explain_tlens(
    model: Any,  # HookedTransformer
    input_ids: torch.Tensor,
    target_token_id: Optional[int],
    target_position: int,
    baseline_token_id: int,
    **gim_kwargs,
) -> torch.Tensor:
    """Compute attributions for TransformerLens HookedTransformer."""
    device = input_ids.device

    # Create corrupt input and get embeddings
    with torch.no_grad():
        corrupt_ids = torch.full_like(input_ids, baseline_token_id)
        corrupt_embedding = model.embed(corrupt_ids)

    clean_embedding = model.embed(input_ids).clone().requires_grad_(True)
    clean_embedding.retain_grad()

    # Hook to replace embedding with our grad-enabled version
    def replace_embed(tensor, hook):
        return clean_embedding

    with GIM(model, **gim_kwargs):
        # Forward pass with hook to inject grad-enabled embedding
        logits = model.run_with_hooks(
            input_ids,
            fwd_hooks=[("hook_embed", replace_embed)]
        )

        # Get target logit
        target_logits = logits[:, target_position, :]
        if target_token_id is None:
            target_token_id = target_logits.argmax(dim=-1).item()

        target = target_logits[0, target_token_id]
        target.backward()

    # Compute attributions: (clean - corrupt) * grad, summed over d_model
    attributions = (
        (clean_embedding.grad * (clean_embedding.detach() - corrupt_embedding))
        .sum(dim=-1)
        .detach()
    )

    return attributions.squeeze(0)


def explain(
    model: Any,
    input_ids: torch.Tensor,
    *,
    target_token_id: Optional[int] = None,
    target_position: int = -1,
    baseline_token_id: Optional[int] = None,
    tokenizer: Optional[Any] = None,
    freeze_norm: bool = True,
    softmax_temperature: Optional[float] = 2.0,
    q_scale: Optional[float] = 0.25,
    k_scale: Optional[float] = 0.25,
    v_scale: Optional[float] = 0.5,
) -> torch.Tensor:
    """
    Compute feature attributions for model inference using GIM gradient modifications.

    Args:
        model: PyTorch nn.Module or TransformerLens HookedTransformer.
        input_ids: Input token IDs of shape [batch, seq_len] or [seq_len].
        target_token_id: Token ID to explain (e.g., predicted token). If None,
                         uses the argmax of logits at target_position.
        target_position: Position in sequence to explain (default: -1 for last token).
        baseline_token_id: Token ID used for baseline/counterfactual input. Required if
                           tokenizer is not provided.
        tokenizer: Optional tokenizer to infer baseline_token_id from (uses space token).
        freeze_norm: If True, detach LayerNorm/RMSNorm statistics during backward.
        softmax_temperature: Temperature for softmax backward pass (default: 2.0).
        q_scale, k_scale, v_scale: Gradient multipliers for attention Q/K/V.

    Returns:
        torch.Tensor: Feature importance scores of shape [seq_len].

    Raises:
        ValueError: If neither baseline_token_id nor tokenizer is provided.
        TypeError: If model type is not supported.

    Example:
        >>> import gim
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
        >>> attributions = gim.explain(model, input_ids, tokenizer=tokenizer)
    """
    # Input validation
    if baseline_token_id is None and tokenizer is None:
        raise ValueError(
            "Either baseline_token_id or tokenizer must be provided."
        )

    if baseline_token_id is None:
        baseline_token_id = _infer_baseline_token_id(tokenizer)

    # Ensure input_ids is 2D [batch, seq]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    # Prepare GIM kwargs
    gim_kwargs = {
        "freeze_norm": freeze_norm,
        "softmax_temperature": softmax_temperature,
        "q_scale": q_scale,
        "k_scale": k_scale,
        "v_scale": v_scale,
    }

    # Dispatch based on model type
    if _is_tlens_model(model):
        return _explain_tlens(
            model, input_ids, target_token_id, target_position,
            baseline_token_id, **gim_kwargs
        )
    elif isinstance(model, nn.Module):
        return _explain_pytorch(
            model, input_ids, target_token_id, target_position,
            baseline_token_id, **gim_kwargs
        )
    else:
        raise TypeError(
            f"Unsupported model type: {type(model)}. "
            "Expected PyTorch nn.Module or TransformerLens HookedTransformer."
        )
