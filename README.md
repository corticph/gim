<p align="center">
  <img src="https://raw.githubusercontent.com/corticph/gim/main/assets/logo.png" alt="GIM Logo" width="300">
</p>

![Python Version](https://img.shields.io/pypi/pyversions/gim-explain)
![PyPI](https://img.shields.io/pypi/v/gim-explain)
![License](https://img.shields.io/github/license/corticph/gim)

# GIM: Gradient Interaction Modifications


## Installation

```bash
pip install gim-explain

# With TransformerLens support
pip install gim-explain[tlens]
```

## Quick Start

### Feature Attribution with `explain()`

```python
import gim
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
attributions = gim.explain(model, input_ids, tokenizer=tokenizer)

# attributions is a tensor of shape [seq_len] with importance scores per token
```

### Using the GIM Context Manager

For more control, use the `GIM` context manager directly. This is useful if you want to use GIM for circuit discovery or network pruning. You wrap the model and run your gradient-based method as usual (e.g., [Edge Attribution Patching](https://github.com/hannamw/EAP-IG/tree/7af394a5662de8b23ad6154716a0cd3993d447a3)). The wrapper will automatically modify the backpropagation. 

```python
import gim
import torch.nn.functional as F

with gim.GIM(model):
    logits = model(input_ids)
    loss = F.cross_entropy(logits[:, -1], target)
    loss.backward()
    # Gradients are now modified by GIM
```

## How It Works

GIM applies four gradient modifications during backpropagation:

1. **Norm Freezing**: Detaches LayerNorm/RMSNorm statistics from the backward pass
2. **Softmax Temperature**: Applies temperature scaling to softmax gradients (softer attention)
3. **Q/K/V Scaling**: Scales gradients for query, key, and value tensors in attention
4. **MLP Gate Scaling**: Scales gradients for the gate and up-projection branches of gated MLP blocks (SwiGLU/GeGLU), at the point where they're multiplied together

As shown in the paper, these modifications improve the quality of gradient-based feature attributions.

MLP gate scaling supports the standard `gate_proj`/`up_proj` convention (Llama, Mistral, Gemma, Qwen2, and most modern open decoder LLMs), Phi3-style fused `gate_up_proj`, and Mixtral-style MoE experts. If a model has a gated MLP that doesn't match a known convention, GIM raises an error instead of silently skipping the correction — pass `gate_scale=None, up_scale=None` to opt out explicitly.

> **Note:** Releases up to and including `0.1.4` had a bug where this MLP gate/up scaling was never implemented — only the Q/K/V attention scaling was applied. On any gated-MLP model (Llama, Mistral, Gemma, Qwen2, Mixtral, Phi3, etc.), this meant `gate_proj`/`up_proj` (and equivalents) gradients were left unscaled. If you computed attributions with an earlier version on a gated-MLP model, consider recomputing them with this fix applied.

## API Reference

### `gim.explain()`

```python
gim.explain(
    model,                          # PyTorch nn.Module or TransformerLens HookedTransformer
    input_ids,                      # Token IDs [batch, seq_len] or [seq_len]
    *,
    target_token_id=None,           # Token to explain (default: argmax prediction)
    target_position=-1,             # Position to explain (default: last token)
    baseline_token_id=None,         # Baseline token for counterfactual
    tokenizer=None,                 # Tokenizer to infer baseline from
    freeze_norm=True,               # Detach norm statistics
    softmax_temperature=2.0,        # Temperature for softmax backward
    q_scale=0.25,                   # Query gradient scale
    k_scale=0.25,                   # Key gradient scale
    v_scale=0.5,                    # Value gradient scale
    gate_scale=0.5,                 # Gated-MLP gate-branch gradient scale
    up_scale=0.5,                   # Gated-MLP up-branch gradient scale
)
```

### `gim.GIM()`

```python
with gim.GIM(
    model,                          # PyTorch nn.Module or TransformerLens HookedTransformer
    *,
    freeze_norm=True,
    softmax_temperature=2.0,
    q_scale=0.25,
    k_scale=0.25,
    v_scale=0.5,
    gate_scale=0.5,
    up_scale=0.5,
):
    # Your forward/backward code here
    pass
```
## Adding Support for a New Model

Most models need **no code changes**. GIM's PyTorch path patches `F.scaled_dot_product_attention` (so any model using it gets Q/K/V scaling automatically) and detects gated MLPs generically by looking for `gate_proj`/`up_proj` attributes (so any model using that naming convention gets gate/up scaling automatically too). Its TransformerLens path works the same way via hook names, for any model built through `HookedTransformer`.

You only need to add something when GIM raises one of these errors:

- `GIM found gate-like Linear layer(s) it doesn't know how to apply the gate/up gradient-scaling rule to: ...` — a PyTorch/HuggingFace model whose gated MLP doesn't use the standard `gate_proj`/`up_proj` naming.
- `GIM found an MLP module '...' of type '...' that it doesn't recognize.` — a TransformerLens model using an MLP class GIM hasn't seen.

Both live in `src/gim/context/mlp.py`. GIM raises rather than silently skipping the correction because applying half the correction (or none) would look identical to correct behavior while being wrong — so treat these errors as "please add this architecture," not as something to suppress.

### PyTorch / HuggingFace: fused gate+up projection

If the model computes the gate and up branches from a single fused `nn.Linear` and splits the output (like Phi3's `gate_up_proj`), add it to `_FUSED_GATE_UP_CLASSES`:

```python
_FUSED_GATE_UP_CLASSES = {
    "Phi3MLP": {"attr": "gate_up_proj", "dim": -1, "order": ("gate", "up")},
    "NewModelMLP": {"attr": "gate_up_proj", "dim": -1, "order": ("gate", "up")},  # add here
}
```

Read the model's actual `forward()` to confirm the attribute name, the `chunk()` dimension, and which half is gate vs. up — getting `order` backwards silently swaps the two scales.

### PyTorch / HuggingFace: MoE experts with non-standard names

If each MoE expert uses different attribute names for gate/up/down (like Mixtral's `w1`/`w2`/`w3`), add it to `_MOE_EXPERT_TRIPLE_CLASSES`:

```python
_MOE_EXPERT_TRIPLE_CLASSES = {
    "MixtralBlockSparseTop2MLP": {"gate": "w1", "up": "w3", "down": "w2"},
    "NewMoeExpertMLP": {"gate": "w1", "up": "w3", "down": "w2"},  # add here
}
```

### TransformerLens: a new MLP class

If TransformerLens adds a new MLP class (check `transformer_lens/factories/mlp_factory.py` for the full list it can construct), add its class name to the matching set in `mlp.py`:

- `_TLENS_KNOWN_PLAIN_MLP_CLASSES` if it isn't gated at all.
- `_TLENS_KNOWN_GATED_MLP_CLASSES` if it exposes `hook_pre`/`hook_pre_linear` like `GatedMLP`.
- `_TLENS_KNOWN_MOE_CLASSES` / `_TLENS_KNOWN_MOE_EXPERT_CLASSES` if it's an MoE wrapper/expert like `MoE`/`MoEGatedMLP`.

If its hook names or roles differ from the existing classes, you'll also need a new branch in `_tlens_gate_up_scales`'s `fwd_hooks` list.

### Test it

Add a case to `tests/test_mlp_gate_architectures.py` using a tiny, randomly-initialized config (no downloads). Compare the gate/up gradients on the **last** layer with vs. without `GIM(..., gate_scale=..., up_scale=...)` — an earlier layer's gradient is a blend with later layers' unscaled residual skip-connections, so only the last layer's gradient is an exact multiple of the baseline (see comments in that file for why).

### A note on attention

Unlike the gate/up path, GIM's Q/K/V scaling has no equivalent safety net: it only works if the model calls `F.scaled_dot_product_attention` (PyTorch) or exposes `hook_q`/`hook_k`/`hook_v` (TransformerLens). A model using a custom attention kernel (e.g. calling FlashAttention or xFormers directly) will silently *not* get Q/K/V scaling. If you're adding support for such a model, check whether its attention path is actually being patched before trusting the results.

## Citation
```bibtex
@misc{edin2025gimimprovedinterpretabilitylarge,
      title={GIM: Improved Interpretability for Large Language Models}, 
      author={Joakim Edin and Róbert Csordás and Tuukka Ruotsalo and Zhengxuan Wu and Maria Maistro and Casper L. Christensen and Jing Huang and Lars Maaløe},
      year={2025},
      eprint={2505.17630},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.17630}, 
}
```

## License

MIT
