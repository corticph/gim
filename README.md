<p align="center">
  <img src="https://raw.githubusercontent.com/corticph/gim/main/assets/logo.png" alt="GIM Logo" width="300">
</p>

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

GIM applies three gradient modifications during backpropagation:

1. **Norm Freezing**: Detaches LayerNorm/RMSNorm statistics from the backward pass
2. **Softmax Temperature**: Applies temperature scaling to softmax gradients (softer attention)
3. **Q/K/V Scaling**: Scales gradients for query, key, and value tensors in attention

As shown in the paper, these modifications improve the quality of gradient-based feature attributions.

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
):
    # Your forward/backward code here
    pass
```
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
