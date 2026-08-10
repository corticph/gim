"""
Tests for GIM's gate/up gradient scaling across real transformer architectures.

Uses tiny, randomly-initialized configs (no downloads) for each architecture.
"""
import pytest
import torch
import torch.nn.functional as F

from gim import GIM
from tests.models import TinyLM

transformers = pytest.importorskip("transformers")

V, B, T = 80, 2, 8


def _tokens():
    torch.manual_seed(0)
    tokens = torch.randint(0, V, (B, T))
    return tokens[:, :-1], tokens[:, 1:]


def _assert_scaled(actual: torch.Tensor, baseline: torch.Tensor, scale: float):
    """Assert actual == scale * baseline element-wise.

    The gate/up gradient scaling is a single scale_grad() applied exactly at
    the multiply's operands; everything downstream into the weight gradient
    is linear, so this holds exactly (not just approximately) - confirmed by
    hand for every architecture below (max abs diff was 0.0 in every case).
    """
    assert torch.allclose(actual, scale * baseline, atol=1e-6), \
        f"max diff {(actual - scale * baseline).abs().max().item()}"


def _find_gate_up_linear_pair(model):
    """Find the LAST module exposing gate_proj/up_proj Linear submodules.

    The last decoder layer's gate/up gradient is a clean function of only its
    own local multiply (nothing downstream re-scales it further), so it's
    exactly scaled. An earlier layer's gradient is a blend of its own scaled
    branch and later layers' unscaled residual skip-connections, so it would
    NOT be exactly scaled - that's inherent to how per-layer gradient scaling
    composes across depth (also true of the existing q/k/v scaling), not
    specific to this fix.
    """
    found = None
    for _, module in model.named_modules():
        if hasattr(module, "gate_proj") and hasattr(module, "up_proj"):
            found = (module.gate_proj, module.up_proj)
    if found is None:
        raise AssertionError("no gate_proj/up_proj found in model")
    return found


def _backward(model, x, y):
    model.zero_grad(set_to_none=True)
    logits = model(x).logits
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()


_STANDARD_ARCH_CONFIGS = ["LlamaConfig", "MistralConfig", "GemmaConfig", "Qwen2Config"]


@pytest.mark.parametrize("config_name", _STANDARD_ARCH_CONFIGS)
def test_standard_gate_proj_up_proj_architectures_scaled(config_name):
    """gate_proj/up_proj gradients are exactly halved with only gate/up scaling enabled."""
    ConfigCls = getattr(transformers, config_name)
    ModelCls = getattr(transformers, config_name.replace("Config", "ForCausalLM"))
    cfg = ConfigCls(
        vocab_size=V, hidden_size=32, intermediate_size=37, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = ModelCls(cfg)
    model.eval()
    x, y = _tokens()
    gate_linear, up_linear = _find_gate_up_linear_pair(model)

    _backward(model, x, y)
    gate_base = gate_linear.weight.grad.clone()
    up_base = up_linear.weight.grad.clone()

    with GIM(model, freeze_norm=False, softmax_temperature=None,
             q_scale=None, k_scale=None, v_scale=None,
             gate_scale=0.5, up_scale=0.5):
        _backward(model, x, y)

    _assert_scaled(gate_linear.weight.grad, gate_base, 0.5)
    _assert_scaled(up_linear.weight.grad, up_base, 0.5)


@pytest.mark.parametrize("config_name", _STANDARD_ARCH_CONFIGS)
def test_standard_gate_proj_up_proj_architectures_default_gim(config_name):
    """Default GIM() (all modifications on) still produces gradients and differs from no-GIM."""
    ConfigCls = getattr(transformers, config_name)
    ModelCls = getattr(transformers, config_name.replace("Config", "ForCausalLM"))
    cfg = ConfigCls(
        vocab_size=V, hidden_size=32, intermediate_size=37, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = ModelCls(cfg)
    model.eval()
    x, y = _tokens()
    gate_linear, up_linear = _find_gate_up_linear_pair(model)

    _backward(model, x, y)
    gate_base = gate_linear.weight.grad.clone()

    with GIM(model):
        _backward(model, x, y)
    assert gate_linear.weight.grad is not None
    assert up_linear.weight.grad is not None
    assert not torch.allclose(gate_linear.weight.grad, gate_base)


def test_asymmetric_gate_up_scales():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=V, hidden_size=32, intermediate_size=37, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    model.eval()
    x, y = _tokens()
    gate_linear, up_linear = _find_gate_up_linear_pair(model)

    _backward(model, x, y)
    gate_base = gate_linear.weight.grad.clone()
    up_base = up_linear.weight.grad.clone()

    with GIM(model, freeze_norm=False, softmax_temperature=None,
             q_scale=None, k_scale=None, v_scale=None,
             gate_scale=0.1, up_scale=0.9):
        _backward(model, x, y)

    _assert_scaled(gate_linear.weight.grad, gate_base, 0.1)
    _assert_scaled(up_linear.weight.grad, up_base, 0.9)


def test_gim_disabled_matches_no_gim_gated_mlp():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=V, hidden_size=32, intermediate_size=37, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    model.eval()
    x, y = _tokens()
    gate_linear, _ = _find_gate_up_linear_pair(model)

    _backward(model, x, y)
    gate_no_gim = gate_linear.weight.grad.clone()

    with GIM(model, freeze_norm=False, softmax_temperature=None,
             q_scale=None, k_scale=None, v_scale=None,
             gate_scale=None, up_scale=None):
        _backward(model, x, y)
    gate_disabled = gate_linear.weight.grad.clone()

    assert torch.allclose(gate_no_gim, gate_disabled, rtol=1e-4)


def test_phi3_fused_gate_up_proj():
    from transformers import Phi3Config, Phi3ForCausalLM

    cfg = Phi3Config(
        vocab_size=V, hidden_size=32, intermediate_size=37, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    model = Phi3ForCausalLM(cfg)
    model.eval()
    x, y = _tokens()
    mlp = None
    for _, module in model.named_modules():
        if hasattr(module, "gate_up_proj"):
            mlp = module  # keep the last match - see _find_gate_up_linear_pair
    assert mlp is not None

    _backward(model, x, y)
    fused_base = mlp.gate_up_proj.weight.grad.clone()
    half = fused_base.shape[0] // 2
    gate_base, up_base = fused_base[:half], fused_base[half:]

    with GIM(model, freeze_norm=False, softmax_temperature=None,
             q_scale=None, k_scale=None, v_scale=None,
             gate_scale=0.1, up_scale=0.9):
        _backward(model, x, y)
    fused_scaled = mlp.gate_up_proj.weight.grad.clone()

    _assert_scaled(fused_scaled[:half], gate_base, 0.1)
    _assert_scaled(fused_scaled[half:], up_base, 0.9)


def test_mixtral_moe_expert_gate_up():
    from transformers import MixtralConfig, MixtralForCausalLM

    cfg = MixtralConfig(
        vocab_size=V, hidden_size=32, intermediate_size=37, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        num_local_experts=4, num_experts_per_tok=2,
    )
    torch.manual_seed(0)
    model = MixtralForCausalLM(cfg)
    model.eval()
    x, y = _tokens()
    experts = model.model.layers[-1].block_sparse_moe.experts

    # set_to_none=False so experts that weren't routed any tokens keep a zero
    # grad instead of None (0 == 0.5 * 0, so allclose still holds for them).
    model.zero_grad(set_to_none=False)
    logits = model(x).logits
    F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).backward()
    w1_base = torch.stack([e.w1.weight.grad.clone() for e in experts])
    w3_base = torch.stack([e.w3.weight.grad.clone() for e in experts])

    model.zero_grad(set_to_none=False)
    with GIM(model, freeze_norm=False, softmax_temperature=None,
             q_scale=None, k_scale=None, v_scale=None,
             gate_scale=0.5, up_scale=0.5):
        logits = model(x).logits
        F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).backward()
    w1_scaled = torch.stack([e.w1.weight.grad.clone() for e in experts])
    w3_scaled = torch.stack([e.w3.weight.grad.clone() for e in experts])

    _assert_scaled(w1_scaled, w1_base, 0.5)
    _assert_scaled(w3_scaled, w3_base, 0.5)


def _backward_tinylm(model, x, y):
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()


def test_non_gated_control_no_error_and_noop():
    """TinyLM's plain GELU MLP has no gate_proj/up_proj: gate/up scaling must no-op, not error."""
    model = TinyLM(vocab_size=V, d_model=32, n_layers=2, n_heads=4)
    model.eval()
    x, y = _tokens()

    _backward_tinylm(model, x, y)
    mlp_grad_no_gim = model.blocks[0].mlp[0].weight.grad.clone()

    with GIM(model, freeze_norm=False, softmax_temperature=None,
             q_scale=None, k_scale=None, v_scale=None,
             gate_scale=0.5, up_scale=0.5):
        _backward_tinylm(model, x, y)
    mlp_grad_with_gate_scale = model.blocks[0].mlp[0].weight.grad.clone()

    assert torch.allclose(mlp_grad_no_gim, mlp_grad_with_gate_scale, rtol=1e-4)


tlens = pytest.importorskip("transformer_lens")


def test_tlens_gated_mlp_scaled():
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=2, d_model=32, n_ctx=64, d_head=8, n_heads=4, d_vocab=V,
        act_fn="silu", gated_mlp=True, normalization_type="RMS",
    )
    torch.manual_seed(0)
    model = HookedTransformer(cfg)
    model.eval()
    x, y = _tokens()

    def run():
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()

    run()
    # Last block: its gate/up gradient is a clean function of only its own
    # local multiply, unlike an earlier block's (see _find_gate_up_linear_pair).
    gate_base = model.blocks[-1].mlp.W_gate.grad.clone()
    up_base = model.blocks[-1].mlp.W_in.grad.clone()

    with GIM(model, freeze_norm=False, softmax_temperature=None,
             q_scale=None, k_scale=None, v_scale=None,
             gate_scale=0.5, up_scale=0.5):
        run()

    _assert_scaled(model.blocks[-1].mlp.W_gate.grad, gate_base, 0.5)
    _assert_scaled(model.blocks[-1].mlp.W_in.grad, up_base, 0.5)


def test_tlens_moe_gate_up_scaled():
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=1, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
        act_fn="silu", num_experts=4, experts_per_token=2, d_mlp=20,
    )
    torch.manual_seed(0)
    model = HookedTransformer(cfg)
    model.eval()
    tokens = torch.randint(0, 50, (1, 12))
    x, y = tokens[:, :-1], tokens[:, 1:]
    experts = model.blocks[0].mlp.experts

    def run():
        # set_to_none=False: experts not routed any tokens keep a zero grad
        # instead of None, so stacking/comparing across experts is safe.
        model.zero_grad(set_to_none=False)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()

    run()
    gate_base = torch.stack([e.W_gate.weight.grad.clone() for e in experts])
    up_base = torch.stack([e.W_in.weight.grad.clone() for e in experts])

    with GIM(model, freeze_norm=False, softmax_temperature=None,
             q_scale=None, k_scale=None, v_scale=None,
             gate_scale=0.5, up_scale=0.5):
        run()
    gate_scaled = torch.stack([e.W_gate.weight.grad.clone() for e in experts])
    up_scaled = torch.stack([e.W_in.weight.grad.clone() for e in experts])

    _assert_scaled(gate_scaled, gate_base, 0.5)
    _assert_scaled(up_scaled, up_base, 0.5)


def test_tlens_unrecognized_mlp_class_raises(monkeypatch):
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from gim.context import mlp as gim_mlp

    cfg = HookedTransformerConfig(
        n_layers=1, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_vocab=50,
        act_fn="silu", gated_mlp=True, normalization_type="RMS",
    )
    model = HookedTransformer(cfg)

    monkeypatch.setattr(gim_mlp, "_TLENS_KNOWN_GATED_MLP_CLASSES", set())

    with pytest.raises(RuntimeError, match="doesn't recognize"):
        with GIM(model, freeze_norm=False, softmax_temperature=None,
                 q_scale=None, k_scale=None, v_scale=None,
                 gate_scale=0.5, up_scale=0.5):
            pass
