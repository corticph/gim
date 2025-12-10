"""
Happy path test for gim.explain() with PyTorch HuggingFace-style models.

Tests feature attribution computation on a small custom transformer.
"""
from gim import explain
from tests.models import TinyLM

import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
V = 1000
T = 8

model = TinyLM(vocab_size=V, d_model=128, n_layers=2, n_heads=4).to(device)
model.eval()

# Test 1: Basic explain with baseline_token_id
print("=" * 60)
print("Test 1: Basic explain with baseline_token_id")
print("=" * 60)
tokens = torch.randint(0, V, (1, T), device=device)
attributions = explain(model, tokens, baseline_token_id=0)
print(f"Input shape: {tokens.shape}")
print(f"Attributions shape: {attributions.shape}")
print(f"Attributions: {attributions}")
print(f"Sum of attributions: {attributions.sum().item():.4f}")
assert attributions.shape == (T,), f"Expected shape ({T},), got {attributions.shape}"
print("PASSED\n")

# Test 2: Explain with specific target token
print("=" * 60)
print("Test 2: Explain with specific target_token_id")
print("=" * 60)
tokens = torch.randint(0, V, (1, T), device=device)
target_token = 42
attributions = explain(model, tokens, baseline_token_id=0, target_token_id=target_token)
print(f"Target token ID: {target_token}")
print(f"Attributions shape: {attributions.shape}")
print(f"Attributions: {attributions}")
assert attributions.shape == (T,), f"Expected shape ({T},), got {attributions.shape}"
print("PASSED\n")

# Test 3: Explain with 1D input (no batch dimension)
print("=" * 60)
print("Test 3: Explain with 1D input tensor")
print("=" * 60)
tokens_1d = torch.randint(0, V, (T,), device=device)
attributions = explain(model, tokens_1d, baseline_token_id=0)
print(f"Input shape: {tokens_1d.shape}")
print(f"Attributions shape: {attributions.shape}")
print(f"Attributions: {attributions}")
assert attributions.shape == (T,), f"Expected shape ({T},), got {attributions.shape}"
print("PASSED\n")

# Test 4: Explain with custom GIM parameters
print("=" * 60)
print("Test 4: Explain with custom GIM parameters")
print("=" * 60)
tokens = torch.randint(0, V, (1, T), device=device)
attributions_default = explain(model, tokens, baseline_token_id=0)
attributions_custom = explain(
    model, tokens,
    baseline_token_id=0,
    freeze_norm=True,
    softmax_temperature=4.0,
    q_scale=0.1,
    k_scale=0.1,
    v_scale=0.8,
)
print(f"Default GIM attributions: {attributions_default}")
print(f"Custom GIM attributions:  {attributions_custom}")
print(f"Attributions differ: {not torch.allclose(attributions_default, attributions_custom)}")
assert attributions_custom.shape == (T,), f"Expected shape ({T},), got {attributions_custom.shape}"
print("PASSED\n")

# Test 5: Explain at different target positions
print("=" * 60)
print("Test 5: Explain at different target positions")
print("=" * 60)
tokens = torch.randint(0, V, (1, T), device=device)
attr_last = explain(model, tokens, baseline_token_id=0, target_position=-1)
attr_mid = explain(model, tokens, baseline_token_id=0, target_position=T // 2)
print(f"Attributions at position -1: {attr_last}")
print(f"Attributions at position {T // 2}: {attr_mid}")
print(f"Attributions differ: {not torch.allclose(attr_last, attr_mid)}")
assert attr_last.shape == (T,) and attr_mid.shape == (T,)
print("PASSED\n")

# Test 6: Verify gradients are non-trivial
print("=" * 60)
print("Test 6: Verify attributions are non-trivial")
print("=" * 60)
tokens = torch.randint(0, V, (1, T), device=device)
attributions = explain(model, tokens, baseline_token_id=0)
print(f"Attributions: {attributions}")
print(f"Any non-zero: {(attributions != 0).any().item()}")
print(f"Max absolute: {attributions.abs().max().item():.6f}")
assert (attributions != 0).any(), "Attributions should not all be zero"
print("PASSED\n")

print("=" * 60)
print("All PyTorch/HF explain() happy path tests PASSED!")
print("=" * 60)
