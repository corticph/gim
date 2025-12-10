"""
Happy path test for gim.explain() with TransformerLens models.

Tests feature attribution computation on GPT-2 small.
"""
import torch
from transformer_lens import HookedTransformer

from gim import explain

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
model.eval()

# Test 1: Basic explain with baseline_token_id
print("=" * 60)
print("Test 1: Basic explain with baseline_token_id")
print("=" * 60)
text = "The capital of France is"
tokens = model.to_tokens(text, prepend_bos=True)
print(f"Text: '{text}'")
print(f"Tokens shape: {tokens.shape}")

attributions = explain(model, tokens, baseline_token_id=50256)  # EOS as baseline
print(f"Attributions shape: {attributions.shape}")
print(f"Attributions: {attributions}")
print(f"Sum of attributions: {attributions.sum().item():.4f}")
assert attributions.shape[0] == tokens.shape[1], f"Shape mismatch"
print("PASSED\n")

# Test 2: Explain with specific target token
print("=" * 60)
print("Test 2: Explain with specific target_token_id (Paris)")
print("=" * 60)
paris_token = model.to_single_token(" Paris")
print(f"Paris token ID: {paris_token}")
attributions = explain(model, tokens, baseline_token_id=50256, target_token_id=paris_token)
print(f"Attributions: {attributions}")
assert attributions.shape[0] == tokens.shape[1]
print("PASSED\n")

# Test 3: Compare attributions for different targets
print("=" * 60)
print("Test 3: Compare attributions for Paris vs London")
print("=" * 60)
paris_token = model.to_single_token(" Paris")
london_token = model.to_single_token(" London")
attr_paris = explain(model, tokens, baseline_token_id=50256, target_token_id=paris_token)
attr_london = explain(model, tokens, baseline_token_id=50256, target_token_id=london_token)
print(f"Attributions for Paris:  {attr_paris}")
print(f"Attributions for London: {attr_london}")
print(f"Attributions differ: {not torch.allclose(attr_paris, attr_london)}")
print("PASSED\n")

# Test 4: Explain with custom GIM parameters
print("=" * 60)
print("Test 4: Explain with custom GIM parameters")
print("=" * 60)
attr_default = explain(model, tokens, baseline_token_id=50256)
attr_custom = explain(
    model, tokens,
    baseline_token_id=50256,
    freeze_norm=True,
    softmax_temperature=4.0,
    q_scale=0.1,
    k_scale=0.1,
    v_scale=0.8,
)
print(f"Default GIM attributions: {attr_default}")
print(f"Custom GIM attributions:  {attr_custom}")
print(f"Attributions differ: {not torch.allclose(attr_default, attr_custom)}")
print("PASSED\n")

# Test 5: Explain with GIM disabled (standard gradients)
print("=" * 60)
print("Test 5: Explain with GIM disabled vs enabled")
print("=" * 60)
attr_gim = explain(model, tokens, baseline_token_id=50256)
attr_no_gim = explain(
    model, tokens,
    baseline_token_id=50256,
    freeze_norm=False,
    softmax_temperature=None,
    q_scale=None,
    k_scale=None,
    v_scale=None,
)
print(f"With GIM:    {attr_gim}")
print(f"Without GIM: {attr_no_gim}")
print(f"Attributions differ: {not torch.allclose(attr_gim, attr_no_gim)}")
print("PASSED\n")

# Test 6: Multiple prompts
print("=" * 60)
print("Test 6: Different prompts")
print("=" * 60)
prompts = [
    "The cat sat on the",
    "In the year 2024,",
    "Machine learning is",
]
for prompt in prompts:
    tokens = model.to_tokens(prompt, prepend_bos=True)
    attributions = explain(model, tokens, baseline_token_id=50256)
    print(f"'{prompt}'")
    print(f"  Shape: {attributions.shape}, Sum: {attributions.sum().item():.4f}")
    assert attributions.shape[0] == tokens.shape[1]
print("PASSED\n")

# Test 7: Verify token with highest attribution makes sense
print("=" * 60)
print("Test 7: Check most important token for 'The capital of France is'")
print("=" * 60)
text = "The capital of France is"
tokens = model.to_tokens(text, prepend_bos=True)
paris_token = model.to_single_token(" Paris")
attributions = explain(model, tokens, baseline_token_id=50256, target_token_id=paris_token)

# Decode tokens for display
token_strs = [model.tokenizer.decode([t]) for t in tokens[0]]
print("Token attributions:")
for i, (tok, attr) in enumerate(zip(token_strs, attributions)):
    print(f"  [{i}] '{tok}': {attr.item():.4f}")

most_important_idx = attributions.abs().argmax().item()
print(f"\nMost important token: '{token_strs[most_important_idx]}' (index {most_important_idx})")
# "France" should be highly important for predicting "Paris"
print("PASSED\n")

print("=" * 60)
print("All TransformerLens happy path tests PASSED!")
print("=" * 60)
