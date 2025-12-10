"""
Happy path test for GIM context manager with PyTorch HuggingFace-style models.

Tests gradient modification on a small custom transformer.
"""
from gim import GIM
from tests.models import TinyLM

import torch
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
V = 1000
B, T = 4, 32

model = TinyLM(vocab_size=V, d_model=128, n_layers=2, n_heads=4).to(device)
tokens = torch.randint(0, V, (B, T), device=device)
x = tokens[:, :-1]  # inputs  [B, T]
y = tokens[:, 1:]   # targets [B, T]


# Test 1: GIM with default parameters
print("=" * 60)
print("Test 1: GIM with default parameters")
print("=" * 60)
with GIM(model):
    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1)
    )
    loss.backward()

print("loss:", float(loss))
print("||dW_Q||:", model.blocks[0].attn.Wq.weight.grad.norm().item())
print("||dW_K||:", model.blocks[0].attn.Wk.weight.grad.norm().item())
print("||dW_V||:", model.blocks[0].attn.Wv.weight.grad.norm().item())
model.zero_grad(set_to_none=True)


# Test 2: GIM with explicit parameters
print("\n" + "=" * 60)
print("Test 2: GIM with explicit parameters")
print("=" * 60)
with GIM(model,
         freeze_norm=True,
         softmax_temperature=2.0,
         q_scale=0.25, k_scale=0.25, v_scale=0.5):
    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1)
    )
    loss.backward()

print("loss:", float(loss))
print("||dW_Q||:", model.blocks[0].attn.Wq.weight.grad.norm().item())
print("||dW_K||:", model.blocks[0].attn.Wk.weight.grad.norm().item())
print("||dW_V||:", model.blocks[0].attn.Wv.weight.grad.norm().item())
model.zero_grad(set_to_none=True)


# Test 3: GIM disabled (all modifications off)
print("\n" + "=" * 60)
print("Test 3: GIM disabled (all modifications off)")
print("=" * 60)
with GIM(model,
         freeze_norm=False,
         softmax_temperature=None,
         q_scale=None, k_scale=None, v_scale=None):
    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1)
    )
    loss.backward()

print("loss:", float(loss))
print("||dW_Q||:", model.blocks[0].attn.Wq.weight.grad.norm().item())
print("||dW_K||:", model.blocks[0].attn.Wk.weight.grad.norm().item())
print("||dW_V||:", model.blocks[0].attn.Wv.weight.grad.norm().item())
model.zero_grad(set_to_none=True)


# Test 4: Without GIM (baseline comparison)
print("\n" + "=" * 60)
print("Test 4: Without GIM (baseline comparison)")
print("=" * 60)
logits = model(x)
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    y.reshape(-1)
)
loss.backward()

print("loss:", float(loss))
print("||dW_Q||:", model.blocks[0].attn.Wq.weight.grad.norm().item())
print("||dW_K||:", model.blocks[0].attn.Wk.weight.grad.norm().item())
print("||dW_V||:", model.blocks[0].attn.Wv.weight.grad.norm().item())
model.zero_grad(set_to_none=True)

print("\n" + "=" * 60)
print("All GIM context manager tests completed!")
print("=" * 60)
