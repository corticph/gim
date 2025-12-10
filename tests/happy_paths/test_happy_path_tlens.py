"""
Happy path test for GIM context manager with TransformerLens models.

Tests gradient modification on GPT-2 small.
"""
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from gim import GIM

device = "cuda" if torch.cuda.is_available() else "cpu"

model = HookedTransformer.from_pretrained("gpt2-small", device=device)
model.zero_grad(set_to_none=True)
model.eval()

texts = [
    "hello world",
    "the cat sat on the mat",
    "transformers are fun",
    "gradient-based methods rock",
]
tokens = model.to_tokens(texts, prepend_bos=True).to(device)   # [B, T+1]
x = tokens[:, :-1]                                             # inputs  [B, T]
y = tokens[:, 1:]                                              # targets [B, T]


with GIM(model):
    logits = model(x)  # [B, T, V]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),  # [B*T, V]
        y.reshape(-1)                         # [B*T]
    )
    loss.backward()

print("loss:", float(loss))
print("||dW_Q||:", model.blocks[0].attn.W_Q.grad.norm().item())
print("||dW_K||:", model.blocks[0].attn.W_K.grad.norm().item())
print("||dW_V||:", model.blocks[0].attn.W_V.grad.norm().item())
model.zero_grad(set_to_none=True)


with GIM(model,
         freeze_norm=True,
         softmax_temperature=2.0,
         q_scale=0.25, k_scale=0.25, v_scale=0.5):
    logits = model(x)  # [B, T, V]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),  # [B*T, V]
        y.reshape(-1)                         # [B*T]
    )
    loss.backward()

print("loss:", float(loss))
print("||dW_Q||:", model.blocks[0].attn.W_Q.grad.norm().item())
print("||dW_K||:", model.blocks[0].attn.W_K.grad.norm().item())
print("||dW_V||:", model.blocks[0].attn.W_V.grad.norm().item())
model.zero_grad(set_to_none=True)

with GIM(model,
         freeze_norm=False,
         softmax_temperature=None,
         q_scale=None, k_scale=None, v_scale=None):
    logits = model(x)  # [B, T, V]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),  # [B*T, V]
        y.reshape(-1)                         # [B*T]
    )
    loss.backward()

print("loss:", float(loss))
print("||dW_Q||:", model.blocks[0].attn.W_Q.grad.norm().item())
print("||dW_K||:", model.blocks[0].attn.W_K.grad.norm().item())
print("||dW_V||:", model.blocks[0].attn.W_V.grad.norm().item())
model.zero_grad(set_to_none=True)


logits = model(x)  # [B, T, V]
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),  # [B*T, V]
    y.reshape(-1)                         # [B*T]
)
loss.backward()

print("loss:", float(loss))
print("||dW_Q||:", model.blocks[0].attn.W_Q.grad.norm().item())
print("||dW_K||:", model.blocks[0].attn.W_K.grad.norm().item())
print("||dW_V||:", model.blocks[0].attn.W_V.grad.norm().item())
model.zero_grad(set_to_none=True)
