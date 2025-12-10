"""Minimal GPT-style language model for testing."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyAttention(nn.Module):
    """Multi-head self-attention using PyTorch's scaled_dot_product_attention."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, is_causal: bool = True):
        B, T, D = x.shape
        q = self.Wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(y)


class TinyBlock(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(self, d_model: int, n_heads: int, mlp_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TinyAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model),
            nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    """Minimal GPT-style language model for testing."""

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 256,
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TinyBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def get_input_embeddings(self):
        return self.tok

    def forward(self, tokens=None, inputs_embeds=None, use_cache=False):
        if inputs_embeds is not None:
            B, T, D = inputs_embeds.shape
            pos = torch.arange(T, device=inputs_embeds.device)
            x = inputs_embeds + self.pos(pos)[None, :, :]
        else:
            B, T = tokens.shape
            pos = torch.arange(T, device=tokens.device)
            x = self.tok(tokens) + self.pos(pos)[None, :, :]

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits
