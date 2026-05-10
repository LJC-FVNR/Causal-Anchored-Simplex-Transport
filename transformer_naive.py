import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from rope import RotaryEmbedding

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_rope = bool(getattr(config, "use_rope", False))
        self.rope = RotaryEmbedding(config.n_embd // config.n_head, base=float(getattr(config, "rope_base", 10000.0))) if self.use_rope else None
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.rope is not None:
            q, k = self.rope(q, k)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

from types import SimpleNamespace

class DistributionSequenceModel(nn.Module):
    """
    堆叠 L 层 DistributionSequenceLayer
    """

    def __init__(
        self,
        n_layer: int,
        Dx: int,
        H: int,
        d_k: int,
        dropout: float = 0.0,
        bias: bool = False,
        eps: float = 1e-8,
        block_size: int = 1024,
        chunk_size: int = 64,
        hidden_dim: int = 512,
        use_pos_emb: bool = True,
        use_rope: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.n_layer = int(n_layer)
        self.block_size = int(block_size)
        self.layers = nn.ModuleList([
            Block(
                config=SimpleNamespace(
                    n_embd=hidden_dim,
                    n_head=H,
                    dropout=dropout,
                    bias=bias,
                    block_size=block_size,
                    use_rope=use_rope,
                    rope_base=rope_base)
            )
            for _ in range(self.n_layer)
        ])

        self.input_proj = nn.Linear(Dx, hidden_dim, bias=bias)
        self.pos_emb = nn.Embedding(block_size, hidden_dim) if use_pos_emb else None
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, Dx, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,Dx) —— 每行是概率向量
        B, T, _ = x.shape
        if T > self.block_size:
            raise ValueError(f"sequence length T={T} exceeds block_size={self.block_size}")
        x = self.input_proj(x)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=x.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_proj(x)
        x = F.softmax(x, dim=-1)  # (B,T,Dx)，每行是概率向量
        return x  # (B,T,Dx)，每行仍在 simplex 上（数值上接近）
