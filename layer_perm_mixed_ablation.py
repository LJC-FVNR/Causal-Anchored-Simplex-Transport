from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RotaryEmbedding


def simplex_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize the last dimension to the probability simplex."""
    x = x.clamp_min(eps)
    return x / x.sum(dim=-1, keepdim=True).clamp_min(eps)


def logit(p: float, eps: float = 1e-6) -> float:
    p = float(min(max(p, eps), 1.0 - eps))
    return math.log(p / (1.0 - p))


def distribution_loss(
    pred: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_type: str = "ce",
    eps: float = 1e-8,
    reduction: str = "mean",
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute cross-entropy or KL between predicted and target distributions."""
    if pred.ndim != 3:
        raise ValueError(f"pred must have shape (B,T,D), got {tuple(pred.shape)}")

    batch_size, seq_len, prob_dim = pred.shape
    pred = pred.clamp_min(eps)
    log_pred = pred.log()

    if mask is not None and (mask.ndim != 2 or tuple(mask.shape) != (batch_size, seq_len)):
        raise ValueError(f"mask must have shape {(batch_size, seq_len)}, got {tuple(mask.shape)}")

    if targets.dtype in (torch.int32, torch.int64, torch.long):
        if targets.ndim != 2 or tuple(targets.shape) != (batch_size, seq_len):
            raise ValueError(f"integer targets must have shape {(batch_size, seq_len)}, got {tuple(targets.shape)}")
        loss = -log_pred.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    else:
        if targets.ndim != 3 or tuple(targets.shape) != (batch_size, seq_len, prob_dim):
            raise ValueError(
                f"distribution targets must have shape {(batch_size, seq_len, prob_dim)}, "
                f"got {tuple(targets.shape)}"
            )
        targets = simplex_normalize(targets, eps=eps)
        if loss_type == "kl":
            loss = (targets * (targets.clamp_min(eps).log() - log_pred)).sum(dim=-1)
        elif loss_type == "ce":
            loss = -(targets * log_pred).sum(dim=-1)
        else:
            raise ValueError("loss_type must be 'ce' or 'kl'")

    if mask is not None:
        mask_f = mask.to(loss.dtype)
        loss = loss * mask_f
        denom = mask_f.sum().clamp_min(1.0)
        if reduction == "mean":
            return loss.sum() / denom
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError("reduction must be mean, sum, or none")

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be mean, sum, or none")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.resid_dropout)
        self.qk_norm = cfg.qk_norm
        self.rope = RotaryEmbedding(self.head_dim, base=cfg.rope_base) if cfg.use_rope else None
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q, k)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.cfg.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.proj(y))


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_proj = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.resid_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.in_proj(x).chunk(2, dim=-1)
        x = F.silu(gate) * val
        x = self.dropout(x)
        x = self.out_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = SwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
