from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ilr_var import _normalize_simplex
from layer_perm_mixed_ablation import distribution_loss
from .tslib_simplex import _iter_context_window_chunks, _require_btd


def _as_int_tuple(values: Sequence[int], *, name: str) -> Tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise ValueError(f"{name} must contain at least one value")
    return parsed


@dataclass
class NHitsSimplexConfig:
    prob_dim: int
    max_seq_len: int = 512
    coordinate_mode: str = "raw_clr"
    num_stacks: int = 3
    blocks_per_stack: int = 2
    hidden_size: int = 512
    num_layers: int = 2
    pool_kernels: Tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.0
    simplex_eps: float = 1e-8
    window_batch_size: int = 256
    loss_type: str = "ce"

    def __post_init__(self) -> None:
        self.coordinate_mode = self.coordinate_mode.strip().lower() or "raw_clr"
        self.pool_kernels = _as_int_tuple(self.pool_kernels, name="pool_kernels")

        if self.prob_dim <= 1:
            raise ValueError("prob_dim must be >= 2")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.coordinate_mode not in {"raw", "log", "clr", "raw_clr"}:
            raise ValueError("coordinate_mode must be one of {'raw', 'log', 'clr', 'raw_clr'}")
        if self.num_stacks <= 0:
            raise ValueError("num_stacks must be positive")
        if self.blocks_per_stack <= 0:
            raise ValueError("blocks_per_stack must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.dropout < 0.0:
            raise ValueError("dropout must be non-negative")
        if self.simplex_eps <= 0.0:
            raise ValueError("simplex_eps must be positive")
        if self.window_batch_size <= 0:
            raise ValueError("window_batch_size must be positive")
        if self.loss_type not in {"ce", "kl"}:
            raise ValueError("loss_type must be 'ce' or 'kl'")
        for kernel in self.pool_kernels:
            if kernel <= 0:
                raise ValueError("pool_kernels must be positive")

    @property
    def input_dim(self) -> int:
        if self.coordinate_mode == "raw_clr":
            return 2 * int(self.prob_dim)
        return int(self.prob_dim)


class NHitsBlock(nn.Module):
    def __init__(
        self,
        *,
        context_len: int,
        input_dim: int,
        prob_dim: int,
        hidden_size: int,
        num_layers: int,
        pool_kernel: int,
        dropout: float,
    ):
        super().__init__()
        self.context_len = int(context_len)
        self.input_dim = int(input_dim)
        self.prob_dim = int(prob_dim)
        self.pool_kernel = max(1, min(int(pool_kernel), self.context_len))
        self.pooled_len = (self.context_len + self.pool_kernel - 1) // self.pool_kernel

        layers: list[nn.Module] = []
        in_features = self.pooled_len * self.input_dim
        for layer_idx in range(int(num_layers)):
            layers.append(nn.Linear(in_features if layer_idx == 0 else int(hidden_size), int(hidden_size)))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
        self.mlp = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(int(hidden_size), self.context_len * self.input_dim)
        self.forecast_head = nn.Linear(int(hidden_size), self.prob_dim)

        nn.init.zeros_(self.backcast_head.weight)
        nn.init.zeros_(self.backcast_head.bias)
        nn.init.zeros_(self.forecast_head.bias)

    def _pool(self, residual: torch.Tensor) -> torch.Tensor:
        if self.pool_kernel == 1:
            return residual
        pooled = F.avg_pool1d(
            residual.transpose(1, 2),
            kernel_size=self.pool_kernel,
            stride=self.pool_kernel,
            ceil_mode=True,
        ).transpose(1, 2)
        if pooled.size(1) != self.pooled_len:
            pooled = F.interpolate(
                pooled.transpose(1, 2),
                size=self.pooled_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return pooled

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if residual.ndim != 3:
            raise ValueError(f"expected residual with shape (B,T,F), got {tuple(residual.shape)}")
        if residual.size(1) != self.context_len or residual.size(2) != self.input_dim:
            raise ValueError(
                f"expected residual shape (*,{self.context_len},{self.input_dim}), got {tuple(residual.shape)}"
            )

        hidden = self.mlp(self._pool(residual).reshape(residual.size(0), -1))
        backcast = self.backcast_head(hidden).view(residual.size(0), self.context_len, self.input_dim)
        forecast_logits = self.forecast_head(hidden)
        return backcast, forecast_logits


class NHitsSimplexBaseline(nn.Module):
    """
    N-HiTS-style residual MLP baseline with a simplex output head.

    For each time step, the model builds a causal fixed-length context window,
    runs a stack of residual MLP blocks, and softmaxes the accumulated one-step
    forecast logits.
    """

    def __init__(self, cfg: NHitsSimplexConfig):
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.input_dim
        blocks = []
        total_blocks = int(cfg.num_stacks) * int(cfg.blocks_per_stack)
        for block_idx in range(total_blocks):
            stack_idx = block_idx // int(cfg.blocks_per_stack)
            pool_kernel = cfg.pool_kernels[min(stack_idx, len(cfg.pool_kernels) - 1)]
            blocks.append(
                NHitsBlock(
                    context_len=cfg.max_seq_len,
                    input_dim=input_dim,
                    prob_dim=cfg.prob_dim,
                    hidden_size=cfg.hidden_size,
                    num_layers=cfg.num_layers,
                    pool_kernel=pool_kernel,
                    dropout=cfg.dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def _coordinates(self, probs: torch.Tensor) -> torch.Tensor:
        probs = _normalize_simplex(probs, eps=self.cfg.simplex_eps)
        if self.cfg.coordinate_mode == "raw":
            return probs

        log_probs = probs.clamp_min(self.cfg.simplex_eps).log()
        if self.cfg.coordinate_mode == "log":
            return log_probs

        clr = log_probs - log_probs.mean(dim=-1, keepdim=True)
        if self.cfg.coordinate_mode == "clr":
            return clr
        if self.cfg.coordinate_mode == "raw_clr":
            return torch.cat([probs, clr], dim=-1)
        raise RuntimeError(f"unsupported coordinate_mode={self.cfg.coordinate_mode!r}")

    def _forecast_window_logits(self, windows: torch.Tensor) -> torch.Tensor:
        residual = self._coordinates(windows)
        forecast_logits = windows.new_zeros(windows.size(0), self.cfg.prob_dim)
        for block in self.blocks:
            backcast, block_logits = block(residual)
            residual = residual - backcast
            forecast_logits = forecast_logits + block_logits
        return forecast_logits.unsqueeze(1)

    def _forecast_logits(self, probs: torch.Tensor) -> torch.Tensor:
        _require_btd(probs)
        batch_size, total_steps, prob_dim = probs.shape
        logits_chunks = []
        for start, end, chunk_windows in _iter_context_window_chunks(
            probs,
            context_len=self.cfg.max_seq_len,
            window_batch_size=self.cfg.window_batch_size,
        ):
            chunk_logits = self._forecast_window_logits(chunk_windows)
            if chunk_logits.ndim != 3 or chunk_logits.size(1) != 1 or chunk_logits.size(2) != prob_dim:
                raise RuntimeError(f"N-HiTS backbone must return shape (N,1,D); got {tuple(chunk_logits.shape)}")
            logits_chunks.append(chunk_logits[:, 0, :].view(batch_size, end - start, prob_dim))
        return torch.cat(logits_chunks, dim=1) if logits_chunks else probs.new_zeros(batch_size, total_steps, prob_dim)

    def forward(
        self,
        probs: torch.Tensor,
        *,
        targets: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        ablation: Optional[Mapping[str, object]] = None,
        return_dict: bool = False,
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        del return_aux
        if ablation:
            raise ValueError("runtime ablations are not supported for MODEL=nhits")

        logits = self._forecast_logits(probs)
        pred = torch.softmax(logits, dim=-1)
        if not return_dict:
            return pred

        out: Dict[str, torch.Tensor] = {
            "pred": pred,
            "logits": logits,
        }

        if targets is not None:
            main_loss = distribution_loss(
                pred,
                targets,
                loss_type=self.cfg.loss_type,
                eps=self.cfg.simplex_eps,
                reduction="mean",
                mask=loss_mask,
            )
            zero = main_loss.new_zeros(())
            out.update(
                {
                    "loss": main_loss,
                    "main_loss": main_loss,
                    "aux_prior_loss": zero,
                    "transport_reg_loss": zero,
                }
            )

        return out
