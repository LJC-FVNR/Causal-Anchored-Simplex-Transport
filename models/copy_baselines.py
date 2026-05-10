from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

from layer_perm_mixed_ablation import distribution_loss


def _require_btd(probs: torch.Tensor, *, name: str = "probs") -> None:
    if probs.ndim != 3:
        raise ValueError(f"expected {name} with shape (B,T,D), got {tuple(probs.shape)}")


@dataclass
class CopyBaselineConfig:
    kind: str
    prob_dim: int
    forecast_lag: int = 1
    simplex_eps: float = 1e-8
    loss_type: str = "ce"

    def __post_init__(self) -> None:
        self.kind = self.kind.strip().lower()
        if self.kind not in {"persistence", "seasonal_naive"}:
            raise ValueError("kind must be one of {'persistence', 'seasonal_naive'}")
        if self.prob_dim <= 1:
            raise ValueError("prob_dim must be >= 2")
        if self.forecast_lag <= 0:
            raise ValueError("forecast_lag must be positive")
        if self.simplex_eps <= 0.0:
            raise ValueError("simplex_eps must be positive")
        if self.loss_type not in {"ce", "kl"}:
            raise ValueError("loss_type must be 'ce' or 'kl'")
        if self.kind == "persistence":
            self.forecast_lag = 1


class CopyBaseline(nn.Module):
    """
    Parameter-free simplex baselines.

    `persistence` predicts p_{t+1} = p_t.
    `seasonal_naive` predicts p_{t+1} = p_{t+1-s} when a full seasonal lag `s`
    is available, and otherwise falls back to persistence for the early steps.
    """

    def __init__(self, cfg: CopyBaselineConfig):
        super().__init__()
        self.cfg = cfg

    def _predict(self, probs: torch.Tensor) -> torch.Tensor:
        _require_btd(probs)
        pred = probs.clone()
        copy_offset = self.cfg.forecast_lag - 1
        if copy_offset <= 0 or probs.size(1) <= copy_offset:
            return pred
        pred[:, copy_offset:, :] = probs[:, : probs.size(1) - copy_offset, :]
        return pred

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
            raise ValueError(f"runtime ablations are not supported for MODEL={self.cfg.kind}")

        pred = self._predict(probs)
        if not return_dict:
            return pred

        logits = pred.clamp_min(self.cfg.simplex_eps).log()
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
