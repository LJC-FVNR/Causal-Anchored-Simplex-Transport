from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

from layer_perm_mixed_ablation import distribution_loss
from transformer_naive import DistributionSequenceModel


@dataclass
class NaiveTransformerConfig:
    prob_dim: int
    max_seq_len: int = 2048
    d_model: int = 512
    n_layer: int = 8
    n_head: int = 8
    dropout: float = 0.1
    bias: bool = False
    simplex_eps: float = 1e-8
    loss_type: str = "ce"
    use_abs_pos_emb: bool = True
    use_rope: bool = True
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if self.prob_dim <= 1:
            raise ValueError("prob_dim must be >= 2")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.use_rope and (self.d_model // self.n_head) % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.loss_type not in {"ce", "kl"}:
            raise ValueError("loss_type must be 'ce' or 'kl'")


class NaiveTransformerBaseline(nn.Module):
    """
    Adapter around transformer_naive.DistributionSequenceModel so that baseline
    models share the same training interface as CAST.
    """

    def __init__(self, cfg: NaiveTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = DistributionSequenceModel(
            n_layer=cfg.n_layer,
            Dx=cfg.prob_dim,
            H=cfg.n_head,
            d_k=cfg.d_model,
            dropout=cfg.dropout,
            bias=cfg.bias,
            eps=cfg.simplex_eps,
            block_size=cfg.max_seq_len,
            hidden_dim=cfg.d_model,
            use_pos_emb=cfg.use_abs_pos_emb,
            use_rope=cfg.use_rope,
            rope_base=cfg.rope_base,
        )

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
            raise ValueError("runtime ablations are not supported for MODEL=transformer_naive")

        pred = self.backbone(probs)
        if not return_dict:
            return pred

        # The baseline implementation only exposes normalized probabilities,
        # so we use log-probabilities as a stand-in for the shared "logits" key.
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
