from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layer_perm_mixed_ablation import (
    RMSNorm,
    TransformerBlock,
    distribution_loss,
    logit,
    simplex_normalize,
)

from .cast_shared import (
    CASTBaseConfig,
    SimplexDynamicsStem,
    SuccessorRetrievalMemory,
    _ordered_tangent_basis,
    _resolve_bool,
    _resolve_float,
    freeze_module,
)


@dataclass
class CASTConfig(CASTBaseConfig):
    """Causal Anchored Simplex Transport configuration."""

    n_retrieval_head: int = 2
    retrieval_head_dim: int = 64

    anchor_persistence_init: float = 0.55
    anchor_lambda_min: float = 0.05
    anchor_lambda_max: float = 0.95

    use_successor_retrieval: bool = True
    use_persistence_mix: bool = True
    use_current_anchor_control: bool = False
    use_tangent_flow: bool = False
    use_local_transport: bool = True
    use_dense_transport_control: bool = False
    use_support_dependent_kernel: bool = True

    local_transport_radius: int = 1
    local_transport_basis_dim: int = 2
    local_transport_identity_bias: float = 5.0
    local_transport_strength_init: float = 0.02
    local_transport_max_strength: float = 0.20
    local_transport_max_mean_shift: float = 0.25
    local_transport_std_shift_fraction: float = 0.10

    aux_prior_loss_weight: float = 0.0
    tangent_reg_weight: float = 0.0
    transport_reg_weight: float = 0.0
    struct_reg_weight: float = 0.0
    struct_smooth_weight: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (0.0 <= self.anchor_lambda_min <= self.anchor_lambda_max <= 1.0):
            raise ValueError("anchor lambda bounds must satisfy 0 <= min <= max <= 1")
        if not (0.0 < self.local_transport_max_strength <= 1.0):
            raise ValueError("local_transport_max_strength must be in (0, 1]")
        if self.local_transport_max_mean_shift < 0.0:
            raise ValueError("local_transport_max_mean_shift must be >= 0")
        if self.local_transport_std_shift_fraction < 0.0:
            raise ValueError("local_transport_std_shift_fraction must be >= 0")
        if self.struct_reg_weight < 0.0:
            raise ValueError("struct_reg_weight must be >= 0")
        if self.struct_smooth_weight < 0.0:
            raise ValueError("struct_smooth_weight must be >= 0")


CAST_ABLATIONS = (
    "full",
    "without_structural_reg",
    "anchor_only",
    "single_head_retrieval",
    "fixed_local_kernel",
    "without_persistence_mix",
)


def apply_named_ablation(cfg: CASTConfig, name: str) -> CASTConfig:
    normalized = name.strip().lower()
    if normalized == "full":
        return replace(
            cfg,
            use_successor_retrieval=True,
            use_persistence_mix=True,
            use_current_anchor_control=False,
            use_tangent_flow=False,
            use_local_transport=True,
            use_dense_transport_control=False,
            use_support_dependent_kernel=True,
            struct_reg_weight=cfg.struct_reg_weight if cfg.struct_reg_weight > 0.0 else 5e-4,
            struct_smooth_weight=cfg.struct_smooth_weight if cfg.struct_smooth_weight > 0.0 else 1e-4,
        )
    if normalized in {"without_structural_reg", "no_structural_reg"}:
        return replace(
            cfg,
            use_successor_retrieval=True,
            use_persistence_mix=True,
            use_current_anchor_control=False,
            use_tangent_flow=False,
            use_local_transport=True,
            use_dense_transport_control=False,
            use_support_dependent_kernel=True,
            struct_reg_weight=0.0,
            struct_smooth_weight=0.0,
        )
    if normalized == "anchor_only":
        return replace(
            cfg,
            use_successor_retrieval=True,
            use_persistence_mix=True,
            use_current_anchor_control=False,
            use_tangent_flow=False,
            use_local_transport=False,
            use_dense_transport_control=False,
            struct_reg_weight=0.0,
            struct_smooth_weight=0.0,
        )
    if normalized in {"single_head_retrieval", "single_head"}:
        return replace(
            cfg,
            n_retrieval_head=1,
            use_successor_retrieval=True,
            use_persistence_mix=True,
            use_current_anchor_control=False,
            use_tangent_flow=False,
            use_local_transport=True,
            use_dense_transport_control=False,
            use_support_dependent_kernel=True,
            struct_reg_weight=0.0,
            struct_smooth_weight=0.0,
        )
    if normalized == "fixed_local_kernel":
        return replace(
            cfg,
            use_successor_retrieval=True,
            use_persistence_mix=True,
            use_current_anchor_control=False,
            use_tangent_flow=False,
            use_local_transport=True,
            use_dense_transport_control=False,
            use_support_dependent_kernel=False,
            struct_reg_weight=0.0,
            struct_smooth_weight=0.0,
        )
    if normalized in {"without_persistence_mix", "no_persistence_mix"}:
        return replace(
            cfg,
            use_successor_retrieval=True,
            use_persistence_mix=False,
            use_current_anchor_control=False,
            use_tangent_flow=False,
            use_local_transport=True,
            use_dense_transport_control=False,
            use_support_dependent_kernel=True,
            struct_reg_weight=0.0,
            struct_smooth_weight=0.0,
        )
    raise ValueError(f"unknown CAST ablation name: {name}")


class BoundedAnchorMixer(nn.Module):
    def __init__(self, cfg: CASTConfig, feature_dim: int):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.d_model + feature_dim, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.proj.weight)
            if self.proj.bias is not None:
                span = max(self.cfg.anchor_lambda_max - self.cfg.anchor_lambda_min, self.cfg.simplex_eps)
                raw_init = (self.cfg.anchor_persistence_init - self.cfg.anchor_lambda_min) / span
                raw_init = min(max(raw_init, self.cfg.simplex_eps), 1.0 - self.cfg.simplex_eps)
                self.proj.bias.fill_(logit(raw_init))

    def forward(
        self,
        probs: torch.Tensor,
        retrieved: torch.Tensor,
        hidden: torch.Tensor,
        features: torch.Tensor,
        *,
        enabled: bool,
        retrieval_enabled: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not retrieval_enabled:
            lam = probs.new_ones(*probs.shape[:2], 1)
            return probs, {"anchor_persistence_weight": lam}
        if enabled:
            branch_in = torch.cat([hidden, features.to(dtype=hidden.dtype)], dim=-1)
            raw = torch.sigmoid(self.proj(branch_in)).to(dtype=probs.dtype)
            lam_min = probs.new_tensor(float(self.cfg.anchor_lambda_min))
            lam_max = probs.new_tensor(float(self.cfg.anchor_lambda_max))
            lam = lam_min + (lam_max - lam_min) * raw
        else:
            lam = probs.new_zeros(*probs.shape[:2], 1)
        anchor = lam * probs + (1.0 - lam) * retrieved
        anchor = simplex_normalize(anchor, eps=self.cfg.simplex_eps)
        return anchor, {"anchor_persistence_weight": lam}


class ConservativeRadiusOneTransport(nn.Module):
    def __init__(self, cfg: CASTConfig, feature_dim: int):
        super().__init__()
        self.cfg = cfg
        self.radius = int(cfg.local_transport_radius)
        self.n_offset = 2 * self.radius + 1
        self.basis_dim = int(cfg.local_transport_basis_dim)
        in_dim = cfg.d_model + feature_dim
        self.offset_head = nn.Linear(in_dim, self.n_offset, bias=cfg.bias)
        self.residual_head = nn.Linear(in_dim, self.basis_dim * self.n_offset, bias=cfg.bias)
        self.strength_head = nn.Linear(in_dim, 1, bias=True)
        self.register_buffer(
            "support_basis",
            _ordered_tangent_basis(cfg.prob_dim, self.basis_dim),
            persistent=False,
        )
        self.register_buffer(
            "offsets",
            torch.arange(-self.radius, self.radius + 1, dtype=torch.float32),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.offset_head.weight)
            if self.offset_head.bias is not None:
                self.offset_head.bias.zero_()
                self.offset_head.bias[self.radius] = self.cfg.local_transport_identity_bias
            nn.init.zeros_(self.residual_head.weight)
            if self.residual_head.bias is not None:
                nn.init.zeros_(self.residual_head.bias)
            nn.init.zeros_(self.strength_head.weight)
            if self.strength_head.bias is not None:
                max_strength = max(float(self.cfg.local_transport_max_strength), self.cfg.simplex_eps)
                raw_init = float(self.cfg.local_transport_strength_init) / max_strength
                raw_init = min(max(raw_init, self.cfg.simplex_eps), 1.0 - self.cfg.simplex_eps)
                self.strength_head.bias.fill_(logit(raw_init))

    def _mean_and_std(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        support = torch.arange(probs.size(-1), device=probs.device, dtype=probs.dtype)
        x = support.view(1, 1, -1)
        mean = (probs * x).sum(dim=-1, keepdim=True)
        var = (probs * (x - mean).square()).sum(dim=-1, keepdim=True).clamp_min(0.0)
        return mean, var.sqrt()

    def forward(
        self,
        source: torch.Tensor,
        hidden: torch.Tensor,
        features: torch.Tensor,
        *,
        enabled: bool,
        support_dependent_kernel: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not enabled:
            alpha = source.new_zeros(*source.shape, self.n_offset)
            alpha[..., self.radius] = 1.0
            zero = source.new_zeros(())
            return source, {
                "transport_kernel": alpha,
                "transport_strength": source.new_zeros(*source.shape[:2], 1),
                "transport_mean_shift": source.new_zeros(*source.shape[:2], 1),
                "transport_struct_reg": zero,
                "transport_smooth_reg": zero,
                "transport_reg": zero,
            }

        branch_in = torch.cat([hidden, features.to(dtype=hidden.dtype)], dim=-1)
        base_logits = self.offset_head(branch_in).unsqueeze(-2).to(dtype=source.dtype)
        if support_dependent_kernel:
            residual_coeff = self.residual_head(branch_in).view(
                source.size(0),
                source.size(1),
                self.basis_dim,
                self.n_offset,
            )
            basis = self.support_basis.to(device=source.device, dtype=source.dtype)
            residual = torch.einsum("dr,btrs->btds", basis, residual_coeff.to(dtype=source.dtype))
            kernel_logits = base_logits + residual
        else:
            kernel_logits = base_logits.expand(-1, -1, source.size(-1), -1)
        alpha = F.softmax(kernel_logits, dim=-1)

        shifted = torch.zeros_like(source)
        src_index = torch.arange(source.size(-1), device=source.device)
        for offset_idx, offset in enumerate(range(-self.radius, self.radius + 1)):
            dest = (src_index + offset).clamp(0, source.size(-1) - 1)
            dest = dest.view(1, 1, -1).expand_as(source)
            shifted.scatter_add_(-1, dest, source * alpha[..., offset_idx])

        raw_strength = torch.sigmoid(self.strength_head(branch_in)).to(dtype=source.dtype)
        max_strength = source.new_tensor(float(self.cfg.local_transport_max_strength))
        strength = max_strength * raw_strength

        source_mean, source_std = self._mean_and_std(source)
        shifted_mean, _ = self._mean_and_std(shifted)
        proposed_delta = shifted_mean - source_mean
        allowed_shift = source.new_tensor(float(self.cfg.local_transport_max_mean_shift))
        allowed_shift = allowed_shift + source_std * float(self.cfg.local_transport_std_shift_fraction)
        guard = (allowed_shift / proposed_delta.abs().clamp_min(self.cfg.simplex_eps)).clamp(max=1.0)
        strength = strength * guard

        out = (1.0 - strength) * source + strength * shifted
        out = simplex_normalize(out, eps=self.cfg.simplex_eps)

        abs_offsets = self.offsets.to(device=source.device, dtype=source.dtype).abs()
        if self.radius > 0:
            shift_reg = (alpha * (abs_offsets / float(self.radius))).mean()
        else:
            shift_reg = alpha.new_zeros(())
        identity_reg = (1.0 - alpha[..., self.radius]).mean()
        smooth_reg = (
            (alpha[:, :, 1:, :] - alpha[:, :, :-1, :]).pow(2).mean()
            if source.size(-1) > 1
            else alpha.new_zeros(())
        )
        strength_reg = strength.mean() / max_strength.clamp_min(self.cfg.simplex_eps)
        mean_shift_reg = (strength * proposed_delta).abs().mean() / allowed_shift.mean().clamp_min(
            self.cfg.simplex_eps
        )
        struct_reg = strength_reg + identity_reg + shift_reg + mean_shift_reg
        return out, {
            "transport_kernel": alpha,
            "transport_strength": strength,
            "transport_mean_shift": strength * proposed_delta,
            "transport_struct_reg": struct_reg,
            "transport_smooth_reg": smooth_reg,
            "transport_reg": struct_reg + smooth_reg,
        }


def freeze_static_ablation_params(model: "CASTModel", cfg: CASTConfig) -> None:
    if not cfg.use_log_stem:
        freeze_module(model.stem.log_proj)
    if not cfg.use_input_stats:
        freeze_module(model.stem.base_stat_proj)
        freeze_module(model.stem.support_stat_proj)
    elif not cfg.use_ordered_support:
        freeze_module(model.stem.support_stat_proj)
    if not cfg.use_successor_retrieval:
        freeze_module(model.retrieval)
    if not cfg.use_persistence_mix:
        freeze_module(model.anchor_mixer)
    if (not cfg.use_local_transport) or (not cfg.use_ordered_support):
        freeze_module(model.local_transport)
    elif not cfg.use_support_dependent_kernel:
        freeze_module(model.local_transport.residual_head)


class CASTModel(nn.Module):
    def __init__(self, cfg: CASTConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = SimplexDynamicsStem(cfg)
        self.feature_dim = self.stem.base_stat_dim + self.stem.support_stat_dim
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.retrieval = SuccessorRetrievalMemory(cfg)
        self.anchor_mixer = BoundedAnchorMixer(cfg, self.feature_dim)
        self.local_transport = ConservativeRadiusOneTransport(cfg, self.feature_dim)

        self.apply(self._init_weights)
        self._rescale_residual_projections()
        self.retrieval.reset_parameters()
        self.anchor_mixer.reset_parameters()
        self.local_transport.reset_parameters()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def _rescale_residual_projections(self) -> None:
        scale = 1.0 / max((2.0 * max(self.cfg.n_layer, 1)) ** 0.5, self.cfg.simplex_eps)
        for block in self.blocks:
            with torch.no_grad():
                block.attn.proj.weight.mul_(scale)
                block.ffn.out_proj.weight.mul_(scale)

    def encode(
        self,
        probs: torch.Tensor,
        *,
        ablation: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        hidden, probs, features, stats = self.stem(probs, ablation=ablation)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        return hidden, probs, features, stats

    def _validate_targets_and_mask(
        self,
        pred_shape: Tuple[int, int, int],
        targets: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
    ) -> None:
        batch_size, seq_len, prob_dim = pred_shape
        if targets is not None:
            if targets.dtype in (torch.int32, torch.int64, torch.long):
                if targets.ndim != 2 or tuple(targets.shape) != (batch_size, seq_len):
                    raise ValueError(
                        f"integer targets must have shape {(batch_size, seq_len)}, got {tuple(targets.shape)}"
                    )
            else:
                if targets.ndim != 3 or tuple(targets.shape) != (batch_size, seq_len, prob_dim):
                    raise ValueError(
                        f"distribution targets must have shape {(batch_size, seq_len, prob_dim)}, "
                        f"got {tuple(targets.shape)}"
                    )
        if loss_mask is not None and (loss_mask.ndim != 2 or tuple(loss_mask.shape) != (batch_size, seq_len)):
            raise ValueError(f"loss_mask must have shape {(batch_size, seq_len)}, got {tuple(loss_mask.shape)}")

    def forward(
        self,
        probs: torch.Tensor,
        *,
        targets: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        ablation: Optional[Mapping[str, Any]] = None,
        return_dict: bool = False,
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        if probs.ndim != 3:
            raise ValueError(f"probs must have shape (B,T,D), got {tuple(probs.shape)}")
        if probs.shape[-1] != self.cfg.prob_dim:
            raise ValueError(f"expected prob_dim={self.cfg.prob_dim}, got {probs.shape[-1]}")

        use_successor_retrieval = _resolve_bool(
            ablation,
            "use_successor_retrieval",
            self.cfg.use_successor_retrieval,
        )
        use_persistence_mix = _resolve_bool(ablation, "use_persistence_mix", self.cfg.use_persistence_mix)
        use_ordered_support = _resolve_bool(ablation, "use_ordered_support", self.cfg.use_ordered_support)
        use_local_transport = _resolve_bool(ablation, "use_local_transport", self.cfg.use_local_transport)
        use_support_dependent_kernel = _resolve_bool(
            ablation,
            "use_support_dependent_kernel",
            self.cfg.use_support_dependent_kernel,
        )

        hidden, probs, features, support_stats = self.encode(probs, ablation=ablation)
        hidden_f = hidden.float()
        features_f = features.float()
        probs_f = probs.float()

        retrieved, retrieval_aux = self.retrieval(
            hidden_f,
            probs_f,
            enabled=use_successor_retrieval,
            current_anchor_control=False,
        )
        anchor, anchor_aux = self.anchor_mixer(
            probs_f,
            retrieved,
            hidden_f,
            features_f,
            enabled=use_persistence_mix,
            retrieval_enabled=use_successor_retrieval,
        )
        pred, transport_aux = self.local_transport(
            anchor,
            hidden_f,
            features_f,
            enabled=use_local_transport and use_ordered_support,
            support_dependent_kernel=use_support_dependent_kernel,
        )
        pred = simplex_normalize(pred, eps=self.cfg.simplex_eps)
        logits = pred.clamp_min(self.cfg.simplex_eps).log()

        self._validate_targets_and_mask(tuple(pred.shape), targets, loss_mask)
        if not return_dict:
            return pred

        out: Dict[str, torch.Tensor] = {
            "pred": pred,
            "logits": logits,
        }

        if return_aux:
            out.update(
                {
                    "hidden": hidden_f,
                    "simplex_features": features_f,
                    "retrieved_successor_anchor": retrieved,
                    "anchor_prior": anchor,
                    "anchor_persistence_weight": anchor_aux["anchor_persistence_weight"],
                    "retrieval_head_gates": retrieval_aux["retrieval_head_gates"],
                    "successor_attention": retrieval_aux["successor_attention"],
                    "transport_kernel": transport_aux["transport_kernel"],
                    "transport_strength": transport_aux["transport_strength"],
                    "transport_mean_shift": transport_aux["transport_mean_shift"],
                    "transport_struct_reg_raw": transport_aux["transport_struct_reg"],
                    "transport_smooth_reg_raw": transport_aux["transport_smooth_reg"],
                    "transport_identity_mass": transport_aux["transport_kernel"][
                        ..., self.cfg.local_transport_radius
                    ].mean(),
                    **{f"support_{key}": value for key, value in support_stats.items()},
                }
            )

        if targets is not None:
            main_loss = distribution_loss(
                pred,
                targets,
                loss_type=self.cfg.loss_type,
                eps=self.cfg.simplex_eps,
                reduction="mean",
                mask=loss_mask,
            )
            struct_reg = transport_aux["transport_struct_reg"]
            smooth_reg = transport_aux["transport_smooth_reg"]
            struct_weight = _resolve_float(ablation, "struct_reg_weight", self.cfg.struct_reg_weight)
            smooth_weight = _resolve_float(ablation, "struct_smooth_weight", self.cfg.struct_smooth_weight)
            regularizer_loss = struct_weight * struct_reg + smooth_weight * smooth_reg
            loss = main_loss + regularizer_loss
            zero = main_loss.new_zeros(())
            out.update(
                {
                    "loss": loss,
                    "main_loss": main_loss,
                    "aux_prior_loss": zero,
                    "transport_reg_loss": regularizer_loss,
                    "transport_struct_reg_loss": struct_reg,
                    "transport_smooth_reg_loss": smooth_reg,
                }
            )

        return out

    def next_step(
        self,
        full_sequence: torch.Tensor,
        *,
        targets: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        ablation: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        if full_sequence.ndim != 3:
            raise ValueError(f"full_sequence must have shape (B,T,D), got {tuple(full_sequence.shape)}")
        if full_sequence.shape[-1] != self.cfg.prob_dim:
            raise ValueError(f"expected prob_dim={self.cfg.prob_dim}, got {full_sequence.shape[-1]}")

        if targets is None:
            if full_sequence.size(1) < 2:
                raise ValueError("full_sequence must have length >= 2 for next-step prediction")
            context = full_sequence[:, :-1]
            targets = full_sequence[:, 1:]
            if loss_mask is not None:
                if loss_mask.ndim != 2 or loss_mask.shape[1] != full_sequence.shape[1]:
                    raise ValueError(
                        f"loss_mask must have shape {(full_sequence.shape[0], full_sequence.shape[1])}, "
                        f"got {tuple(loss_mask.shape)}"
                    )
                loss_mask = loss_mask[:, 1:]
        else:
            context = full_sequence

        return self.forward(
            context,
            targets=targets,
            loss_mask=loss_mask,
            return_aux=return_aux,
            ablation=ablation,
            return_dict=True,
        )


__all__ = [
    "CAST_ABLATIONS",
    "CASTConfig",
    "CASTModel",
    "apply_named_ablation",
    "freeze_static_ablation_params",
]
