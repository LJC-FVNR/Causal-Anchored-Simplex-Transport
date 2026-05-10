from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RotaryEmbedding

from layer_perm_mixed_ablation import (
    RMSNorm,
    TransformerBlock,
    distribution_loss,
    logit,
    simplex_normalize,
)


def _resolve_bool(ablation: Optional[Mapping[str, Any]], key: str, default: bool) -> bool:
    if ablation is not None and key in ablation:
        return bool(ablation[key])
    return bool(default)


def _resolve_float(ablation: Optional[Mapping[str, Any]], key: str, default: float) -> float:
    if ablation is not None and key in ablation:
        return float(ablation[key])
    return float(default)


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask_f = mask.to(dtype=values.dtype, device=values.device)
    while mask_f.ndim < values.ndim:
        mask_f = mask_f.unsqueeze(-1)
    return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def _center_normalize_columns(basis: torch.Tensor) -> torch.Tensor:
    basis = basis - basis.mean(dim=0, keepdim=True)
    return basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-6)


def _ordered_tangent_basis(prob_dim: int, basis_dim: int) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, prob_dim)
    centered = x - x.mean()
    cols = [
        centered,
        centered.square(),
        centered.pow(3),
        torch.sin(math.pi * x),
        torch.cos(math.pi * x),
        torch.sin(2.0 * math.pi * x),
        torch.cos(2.0 * math.pi * x),
        torch.sin(3.0 * math.pi * x),
        torch.cos(3.0 * math.pi * x),
    ]
    power = 4
    while len(cols) < basis_dim:
        cols.append(centered.pow(power))
        power += 1
    return _center_normalize_columns(torch.stack(cols[:basis_dim], dim=-1))


@dataclass
class CASTBaseConfig:
    prob_dim: int
    max_seq_len: int = 2048
    normalize_input: bool = True
    simplex_eps: float = 1e-8
    use_input_stats: bool = True
    use_log_stem: bool = True
    use_ordered_support: bool = True

    d_model: int = 512
    n_layer: int = 8
    n_head: int = 8
    d_ff: Optional[int] = None
    dropout: float = 0.1
    attn_dropout: float = 0.0
    resid_dropout: float = 0.1
    norm_eps: float = 1e-5
    bias: bool = True
    qk_norm: bool = False
    use_rope: bool = True
    rope_base: float = 10000.0

    n_retrieval_head: int = 4
    retrieval_head_dim: int = 64
    retrieval_attn_dropout: float = 0.0

    anchor_persistence_init: float = 0.50
    use_successor_retrieval: bool = True
    use_persistence_mix: bool = True
    use_current_anchor_control: bool = False

    use_tangent_flow: bool = True
    tangent_basis_dim: int = 8
    tangent_max_amplitude: float = 0.75
    tangent_amplitude_init: float = 0.05
    tangent_reg_weight: float = 1e-4

    use_local_transport: bool = True
    local_transport_radius: int = 3
    local_transport_basis_dim: int = 4
    local_transport_identity_bias: float = 4.0
    local_transport_strength_init: float = 0.05
    use_dense_transport_control: bool = False
    dense_transport_rho_init: float = 0.05

    loss_type: str = "ce"
    aux_prior_loss_weight: float = 0.0
    transport_reg_weight: float = 1e-4

    init_std: float = 0.02

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
            raise ValueError("RoPE requires an even backbone attention head dimension")
        if self.n_retrieval_head <= 0:
            raise ValueError("n_retrieval_head must be positive")
        if self.retrieval_head_dim <= 0:
            raise ValueError("retrieval_head_dim must be positive")
        if self.use_rope and self.retrieval_head_dim % 2 != 0:
            raise ValueError("RoPE requires an even retrieval head dimension")
        if not (0.0 < self.anchor_persistence_init < 1.0):
            raise ValueError("anchor_persistence_init must be in (0, 1)")
        if self.tangent_basis_dim <= 0:
            raise ValueError("tangent_basis_dim must be positive")
        if self.tangent_max_amplitude <= 0.0:
            raise ValueError("tangent_max_amplitude must be positive")
        if not (0.0 < self.tangent_amplitude_init < self.tangent_max_amplitude):
            raise ValueError("tangent_amplitude_init must be in (0, tangent_max_amplitude)")
        if self.local_transport_radius < 0:
            raise ValueError("local_transport_radius must be >= 0")
        if self.local_transport_basis_dim <= 0:
            raise ValueError("local_transport_basis_dim must be positive")
        if not (0.0 < self.local_transport_strength_init < 1.0):
            raise ValueError("local_transport_strength_init must be in (0, 1)")
        if not (0.0 < self.dense_transport_rho_init < 1.0):
            raise ValueError("dense_transport_rho_init must be in (0, 1)")
        if self.loss_type not in {"ce", "kl"}:
            raise ValueError("loss_type must be 'ce' or 'kl'")
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")


CAST_SHARED_ABLATIONS = (
    "full",
    "shared_full",
    "shared_successor_anchor",
    "shared_successor_tangent",
    "shared_successor_local",
    "shared_successor_tangent_local",
    "shared_no_persistence_mix",
    "shared_current_anchor_control",
    "shared_dense_transport_control",
)


def apply_named_ablation(cfg: CASTBaseConfig, name: str) -> CASTBaseConfig:
    normalized = name.strip().lower()
    if normalized in {"full", "shared_full", "shared_successor_tangent_local"}:
        return replace(
            cfg,
            use_successor_retrieval=True,
            use_persistence_mix=True,
            use_current_anchor_control=False,
            use_tangent_flow=True,
            use_local_transport=True,
            use_dense_transport_control=False,
        )
    if normalized == "shared_successor_anchor":
        return replace(
            cfg,
            use_successor_retrieval=True,
            use_persistence_mix=True,
            use_current_anchor_control=False,
            use_tangent_flow=False,
            use_local_transport=False,
            use_dense_transport_control=False,
        )
    if normalized == "shared_successor_tangent":
        return replace(
            cfg,
            use_successor_retrieval=True,
            use_persistence_mix=True,
            use_current_anchor_control=False,
            use_tangent_flow=True,
            use_local_transport=False,
            use_dense_transport_control=False,
        )
    if normalized == "shared_successor_local":
        return replace(
            cfg,
            use_successor_retrieval=True,
            use_persistence_mix=True,
            use_current_anchor_control=False,
            use_tangent_flow=False,
            use_local_transport=True,
            use_dense_transport_control=False,
        )
    if normalized == "shared_no_persistence_mix":
        return replace(
            cfg,
            use_successor_retrieval=True,
            use_persistence_mix=False,
            use_current_anchor_control=False,
            use_tangent_flow=True,
            use_local_transport=True,
            use_dense_transport_control=False,
        )
    if normalized == "shared_current_anchor_control":
        return replace(
            cfg,
            use_successor_retrieval=True,
            use_persistence_mix=True,
            use_current_anchor_control=True,
            use_tangent_flow=True,
            use_local_transport=True,
            use_dense_transport_control=False,
        )
    if normalized == "shared_dense_transport_control":
        return replace(
            cfg,
            use_successor_retrieval=True,
            use_persistence_mix=True,
            use_current_anchor_control=False,
            use_tangent_flow=True,
            use_local_transport=True,
            use_dense_transport_control=True,
        )
    raise ValueError(f"unknown CAST shared ablation name: {name}")


def freeze_module(module: Optional[nn.Module]) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(False)


def freeze_static_ablation_params(model: "CASTSharedTransportModel", cfg: CASTBaseConfig) -> None:
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
    if not cfg.use_tangent_flow:
        freeze_module(model.tangent_flow)
    elif cfg.use_ordered_support:
        model.tangent_flow.learned_basis.requires_grad_(False)
    if (not cfg.use_local_transport) or cfg.use_dense_transport_control:
        freeze_module(model.local_transport)
    if (not cfg.use_local_transport) or (not cfg.use_dense_transport_control):
        freeze_module(model.dense_transport)


class SimplexDynamicsStem(nn.Module):
    base_stat_dim = 4
    support_stat_dim = 14

    def __init__(self, cfg: CASTBaseConfig):
        super().__init__()
        self.cfg = cfg
        self.raw_proj = nn.Linear(cfg.prob_dim, cfg.d_model, bias=cfg.bias)
        self.log_proj = nn.Linear(cfg.prob_dim, cfg.d_model, bias=cfg.bias)
        self.base_stat_proj = nn.Linear(self.base_stat_dim, cfg.d_model, bias=cfg.bias)
        self.support_stat_proj = nn.Linear(self.support_stat_dim, cfg.d_model, bias=cfg.bias)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("support", torch.linspace(0.0, 1.0, cfg.prob_dim), persistent=False)

    def _quantile_position(self, cdf: torch.Tensor, threshold: float, support: torch.Tensor) -> torch.Tensor:
        idx = (cdf >= threshold).to(torch.int64).argmax(dim=-1)
        return support[idx].unsqueeze(-1)

    def _base_stats(self, p: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
        entropy = -(p * log_p).sum(dim=-1, keepdim=True) / math.log(p.shape[-1])
        max_p = p.max(dim=-1, keepdim=True).values
        sq_mass = p.square().sum(dim=-1, keepdim=True)
        top2 = p.topk(k=2, dim=-1).values
        gap = top2[..., :1] - top2[..., 1:2]
        return torch.cat([entropy, max_p, sq_mass, gap], dim=-1)

    def _support_stats(self, p: torch.Tensor, log_p: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        support = self.support.to(device=p.device, dtype=p.dtype)
        x = support.view(1, 1, -1)
        mean = (p * x).sum(dim=-1, keepdim=True)
        centered = x - mean
        var = (p * centered.square()).sum(dim=-1, keepdim=True).clamp_min(self.cfg.simplex_eps)
        std = var.sqrt()
        skew = ((p * centered.pow(3)).sum(dim=-1, keepdim=True) / std.pow(3).clamp_min(1e-6)).clamp(-5.0, 5.0) / 5.0
        cdf = p.cumsum(dim=-1)
        q01 = self._quantile_position(cdf, 0.01, support)
        q10 = self._quantile_position(cdf, 0.10, support)
        q50 = self._quantile_position(cdf, 0.50, support)
        q90 = self._quantile_position(cdf, 0.90, support)
        q99 = self._quantile_position(cdf, 0.99, support)
        width90 = (q90 - q10).clamp_min(0.0)
        width98 = (q99 - q01).clamp_min(0.0)
        tail_left = p[..., support <= 0.10].sum(dim=-1, keepdim=True)
        tail_right = p[..., support >= 0.90].sum(dim=-1, keepdim=True)
        boundary_left = p[..., :1]
        boundary_right = p[..., -1:]
        features = torch.cat(
            [
                mean,
                std,
                skew,
                q01,
                q10,
                q50,
                q90,
                q99,
                width90,
                width98,
                tail_left,
                tail_right,
                boundary_left,
                boundary_right,
            ],
            dim=-1,
        )
        stats = {
            "mean": mean,
            "std": std,
            "skew": skew,
            "q01": q01,
            "q10": q10,
            "q50": q50,
            "q90": q90,
            "q99": q99,
            "width90": width90,
            "width98": width98,
        }
        return features, stats

    def forward(
        self,
        p: torch.Tensor,
        *,
        ablation: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if p.ndim != 3:
            raise ValueError(f"expected p shape (B,T,D), got {tuple(p.shape)}")
        batch_size, seq_len, prob_dim = p.shape
        if prob_dim != self.cfg.prob_dim:
            raise ValueError(f"expected prob_dim={self.cfg.prob_dim}, got {prob_dim}")
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"sequence length T={seq_len} exceeds max_seq_len={self.cfg.max_seq_len}")

        if self.cfg.normalize_input:
            p = simplex_normalize(p, eps=self.cfg.simplex_eps)

        use_log_stem = _resolve_bool(ablation, "use_log_stem", self.cfg.use_log_stem)
        use_input_stats = _resolve_bool(ablation, "use_input_stats", self.cfg.use_input_stats)
        use_ordered_support = _resolve_bool(ablation, "use_ordered_support", self.cfg.use_ordered_support)

        log_p = p.clamp_min(self.cfg.simplex_eps).log()
        pos = torch.arange(seq_len, device=p.device)
        hidden = self.raw_proj(p) + self.pos_emb(pos)[None, :, :]
        if use_log_stem:
            hidden = hidden + self.log_proj(log_p)

        base_features = self._base_stats(p.float(), log_p.float()).to(dtype=hidden.dtype)
        if use_input_stats:
            hidden = hidden + self.base_stat_proj(base_features)

        if use_ordered_support:
            support_features, stats = self._support_stats(p.float(), log_p.float())
            support_features = support_features.to(device=p.device, dtype=hidden.dtype)
        else:
            support_features = p.new_zeros(batch_size, seq_len, self.support_stat_dim, dtype=hidden.dtype)
            stats = {}
        if use_input_stats and use_ordered_support:
            hidden = hidden + self.support_stat_proj(support_features)

        hidden = self.dropout(hidden)
        features = torch.cat([base_features, support_features], dim=-1)
        return hidden, p, features, stats


class SuccessorRetrievalMemory(nn.Module):
    def __init__(self, cfg: CASTBaseConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_retrieval_head
        self.qk_dim = cfg.retrieval_head_dim
        self.q_proj = nn.Linear(cfg.d_model, self.n_head * self.qk_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, self.n_head * self.qk_dim, bias=cfg.bias)
        self.head_gate = nn.Linear(cfg.d_model, self.n_head, bias=cfg.bias)
        self.rope = RotaryEmbedding(self.qk_dim, base=cfg.rope_base) if cfg.use_rope else None
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.qk_dim, eps=cfg.norm_eps)
            self.k_norm = RMSNorm(self.qk_dim, eps=cfg.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.head_gate.weight)
            if self.head_gate.bias is not None:
                nn.init.zeros_(self.head_gate.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        probs: torch.Tensor,
        *,
        enabled: bool,
        current_anchor_control: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, seq_len, prob_dim = probs.shape
        if not enabled:
            gates = probs.new_full((batch_size, seq_len, self.n_head), 1.0 / self.n_head)
            return probs, {
                "retrieval_head_gates": gates,
                "successor_attention": probs.new_zeros(batch_size, self.n_head, seq_len, seq_len),
            }

        q = self.q_proj(hidden).view(batch_size, seq_len, self.n_head, self.qk_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(batch_size, seq_len, self.n_head, self.qk_dim).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q, k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.qk_dim))
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=probs.device)
        if current_anchor_control:
            mask = torch.tril(mask, diagonal=0)
            values = probs
        else:
            mask = torch.tril(mask, diagonal=-1)
            values = torch.empty_like(probs)
            if seq_len > 1:
                values[:, :-1] = probs[:, 1:]
                values[:, -1] = probs[:, -1]
            else:
                values[:, 0] = probs[:, 0]

        has_valid = mask.any(dim=-1)
        safe_mask = mask.clone()
        if not bool(has_valid.all()):
            safe_mask[~has_valid, 0] = True

        safe_mask_b = safe_mask.view(1, 1, seq_len, seq_len)
        neg_inf = torch.finfo(scores.dtype).min
        masked_scores = torch.where(safe_mask_b, scores, torch.full_like(scores, neg_inf))
        attn = F.softmax(masked_scores, dim=-1)
        if self.cfg.retrieval_attn_dropout > 0.0:
            attn = F.dropout(attn, p=self.cfg.retrieval_attn_dropout, training=self.training)

        head_out = torch.einsum("bhts,bsd->bhtd", attn.to(dtype=values.dtype), values)
        head_out = head_out.transpose(1, 2).contiguous()
        gates = F.softmax(self.head_gate(hidden), dim=-1).to(dtype=head_out.dtype)
        retrieved = torch.einsum("bth,bthd->btd", gates, head_out)
        if not current_anchor_control and not bool(has_valid.all()):
            fallback = (~has_valid).view(1, seq_len, 1)
            retrieved = torch.where(fallback, probs.to(dtype=retrieved.dtype), retrieved)
        retrieved = simplex_normalize(retrieved, eps=self.cfg.simplex_eps)
        return retrieved, {"retrieval_head_gates": gates, "successor_attention": attn}


class AnchorMixer(nn.Module):
    def __init__(self, cfg: CASTBaseConfig, feature_dim: int):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.d_model + feature_dim, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.proj.weight)
            if self.proj.bias is not None:
                self.proj.bias.fill_(logit(self.cfg.anchor_persistence_init))

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
            lam = torch.sigmoid(self.proj(branch_in)).to(dtype=probs.dtype)
        else:
            lam = probs.new_zeros(*probs.shape[:2], 1)
        anchor = lam * probs + (1.0 - lam) * retrieved
        anchor = simplex_normalize(anchor, eps=self.cfg.simplex_eps)
        return anchor, {"anchor_persistence_weight": lam}


class TangentFlowCorrection(nn.Module):
    def __init__(self, cfg: CASTBaseConfig, feature_dim: int):
        super().__init__()
        self.cfg = cfg
        self.coeff_head = nn.Linear(cfg.d_model + feature_dim, cfg.tangent_basis_dim, bias=cfg.bias)
        self.amplitude_head = nn.Linear(cfg.d_model + feature_dim, 1, bias=True)
        self.learned_basis = nn.Parameter(torch.empty(cfg.prob_dim, cfg.tangent_basis_dim))
        self.register_buffer(
            "ordered_basis",
            _ordered_tangent_basis(cfg.prob_dim, cfg.tangent_basis_dim),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.coeff_head.weight)
            if self.coeff_head.bias is not None:
                nn.init.zeros_(self.coeff_head.bias)
            nn.init.zeros_(self.amplitude_head.weight)
            if self.amplitude_head.bias is not None:
                init_ratio = self.cfg.tangent_amplitude_init / self.cfg.tangent_max_amplitude
                self.amplitude_head.bias.fill_(logit(init_ratio))
            nn.init.normal_(self.learned_basis, mean=0.0, std=self.cfg.init_std)

    def _basis(self, device: torch.device, dtype: torch.dtype, *, ordered: bool) -> torch.Tensor:
        if ordered:
            return self.ordered_basis.to(device=device, dtype=dtype)
        return _center_normalize_columns(self.learned_basis.to(device=device, dtype=dtype))

    def forward(
        self,
        anchor: torch.Tensor,
        hidden: torch.Tensor,
        features: torch.Tensor,
        *,
        enabled: bool,
        ordered: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not enabled:
            return anchor, {
                "tangent_coeff": anchor.new_zeros(*anchor.shape[:2], self.cfg.tangent_basis_dim),
                "tangent_amplitude": anchor.new_zeros(*anchor.shape[:2], 1),
                "tangent_reg": anchor.new_zeros(()),
            }

        branch_in = torch.cat([hidden, features.to(dtype=hidden.dtype)], dim=-1)
        coeff = self.coeff_head(branch_in).to(dtype=anchor.dtype)
        amplitude = (
            self.cfg.tangent_max_amplitude
            * torch.sigmoid(self.amplitude_head(branch_in)).to(dtype=anchor.dtype)
        )
        basis = self._basis(anchor.device, anchor.dtype, ordered=ordered)
        field = torch.einsum("btr,dr->btd", coeff, basis)
        centered_field = field - (anchor * field).sum(dim=-1, keepdim=True)
        logits = anchor.clamp_min(self.cfg.simplex_eps).log() + amplitude * centered_field
        out = torch.softmax(logits, dim=-1)
        tv_reg = (
            (centered_field[..., 1:] - centered_field[..., :-1]).pow(2).mean()
            if ordered and centered_field.size(-1) > 1
            else centered_field.new_zeros(())
        )
        tangent_reg = coeff.pow(2).mean() + amplitude.pow(2).mean() + tv_reg
        return out, {
            "tangent_coeff": coeff,
            "tangent_amplitude": amplitude,
            "tangent_field": centered_field,
            "tangent_reg": tangent_reg,
        }


class BandedLocalTransport(nn.Module):
    def __init__(self, cfg: CASTBaseConfig, feature_dim: int):
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
                self.strength_head.bias.fill_(logit(self.cfg.local_transport_strength_init))

    def forward(
        self,
        source: torch.Tensor,
        hidden: torch.Tensor,
        features: torch.Tensor,
        *,
        enabled: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not enabled:
            alpha = source.new_zeros(*source.shape, self.n_offset)
            alpha[..., self.radius] = 1.0
            return source, {
                "transport_kernel": alpha,
                "transport_strength": source.new_zeros(*source.shape[:2], 1),
                "transport_reg": source.new_zeros(()),
            }

        branch_in = torch.cat([hidden, features.to(dtype=hidden.dtype)], dim=-1)
        base_logits = self.offset_head(branch_in).unsqueeze(-2).to(dtype=source.dtype)
        residual_coeff = self.residual_head(branch_in).view(
            source.size(0),
            source.size(1),
            self.basis_dim,
            self.n_offset,
        )
        basis = self.support_basis.to(device=source.device, dtype=source.dtype)
        residual = torch.einsum("dr,btrs->btds", basis, residual_coeff.to(dtype=source.dtype))
        alpha = F.softmax(base_logits + residual, dim=-1)
        shifted = torch.zeros_like(source)
        src_index = torch.arange(source.size(-1), device=source.device)
        for offset_idx, offset in enumerate(range(-self.radius, self.radius + 1)):
            dest = (src_index + offset).clamp(0, source.size(-1) - 1)
            dest = dest.view(1, 1, -1).expand_as(source)
            shifted.scatter_add_(-1, dest, source * alpha[..., offset_idx])

        strength = torch.sigmoid(self.strength_head(branch_in)).to(dtype=source.dtype)
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
        strength_reg = strength.mean()
        transport_reg = strength_reg + identity_reg + shift_reg + smooth_reg
        return out, {
            "transport_kernel": alpha,
            "transport_strength": strength,
            "transport_reg": transport_reg,
        }


class DenseTransportControl(nn.Module):
    def __init__(self, cfg: CASTBaseConfig):
        super().__init__()
        self.cfg = cfg
        self.logits = nn.Parameter(torch.zeros(cfg.prob_dim, cfg.prob_dim))
        self.rho_logit = nn.Parameter(torch.tensor(logit(cfg.dense_transport_rho_init)))
        self.register_buffer("eye", torch.eye(cfg.prob_dim), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.logits.normal_(mean=0.0, std=self.cfg.init_std)
            idx = torch.arange(self.cfg.prob_dim, device=self.logits.device)
            self.logits[idx, idx] += 4.0

    def transport(self) -> torch.Tensor:
        dense = F.softmax(self.logits, dim=-1)
        rho = torch.sigmoid(self.rho_logit)
        eye = self.eye.to(device=dense.device, dtype=dense.dtype)
        return (1.0 - rho) * eye + rho * dense

    def forward(self, source: torch.Tensor, *, enabled: bool) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not enabled:
            return source, {"transport_reg": source.new_zeros(()), "dense_transport": self.eye.to(source.device)}
        transport = self.transport().to(device=source.device, dtype=source.dtype)
        out = torch.einsum("btd,de->bte", source, transport)
        out = simplex_normalize(out, eps=self.cfg.simplex_eps)
        eye = self.eye.to(device=transport.device, dtype=transport.dtype)
        reg = (transport - eye).pow(2).mean()
        return out, {"transport_reg": reg, "dense_transport": transport}


class CASTSharedTransportModel(nn.Module):
    def __init__(self, cfg: CASTBaseConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = SimplexDynamicsStem(cfg)
        self.feature_dim = self.stem.base_stat_dim + self.stem.support_stat_dim
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.retrieval = SuccessorRetrievalMemory(cfg)
        self.anchor_mixer = AnchorMixer(cfg, self.feature_dim)
        self.tangent_flow = TangentFlowCorrection(cfg, self.feature_dim)
        self.local_transport = BandedLocalTransport(cfg, self.feature_dim)
        self.dense_transport = DenseTransportControl(cfg)

        self.apply(self._init_weights)
        self._rescale_residual_projections()
        self.retrieval.reset_parameters()
        self.anchor_mixer.reset_parameters()
        self.tangent_flow.reset_parameters()
        self.local_transport.reset_parameters()
        self.dense_transport.reset_parameters()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def _rescale_residual_projections(self) -> None:
        scale = 1.0 / math.sqrt(2.0 * max(self.cfg.n_layer, 1))
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
                        f"distribution targets must have shape {(batch_size, seq_len, prob_dim)}, got {tuple(targets.shape)}"
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
        use_current_anchor_control = _resolve_bool(
            ablation,
            "use_current_anchor_control",
            self.cfg.use_current_anchor_control,
        )
        use_tangent_flow = _resolve_bool(ablation, "use_tangent_flow", self.cfg.use_tangent_flow)
        use_ordered_support = _resolve_bool(ablation, "use_ordered_support", self.cfg.use_ordered_support)
        use_local_transport = _resolve_bool(ablation, "use_local_transport", self.cfg.use_local_transport)
        use_dense_transport_control = _resolve_bool(
            ablation,
            "use_dense_transport_control",
            self.cfg.use_dense_transport_control,
        )

        hidden, probs, features, support_stats = self.encode(probs, ablation=ablation)
        hidden_f = hidden.float()
        features_f = features.float()
        probs_f = probs.float()

        retrieved, retrieval_aux = self.retrieval(
            hidden_f,
            probs_f,
            enabled=use_successor_retrieval,
            current_anchor_control=use_current_anchor_control,
        )
        anchor, anchor_aux = self.anchor_mixer(
            probs_f,
            retrieved,
            hidden_f,
            features_f,
            enabled=use_persistence_mix,
            retrieval_enabled=use_successor_retrieval,
        )
        tangent_pred, tangent_aux = self.tangent_flow(
            anchor,
            hidden_f,
            features_f,
            enabled=use_tangent_flow,
            ordered=use_ordered_support,
        )

        if use_dense_transport_control:
            pred, transport_aux = self.dense_transport(
                tangent_pred,
                enabled=use_local_transport,
            )
            transport_aux.setdefault("transport_kernel", probs_f.new_zeros(*probs_f.shape, 1))
            transport_aux.setdefault("transport_strength", probs_f.new_zeros(*probs_f.shape[:2], 1))
        else:
            pred, transport_aux = self.local_transport(
                tangent_pred,
                hidden_f,
                features_f,
                enabled=use_local_transport and use_ordered_support,
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
                    "tangent_prior": tangent_pred,
                    "anchor_persistence_weight": anchor_aux["anchor_persistence_weight"],
                    "retrieval_head_gates": retrieval_aux["retrieval_head_gates"],
                    "successor_attention": retrieval_aux["successor_attention"],
                    "tangent_coeff": tangent_aux["tangent_coeff"],
                    "tangent_amplitude": tangent_aux["tangent_amplitude"],
                    "tangent_field": tangent_aux.get("tangent_field"),
                    "transport_kernel": transport_aux.get("transport_kernel"),
                    "transport_strength": transport_aux.get("transport_strength"),
                    "dense_transport": transport_aux.get("dense_transport"),
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
            anchor_prior_loss = distribution_loss(
                anchor,
                targets,
                loss_type=self.cfg.loss_type,
                eps=self.cfg.simplex_eps,
                reduction="mean",
                mask=loss_mask,
            )
            tangent_reg = tangent_aux["tangent_reg"]
            transport_reg = transport_aux["transport_reg"]
            aux_prior_loss_weight = _resolve_float(
                ablation,
                "aux_prior_loss_weight",
                self.cfg.aux_prior_loss_weight,
            )
            tangent_reg_weight = _resolve_float(
                ablation,
                "tangent_reg_weight",
                self.cfg.tangent_reg_weight,
            )
            transport_reg_weight = _resolve_float(
                ablation,
                "transport_reg_weight",
                self.cfg.transport_reg_weight,
            )
            loss = (
                main_loss
                + aux_prior_loss_weight * anchor_prior_loss
                + tangent_reg_weight * tangent_reg
                + transport_reg_weight * transport_reg
            )
            out.update(
                {
                    "loss": loss,
                    "main_loss": main_loss,
                    "aux_prior_loss": anchor_prior_loss,
                    "transport_reg_loss": tangent_reg + transport_reg,
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
    "CAST_SHARED_ABLATIONS",
    "CASTBaseConfig",
    "CASTSharedTransportModel",
    "apply_named_ablation",
    "freeze_static_ablation_params",
]
