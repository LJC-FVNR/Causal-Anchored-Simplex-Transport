from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .ilr_var import (
    _align_fit_sequence,
    _helmert_basis,
    _ilr_to_simplex,
    _iter_sequences_from_split_payload,
    _normalize_simplex,
    _simplex_to_ilr,
)
from layer_perm_mixed_ablation import distribution_loss
from tasks.base import ModelFitSequence


def _as_float_tuple(values: Sequence[float], *, name: str) -> Tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if not parsed:
        raise ValueError(f"{name} must contain at least one value")
    return parsed


@dataclass
class CompositionalETSConfig:
    prob_dim: int
    mode: str = "auto"
    seasonal_period: int = 1
    alpha_grid: Tuple[float, ...] = (0.1, 0.2, 0.4, 0.6, 0.8)
    beta_grid: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)
    gamma_grid: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)
    phi_grid: Tuple[float, ...] = (0.8, 0.9, 0.98, 1.0)
    simplex_eps: float = 1e-8
    loss_type: str = "ce"
    fit_split: str = "train"
    selection_metric: str = "kl"
    fallback: str = "persistence"
    effective_mode: str = "persistence"
    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0
    phi: float = 1.0
    fitted_num_samples: int = 0
    selected_score: Optional[float] = None

    def __post_init__(self) -> None:
        self.mode = self.mode.strip().lower()
        self.fit_split = self.fit_split.strip().lower() or "train"
        self.selection_metric = self.selection_metric.strip().lower() or "kl"
        self.effective_mode = self.effective_mode.strip().lower()
        self.fallback = self.fallback.strip().lower()
        self.alpha_grid = _as_float_tuple(self.alpha_grid, name="alpha_grid")
        self.beta_grid = _as_float_tuple(self.beta_grid, name="beta_grid")
        self.gamma_grid = _as_float_tuple(self.gamma_grid, name="gamma_grid")
        self.phi_grid = _as_float_tuple(self.phi_grid, name="phi_grid")

        if self.prob_dim <= 1:
            raise ValueError("prob_dim must be >= 2")
        if self.mode not in {"auto", "level", "trend", "damped", "seasonal"}:
            raise ValueError("mode must be one of {'auto', 'level', 'trend', 'damped', 'seasonal'}")
        if self.effective_mode not in {"persistence", "level", "trend", "damped", "seasonal"}:
            raise ValueError(
                "effective_mode must be one of {'persistence', 'level', 'trend', 'damped', 'seasonal'}"
            )
        if self.seasonal_period < 1:
            raise ValueError("seasonal_period must be >= 1")
        if self.simplex_eps <= 0.0:
            raise ValueError("simplex_eps must be positive")
        if self.loss_type not in {"ce", "kl"}:
            raise ValueError("loss_type must be 'ce' or 'kl'")
        if self.selection_metric != "kl":
            raise ValueError("currently only selection_metric='kl' is supported")
        if self.fallback != "persistence":
            raise ValueError("currently only fallback='persistence' is supported")
        for name, values in (
            ("alpha_grid", self.alpha_grid),
            ("beta_grid", self.beta_grid),
            ("gamma_grid", self.gamma_grid),
            ("phi_grid", self.phi_grid),
        ):
            for value in values:
                if not (0.0 <= value <= 1.0):
                    raise ValueError(f"{name} values must be in [0, 1], got {value}")


@dataclass(frozen=True)
class _ETSCandidate:
    mode: str
    alpha: float
    beta: float = 0.0
    gamma: float = 0.0
    phi: float = 1.0


class CompositionalETSBaseline(nn.Module):
    """
    Classical compositional exponential-smoothing baseline.

    The model transforms probability vectors to ilr coordinates, selects a
    shared additive ETS recursion on the training split, and maps each one-step
    forecast back to the simplex.
    """

    def __init__(self, cfg: CompositionalETSConfig, *, basis: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("basis", basis.to(torch.float32))

    @classmethod
    def fit_from_split_payload(cls, cfg: CompositionalETSConfig, split_payload) -> "CompositionalETSBaseline":
        sequence_factory = lambda: _iter_sequences_from_split_payload(split_payload)
        return cls._fit_from_sequence_factory(cfg, sequence_factory)

    @classmethod
    def fit_from_task(cls, cfg: CompositionalETSConfig, task) -> "CompositionalETSBaseline":
        sequence_factory = lambda: task.iter_model_fit_sequences(cfg.fit_split, model_name="compositional_ets")
        return cls._fit_from_sequence_factory(cfg, sequence_factory)

    @classmethod
    def _fit_from_sequence_factory(
        cls,
        cfg: CompositionalETSConfig,
        sequence_factory: Callable[[], Iterator[ModelFitSequence]],
    ) -> "CompositionalETSBaseline":
        basis = _helmert_basis(cfg.prob_dim, dtype=torch.float64)
        fit_sequences = cls._collect_fit_sequences(cfg, basis, sequence_factory)
        candidates = cls._candidate_grid(cfg)

        best_candidate: Optional[_ETSCandidate] = None
        best_score = float("inf")
        best_num_samples = 0

        for candidate in candidates:
            score, num_samples = cls._score_candidate(cfg, basis, fit_sequences, candidate)
            if num_samples <= 0:
                continue
            if best_candidate is None or score < best_score:
                best_candidate = candidate
                best_score = score
                best_num_samples = num_samples

        if best_candidate is None:
            fitted_cfg = replace(
                cfg,
                effective_mode="persistence",
                alpha=1.0,
                beta=0.0,
                gamma=0.0,
                phi=1.0,
                fitted_num_samples=0,
                selected_score=None,
            )
            return cls(fitted_cfg, basis=basis)

        fitted_cfg = replace(
            cfg,
            effective_mode=best_candidate.mode,
            alpha=best_candidate.alpha,
            beta=best_candidate.beta,
            gamma=best_candidate.gamma,
            phi=best_candidate.phi,
            fitted_num_samples=best_num_samples,
            selected_score=best_score,
        )
        return cls(fitted_cfg, basis=basis)

    @staticmethod
    def _collect_fit_sequences(
        cfg: CompositionalETSConfig,
        basis: torch.Tensor,
        sequence_factory: Callable[[], Iterator[ModelFitSequence]],
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        collected: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for example in sequence_factory():
            context, targets, mask = _align_fit_sequence(example)
            if context is None or targets is None or mask is None:
                continue
            if context.size(-1) != cfg.prob_dim:
                raise ValueError(
                    f"fit sequence prob_dim mismatch: expected {cfg.prob_dim}, got {context.size(-1)}"
                )
            if context.size(0) == 0:
                continue
            if not bool(mask.any().item()):
                continue

            context = _normalize_simplex(context.to(dtype=basis.dtype), eps=cfg.simplex_eps)
            targets = _normalize_simplex(targets.to(dtype=basis.dtype), eps=cfg.simplex_eps)
            z_context = _simplex_to_ilr(context, basis, cfg.simplex_eps)
            collected.append((z_context.contiguous(), targets.contiguous(), mask.to(torch.bool).contiguous()))
        return collected

    @staticmethod
    def _candidate_grid(cfg: CompositionalETSConfig) -> list[_ETSCandidate]:
        if cfg.mode == "auto":
            modes = ["level", "trend", "damped"]
            if cfg.seasonal_period > 1:
                modes.append("seasonal")
        elif cfg.mode == "seasonal" and cfg.seasonal_period <= 1:
            modes = ["trend"]
        else:
            modes = [cfg.mode]

        candidates: list[_ETSCandidate] = []
        seen: set[tuple[str, float, float, float, float]] = set()

        def add(candidate: _ETSCandidate) -> None:
            key = (
                candidate.mode,
                round(candidate.alpha, 12),
                round(candidate.beta, 12),
                round(candidate.gamma, 12),
                round(candidate.phi, 12),
            )
            if key not in seen:
                candidates.append(candidate)
                seen.add(key)

        for mode in modes:
            if mode == "level":
                for alpha in cfg.alpha_grid:
                    add(_ETSCandidate(mode="level", alpha=alpha))
            elif mode == "trend":
                for alpha in cfg.alpha_grid:
                    for beta in cfg.beta_grid:
                        add(_ETSCandidate(mode="trend", alpha=alpha, beta=beta, phi=1.0))
            elif mode == "damped":
                for alpha in cfg.alpha_grid:
                    for beta in cfg.beta_grid:
                        for phi in cfg.phi_grid:
                            add(_ETSCandidate(mode="damped", alpha=alpha, beta=beta, phi=phi))
            elif mode == "seasonal":
                for alpha in cfg.alpha_grid:
                    for beta in cfg.beta_grid:
                        for gamma in cfg.gamma_grid:
                            for phi in cfg.phi_grid:
                                add(
                                    _ETSCandidate(
                                        mode="seasonal",
                                        alpha=alpha,
                                        beta=beta,
                                        gamma=gamma,
                                        phi=phi,
                                    )
                                )
            else:
                raise RuntimeError(f"unsupported ETS candidate mode={mode!r}")
        return candidates

    @classmethod
    def _score_candidate(
        cls,
        cfg: CompositionalETSConfig,
        basis: torch.Tensor,
        fit_sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        candidate: _ETSCandidate,
    ) -> tuple[float, int]:
        total = 0.0
        count = 0
        for z_context, targets, mask in fit_sequences:
            pred_z = cls._predict_ilr_coords(
                z_context.unsqueeze(0),
                mode=candidate.mode,
                seasonal_period=cfg.seasonal_period,
                alpha=candidate.alpha,
                beta=candidate.beta,
                gamma=candidate.gamma,
                phi=candidate.phi,
            ).squeeze(0)
            pred = _ilr_to_simplex(pred_z, basis, cfg.simplex_eps)

            target = targets.clamp_min(cfg.simplex_eps)
            target = target / target.sum(dim=-1, keepdim=True).clamp_min(cfg.simplex_eps)
            pred = pred.clamp_min(cfg.simplex_eps)
            pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(cfg.simplex_eps)
            loss = (target * (target.log() - pred.log())).sum(dim=-1)
            selected = loss[mask]
            if selected.numel() == 0:
                continue
            total += float(selected.sum().item())
            count += int(selected.numel())

        if count <= 0:
            return float("inf"), 0
        return total / float(count), count

    @staticmethod
    def _predict_ilr_coords(
        z: torch.Tensor,
        *,
        mode: str,
        seasonal_period: int,
        alpha: float,
        beta: float,
        gamma: float,
        phi: float,
    ) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"expected ilr coordinates with shape (B,T,H), got {tuple(z.shape)}")
        if z.size(1) == 0:
            return z.clone()

        mode = mode.strip().lower()
        if mode not in {"level", "trend", "damped", "seasonal"}:
            raise ValueError(f"unsupported ETS mode={mode!r}")

        batch_size, steps, ilr_dim = z.shape
        pred = torch.empty_like(z)
        level = z[:, 0, :].clone()
        trend = torch.zeros_like(level)

        use_seasonal = mode == "seasonal" and seasonal_period > 1
        if use_seasonal:
            season = z.new_zeros(batch_size, int(seasonal_period), ilr_dim)
        else:
            season = None

        alpha_f = float(alpha)
        beta_f = float(beta)
        gamma_f = float(gamma)
        phi_f = float(phi if mode in {"damped", "seasonal"} else 1.0)

        for step in range(steps):
            obs = z[:, step, :]

            if mode == "level":
                level = alpha_f * obs + (1.0 - alpha_f) * level
                forecast = level
            elif use_seasonal and season is not None:
                phase = step % int(seasonal_period)
                next_phase = (step + 1) % int(seasonal_period)
                seasonal_component = season[:, phase, :]
                prev_level = level
                level = alpha_f * (obs - seasonal_component) + (1.0 - alpha_f) * (level + phi_f * trend)
                trend = beta_f * (level - prev_level) + (1.0 - beta_f) * phi_f * trend
                season[:, phase, :] = gamma_f * (obs - level) + (1.0 - gamma_f) * seasonal_component
                forecast = level + phi_f * trend + season[:, next_phase, :]
            else:
                prev_level = level
                level = alpha_f * obs + (1.0 - alpha_f) * (level + phi_f * trend)
                trend = beta_f * (level - prev_level) + (1.0 - beta_f) * phi_f * trend
                forecast = level + phi_f * trend

            pred[:, step, :] = forecast

        return pred

    def _predict_sequence(self, probs: torch.Tensor) -> torch.Tensor:
        probs = _normalize_simplex(probs, eps=self.cfg.simplex_eps)
        if self.cfg.fitted_num_samples <= 0 or self.cfg.effective_mode == "persistence":
            return probs.clone()

        basis = self.basis.to(device=probs.device, dtype=probs.dtype)
        z = _simplex_to_ilr(probs, basis, self.cfg.simplex_eps)
        pred_z = self._predict_ilr_coords(
            z,
            mode=self.cfg.effective_mode,
            seasonal_period=self.cfg.seasonal_period,
            alpha=self.cfg.alpha,
            beta=self.cfg.beta,
            gamma=self.cfg.gamma,
            phi=self.cfg.phi,
        )
        return _ilr_to_simplex(pred_z, basis, self.cfg.simplex_eps)

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
            raise ValueError("runtime ablations are not supported for MODEL=compositional_ets")

        pred = self._predict_sequence(probs)
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
