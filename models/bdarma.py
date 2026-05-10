from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ilr_var import _align_fit_sequence, _fit_var_parameters, _iter_sequences_from_split_payload
from layer_perm_mixed_ablation import distribution_loss
from tasks.base import ModelFitSequence


def _normalize_simplex(probs: torch.Tensor, eps: float) -> torch.Tensor:
    if not torch.is_floating_point(probs):
        probs = probs.float()
    probs = probs.clamp_min(0.0)
    denom = probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return probs / denom


def _strictly_positive_simplex(probs: torch.Tensor, eps: float) -> torch.Tensor:
    probs = _normalize_simplex(probs, eps=eps)
    probs = probs.clamp_min(eps)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


def _choose_reference_index(sequences: Sequence[torch.Tensor], *, eps: float) -> int:
    if not sequences:
        raise ValueError("at least one sequence is required to choose an ALR reference")

    total = torch.zeros_like(sequences[0][0], dtype=torch.float64)
    count = 0
    for seq in sequences:
        positive = _strictly_positive_simplex(seq.to(torch.float64), eps)
        total = total + positive.sum(dim=0)
        count += int(positive.size(0))

    if count <= 0:
        raise ValueError("unable to choose an ALR reference from empty sequences")
    return int(torch.argmax(total / float(count)).item())


def _simplex_to_alr(probs: torch.Tensor, ref_index: int, eps: float) -> torch.Tensor:
    probs = _strictly_positive_simplex(probs, eps)
    prob_dim = probs.size(-1)
    if ref_index < 0 or ref_index >= prob_dim:
        raise ValueError(f"ref_index must be in [0, {prob_dim}), got {ref_index}")

    ref = probs[..., ref_index : ref_index + 1]
    left = probs[..., :ref_index]
    right = probs[..., ref_index + 1 :]
    others = torch.cat([left, right], dim=-1)
    return others.log() - ref.log()


def _alr_to_simplex(coords: torch.Tensor, ref_index: int, prob_dim: int, eps: float) -> torch.Tensor:
    if coords.size(-1) != prob_dim - 1:
        raise ValueError(
            f"ALR coordinates must have last dimension {prob_dim - 1}, got {coords.size(-1)}"
        )

    zeros = torch.zeros(*coords.shape[:-1], 1, dtype=coords.dtype, device=coords.device)
    if ref_index == 0:
        logits = torch.cat([zeros, coords], dim=-1)
    elif ref_index == prob_dim - 1:
        logits = torch.cat([coords, zeros], dim=-1)
    else:
        logits = torch.cat([coords[..., :ref_index], zeros, coords[..., ref_index:]], dim=-1)
    return _normalize_simplex(torch.softmax(logits, dim=-1), eps=eps)


def _dirichlet_log_prob(probs: torch.Tensor, alpha: torch.Tensor, eps: float) -> torch.Tensor:
    probs = _strictly_positive_simplex(probs, eps)
    alpha = alpha.clamp_min(eps)
    alpha_sum = alpha.sum(dim=-1)
    return (
        torch.lgamma(alpha_sum)
        - torch.lgamma(alpha).sum(dim=-1)
        + ((alpha - 1.0) * probs.log()).sum(dim=-1)
    )


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-12)
    return math.log(math.expm1(value))


def _coefficients_to_raw(coefficients: torch.Tensor, *, coef_bound: float) -> torch.Tensor:
    if coefficients.numel() == 0:
        return coefficients.clone()
    scaled = (coefficients / max(coef_bound, 1e-6)).clamp(min=-0.999999, max=0.999999)
    return torch.atanh(scaled)


def _raw_to_coefficients(raw_coefficients: torch.Tensor, *, coef_bound: float) -> torch.Tensor:
    if raw_coefficients.numel() == 0:
        return raw_coefficients
    return float(coef_bound) * torch.tanh(raw_coefficients)


@dataclass
class _PreparedSequence:
    probs: torch.Tensor
    alr: torch.Tensor
    loss_mask: torch.Tensor


def _prepare_sequences(
    sequence_factory: Callable[[], Iterator[ModelFitSequence]],
    *,
    prob_dim: int,
    simplex_eps: float,
    dirichlet_eps: float,
) -> Tuple[List[_PreparedSequence], int]:
    full_sequences: List[torch.Tensor] = []
    full_masks: List[torch.Tensor] = []

    for example in sequence_factory():
        context, targets, mask = _align_fit_sequence(example)
        if context is None or targets is None:
            continue
        if context.size(-1) != prob_dim or targets.size(-1) != prob_dim:
            raise ValueError(
                f"B-DARMA prob_dim mismatch: expected {prob_dim}, got {context.size(-1)} / {targets.size(-1)}"
            )

        full = torch.cat([context[:1], targets], dim=0).to(torch.float64)
        full = _strictly_positive_simplex(full, dirichlet_eps)

        if mask is None:
            aligned_mask = torch.ones(targets.size(0), dtype=torch.bool)
        else:
            aligned_mask = mask.to(torch.bool).contiguous()

        full_mask = torch.zeros(full.size(0), dtype=torch.bool)
        full_mask[1:] = aligned_mask

        full_sequences.append(full)
        full_masks.append(full_mask)

    if not full_sequences:
        raise ValueError("B-DARMA did not receive any valid fit sequences")

    ref_index = _choose_reference_index(full_sequences, eps=max(simplex_eps, dirichlet_eps))
    prepared: List[_PreparedSequence] = []
    for full, full_mask in zip(full_sequences, full_masks):
        prepared.append(
            _PreparedSequence(
                probs=full,
                alr=_simplex_to_alr(full, ref_index, max(simplex_eps, dirichlet_eps)),
                loss_mask=full_mask,
            )
        )
    return prepared, ref_index


def _accumulate_alr_ar_sufficient_statistics(
    prepared_sequences: Sequence[_PreparedSequence],
    *,
    lag_order: int,
) -> Dict[str, torch.Tensor | int]:
    if lag_order < 1:
        raise ValueError("lag_order must be >= 1")

    alr_dim = int(prepared_sequences[0].alr.size(-1))
    feature_dim = lag_order * alr_dim
    xtx = torch.zeros((feature_dim + 1, feature_dim + 1), dtype=torch.float64)
    xty = torch.zeros((feature_dim + 1, alr_dim), dtype=torch.float64)
    yty = torch.zeros((alr_dim, alr_dim), dtype=torch.float64)
    n_samples = 0

    for prepared in prepared_sequences:
        z = prepared.alr
        if z.size(0) <= lag_order:
            continue

        valid_idx = torch.nonzero(prepared.loss_mask, as_tuple=False).squeeze(-1)
        valid_idx = valid_idx[valid_idx >= lag_order]
        if valid_idx.numel() == 0:
            continue

        lag_source = z[:-1]
        lag_blocks = lag_source.unfold(0, lag_order, 1).permute(0, 2, 1)
        x_block = lag_blocks[valid_idx - lag_order].reshape(valid_idx.numel(), -1)
        y_block = z[valid_idx]
        ones = torch.ones((valid_idx.numel(), 1), dtype=torch.float64)
        design = torch.cat([ones, x_block], dim=1)

        xtx += design.transpose(0, 1) @ design
        xty += design.transpose(0, 1) @ y_block
        yty += y_block.transpose(0, 1) @ y_block
        n_samples += int(valid_idx.numel())

    if n_samples <= 0:
        raise ValueError(f"lag_order={lag_order} yields no valid ALR autoregression samples")

    return {
        "xtx": xtx,
        "xty": xty,
        "yty": yty,
        "num_samples": n_samples,
    }


def _default_alr_mean(prepared_sequences: Sequence[_PreparedSequence]) -> torch.Tensor:
    numerator = torch.zeros(prepared_sequences[0].alr.size(-1), dtype=torch.float64)
    denominator = 0
    for prepared in prepared_sequences:
        valid = torch.nonzero(prepared.loss_mask, as_tuple=False).squeeze(-1)
        if valid.numel() == 0:
            continue
        numerator += prepared.alr[valid].sum(dim=0)
        denominator += int(valid.numel())

    if denominator <= 0:
        for prepared in prepared_sequences:
            numerator += prepared.alr.sum(dim=0)
            denominator += int(prepared.alr.size(0))

    return numerator / max(denominator, 1)


def _initialize_ar_parameters(
    prepared_sequences: Sequence[_PreparedSequence],
    *,
    ar_order: int,
    ridge: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    alr_dim = int(prepared_sequences[0].alr.size(-1))
    default_mu = _default_alr_mean(prepared_sequences)
    if ar_order <= 0:
        return default_mu, torch.zeros((0, alr_dim, alr_dim), dtype=torch.float64)

    stats = _accumulate_alr_ar_sufficient_statistics(prepared_sequences, lag_order=ar_order)
    fit = _fit_var_parameters(
        stats["xtx"],
        stats["xty"],
        stats["yty"],
        lag_order=ar_order,
        ridge=ridge,
        num_samples=int(stats["num_samples"]),
    )
    coefficients = fit["coefficients"].to(torch.float64).flip(0).contiguous()
    intercept = fit["intercept"].to(torch.float64)

    system = torch.eye(alr_dim, dtype=torch.float64) - coefficients.sum(dim=0)
    system = system + max(float(ridge), 1e-6) * torch.eye(alr_dim, dtype=torch.float64)
    try:
        mu = torch.linalg.solve(system, intercept)
    except RuntimeError:
        mu = torch.linalg.lstsq(system, intercept.unsqueeze(-1)).solution.squeeze(-1)

    if not torch.isfinite(mu).all():
        mu = default_mu
    return mu.contiguous(), coefficients


def _compute_observed_states(
    alr_sequence: torch.Tensor,
    mean_alr: torch.Tensor,
    ar_coefficients: torch.Tensor,
    ma_coefficients: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    length, alr_dim = alr_sequence.shape
    ar_order = int(ar_coefficients.size(0))
    ma_order = int(ma_coefficients.size(0))
    memory = max(ar_order, ma_order)

    eta = torch.zeros((length, alr_dim), dtype=alr_sequence.dtype, device=alr_sequence.device)
    residual = torch.zeros_like(eta)

    if memory > 0:
        eta[:memory] = alr_sequence[:memory]

    for t in range(memory, length):
        eta_t = mean_alr
        for lag in range(ar_order):
            eta_t = eta_t + ar_coefficients[lag] @ (alr_sequence[t - lag - 1] - mean_alr)
        for lag in range(ma_order):
            eta_t = eta_t + ma_coefficients[lag] @ residual[t - lag - 1]
        eta[t] = eta_t
        residual[t] = alr_sequence[t] - eta_t

    if memory == 0:
        residual = alr_sequence - eta
    return eta, residual


def _forecast_next_alr(
    alr_sequence: torch.Tensor,
    residual: torch.Tensor,
    step_index: int,
    mean_alr: torch.Tensor,
    ar_coefficients: torch.Tensor,
    ma_coefficients: torch.Tensor,
) -> torch.Tensor:
    eta_next = mean_alr
    for lag in range(int(ar_coefficients.size(0))):
        eta_next = eta_next + ar_coefficients[lag] @ (alr_sequence[step_index - lag] - mean_alr)
    for lag in range(int(ma_coefficients.size(0))):
        eta_next = eta_next + ma_coefficients[lag] @ residual[step_index - lag]
    return eta_next


def _sequence_neg_log_likelihood(
    prepared: _PreparedSequence,
    *,
    mean_alr: torch.Tensor,
    ar_coefficients: torch.Tensor,
    ma_coefficients: torch.Tensor,
    concentration: torch.Tensor,
    ref_index: int,
    prob_dim: int,
    simplex_eps: float,
) -> Tuple[torch.Tensor, int]:
    eta, _ = _compute_observed_states(prepared.alr, mean_alr, ar_coefficients, ma_coefficients)
    memory = max(int(ar_coefficients.size(0)), int(ma_coefficients.size(0)))
    if prepared.probs.size(0) <= memory:
        return mean_alr.new_zeros(()), 0

    valid_idx = torch.nonzero(prepared.loss_mask, as_tuple=False).squeeze(-1)
    valid_idx = valid_idx[valid_idx >= memory]
    if valid_idx.numel() == 0:
        return mean_alr.new_zeros(()), 0

    mean_simplex = _alr_to_simplex(eta[valid_idx], ref_index, prob_dim, simplex_eps)
    alpha = concentration * mean_simplex
    log_prob = _dirichlet_log_prob(prepared.probs[valid_idx], alpha, simplex_eps)
    return -log_prob.sum(), int(valid_idx.numel())


def _prior_penalty(
    *,
    mean_alr: torch.Tensor,
    ar_coefficients: torch.Tensor,
    ma_coefficients: torch.Tensor,
    concentration: torch.Tensor,
    cfg: "BDARMAConfig",
) -> torch.Tensor:
    penalty = (mean_alr / float(cfg.prior_scale_mean)).pow(2).sum()
    if ar_coefficients.numel() > 0:
        penalty = penalty + (ar_coefficients / float(cfg.prior_scale_ar)).pow(2).sum()
    if ma_coefficients.numel() > 0:
        penalty = penalty + (ma_coefficients / float(cfg.prior_scale_ma)).pow(2).sum()

    log_concentration = concentration.log()
    target_log_concentration = math.log(max(float(cfg.init_concentration), cfg.min_concentration + 1e-8))
    penalty = penalty + ((log_concentration - target_log_concentration) / float(cfg.prior_scale_log_concentration)).pow(2)
    return 0.5 * penalty


def _evaluate_objective(
    sequences: Sequence[_PreparedSequence],
    *,
    mean_alr: torch.Tensor,
    ar_coefficients: torch.Tensor,
    ma_coefficients: torch.Tensor,
    concentration: torch.Tensor,
    ref_index: int,
    cfg: "BDARMAConfig",
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    total_nll = mean_alr.new_zeros(())
    total_count = 0
    for prepared in sequences:
        seq_nll, seq_count = _sequence_neg_log_likelihood(
            prepared,
            mean_alr=mean_alr,
            ar_coefficients=ar_coefficients,
            ma_coefficients=ma_coefficients,
            concentration=concentration,
            ref_index=ref_index,
            prob_dim=cfg.prob_dim,
            simplex_eps=max(cfg.simplex_eps, cfg.dirichlet_eps),
        )
        total_nll = total_nll + seq_nll
        total_count += seq_count

    if total_count <= 0:
        raise ValueError("B-DARMA objective has no valid observations")

    prior = _prior_penalty(
        mean_alr=mean_alr,
        ar_coefficients=ar_coefficients,
        ma_coefficients=ma_coefficients,
        concentration=concentration,
        cfg=cfg,
    )
    objective = total_nll / float(total_count) + prior / float(total_count)
    return objective, total_nll, total_count, prior


@dataclass
class BDARMAConfig:
    prob_dim: int
    ar_order: int = 1
    ma_order: int = 0
    max_ar_order: int = 4
    max_ma_order: int = 0
    ar_order_locked: bool = False
    ma_order_locked: bool = False
    order_selection: str = "aic"
    ridge: float = 1e-3
    simplex_eps: float = 1e-8
    dirichlet_eps: float = 1e-6
    loss_type: str = "ce"
    fit_split: str = "train"
    fallback: str = "persistence"
    fit_max_steps: int = 400
    fit_eval_interval: int = 10
    fit_patience: int = 80
    fit_sequence_batch_size: int = 16
    fit_lr: float = 5e-2
    fit_grad_clip: float = 10.0
    coef_bound: float = 0.95
    init_concentration: float = 50.0
    min_concentration: float = 1e-2
    prior_scale_mean: float = 2.0
    prior_scale_ar: float = 0.5
    prior_scale_ma: float = 0.5
    prior_scale_log_concentration: float = 2.0
    fitted_num_samples: int = 0
    selected_score: Optional[float] = None
    effective_ar_order: int = 0
    effective_ma_order: int = 0
    reference_index: int = 0

    def __post_init__(self) -> None:
        if self.prob_dim <= 1:
            raise ValueError("prob_dim must be >= 2")
        if self.ar_order < 0 or self.ma_order < 0:
            raise ValueError("ar_order and ma_order must be >= 0")
        if self.max_ar_order < 0 or self.max_ma_order < 0:
            raise ValueError("max_ar_order and max_ma_order must be >= 0")
        if self.order_selection not in {"aic", "bic", "fixed"}:
            raise ValueError("order_selection must be 'aic', 'bic', or 'fixed'")
        if self.loss_type not in {"ce", "kl"}:
            raise ValueError("loss_type must be 'ce' or 'kl'")
        if self.fallback != "persistence":
            raise ValueError("currently only fallback='persistence' is supported")
        if self.fit_max_steps < 1:
            raise ValueError("fit_max_steps must be >= 1")
        if self.fit_eval_interval < 1:
            raise ValueError("fit_eval_interval must be >= 1")
        if self.fit_patience < 1:
            raise ValueError("fit_patience must be >= 1")
        if self.fit_sequence_batch_size < 1:
            raise ValueError("fit_sequence_batch_size must be >= 1")
        if self.fit_lr <= 0.0:
            raise ValueError("fit_lr must be > 0")
        if self.fit_grad_clip <= 0.0:
            raise ValueError("fit_grad_clip must be > 0")
        if self.coef_bound <= 0.0:
            raise ValueError("coef_bound must be > 0")
        if self.init_concentration <= 0.0:
            raise ValueError("init_concentration must be > 0")
        if self.min_concentration <= 0.0:
            raise ValueError("min_concentration must be > 0")
        if self.prior_scale_mean <= 0.0 or self.prior_scale_ar <= 0.0 or self.prior_scale_ma <= 0.0:
            raise ValueError("prior scales must be > 0")
        if self.prior_scale_log_concentration <= 0.0:
            raise ValueError("prior_scale_log_concentration must be > 0")
        if not (0 <= self.reference_index < self.prob_dim):
            raise ValueError(f"reference_index must be in [0, {self.prob_dim}), got {self.reference_index}")


class BDARMABaseline(nn.Module):
    """
    Practical MAP B-DARMA baseline in ALR coordinates.

    This keeps the key B-DARMA structure used in the Stan formulations:
    a Dirichlet observation model whose mean evolves through an ALR-space
    ARMA recursion. The implementation here uses a penalized MAP fit inside the
    repo so the benchmark can run without a separate Stan runtime.
    """

    def __init__(
        self,
        cfg: BDARMAConfig,
        *,
        mean_alr: Optional[torch.Tensor] = None,
        ar_coefficients: Optional[torch.Tensor] = None,
        ma_coefficients: Optional[torch.Tensor] = None,
        concentration: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.reference_index = int(cfg.reference_index)

        alr_dim = cfg.prob_dim - 1
        if mean_alr is None:
            mean_alr = torch.zeros(alr_dim, dtype=torch.float32)
        if ar_coefficients is None:
            ar_coefficients = torch.zeros((cfg.effective_ar_order, alr_dim, alr_dim), dtype=torch.float32)
        if ma_coefficients is None:
            ma_coefficients = torch.zeros((cfg.effective_ma_order, alr_dim, alr_dim), dtype=torch.float32)
        if concentration is None:
            concentration = torch.tensor(float(cfg.init_concentration), dtype=torch.float32)

        self.register_buffer("mean_alr", mean_alr.to(torch.float32))
        self.register_buffer("ar_coefficients", ar_coefficients.to(torch.float32))
        self.register_buffer("ma_coefficients", ma_coefficients.to(torch.float32))
        self.register_buffer("concentration", concentration.to(torch.float32).reshape(()))

    @classmethod
    def fit_from_split_payload(cls, cfg: BDARMAConfig, split_payload) -> "BDARMABaseline":
        sequence_factory = lambda: _iter_sequences_from_split_payload(split_payload)
        return cls._fit_from_sequence_factory(cfg, sequence_factory)

    @classmethod
    def fit_from_task(cls, cfg: BDARMAConfig, task) -> "BDARMABaseline":
        sequence_factory = lambda: task.iter_model_fit_sequences(cfg.fit_split, model_name="bdarma")
        return cls._fit_from_sequence_factory(cfg, sequence_factory)

    @classmethod
    def _fit_from_sequence_factory(
        cls,
        cfg: BDARMAConfig,
        sequence_factory: Callable[[], Iterator[ModelFitSequence]],
    ) -> "BDARMABaseline":
        prepared_sequences, ref_index = _prepare_sequences(
            sequence_factory,
            prob_dim=cfg.prob_dim,
            simplex_eps=cfg.simplex_eps,
            dirichlet_eps=cfg.dirichlet_eps,
        )

        if cfg.order_selection == "fixed":
            candidate_orders = [(int(cfg.ar_order), int(cfg.ma_order))]
        else:
            ar_candidates = (
                [int(cfg.ar_order)]
                if cfg.ar_order_locked
                else list(range(0 if cfg.max_ar_order == 0 else 1, int(cfg.max_ar_order) + 1))
            )
            ma_candidates = (
                [int(cfg.ma_order)]
                if cfg.ma_order_locked
                else list(range(0, int(cfg.max_ma_order) + 1))
            )
            candidate_orders = [(p, q) for p in ar_candidates for q in ma_candidates if p > 0 or q > 0]
            if not candidate_orders:
                candidate_orders = [(0, 0)]

        criterion = "aic" if cfg.order_selection == "aic" else "bic"
        best_fit: Optional[Dict[str, object]] = None
        best_order: Optional[Tuple[int, int]] = None
        best_score = float("inf")

        for ar_order, ma_order in candidate_orders:
            try:
                fit = cls._fit_candidate(
                    cfg,
                    prepared_sequences=prepared_sequences,
                    ref_index=ref_index,
                    ar_order=int(ar_order),
                    ma_order=int(ma_order),
                )
            except ValueError:
                continue

            score = float(fit[criterion])
            if best_fit is None or score < best_score:
                best_fit = fit
                best_order = (int(ar_order), int(ma_order))
                best_score = score

        if best_fit is None or best_order is None:
            fitted_cfg = replace(
                cfg,
                fitted_num_samples=0,
                selected_score=None,
                effective_ar_order=0,
                effective_ma_order=0,
                reference_index=ref_index,
            )
            return cls(fitted_cfg)

        fitted_cfg = replace(
            cfg,
            fitted_num_samples=int(best_fit["num_samples"]),
            selected_score=best_score,
            effective_ar_order=best_order[0],
            effective_ma_order=best_order[1],
            reference_index=ref_index,
            order_selection="fixed" if cfg.order_selection == "fixed" else cfg.order_selection,
        )
        return cls(
            fitted_cfg,
            mean_alr=best_fit["mean_alr"],
            ar_coefficients=best_fit["ar_coefficients"],
            ma_coefficients=best_fit["ma_coefficients"],
            concentration=best_fit["concentration"],
        )

    @classmethod
    def _fit_candidate(
        cls,
        cfg: BDARMAConfig,
        *,
        prepared_sequences: Sequence[_PreparedSequence],
        ref_index: int,
        ar_order: int,
        ma_order: int,
    ) -> Dict[str, object]:
        alr_dim = cfg.prob_dim - 1
        if ar_order == 0 and ma_order == 0:
            init_mean_alr = _default_alr_mean(prepared_sequences)
            init_ar = torch.zeros((0, alr_dim, alr_dim), dtype=torch.float64)
        else:
            init_mean_alr, init_ar = _initialize_ar_parameters(
                prepared_sequences,
                ar_order=ar_order,
                ridge=cfg.ridge,
            )
        init_ma = torch.zeros((ma_order, alr_dim, alr_dim), dtype=torch.float64)

        mean_alr_param = nn.Parameter(init_mean_alr.clone())
        raw_ar_param = nn.Parameter(_coefficients_to_raw(init_ar, coef_bound=cfg.coef_bound))
        raw_ma_param = nn.Parameter(_coefficients_to_raw(init_ma, coef_bound=cfg.coef_bound))
        raw_concentration_param = nn.Parameter(
            torch.tensor(
                _inverse_softplus(max(float(cfg.init_concentration) - float(cfg.min_concentration), 1e-6)),
                dtype=torch.float64,
            )
        )

        params = [mean_alr_param, raw_concentration_param]
        if raw_ar_param.numel() > 0:
            params.append(raw_ar_param)
        if raw_ma_param.numel() > 0:
            params.append(raw_ma_param)
        optimizer = torch.optim.Adam(params, lr=float(cfg.fit_lr))

        num_sequences = len(prepared_sequences)
        batch_size = min(int(cfg.fit_sequence_batch_size), num_sequences)
        eval_interval = max(1, min(int(cfg.fit_eval_interval), int(cfg.fit_max_steps)))
        generator = torch.Generator()
        generator.manual_seed(0)

        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_objective = float("inf")
        stale_evals = 0

        def current_coefficients() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            concentration = float(cfg.min_concentration) + F.softplus(raw_concentration_param)
            return (
                mean_alr_param,
                _raw_to_coefficients(raw_ar_param, coef_bound=cfg.coef_bound),
                _raw_to_coefficients(raw_ma_param, coef_bound=cfg.coef_bound),
                concentration,
            )

        for step in range(1, int(cfg.fit_max_steps) + 1):
            if batch_size >= num_sequences:
                batch = prepared_sequences
            else:
                chosen = torch.randperm(num_sequences, generator=generator)[:batch_size].tolist()
                batch = [prepared_sequences[idx] for idx in chosen]

            optimizer.zero_grad()
            mean_alr, ar_coefficients, ma_coefficients, concentration = current_coefficients()
            objective, _, count, _ = _evaluate_objective(
                batch,
                mean_alr=mean_alr,
                ar_coefficients=ar_coefficients,
                ma_coefficients=ma_coefficients,
                concentration=concentration,
                ref_index=ref_index,
                cfg=cfg,
            )
            if count <= 0 or not torch.isfinite(objective):
                raise ValueError(f"B-DARMA objective became invalid for p={ar_order}, q={ma_order}")

            objective.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(cfg.fit_grad_clip))
            optimizer.step()

            if step == 1 or step % eval_interval == 0 or step == int(cfg.fit_max_steps):
                with torch.no_grad():
                    mean_alr_eval, ar_eval, ma_eval, concentration_eval = current_coefficients()
                    full_objective, total_nll, total_count, _ = _evaluate_objective(
                        prepared_sequences,
                        mean_alr=mean_alr_eval,
                        ar_coefficients=ar_eval,
                        ma_coefficients=ma_eval,
                        concentration=concentration_eval,
                        ref_index=ref_index,
                        cfg=cfg,
                    )

                score = float(full_objective.item())
                if math.isfinite(score) and score < best_objective - 1e-8:
                    best_objective = score
                    stale_evals = 0
                    best_state = {
                        "mean_alr": mean_alr_eval.detach().clone(),
                        "ar_coefficients": ar_eval.detach().clone(),
                        "ma_coefficients": ma_eval.detach().clone(),
                        "concentration": concentration_eval.detach().clone(),
                        "total_nll": total_nll.detach().clone(),
                        "total_count": torch.tensor(int(total_count)),
                    }
                else:
                    stale_evals += 1

                if stale_evals * eval_interval >= int(cfg.fit_patience):
                    break

        if best_state is None:
            raise ValueError(f"B-DARMA optimization did not converge for p={ar_order}, q={ma_order}")

        total_nll = float(best_state["total_nll"].item())
        total_count = int(best_state["total_count"].item())
        num_params = alr_dim + ar_order * alr_dim * alr_dim + ma_order * alr_dim * alr_dim + 1
        aic = 2.0 * total_nll + 2.0 * num_params
        bic = 2.0 * total_nll + math.log(max(total_count, 2)) * num_params

        return {
            "mean_alr": best_state["mean_alr"].to(torch.float32),
            "ar_coefficients": best_state["ar_coefficients"].to(torch.float32),
            "ma_coefficients": best_state["ma_coefficients"].to(torch.float32),
            "concentration": best_state["concentration"].to(torch.float32),
            "num_samples": total_count,
            "aic": aic,
            "bic": bic,
        }

    def _predict_sequence(self, probs: torch.Tensor) -> torch.Tensor:
        probs = _normalize_simplex(probs, eps=self.cfg.simplex_eps)
        pred = probs.clone()

        if self.cfg.fitted_num_samples <= 0 and self.cfg.effective_ar_order == 0 and self.cfg.effective_ma_order == 0:
            return pred

        if probs.size(1) == 0:
            return pred

        smoothed = _strictly_positive_simplex(probs, max(self.cfg.simplex_eps, self.cfg.dirichlet_eps))
        ar_order = int(self.ar_coefficients.size(0))
        ma_order = int(self.ma_coefficients.size(0))
        memory = max(ar_order, ma_order)

        mean_alr = self.mean_alr.to(device=probs.device, dtype=probs.dtype)
        ar_coefficients = self.ar_coefficients.to(device=probs.device, dtype=probs.dtype)
        ma_coefficients = self.ma_coefficients.to(device=probs.device, dtype=probs.dtype)

        if ar_order == 0 and ma_order == 0 and self.cfg.fitted_num_samples > 0:
            constant = _alr_to_simplex(mean_alr, self.reference_index, self.cfg.prob_dim, self.cfg.simplex_eps)
            pred[:] = constant.view(1, 1, -1).expand_as(pred)
            return pred

        start = max(memory - 1, 0)
        if probs.size(1) <= start:
            return pred

        for batch_idx in range(probs.size(0)):
            alr_sequence = _simplex_to_alr(
                smoothed[batch_idx],
                self.reference_index,
                max(self.cfg.simplex_eps, self.cfg.dirichlet_eps),
            )
            _, residual = _compute_observed_states(alr_sequence, mean_alr, ar_coefficients, ma_coefficients)

            for step_idx in range(start, int(alr_sequence.size(0))):
                eta_next = _forecast_next_alr(
                    alr_sequence,
                    residual,
                    step_idx,
                    mean_alr,
                    ar_coefficients,
                    ma_coefficients,
                )
                pred[batch_idx, step_idx] = _alr_to_simplex(
                    eta_next,
                    self.reference_index,
                    self.cfg.prob_dim,
                    self.cfg.simplex_eps,
                )

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
            raise ValueError("runtime ablations are not supported for MODEL=bdarma")

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
