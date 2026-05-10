from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Dict, Iterator, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from layer_perm_mixed_ablation import distribution_loss
from tasks.base import ModelFitSequence


def _normalize_simplex(probs: torch.Tensor, eps: float) -> torch.Tensor:
    if not torch.is_floating_point(probs):
        probs = probs.float()
    probs = probs.clamp_min(0.0)
    denom = probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return probs / denom


def _helmert_basis(prob_dim: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    if prob_dim <= 1:
        raise ValueError("prob_dim must be >= 2")

    basis = torch.zeros((prob_dim, prob_dim - 1), dtype=dtype)
    for col in range(prob_dim - 1):
        level = col + 1
        scale = math.sqrt(level * (level + 1))
        basis[:level, col] = 1.0 / scale
        basis[level, col] = -float(level) / scale
    return basis


def _simplex_to_ilr(probs: torch.Tensor, basis: torch.Tensor, eps: float) -> torch.Tensor:
    probs = _normalize_simplex(probs, eps=eps).clamp_min(eps)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    log_probs = probs.log()
    clr = log_probs - log_probs.mean(dim=-1, keepdim=True)
    return torch.matmul(clr, basis)


def _ilr_to_simplex(coords: torch.Tensor, basis: torch.Tensor, eps: float) -> torch.Tensor:
    clr = torch.matmul(coords, basis.transpose(0, 1))
    probs = torch.softmax(clr, dim=-1)
    return _normalize_simplex(probs, eps=eps)


def _iter_sequences_from_split_payload(split_payload) -> Iterator[ModelFitSequence]:
    inputs = split_payload.inputs.float().contiguous()
    lengths = split_payload.lengths.long().contiguous()
    targets = split_payload.targets
    loss_mask = split_payload.loss_mask

    if targets is not None:
        targets = targets.float().contiguous()
    if loss_mask is not None:
        loss_mask = loss_mask.to(torch.bool).contiguous()

    for seq_idx in range(inputs.size(0)):
        seq_len = int(lengths[seq_idx].item())
        if seq_len <= 0:
            continue

        seq_inputs = inputs[seq_idx, :seq_len].float().contiguous()
        seq_targets = targets[seq_idx, :seq_len].float().contiguous() if targets is not None else None
        seq_mask = (
            loss_mask[seq_idx, :seq_len].to(torch.bool).contiguous()
            if loss_mask is not None
            else None
        )
        yield ModelFitSequence(
            inputs=seq_inputs,
            targets=seq_targets,
            loss_mask=seq_mask,
        )


def _align_fit_sequence(example: ModelFitSequence) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    inputs = example.inputs.float().contiguous()
    if inputs.ndim != 2:
        raise ValueError("ilr-VAR expects each fit sequence to have shape (T,D)")

    if example.targets is None:
        if inputs.size(0) < 2:
            return None, None, None
        context = inputs[:-1].contiguous()
        targets = inputs[1:].contiguous()
        mask = example.loss_mask[1:].to(torch.bool).contiguous() if example.loss_mask is not None else None
    else:
        targets = example.targets.float().contiguous()
        context = inputs
        mask = example.loss_mask.to(torch.bool).contiguous() if example.loss_mask is not None else None

    if targets.ndim != 2:
        raise ValueError("ilr-VAR requires distribution-valued targets with shape (T,D)")
    if targets.shape != context.shape:
        raise ValueError(
            f"ilr-VAR requires aligned context/targets with the same shape, got {tuple(context.shape)} vs {tuple(targets.shape)}"
        )

    if mask is None:
        mask = torch.ones(targets.size(0), dtype=torch.bool)
    elif mask.ndim != 1 or mask.size(0) != targets.size(0):
        raise ValueError(
            f"ilr-VAR expects per-sequence loss masks with shape {(targets.size(0),)}, got {tuple(mask.shape)}"
        )

    return context, targets, mask


def _accumulate_var_sufficient_statistics(
    sequence_factory: Callable[[], Iterator[ModelFitSequence]],
    basis: torch.Tensor,
    *,
    lag_order: int,
    simplex_eps: float,
    chunk_size: int,
) -> Dict[str, torch.Tensor | int]:
    if lag_order < 1:
        raise ValueError("lag_order must be >= 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    ilr_dim = int(basis.size(1))
    feature_dim = lag_order * ilr_dim
    xtx = torch.zeros((feature_dim + 1, feature_dim + 1), dtype=basis.dtype)
    xty = torch.zeros((feature_dim + 1, ilr_dim), dtype=basis.dtype)
    yty = torch.zeros((ilr_dim, ilr_dim), dtype=basis.dtype)
    n_samples = 0
    min_index = lag_order - 1

    for example in sequence_factory():
        context, targets, mask = _align_fit_sequence(example)
        if context is None or targets is None or mask is None:
            continue
        if context.size(-1) != basis.size(0):
            raise ValueError(
                f"fit sequence prob_dim mismatch: expected {basis.size(0)}, got {context.size(-1)}"
            )

        valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        valid_idx = valid_idx[valid_idx >= min_index]
        if valid_idx.numel() == 0:
            continue

        z_context = _simplex_to_ilr(context.to(dtype=basis.dtype), basis, simplex_eps)
        z_targets = _simplex_to_ilr(targets.to(dtype=basis.dtype), basis, simplex_eps)
        lag_blocks = z_context.unfold(0, lag_order, 1).permute(0, 2, 1)

        for start in range(0, valid_idx.numel(), chunk_size):
            chunk_idx = valid_idx[start : start + chunk_size]
            x_block = lag_blocks[chunk_idx - min_index].reshape(chunk_idx.numel(), -1)
            y_block = z_targets[chunk_idx]
            ones = torch.ones((chunk_idx.numel(), 1), dtype=basis.dtype)
            design = torch.cat([ones, x_block], dim=1)

            xtx += design.transpose(0, 1) @ design
            xty += design.transpose(0, 1) @ y_block
            yty += y_block.transpose(0, 1) @ y_block
            n_samples += int(chunk_idx.numel())

    if n_samples <= 0:
        raise ValueError(f"lag_order={lag_order} yields no valid training samples")

    return {
        "xtx": xtx,
        "xty": xty,
        "yty": yty,
        "num_samples": n_samples,
    }


def _fit_var_parameters(
    xtx: torch.Tensor,
    xty: torch.Tensor,
    yty: torch.Tensor,
    *,
    lag_order: int,
    ridge: float,
    num_samples: int,
) -> Dict[str, object]:
    if xtx.ndim != 2 or xty.ndim != 2 or yty.ndim != 2:
        raise ValueError("sufficient statistics must be rank-2")
    if lag_order < 1:
        raise ValueError("lag_order must be >= 1")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    ilr_dim = int(xty.size(1))
    feature_dim = int(xtx.size(0) - 1)

    penalty = torch.eye(xtx.size(0), dtype=xtx.dtype)
    penalty[0, 0] = 0.0
    system = xtx + float(ridge) * penalty
    beta = torch.linalg.solve(system, xty)

    intercept = beta[0]
    coef_flat = beta[1:]
    coeffs = coef_flat.view(lag_order, ilr_dim, ilr_dim)

    residual_ss = yty - xty.transpose(0, 1) @ beta - beta.transpose(0, 1) @ xty + beta.transpose(0, 1) @ xtx @ beta
    residual_ss = 0.5 * (residual_ss + residual_ss.transpose(0, 1))
    sigma = residual_ss / max(num_samples, 1)
    sigma = sigma + 1e-9 * torch.eye(ilr_dim, dtype=xtx.dtype)
    sign, logabsdet = torch.linalg.slogdet(sigma)
    if sign <= 0:
        logabsdet = torch.logdet(sigma + 1e-6 * torch.eye(ilr_dim, dtype=xtx.dtype))

    num_params = (feature_dim + 1) * ilr_dim
    aic = float(num_samples * logabsdet.item() + 2.0 * num_params)
    bic = float(num_samples * logabsdet.item() + math.log(max(num_samples, 2)) * num_params)

    return {
        "intercept": intercept,
        "coefficients": coeffs,
        "num_samples": num_samples,
        "aic": aic,
        "bic": bic,
    }


@dataclass
class ILRVARConfig:
    prob_dim: int
    lag_order: int = 1
    max_lags: int = 8
    order_selection: str = "aic"
    ridge: float = 1e-4
    simplex_eps: float = 1e-8
    loss_type: str = "ce"
    fit_split: str = "train"
    fit_chunk_size: int = 1024
    fallback: str = "persistence"
    fitted_num_samples: int = 0
    selected_score: Optional[float] = None
    effective_order: int = 0

    def __post_init__(self) -> None:
        if self.prob_dim <= 1:
            raise ValueError("prob_dim must be >= 2")
        if self.max_lags < 1:
            raise ValueError("max_lags must be >= 1")
        if self.lag_order < 0:
            raise ValueError("lag_order must be >= 0")
        if self.order_selection not in {"aic", "bic", "fixed"}:
            raise ValueError("order_selection must be 'aic', 'bic', or 'fixed'")
        if self.loss_type not in {"ce", "kl"}:
            raise ValueError("loss_type must be 'ce' or 'kl'")
        if self.fit_chunk_size < 1:
            raise ValueError("fit_chunk_size must be >= 1")
        if self.fallback != "persistence":
            raise ValueError("currently only fallback='persistence' is supported")


class ILRVARBaseline(nn.Module):
    def __init__(
        self,
        cfg: ILRVARConfig,
        *,
        basis: torch.Tensor,
        intercept: Optional[torch.Tensor] = None,
        coefficients: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg = cfg
        ilr_dim = cfg.prob_dim - 1

        self.register_buffer("basis", basis.to(torch.float32))
        if intercept is None:
            intercept = torch.zeros(ilr_dim, dtype=torch.float32)
        if coefficients is None:
            coefficients = torch.zeros((cfg.effective_order, ilr_dim, ilr_dim), dtype=torch.float32)
        self.register_buffer("intercept", intercept.to(torch.float32))
        self.register_buffer("coefficients", coefficients.to(torch.float32))

    @classmethod
    def fit_from_split_payload(cls, cfg: ILRVARConfig, split_payload) -> "ILRVARBaseline":
        sequence_factory = lambda: _iter_sequences_from_split_payload(split_payload)
        return cls._fit_from_sequence_factory(cfg, sequence_factory)

    @classmethod
    def fit_from_task(cls, cfg: ILRVARConfig, task) -> "ILRVARBaseline":
        sequence_factory = lambda: task.iter_model_fit_sequences(cfg.fit_split, model_name="ilr_var")
        return cls._fit_from_sequence_factory(cfg, sequence_factory)

    @classmethod
    def _fit_from_sequence_factory(
        cls,
        cfg: ILRVARConfig,
        sequence_factory: Callable[[], Iterator[ModelFitSequence]],
    ) -> "ILRVARBaseline":
        basis = _helmert_basis(cfg.prob_dim, dtype=torch.float64)
        if cfg.order_selection == "fixed":
            candidate_orders = [max(int(cfg.lag_order), 1)]
        else:
            candidate_orders = list(range(1, int(cfg.max_lags) + 1))

        criterion = "aic" if cfg.order_selection == "aic" else "bic"
        best_fit: Optional[Dict[str, object]] = None
        best_order: Optional[int] = None
        best_score = float("inf")

        for lag_order in candidate_orders:
            try:
                stats = _accumulate_var_sufficient_statistics(
                    sequence_factory,
                    basis,
                    lag_order=lag_order,
                    simplex_eps=cfg.simplex_eps,
                    chunk_size=cfg.fit_chunk_size,
                )
            except ValueError:
                continue

            fit = _fit_var_parameters(
                stats["xtx"],
                stats["xty"],
                stats["yty"],
                lag_order=lag_order,
                ridge=cfg.ridge,
                num_samples=int(stats["num_samples"]),
            )
            score = float(fit[criterion])
            if best_fit is None or score < best_score:
                best_fit = fit
                best_order = lag_order
                best_score = score

        if best_fit is None or best_order is None:
            fitted_cfg = replace(
                cfg,
                lag_order=0,
                effective_order=0,
                fitted_num_samples=0,
                selected_score=None,
            )
            return cls(fitted_cfg, basis=basis)

        fitted_cfg = replace(
            cfg,
            lag_order=best_order,
            effective_order=best_order,
            fitted_num_samples=int(best_fit["num_samples"]),
            selected_score=best_score,
            order_selection="fixed" if cfg.order_selection == "fixed" else cfg.order_selection,
        )
        return cls(
            fitted_cfg,
            basis=basis,
            intercept=best_fit["intercept"],
            coefficients=best_fit["coefficients"],
        )

    def _predict_sequence(self, probs: torch.Tensor) -> torch.Tensor:
        probs = _normalize_simplex(probs, eps=self.cfg.simplex_eps)
        pred = probs.clone()

        if self.cfg.effective_order < 1:
            return pred

        z = _simplex_to_ilr(probs, self.basis.to(device=probs.device, dtype=probs.dtype), self.cfg.simplex_eps)
        start = self.cfg.effective_order - 1
        if z.size(1) <= start:
            return pred

        lag_windows = z.unfold(1, self.cfg.effective_order, 1).permute(0, 1, 3, 2)
        pred_z = torch.einsum(
            "btlk,lkh->bth",
            lag_windows,
            self.coefficients.to(device=z.device, dtype=z.dtype),
        )
        pred_z = pred_z + self.intercept.to(device=z.device, dtype=z.dtype).view(1, 1, -1)
        pred[:, start:] = _ilr_to_simplex(
            pred_z,
            self.basis.to(device=z.device, dtype=z.dtype),
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
            raise ValueError("runtime ablations are not supported for MODEL=ilr_var")

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
