from __future__ import annotations

from typing import Dict

import torch

EPS = 1e-8


def normalize_simplex(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    x = x.float().clamp_min(0.0)
    denom = x.sum(dim=-1, keepdim=True).clamp_min(eps)
    return x / denom


def kl_divergence(target: torch.Tensor, pred: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    target = normalize_simplex(target, eps=eps)
    pred = normalize_simplex(pred, eps=eps)
    return (target * (target.clamp_min(eps).log() - pred.clamp_min(eps).log())).sum(dim=-1)


def js_divergence(target: torch.Tensor, pred: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    target = normalize_simplex(target, eps=eps)
    pred = normalize_simplex(pred, eps=eps)
    mean = 0.5 * (target + pred)
    return 0.5 * kl_divergence(target, mean, eps=eps) + 0.5 * kl_divergence(pred, mean, eps=eps)


def l1_distance(target: torch.Tensor, pred: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    del eps
    target = normalize_simplex(target)
    pred = normalize_simplex(pred)
    return (target - pred).abs().sum(dim=-1)


def bray_curtis_distance(target: torch.Tensor, pred: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    del eps
    target = normalize_simplex(target)
    pred = normalize_simplex(pred)
    numerator = (target - pred).abs().sum(dim=-1)
    denominator = (target + pred).sum(dim=-1).clamp_min(EPS)
    return numerator / denominator


def wasserstein_1_distance(target: torch.Tensor, pred: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    target = normalize_simplex(target, eps=eps)
    pred = normalize_simplex(pred, eps=eps)
    return (target.cumsum(dim=-1) - pred.cumsum(dim=-1)).abs().sum(dim=-1)


def compute_pointwise_metrics(
    target: torch.Tensor,
    pred: torch.Tensor,
    *,
    include_w1: bool,
) -> Dict[str, torch.Tensor]:
    metrics: Dict[str, torch.Tensor] = {
        "kl": kl_divergence(target, pred),
        "jsd": js_divergence(target, pred),
        "l1": l1_distance(target, pred),
        "bray_curtis": bray_curtis_distance(target, pred),
    }
    if include_w1:
        metrics["w1"] = wasserstein_1_distance(target, pred)
    return metrics

