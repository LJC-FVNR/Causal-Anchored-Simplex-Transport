from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _scored_indices(mask_row: torch.Tensor) -> np.ndarray:
    return torch.nonzero(mask_row, as_tuple=False).squeeze(-1).cpu().numpy()


def _sequence_label(artifact: Dict[str, object], sequence_index: int) -> str:
    labels = artifact.get("sequence_labels") or []
    if sequence_index < len(labels):
        return str(labels[sequence_index])
    return f"sequence_{sequence_index}"


def _time_labels(artifact: Dict[str, object], sequence_index: int) -> List[str]:
    time_labels = artifact.get("time_labels") or []
    if sequence_index < len(time_labels):
        return [str(value) for value in time_labels[sequence_index]]
    target_len = int(artifact["targets"].shape[1])
    return [f"t{idx}" for idx in range(1, target_len + 1)]


def _token_labels(artifact: Dict[str, object]) -> List[str]:
    labels = artifact.get("token_labels") or []
    if labels:
        return [str(value) for value in labels]
    dim = int(artifact["targets"].shape[-1])
    return [f"token_{idx}" for idx in range(dim)]


def _support_values(artifact: Dict[str, object]) -> np.ndarray:
    support_values = artifact.get("support_values")
    if support_values is not None:
        return support_values.float().cpu().numpy()
    dim = int(artifact["targets"].shape[-1])
    return np.arange(dim, dtype=np.float64)


def _expected_support_for_steps(artifact: Dict[str, object], sequence_index: int) -> np.ndarray:
    support_values = _support_values(artifact)
    targets = artifact["targets"][sequence_index].float().cpu().numpy()
    return (targets * support_values[None, :]).sum(axis=-1)


def _snapshot_index(
    artifacts: List[Dict[str, object]],
    sequence_index: int,
    *,
    policy: str,
) -> int:
    reference = artifacts[0]
    if policy == "lowest_mean_support":
        support_curve = _expected_support_for_steps(reference, sequence_index)
        scored_idx = _scored_indices(reference["loss_mask"][sequence_index])
        return int(scored_idx[int(np.nanargmin(support_curve[scored_idx]))])
    if policy == "highest_mean_support":
        support_curve = _expected_support_for_steps(reference, sequence_index)
        scored_idx = _scored_indices(reference["loss_mask"][sequence_index])
        return int(scored_idx[int(np.nanargmax(support_curve[scored_idx]))])

    avg_jsd = None
    for artifact in artifacts:
        jsd = artifact["point_metrics"]["jsd"][sequence_index].clone()
        mask = artifact["loss_mask"][sequence_index]
        jsd[~mask] = float("nan")
        values = jsd.numpy()
        avg_jsd = values if avg_jsd is None else np.nanmean(np.stack([avg_jsd, values], axis=0), axis=0)
    return int(np.nanargmax(avg_jsd))


def _hardest_sequence_index(artifacts: List[Dict[str, object]]) -> int:
    avg_jsd = None
    for artifact in artifacts:
        jsd = artifact["point_metrics"]["jsd"].clone()
        mask = artifact["loss_mask"]
        jsd[~mask] = float("nan")
        values = torch.nanmean(jsd, dim=1).cpu().numpy()
        avg_jsd = values if avg_jsd is None else np.nanmean(np.stack([avg_jsd, values], axis=0), axis=0)
    return int(np.nanargmax(avg_jsd))


def _lowest_support_sequence_index(artifacts: List[Dict[str, object]]) -> int:
    reference = artifacts[0]
    support_values = _support_values(reference)
    targets = reference["targets"].float().cpu().numpy()
    mask = reference["loss_mask"].cpu().numpy()
    mean_support = (targets * support_values[None, None, :]).sum(axis=-1)
    mean_support = np.where(mask, mean_support, np.nan)
    return int(np.nanargmin(np.nanmean(mean_support, axis=1)))


def _highest_support_sequence_index(artifacts: List[Dict[str, object]]) -> int:
    reference = artifacts[0]
    support_values = _support_values(reference)
    targets = reference["targets"].float().cpu().numpy()
    mask = reference["loss_mask"].cpu().numpy()
    mean_support = (targets * support_values[None, None, :]).sum(axis=-1)
    mean_support = np.where(mask, mean_support, np.nan)
    return int(np.nanargmax(np.nanmean(mean_support, axis=1)))


def _select_sequence_index(artifacts: List[Dict[str, object]], policy: str) -> int:
    if policy == "lowest_mean_support":
        return _lowest_support_sequence_index(artifacts)
    if policy == "highest_mean_support":
        return _highest_support_sequence_index(artifacts)
    return _hardest_sequence_index(artifacts)


def _expand_window(low: int, high: int, dim: int, min_width: int) -> Tuple[int, int]:
    width = high - low + 1
    if width >= min_width:
        return low, high
    deficit = min_width - width
    grow_left = deficit // 2
    grow_right = deficit - grow_left
    low = max(0, low - grow_left)
    high = min(dim - 1, high + grow_right)
    width = high - low + 1
    if width >= min_width:
        return low, high
    if low == 0:
        high = min(dim - 1, min_width - 1)
    elif high == dim - 1:
        low = max(0, dim - min_width)
    return low, high


def _support_window_from_arrays(
    arrays: List[np.ndarray],
    *,
    pad: int,
    min_width: int,
    quantile_low: float = 0.001,
    quantile_high: float = 0.999,
) -> Tuple[int, int]:
    dim = int(arrays[0].shape[-1])
    aggregate = np.zeros(dim, dtype=np.float64)
    for array in arrays:
        flat = np.asarray(array, dtype=np.float64).reshape(-1, dim)
        aggregate += flat.sum(axis=0)
    total = aggregate.sum()
    if not np.isfinite(total) or total <= 0:
        return 0, dim - 1
    cumulative = np.cumsum(aggregate) / total
    low = int(np.searchsorted(cumulative, quantile_low, side="left"))
    high = int(np.searchsorted(cumulative, quantile_high, side="left"))
    low = max(0, low - pad)
    high = min(dim - 1, high + pad)
    return _expand_window(low, high, dim, min_width)


def _ordered_support_window(
    reference: Dict[str, object],
    artifacts: List[Dict[str, object]],
    sequence_index: int,
    scored_idx: np.ndarray,
    *,
    pad: int,
    min_width: int,
    quantile_low: float = 0.001,
    quantile_high: float = 0.999,
) -> Tuple[int, int]:
    arrays: List[np.ndarray] = [reference["targets"][sequence_index, scored_idx].float().cpu().numpy()]
    for artifact in artifacts:
        arrays.append(artifact["predictions"][sequence_index, scored_idx].float().cpu().numpy())
    return _support_window_from_arrays(
        arrays,
        pad=pad,
        min_width=min_width,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )


def _set_sparse_tick_labels(ax: plt.Axes, positions: List[int], labels: List[str], *, axis: str) -> None:
    if axis == "x":
        if not positions:
            return
        stride = max(1, math.ceil(len(positions) / 16))
        tick_positions = positions[::stride]
        if tick_positions[-1] != positions[-1]:
            tick_positions.append(positions[-1])
        tick_labels = [labels[positions.index(pos)] for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    else:
        if not positions:
            return
        stride = max(1, math.ceil(len(positions) / 20))
        tick_positions = positions[::stride]
        if tick_positions[-1] != positions[-1]:
            tick_positions.append(positions[-1])
        tick_labels = [labels[positions.index(pos)] for pos in tick_positions]
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)


def plot_artifact_comparison(
    *,
    artifacts: List[Dict[str, object]],
    output_dir: Path,
    sequence_index: int = 0,
    snapshot_index: int = -1,
    top_k: int = 8,
    sequence_policy: str = "hardest",
    snapshot_policy: str = "hardest_jsd",
    ordered_support_mode: str = "full",
    ordered_support_pad: int = 24,
    ordered_support_min_window: int = 64,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not artifacts:
        raise ValueError("at least one artifact is required")

    if sequence_index < 0:
        sequence_index = _select_sequence_index(artifacts, sequence_policy)

    reference = artifacts[0]
    sequence_label = _sequence_label(reference, sequence_index)
    time_labels = _time_labels(reference, sequence_index)
    token_labels = _token_labels(reference)
    scored_idx = _scored_indices(reference["loss_mask"][sequence_index])
    if scored_idx.size == 0:
        raise ValueError(f"sequence_index={sequence_index} has no scored targets")
    snapshot_idx = (
        snapshot_index
        if snapshot_index >= 0
        else _snapshot_index(artifacts, sequence_index, policy=snapshot_policy)
    )
    snapshot_label = time_labels[snapshot_idx] if snapshot_idx < len(time_labels) else f"t{snapshot_idx + 1}"
    produced: List[Path] = []

    # 1. Error timeline
    fig, ax = plt.subplots(figsize=(10, 4))
    x_labels = [time_labels[idx] if idx < len(time_labels) else f"t{idx + 1}" for idx in scored_idx]
    for artifact in artifacts:
        model_label = str(artifact["model_label"])
        jsd = artifact["point_metrics"]["jsd"][sequence_index, scored_idx].cpu().numpy()
        ax.plot(range(len(scored_idx)), jsd, marker="o", linewidth=1.8, label=model_label)
    ax.set_title(f"Per-step JSD on held-out targets: {sequence_label}")
    ax.set_ylabel("JSD")
    ax.set_xlabel("Target time")
    ax.set_xticks(range(len(scored_idx)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "error_timeline_jsd.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    produced.append(path)

    target = reference["targets"][sequence_index, snapshot_idx].float().cpu().numpy()
    ordered_support = bool(reference.get("ordered_support", False))
    support_window = None
    heatmap_window = None
    if ordered_support and ordered_support_mode == "focus":
        support_window = _ordered_support_window(
            reference,
            artifacts,
            sequence_index,
            scored_idx,
            pad=ordered_support_pad,
            min_width=ordered_support_min_window,
            quantile_low=0.002,
            quantile_high=0.998,
        )
        heatmap_window = _ordered_support_window(
            reference,
            artifacts,
            sequence_index,
            scored_idx,
            pad=max(8, ordered_support_pad // 3),
            min_width=max(40, ordered_support_min_window // 2),
            quantile_low=0.01,
            quantile_high=0.99,
        )

    # 2. Distribution snapshot
    if ordered_support:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        if support_window is None:
            selected = np.arange(len(token_labels))
        else:
            low, high = support_window
            selected = np.arange(low, high + 1)
        x = np.arange(len(selected))
        ax.plot(x, target[selected], color="black", linewidth=2.4, label="target")
        for artifact in artifacts:
            pred = artifact["predictions"][sequence_index, snapshot_idx].float().cpu().numpy()
            ax.plot(x, pred[selected], linewidth=1.8, label=str(artifact["model_label"]))
        ax.set_title(f"Distribution snapshot at {snapshot_label}: {sequence_label}")
        ax.set_ylabel("Probability mass")
        ax.set_xlabel("Bin")
        selected_labels = [token_labels[idx] for idx in selected]
        _set_sparse_tick_labels(ax, list(x), selected_labels, axis="x")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"snapshot_{snapshot_label}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        produced.append(path)
    else:
        union = set(np.argsort(target)[-top_k:].tolist())
        for artifact in artifacts:
            pred = artifact["predictions"][sequence_index, snapshot_idx].float().cpu().numpy()
            union.update(np.argsort(pred)[-top_k:].tolist())
        selected = sorted(union, key=lambda idx: float(target[idx]), reverse=True)[:top_k]
        labels = [token_labels[idx] for idx in selected]
        x = np.arange(len(selected))
        width = 0.8 / (len(artifacts) + 1)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.bar(x, target[selected], width=width, label="target", color="black", alpha=0.75)
        for offset, artifact in enumerate(artifacts, start=1):
            pred = artifact["predictions"][sequence_index, snapshot_idx].float().cpu().numpy()
            ax.bar(x + offset * width, pred[selected], width=width, label=str(artifact["model_label"]), alpha=0.8)
        ax.set_title(f"Top-{top_k} component comparison at {snapshot_label}: {sequence_label}")
        ax.set_ylabel("Probability mass")
        ax.set_xlabel("Component")
        ax.set_xticks(x + width * len(artifacts) / 2)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"snapshot_topk_{snapshot_label}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        produced.append(path)

    # 3. Target / prediction / error heatmap
    max_dims = 16
    scored_targets = reference["targets"][sequence_index, scored_idx].float().cpu().numpy()
    if ordered_support:
        if heatmap_window is None:
            selected_dims = list(range(len(token_labels)))
        else:
            low, high = heatmap_window
            selected_dims = list(range(low, high + 1))
    else:
        selected_dims = np.argsort(scored_targets.sum(axis=0))[-max_dims:].tolist()
        selected_dims = sorted(selected_dims)
    heatmap_labels = [token_labels[idx] for idx in selected_dims]
    num_rows = len(artifacts) + 1
    fig, axes = plt.subplots(
        num_rows,
        1,
        figsize=(12, 2.2 * num_rows),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    target_view = scored_targets[:, selected_dims].T
    target_im = axes[0].imshow(target_view, aspect="auto", origin="lower", interpolation="nearest")
    axes[0].set_title(f"Target heatmap ({sequence_label})")
    _set_sparse_tick_labels(axes[0], list(range(len(selected_dims))), heatmap_labels, axis="y")
    error_im = None
    for row, artifact in enumerate(artifacts, start=1):
        pred_view = artifact["predictions"][sequence_index, scored_idx][:, selected_dims].float().cpu().numpy().T
        err_view = np.abs(pred_view - target_view)
        error_im = axes[row].imshow(err_view, aspect="auto", origin="lower", interpolation="nearest")
        axes[row].set_title(f"|prediction - target|: {artifact['model_label']}")
        _set_sparse_tick_labels(axes[row], list(range(len(selected_dims))), heatmap_labels, axis="y")
    axes[-1].set_xticks(range(len(scored_idx)))
    axes[-1].set_xticklabels(x_labels, rotation=45, ha="right")
    for ax in axes:
        ax.set_xlabel("Target time")
    fig.colorbar(target_im, ax=axes[0], shrink=0.75, pad=0.01, label="Target mass")
    if error_im is not None:
        fig.colorbar(error_im, ax=axes[1:], shrink=0.75, pad=0.01, label="Absolute error")
    path = output_dir / "target_and_error_heatmaps.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    produced.append(path)

    return produced
