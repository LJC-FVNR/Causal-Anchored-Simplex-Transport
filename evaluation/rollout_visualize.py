from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _sequence_label(artifact: Dict[str, object], sequence_index: int) -> str:
    labels = artifact.get("sequence_labels") or []
    if sequence_index < len(labels):
        return str(labels[sequence_index])
    return f"sequence_{sequence_index}"


def _time_labels(artifact: Dict[str, object], sequence_index: int) -> List[str]:
    labels = artifact.get("time_labels") or []
    if sequence_index < len(labels):
        return [str(value) for value in labels[sequence_index]]
    horizon = int(artifact["targets"].shape[1])
    return [f"t+{idx}" for idx in range(1, horizon + 1)]


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


def _expected_support_curve(artifact: Dict[str, object], sequence_index: int) -> np.ndarray:
    support_values = _support_values(artifact)
    targets = artifact["targets"][sequence_index].float().cpu().numpy()
    return (targets * support_values[None, :]).sum(axis=-1)


def _hardest_sequence_index(artifacts: List[Dict[str, object]]) -> int:
    avg_jsd = None
    for artifact in artifacts:
        jsd = artifact["point_metrics"]["jsd"].float().mean(dim=1).cpu().numpy()
        avg_jsd = jsd if avg_jsd is None else np.mean(np.stack([avg_jsd, jsd], axis=0), axis=0)
    return int(np.argmax(avg_jsd))


def _lowest_support_sequence_index(artifacts: List[Dict[str, object]]) -> int:
    reference = artifacts[0]
    mean_support = np.stack(
        [_expected_support_curve(reference, idx) for idx in range(reference["targets"].shape[0])],
        axis=0,
    ).mean(axis=1)
    return int(np.argmin(mean_support))


def _highest_support_sequence_index(artifacts: List[Dict[str, object]]) -> int:
    reference = artifacts[0]
    mean_support = np.stack(
        [_expected_support_curve(reference, idx) for idx in range(reference["targets"].shape[0])],
        axis=0,
    ).mean(axis=1)
    return int(np.argmax(mean_support))


def _select_sequence_index(artifacts: List[Dict[str, object]], policy: str) -> int:
    if policy == "lowest_mean_support":
        return _lowest_support_sequence_index(artifacts)
    if policy == "highest_mean_support":
        return _highest_support_sequence_index(artifacts)
    return _hardest_sequence_index(artifacts)


def _snapshot_index(
    artifacts: List[Dict[str, object]],
    sequence_index: int,
    *,
    policy: str,
) -> int:
    reference = artifacts[0]
    if policy == "lowest_mean_support":
        support_curve = _expected_support_curve(reference, sequence_index)
        return int(np.argmin(support_curve))
    if policy == "highest_mean_support":
        support_curve = _expected_support_curve(reference, sequence_index)
        return int(np.argmax(support_curve))

    avg_jsd = None
    for artifact in artifacts:
        jsd = artifact["point_metrics"]["jsd"][sequence_index].float().cpu().numpy()
        avg_jsd = jsd if avg_jsd is None else np.mean(np.stack([avg_jsd, jsd], axis=0), axis=0)
    return int(np.argmax(avg_jsd))


def _rollout_xticks(labels: List[str]) -> tuple[List[int], List[str]]:
    if len(labels) <= 12:
        return list(range(len(labels))), labels
    stride = max(1, math.ceil(len(labels) / 8))
    idxs = list(range(0, len(labels), stride))
    if idxs[-1] != len(labels) - 1:
        idxs.append(len(labels) - 1)
    return idxs, [labels[idx] for idx in idxs]


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
    artifacts: List[Dict[str, object]],
    sequence_index: int,
    *,
    pad: int,
    min_width: int,
    quantile_low: float = 0.001,
    quantile_high: float = 0.999,
) -> Tuple[int, int]:
    arrays: List[np.ndarray] = [artifacts[0]["targets"][sequence_index].float().cpu().numpy()]
    for artifact in artifacts:
        arrays.append(artifact["predictions"][sequence_index].float().cpu().numpy())
    return _support_window_from_arrays(
        arrays,
        pad=pad,
        min_width=min_width,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )


def _set_sparse_tick_labels(ax: plt.Axes, positions: List[int], labels: List[str], *, axis: str) -> None:
    if not positions:
        return
    if axis == "x":
        stride = max(1, math.ceil(len(positions) / 16))
        tick_positions = positions[::stride]
        if tick_positions[-1] != positions[-1]:
            tick_positions.append(positions[-1])
        tick_labels = [labels[positions.index(pos)] for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    else:
        stride = max(1, math.ceil(len(positions) / 20))
        tick_positions = positions[::stride]
        if tick_positions[-1] != positions[-1]:
            tick_positions.append(positions[-1])
        tick_labels = [labels[positions.index(pos)] for pos in tick_positions]
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)


def _select_component_dims(
    artifacts: List[Dict[str, object]],
    sequence_index: int,
    *,
    top_k: int,
    max_dims: int,
) -> List[int]:
    target = artifacts[0]["targets"][sequence_index].float().cpu().numpy()
    union = set(np.argsort(target.mean(axis=0))[-max_dims:].tolist())
    for artifact in artifacts:
        pred = artifact["predictions"][sequence_index].float().cpu().numpy()
        union.update(np.argsort(pred.mean(axis=0))[-max_dims:].tolist())
    selected = sorted(union, key=lambda idx: float(target[:, idx].mean()), reverse=True)
    return sorted(selected[:top_k if top_k > 0 else max_dims])


def plot_rollout_artifact_comparison(
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
        raise ValueError("at least one rollout artifact is required")

    if sequence_index < 0:
        sequence_index = _select_sequence_index(artifacts, sequence_policy)

    reference = artifacts[0]
    ordered_support = bool(reference.get("ordered_support", False))
    sequence_label = _sequence_label(reference, sequence_index)
    time_labels = _time_labels(reference, sequence_index)
    token_labels = _token_labels(reference)
    snapshot_idx = (
        snapshot_index
        if snapshot_index >= 0
        else _snapshot_index(artifacts, sequence_index, policy=snapshot_policy)
    )
    snapshot_label = time_labels[snapshot_idx] if snapshot_idx < len(time_labels) else f"t+{snapshot_idx + 1}"
    produced: List[Path] = []
    support_window = None
    heatmap_window = None
    if ordered_support and ordered_support_mode == "focus":
        support_window = _ordered_support_window(
            artifacts,
            sequence_index,
            pad=ordered_support_pad,
            min_width=ordered_support_min_window,
            quantile_low=0.002,
            quantile_high=0.998,
        )
        heatmap_window = _ordered_support_window(
            artifacts,
            sequence_index,
            pad=max(8, ordered_support_pad // 3),
            min_width=max(40, ordered_support_min_window // 2),
            quantile_low=0.01,
            quantile_high=0.99,
        )

    # 1. Error curves by rollout horizon.
    metrics_to_plot = ["jsd", "l1"]
    if "w1" in reference.get("per_horizon_metrics", {}):
        metrics_to_plot.append("w1")
    titles = {
        "jsd": "JSD",
        "l1": "L1",
        "w1": "Wasserstein-1",
    }
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 3.2 * len(metrics_to_plot)), sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(1, len(reference["per_horizon_metrics"]["jsd"]) + 1)
    xticks, xticklabels = _rollout_xticks(time_labels)
    for ax, metric in zip(axes, metrics_to_plot):
        for artifact in artifacts:
            values = artifact["per_horizon_metrics"][metric].float().cpu().numpy()
            ax.plot(x, values, linewidth=2.0, label=str(artifact["model_label"]))
        ax.set_ylabel(titles[metric])
        ax.set_title(f"{titles[metric]} vs rollout horizon")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Forecast step")
    axes[-1].set_xticks([idx + 1 for idx in xticks])
    axes[-1].set_xticklabels(xticklabels, rotation=45, ha="right")
    axes[0].legend()
    fig.tight_layout()
    path = output_dir / "rollout_error_by_horizon.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    produced.append(path)

    # 2. Distribution snapshot at the hardest horizon for the chosen sequence.
    target_snapshot = reference["targets"][sequence_index, snapshot_idx].float().cpu().numpy()
    if ordered_support:
        if support_window is None:
            selected = np.arange(len(token_labels))
        else:
            low, high = support_window
            selected = np.arange(low, high + 1)
        x_bins = np.arange(len(selected))
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(x_bins, target_snapshot[selected], color="black", linewidth=2.4, label="target")
        for artifact in artifacts:
            pred_snapshot = artifact["predictions"][sequence_index, snapshot_idx].float().cpu().numpy()
            ax.plot(x_bins, pred_snapshot[selected], linewidth=1.8, label=str(artifact["model_label"]))
        ax.set_title(f"Rollout snapshot at {snapshot_label}: {sequence_label}")
        ax.set_ylabel("Probability mass")
        ax.set_xlabel("Support bin")
        selected_labels = [token_labels[idx] for idx in selected]
        _set_sparse_tick_labels(ax, list(x_bins), selected_labels, axis="x")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"rollout_snapshot_{snapshot_label}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        produced.append(path)
    else:
        union = set(np.argsort(target_snapshot)[-top_k:].tolist())
        for artifact in artifacts:
            pred_snapshot = artifact["predictions"][sequence_index, snapshot_idx].float().cpu().numpy()
            union.update(np.argsort(pred_snapshot)[-top_k:].tolist())
        selected = sorted(union, key=lambda idx: float(target_snapshot[idx]), reverse=True)[:top_k]
        labels = [token_labels[idx] for idx in selected]
        x_bins = np.arange(len(selected))
        width = 0.8 / (len(artifacts) + 1)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.bar(x_bins, target_snapshot[selected], width=width, label="target", color="black", alpha=0.75)
        for offset, artifact in enumerate(artifacts, start=1):
            pred_snapshot = artifact["predictions"][sequence_index, snapshot_idx].float().cpu().numpy()
            ax.bar(
                x_bins + offset * width,
                pred_snapshot[selected],
                width=width,
                label=str(artifact["model_label"]),
                alpha=0.8,
            )
        ax.set_title(f"Rollout top-{top_k} snapshot at {snapshot_label}: {sequence_label}")
        ax.set_ylabel("Probability mass")
        ax.set_xlabel("Component")
        ax.set_xticks(x_bins + width * len(artifacts) / 2)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"rollout_snapshot_topk_{snapshot_label}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        produced.append(path)

    # 3. Target vs prediction heatmaps on the rollout horizon.
    if ordered_support and heatmap_window is not None:
        low, high = heatmap_window
        selected_dims = list(range(low, high + 1))
    else:
        selected_dims = _select_component_dims(
            artifacts,
            sequence_index,
            top_k=max(12, min(top_k, 24)),
            max_dims=24,
        )
    heatmap_labels = [token_labels[idx] for idx in selected_dims]
    target_heat = reference["targets"][sequence_index, :, selected_dims].float().cpu().numpy().T
    num_rows = len(artifacts) + 1
    fig, axes = plt.subplots(
        num_rows,
        1,
        figsize=(12, 2.4 * num_rows),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    target_im = axes[0].imshow(target_heat, aspect="auto", origin="lower", interpolation="nearest")
    axes[0].set_title(f"Rollout target heatmap ({sequence_label})")
    _set_sparse_tick_labels(axes[0], list(range(len(selected_dims))), heatmap_labels, axis="y")
    for row, artifact in enumerate(artifacts, start=1):
        pred_heat = artifact["predictions"][sequence_index, :, selected_dims].float().cpu().numpy().T
        axes[row].imshow(pred_heat, aspect="auto", origin="lower", interpolation="nearest")
        axes[row].set_title(f"Rollout prediction heatmap: {artifact['model_label']}")
        _set_sparse_tick_labels(axes[row], list(range(len(selected_dims))), heatmap_labels, axis="y")
    axes[-1].set_xlabel("Forecast time")
    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels(xticklabels, rotation=45, ha="right")
    fig.colorbar(target_im, ax=axes, shrink=0.75, pad=0.01, label="Probability mass")
    path = output_dir / "rollout_target_and_prediction_heatmaps.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    produced.append(path)

    # 4. Ordered-support expected value drift or unordered top-k component trajectories.
    if ordered_support:
        support_values = torch.from_numpy(_support_values(reference)).to(torch.float32)

        def expected_support(dist: torch.Tensor) -> np.ndarray:
            return (dist.float() * support_values.unsqueeze(0)).sum(dim=-1).cpu().numpy()

        anchor = expected_support(reference["contexts"][sequence_index, -1:].float())[0]
        fig, ax = plt.subplots(figsize=(10, 4.2))
        x_values = np.arange(0, len(time_labels) + 1)
        target_curve = np.concatenate([[anchor], expected_support(reference["targets"][sequence_index])], axis=0)
        ax.plot(x_values, target_curve, color="black", linewidth=2.4, label="target")
        for artifact in artifacts:
            pred_curve = np.concatenate([[anchor], expected_support(artifact["predictions"][sequence_index])], axis=0)
            ax.plot(x_values, pred_curve, linewidth=1.8, label=str(artifact["model_label"]))
        ax.set_title(f"Expected support drift across rollout: {sequence_label}")
        ax.set_ylabel("Expected support")
        ax.set_xlabel("Forecast time")
        anchor_labels = ["ctx"] + time_labels
        drift_ticks, drift_ticklabels = _rollout_xticks(anchor_labels)
        ax.set_xticks(drift_ticks)
        ax.set_xticklabels(drift_ticklabels, rotation=45, ha="right")
        if support_window is not None:
            low, high = support_window
            lower = float(support_values[low].item())
            upper = float(support_values[high].item())
            if upper > lower:
                pad = max(1.0, 0.05 * (upper - lower))
                ax.set_ylim(lower - pad, upper + pad)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = output_dir / "rollout_expected_support.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        produced.append(path)
    else:
        selected = _select_component_dims(artifacts, sequence_index, top_k=min(top_k, 4), max_dims=8)
        num_axes = len(selected)
        fig, axes = plt.subplots(num_axes, 1, figsize=(10, 2.8 * num_axes), sharex=True)
        axes = np.atleast_1d(axes)
        x_values = np.arange(1, len(time_labels) + 1)
        for ax, dim_idx in zip(axes, selected):
            target_curve = reference["targets"][sequence_index, :, dim_idx].float().cpu().numpy()
            ax.plot(x_values, target_curve, color="black", linewidth=2.2, label="target")
            for artifact in artifacts:
                pred_curve = artifact["predictions"][sequence_index, :, dim_idx].float().cpu().numpy()
                ax.plot(x_values, pred_curve, linewidth=1.6, label=str(artifact["model_label"]))
            ax.set_ylabel(token_labels[dim_idx])
            ax.grid(alpha=0.25)
        axes[0].set_title(f"Top component trajectories across rollout: {sequence_label}")
        axes[-1].set_xlabel("Forecast time")
        axes[-1].set_xticks([idx + 1 for idx in xticks])
        axes[-1].set_xticklabels(xticklabels, rotation=45, ha="right")
        axes[0].legend()
        fig.tight_layout()
        path = output_dir / "rollout_topk_component_trajectories.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        produced.append(path)

    return produced
