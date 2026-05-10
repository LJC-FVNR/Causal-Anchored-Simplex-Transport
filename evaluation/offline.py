from __future__ import annotations

import json
from pathlib import Path
import os
from typing import Dict, Optional, Tuple

import torch

from .metrics import compute_pointwise_metrics
from .sidecars import build_artifact_labels

LOSS_KEYS = ("loss", "main_loss", "aux_prior_loss", "transport_reg_loss")


def _model_max_seq_len(model) -> Optional[int]:
    cfg = getattr(model, "cfg", None)
    max_seq_len = getattr(cfg, "max_seq_len", None)
    if max_seq_len is not None:
        return int(max_seq_len)
    return None


def _align_split_payload(split_payload) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = split_payload.inputs.float().contiguous()
    lengths = split_payload.lengths.long().contiguous()

    if split_payload.targets is None:
        if inputs.size(1) < 2:
            raise ValueError("offline evaluation requires sequence length >= 2")
        context = inputs[:, :-1].contiguous()
        targets = inputs[:, 1:].contiguous()
        lengths = (lengths - 1).clamp_min(0)
        if split_payload.loss_mask is not None:
            mask = split_payload.loss_mask[:, 1:].to(torch.bool).contiguous()
        else:
            mask = None
    else:
        context = inputs
        targets = split_payload.targets.float().contiguous()
        mask = split_payload.loss_mask.to(torch.bool).contiguous() if split_payload.loss_mask is not None else None

    if targets.ndim != 3:
        raise ValueError("offline probability evaluation requires distribution-valued targets with shape (N,T,D)")

    if mask is None:
        positions = torch.arange(targets.size(1), dtype=torch.long).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)

    return context, targets, mask, lengths


def _evaluate_long_context_batch(
    model,
    *,
    batch_inputs: torch.Tensor,
    batch_targets: torch.Tensor,
    batch_mask: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    max_seq_len: int,
) -> Tuple[Dict[str, float], float, torch.Tensor]:
    """Evaluate long sequences with a rolling context window.

    Models with learned absolute positional embeddings cannot be safely rebuilt
    with a longer max length at evaluation time. For sequences longer than the
    trained context limit, score each next-step target using the latest valid
    context window and place the last-position prediction back into the full
    artifact timeline.
    """

    if max_seq_len < 1:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

    batch_size, seq_len, prob_dim = batch_inputs.shape
    pred_full = batch_inputs.new_zeros(batch_size, seq_len, prob_dim)
    loss_sums = {key: 0.0 for key in LOSS_KEYS}
    total_weight = 0.0

    model.eval()
    prefix_len = min(seq_len, max_seq_len)
    prefix_mask = batch_mask[:, :prefix_len]
    if bool(prefix_mask.any().item()):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            batch_out = model(
                batch_inputs[:, :prefix_len],
                targets=batch_targets[:, :prefix_len],
                loss_mask=prefix_mask,
                return_dict=True,
            )

        pred_full[:, :prefix_len] = batch_out["pred"].detach().float()
        weight = float(prefix_mask.sum().item())
        total_weight += weight
        for key in LOSS_KEYS:
            loss_sums[key] += float(batch_out[key].detach().float().item()) * weight

    if seq_len <= max_seq_len:
        return loss_sums, total_weight, pred_full

    window_batch_size = int(os.environ.get("OFFLINE_LONG_WINDOW_BATCH_SIZE", "128"))
    window_batch_size = max(batch_size, window_batch_size)
    positions_per_chunk = max(1, window_batch_size // batch_size)

    for pos_start in range(max_seq_len, seq_len, positions_per_chunk):
        positions = list(range(pos_start, min(seq_len, pos_start + positions_per_chunk)))
        num_positions = len(positions)
        windows = torch.stack(
            [batch_inputs[:, t - max_seq_len + 1 : t + 1] for t in positions],
            dim=1,
        ).reshape(batch_size * num_positions, max_seq_len, prob_dim)
        window_targets = torch.zeros_like(windows)
        target_last = torch.stack([batch_targets[:, t] for t in positions], dim=1).reshape(
            batch_size * num_positions,
            prob_dim,
        )
        window_targets[:, -1] = target_last
        window_mask = torch.zeros(
            batch_size,
            num_positions,
            max_seq_len,
            dtype=torch.bool,
            device=batch_inputs.device,
        )
        window_mask[:, :, -1] = torch.stack([batch_mask[:, t] for t in positions], dim=1)
        window_mask = window_mask.reshape(batch_size * num_positions, max_seq_len)
        if not bool(window_mask.any().item()):
            continue

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            batch_out = model(
                windows,
                targets=window_targets,
                loss_mask=window_mask,
                return_dict=True,
            )

        pred_last = batch_out["pred"][:, -1].detach().float().reshape(batch_size, num_positions, prob_dim)
        for offset, t in enumerate(positions):
            pred_full[:, t] = pred_last[:, offset]

        weight = float(window_mask.sum().item())
        total_weight += weight
        for key in LOSS_KEYS:
            loss_sums[key] += float(batch_out[key].detach().float().item()) * weight

    return loss_sums, total_weight, pred_full


def format_offline_metric_line(loss_metrics: Dict[str, float], distribution_metrics: Dict[str, float]) -> str:
    pieces = [
        f"main {loss_metrics['main_loss']:.4f}",
        f"kl {distribution_metrics['kl']:.4f}",
        f"jsd {distribution_metrics['jsd']:.4f}",
        f"l1 {distribution_metrics['l1']:.4f}",
        f"bc {distribution_metrics['bray_curtis']:.4f}",
    ]
    if "w1" in distribution_metrics:
        pieces.append(f"w1 {distribution_metrics['w1']:.4f}")
    pieces.append(f"total {loss_metrics['loss']:.4f}")
    pieces.append(f"aux {loss_metrics['aux_prior_loss']:.4f}")
    pieces.append(f"reg {loss_metrics['transport_reg_loss']:.6f}")
    return " | ".join(pieces)


@torch.no_grad()
def evaluate_probability_split(
    model,
    task,
    *,
    split: str,
    device: torch.device,
    use_amp: bool,
    batch_size: int,
    include_artifact: bool,
    run_tag: str,
    model_label: str,
    checkpoint_kind: str,
) -> Tuple[Dict[str, float], Dict[str, float], Optional[Dict[str, object]]]:
    split_payload = task.get_split_payload(split)
    context, targets, loss_mask, lengths = _align_split_payload(split_payload)
    ordered_support = task.ordered_probability_support() is not None

    total_weight = 0.0
    loss_sums = {key: 0.0 for key in LOSS_KEYS}
    metric_sums: Dict[str, float] = {}

    pred_chunks = []
    metric_chunks: Dict[str, list[torch.Tensor]] = {}
    max_seq_len = _model_max_seq_len(model)

    model.eval()
    for start in range(0, context.size(0), batch_size):
        end = min(start + batch_size, context.size(0))
        batch_inputs = context[start:end].to(device)
        batch_targets = targets[start:end].to(device)
        batch_mask = loss_mask[start:end].to(device)

        if max_seq_len is not None and batch_inputs.size(1) > max_seq_len:
            batch_loss_sums, weight, pred_device = _evaluate_long_context_batch(
                model,
                batch_inputs=batch_inputs,
                batch_targets=batch_targets,
                batch_mask=batch_mask,
                device=device,
                use_amp=use_amp,
                max_seq_len=max_seq_len,
            )
            total_weight += weight
            for key in LOSS_KEYS:
                loss_sums[key] += batch_loss_sums[key]
            pred = pred_device.detach().float().cpu()
        else:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                batch_out = model(
                    batch_inputs,
                    targets=batch_targets,
                    loss_mask=batch_mask,
                    return_dict=True,
                )

            weight = float(batch_mask.sum().item())
            total_weight += weight
            for key in LOSS_KEYS:
                loss_sums[key] += float(batch_out[key].detach().float().item()) * weight

            pred = batch_out["pred"].detach().float().cpu()
        target = batch_targets.detach().float().cpu()
        mask_cpu = batch_mask.detach().cpu()

        point_metrics = compute_pointwise_metrics(target, pred, include_w1=ordered_support)
        for key, value in point_metrics.items():
            masked_sum = torch.where(mask_cpu, value, torch.zeros_like(value)).sum().item()
            metric_sums[key] = metric_sums.get(key, 0.0) + float(masked_sum)
            if include_artifact:
                metric_copy = value.clone()
                metric_copy[~mask_cpu] = float("nan")
                metric_chunks.setdefault(key, []).append(metric_copy)

        if include_artifact:
            pred_chunks.append(pred)

    if total_weight <= 0:
        raise ValueError(f"split={split} has no scored targets for offline evaluation")

    loss_metrics = {key: value / total_weight for key, value in loss_sums.items()}
    distribution_metrics = {key: value / total_weight for key, value in metric_sums.items()}

    artifact = None
    if include_artifact:
        dataset_dir = Path(task.dataset_dir).expanduser().resolve()
        labels = {
            "sequence_labels": getattr(split_payload, "sequence_labels", None),
            "time_labels": getattr(split_payload, "time_labels", None),
            "token_labels": getattr(split_payload, "token_labels", None),
        }
        if (
            labels["sequence_labels"] is None
            or labels["time_labels"] is None
            or labels["token_labels"] is None
        ):
            labels = build_artifact_labels(
                dataset_dir=dataset_dir,
                num_sequences=context.size(0),
                prob_dim=context.size(-1),
                target_lengths=lengths.tolist(),
            )
        artifact = {
            "run_tag": run_tag,
            "dataset": task.name,
            "model_label": model_label,
            "checkpoint_kind": checkpoint_kind,
            "split": split,
            "dataset_dir": str(dataset_dir),
            "loss_metrics": loss_metrics,
            "distribution_metrics": distribution_metrics,
            "inputs": context.to(torch.float16),
            "targets": targets.to(torch.float16),
            "predictions": torch.cat(pred_chunks, dim=0).to(torch.float16),
            "loss_mask": loss_mask.to(torch.bool),
            "lengths": lengths,
            "ordered_support": ordered_support,
            "point_metrics": {key: torch.cat(value, dim=0).to(torch.float32) for key, value in metric_chunks.items()},
            **labels,
        }

    return loss_metrics, distribution_metrics, artifact


def save_evaluation_artifact(
    artifact: Dict[str, object],
    *,
    artifact_dir: Path,
    run_tag: str,
    split: str,
) -> tuple[Path, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_tag}_{split}"
    artifact_path = artifact_dir / f"{stem}.pt"
    metrics_path = artifact_dir / f"{stem}.metrics.json"
    torch.save(artifact, artifact_path)
    metrics_payload = {
        "run_tag": artifact["run_tag"],
        "dataset": artifact["dataset"],
        "model_label": artifact["model_label"],
        "checkpoint_kind": artifact["checkpoint_kind"],
        "split": artifact["split"],
        "loss_metrics": artifact["loss_metrics"],
        "distribution_metrics": artifact["distribution_metrics"],
        "artifact_path": str(artifact_path),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return artifact_path, metrics_path
