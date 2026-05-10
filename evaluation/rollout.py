from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .metrics import compute_pointwise_metrics, normalize_simplex
from .sidecars import infer_token_labels


def format_rollout_metric_line(
    metrics: Dict[str, float],
    per_horizon_metrics: Dict[str, List[float]],
) -> str:
    pieces = [
        f"examples {int(metrics['num_examples'])}",
        f"context {int(metrics['context_steps'])}",
        f"horizon {int(metrics['horizon_steps'])}",
        f"kl {metrics['kl']:.4f}",
        f"jsd {metrics['jsd']:.4f}",
        f"l1 {metrics['l1']:.4f}",
        f"bc {metrics['bray_curtis']:.4f}",
    ]
    if "w1" in metrics:
        pieces.append(f"w1 {metrics['w1']:.4f}")

    jsd_curve = per_horizon_metrics.get("jsd")
    if jsd_curve:
        pieces.append(f"jsd@1 {jsd_curve[0]:.4f}")
        pieces.append(f"jsd@H {jsd_curve[-1]:.4f}")

    return " | ".join(pieces)


@torch.no_grad()
def autoregressive_rollout(
    model,
    contexts: torch.Tensor,
    *,
    horizon_steps: int,
    max_context_steps: int,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    history = contexts.to(device=device, dtype=torch.float32)
    pred_steps = []

    for _ in range(horizon_steps):
        current = history[:, -max_context_steps:].contiguous()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            batch_out = model(current, return_dict=True)

        next_pred = normalize_simplex(batch_out["pred"][:, -1:].detach().float())
        pred_steps.append(next_pred)
        history = torch.cat([history, next_pred], dim=1)

    return torch.cat(pred_steps, dim=1)


@torch.no_grad()
def evaluate_rollout_examples(
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
    max_examples: int,
    context_steps: int,
    horizon_steps: int,
    max_context_steps: int,
) -> Tuple[Dict[str, float], Dict[str, List[float]], Optional[Dict[str, object]]]:
    examples = task.get_rollout_examples(
        split,
        max_examples=max_examples,
        context_steps=context_steps,
        horizon_steps=horizon_steps,
    )
    if not examples:
        raise ValueError(f"split={split} yielded no rollout examples")

    contexts = torch.stack([example.context.float() for example in examples], dim=0).contiguous()
    targets = torch.stack([example.targets.float() for example in examples], dim=0).contiguous()
    if contexts.ndim != 3 or targets.ndim != 3:
        raise ValueError("rollout evaluation requires context and target tensors with shape (N,T,D)")
    if contexts.size(1) != context_steps or targets.size(1) != horizon_steps:
        raise ValueError("task returned rollout examples with unexpected context/horizon shapes")

    ordered_support_values = task.ordered_probability_support()
    ordered_support = ordered_support_values is not None

    overall_sums: Dict[str, float] = {}
    horizon_sums: Dict[str, torch.Tensor] = {}
    total_steps = 0
    total_examples = 0

    prediction_chunks: List[torch.Tensor] = []
    metric_chunks: Dict[str, List[torch.Tensor]] = {}

    model.eval()
    for start in range(0, len(examples), batch_size):
        end = min(start + batch_size, len(examples))
        batch_context = contexts[start:end]
        batch_targets = targets[start:end]

        pred = autoregressive_rollout(
            model,
            batch_context,
            horizon_steps=horizon_steps,
            max_context_steps=max_context_steps,
            device=device,
            use_amp=use_amp,
        ).cpu()

        point_metrics = compute_pointwise_metrics(
            batch_targets,
            pred,
            include_w1=ordered_support,
        )

        total_examples += pred.size(0)
        total_steps += pred.size(0) * pred.size(1)
        for key, value in point_metrics.items():
            overall_sums[key] = overall_sums.get(key, 0.0) + float(value.sum().item())
            horizon_total = value.sum(dim=0)
            if key not in horizon_sums:
                horizon_sums[key] = horizon_total.clone()
            else:
                horizon_sums[key] += horizon_total

            if include_artifact:
                metric_chunks.setdefault(key, []).append(value.to(torch.float32))

        if include_artifact:
            prediction_chunks.append(pred.to(torch.float16))

    if total_steps <= 0:
        raise ValueError("rollout evaluation produced zero scored steps")

    metrics = {
        key: value / float(total_steps)
        for key, value in overall_sums.items()
    }
    metrics.update(
        {
            "num_examples": float(total_examples),
            "context_steps": float(context_steps),
            "horizon_steps": float(horizon_steps),
        }
    )

    per_horizon_metrics = {
        key: (value / float(total_examples)).tolist()
        for key, value in horizon_sums.items()
    }

    artifact = None
    if include_artifact:
        dataset_dir = Path(task.dataset_dir).expanduser().resolve()
        artifact = {
            "kind": "rollout_eval",
            "run_tag": run_tag,
            "dataset": task.name,
            "model_label": model_label,
            "checkpoint_kind": checkpoint_kind,
            "split": split,
            "dataset_dir": str(dataset_dir),
            "context_steps": int(context_steps),
            "horizon_steps": int(horizon_steps),
            "num_examples": int(total_examples),
            "distribution_metrics": {
                key: float(value)
                for key, value in metrics.items()
                if key not in {"num_examples", "context_steps", "horizon_steps"}
            },
            "per_horizon_metrics": {
                key: torch.tensor(values, dtype=torch.float32)
                for key, values in per_horizon_metrics.items()
            },
            "contexts": contexts.to(torch.float16),
            "targets": targets.to(torch.float16),
            "predictions": torch.cat(prediction_chunks, dim=0),
            "ordered_support": ordered_support,
            "support_values": (
                ordered_support_values.detach().cpu().to(torch.float32)
                if ordered_support_values is not None
                else None
            ),
            "point_metrics": {
                key: torch.cat(chunks, dim=0)
                for key, chunks in metric_chunks.items()
            },
            "sequence_labels": [example.sequence_label for example in examples],
            "time_labels": [list(example.time_labels) for example in examples],
            "token_labels": infer_token_labels(dataset_dir=dataset_dir, prob_dim=targets.size(-1)),
        }

    return metrics, per_horizon_metrics, artifact


def save_rollout_artifact(
    artifact: Dict[str, object],
    *,
    artifact_dir: Path,
    run_tag: str,
    split: str,
) -> tuple[Path, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_tag}_{split}_rollout"
    artifact_path = artifact_dir / f"{stem}.pt"
    metrics_path = artifact_dir / f"{stem}.metrics.json"
    torch.save(artifact, artifact_path)

    metrics_payload = {
        "kind": artifact["kind"],
        "run_tag": artifact["run_tag"],
        "dataset": artifact["dataset"],
        "model_label": artifact["model_label"],
        "checkpoint_kind": artifact["checkpoint_kind"],
        "split": artifact["split"],
        "context_steps": artifact["context_steps"],
        "horizon_steps": artifact["horizon_steps"],
        "num_examples": artifact["num_examples"],
        "distribution_metrics": artifact["distribution_metrics"],
        "per_horizon_metrics": {
            key: value.tolist()
            for key, value in artifact["per_horizon_metrics"].items()
        },
        "artifact_path": str(artifact_path),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return artifact_path, metrics_path
