from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from evaluation import (
    evaluate_probability_split,
    evaluate_rollout_examples,
    format_offline_metric_line,
    format_rollout_metric_line,
    save_evaluation_artifact,
    save_rollout_artifact,
)
from models import build_model
from tasks import build_task
from training.config import TrainingConfig
from training.distributed import cleanup_distributed, setup_distributed

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

LOSS_KEYS = ("loss", "main_loss", "aux_prior_loss", "transport_reg_loss")


def get_lr(step: int, config: TrainingConfig) -> float:
    if config.warmup_iters > 0 and step <= config.warmup_iters:
        return config.learning_rate * step / max(1, config.warmup_iters)
    return config.learning_rate


def maybe_all_reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value


@torch.no_grad()
def estimate_metrics(model, task, config, device, rank, world_size, use_amp, *, splits=("train", "val")):
    model.eval()
    out = {}
    for split in splits:
        split_metrics = {}
        for key in LOSS_KEYS:
            values = []
            for _ in range(config.eval_iters):
                batch = task.sample_batch(
                    split,
                    config.batch_size,
                    config.block_size,
                    device,
                    rank=rank,
                    world_size=world_size,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_amp,
                ):
                    batch_out = model(
                        batch.inputs,
                        targets=batch.targets,
                        loss_mask=batch.loss_mask,
                        return_dict=True,
                    )
                values.append(batch_out[key].detach().float())

            metric = torch.stack(values).mean()
            split_metrics[key] = float(maybe_all_reduce_mean(metric, world_size).item())
        out[split] = split_metrics

    model.train()
    return out


def format_metric_line(task, metrics):
    extra = task.extra_metric_items(metrics["main_loss"])
    extra_text = ""
    if extra:
        rendered = ", ".join(f"{name}={value:.3f}" for name, value in extra.items())
        extra_text = f" ({rendered})"

    return (
        f"total {metrics['loss']:.4f} | "
        f"main {metrics['main_loss']:.4f}{extra_text} | "
        f"aux {metrics['aux_prior_loss']:.4f} | "
        f"reg {metrics['transport_reg_loss']:.6f}"
    )


def log_dataset_summary(task) -> None:
    summary = task.dataset_summary()
    rendered = ", ".join(f"{key}={value}" for key, value in summary.items())
    print(f"[task] {rendered}")


def cpu_state_dict(module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}


def resolved_rollout_steps(config: TrainingConfig) -> tuple[int, int]:
    context_steps = config.rollout_context_steps
    horizon_steps = config.rollout_horizon_steps

    if context_steps is None:
        context_steps = min(config.model_max_seq_len, 64)
    if horizon_steps is None:
        horizon_steps = min(config.model_max_seq_len, 64)

    return int(context_steps), int(horizon_steps)


def completed_eval_metrics_path(config: TrainingConfig) -> Optional[Path]:
    if not config.skip_completed_identical:
        return None

    search_roots: list[Path] = [Path(config.eval_artifact_dir).expanduser()]
    if config.completed_search_root:
        search_roots.append(Path(config.completed_search_root).expanduser())

    patterns = [
        f"{config.run_tag}_test.metrics.json",
        f"{config.run_tag}_val.metrics.json",
    ]
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        for path in _completed_metric_candidates(root, patterns):
            resolved = _safe_resolve(path)
            if resolved is None or resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(path)

    valid = [path for path in candidates if _is_matching_completed_metric(path, config)]
    if not valid:
        return None

    current_eval_dir = _safe_resolve(Path(config.eval_artifact_dir).expanduser())

    def priority(path: Path) -> tuple[int, int, float, str]:
        split_rank = 0 if "_test.metrics.json" in path.name else 1
        parent = _safe_resolve(path.parent)
        local_rank = 0 if current_eval_dir is not None and parent == current_eval_dir else 1
        try:
            mtime_rank = -path.stat().st_mtime
        except OSError:
            mtime_rank = 0.0
        return (split_rank, local_rank, mtime_rank, str(path))

    return sorted(valid, key=priority)[0]


def _completed_metric_candidates(root: Path, patterns: list[str]) -> list[Path]:
    if not root.exists():
        return []

    if root.name == "eval_artifacts":
        return [path for pattern in patterns for path in root.glob(pattern)]

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(root.glob(f"**/eval_artifacts/{pattern}"))
    return candidates


def _is_matching_completed_metric(path: Path, config: TrainingConfig) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if payload.get("run_tag") != config.run_tag:
        return False

    payload_dataset = payload.get("dataset")
    if payload_dataset is not None and str(payload_dataset).lower() != config.dataset_name:
        return False

    if payload.get("split") not in {"test", "val"}:
        return False

    return isinstance(payload.get("loss_metrics"), dict) and isinstance(
        payload.get("distribution_metrics"),
        dict,
    )


def materialize_reused_metrics(config: TrainingConfig, eval_metrics_path: Path) -> list[Path]:
    copied = [_copy_metrics_record(eval_metrics_path, Path(config.eval_artifact_dir).expanduser())]
    rollout_metrics_path = _matching_completed_rollout_metrics_path(config, eval_metrics_path)
    if rollout_metrics_path is not None:
        copied.append(_copy_metrics_record(rollout_metrics_path, Path(config.rollout_artifact_dir).expanduser()))
    return copied


def _copy_metrics_record(source: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    source_resolved = _safe_resolve(source)
    target_resolved = _safe_resolve(target)
    if source_resolved is not None and target_resolved == source_resolved:
        return target
    shutil.copy2(source, target)
    return target


def _matching_completed_rollout_metrics_path(config: TrainingConfig, eval_metrics_path: Path) -> Optional[Path]:
    try:
        eval_payload = json.loads(eval_metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    split = str(eval_payload.get("split", "test"))
    patterns = [f"{config.run_tag}_{split}_rollout.metrics.json"]
    if split != "test":
        patterns.append(f"{config.run_tag}_test_rollout.metrics.json")
    if split != "val":
        patterns.append(f"{config.run_tag}_val_rollout.metrics.json")

    search_roots = [eval_metrics_path.parent.parent / "rollout_artifacts"]
    if config.completed_search_root:
        search_roots.append(Path(config.completed_search_root).expanduser())

    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        for path in _completed_rollout_metric_candidates(root, patterns):
            resolved = _safe_resolve(path)
            if resolved is None or resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(path)

    valid = [path for path in candidates if _is_matching_completed_rollout_metric(path, config)]
    if not valid:
        return None

    def priority(path: Path) -> tuple[int, float, str]:
        split_rank = 0 if f"_{split}_rollout.metrics.json" in path.name else 1
        try:
            mtime_rank = -path.stat().st_mtime
        except OSError:
            mtime_rank = 0.0
        return (split_rank, mtime_rank, str(path))

    return sorted(valid, key=priority)[0]


def _completed_rollout_metric_candidates(root: Path, patterns: list[str]) -> list[Path]:
    if not root.exists():
        return []

    if root.name == "rollout_artifacts":
        return [path for pattern in patterns for path in root.glob(pattern)]

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(root.glob(f"**/rollout_artifacts/{pattern}"))
    return candidates


def _is_matching_completed_rollout_metric(path: Path, config: TrainingConfig) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if payload.get("run_tag") != config.run_tag:
        return False
    if payload.get("split") not in {"test", "val"}:
        return False
    return isinstance(payload.get("distribution_metrics"), dict)


def _safe_resolve(path: Path) -> Optional[Path]:
    try:
        return path.resolve()
    except OSError:
        return None


def main() -> None:
    device, rank, world_size, local_rank, distributed = setup_distributed()
    config = TrainingConfig.from_env()

    try:
        if rank == 0:
            print(
                f"[dist] world_size={world_size}, rank={rank}, local_rank={local_rank}, "
                f"device={device}, dataset={config.dataset_name}, model={config.model_name}"
            )

        existing_metrics = completed_eval_metrics_path(config) if rank == 0 else None
        if distributed and world_size > 1:
            skip_payload = [str(existing_metrics) if existing_metrics is not None else ""]
            dist.broadcast_object_list(skip_payload, src=0)
            existing_metrics = Path(skip_payload[0]) if skip_payload[0] else None
        if existing_metrics is not None:
            if rank == 0:
                reused_records = materialize_reused_metrics(config, existing_metrics)
                print(
                    f"[skip_completed][{config.run_tag}] "
                    f"found completed identical setting: {existing_metrics}"
                )
                for metrics_path in reused_records:
                    print(f"[skip_completed][{config.run_tag}] reused metrics record: {metrics_path}")
                print(f"[skip_completed][{config.run_tag}] exiting before task/model setup")
            if distributed and world_size > 1:
                dist.barrier()
            return

        torch.manual_seed(config.seed + rank)

        task = build_task(config.dataset_name)
        task.prepare(rank=rank, world_size=world_size)
        max_block_size = task.max_block_size()
        if max_block_size is not None and config.block_size > max_block_size:
            if rank == 0:
                print(
                    f"[task] reducing block_size from {config.block_size} to {max_block_size} "
                    f"to fit available sequence lengths"
                )
            config = replace(config, block_size=max_block_size)
        if rank == 0:
            log_dataset_summary(task)

        model_build = build_model(config, task)

        if rank == 0:
            print(f"[model] selected={model_build.name}")
            print(f"[model] run_label={model_build.run_label}")
            print(f"[model] config={model_build.config}")

        base_model = model_build.model.to(device)
        model_build.freeze_for_training()

        trainable_params = [param for param in base_model.parameters() if param.requires_grad]
        requires_training = bool(model_build.requires_training) and bool(trainable_params)

        if rank == 0:
            n_trainable = sum(param.numel() for param in trainable_params)
            n_total = sum(param.numel() for param in base_model.parameters())
            print(f"[model] trainable params={n_trainable/1e6:.3f}M / total={n_total/1e6:.3f}M")

        model = base_model
        use_amp = device.type == "cuda" and config.use_amp and torch.cuda.is_bf16_supported()
        eval_splits = tuple(split for split in ("train", "val") if split in task.available_splits())
        checkpoint_path = Path(config.checkpoint_dir).expanduser().resolve() / f"{config.run_tag}.best.pt"
        best_val_main = float("inf")
        best_val_step: Optional[int] = None
        checkpoint_kind = "fitted"
        external_checkpoint_raw = os.environ.get("LOAD_CHECKPOINT_PATH", "").strip()
        eval_only_from_checkpoint = os.environ.get("EVAL_ONLY_FROM_CHECKPOINT", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        loaded_external_checkpoint = False

        if external_checkpoint_raw:
            external_checkpoint_path = Path(external_checkpoint_raw).expanduser().resolve()
            if not external_checkpoint_path.exists():
                raise FileNotFoundError(f"LOAD_CHECKPOINT_PATH does not exist: {external_checkpoint_path}")

            checkpoint_payload = torch.load(external_checkpoint_path, map_location=device)
            model_state_dict = checkpoint_payload.get("model_state_dict")
            if not isinstance(model_state_dict, dict):
                raise KeyError(
                    f"checkpoint at {external_checkpoint_path} does not contain a model_state_dict payload"
                )

            base_model.load_state_dict(model_state_dict)
            loaded_external_checkpoint = True
            checkpoint_kind = "external_checkpoint"
            if rank == 0:
                checkpoint_step = checkpoint_payload.get("step", "n/a")
                print(
                    f"[checkpoint][{model_build.run_label}][{task.name}] "
                    f"loaded external checkpoint from {external_checkpoint_path} (step {checkpoint_step})"
                )

            if eval_only_from_checkpoint:
                requires_training = False

        if requires_training:
            compiled_model = base_model
            if config.use_compile:
                try:
                    compiled_model = torch.compile(base_model)
                    if rank == 0:
                        print("[model] using torch.compile")
                except Exception as exc:
                    if rank == 0:
                        print(f"[warn] torch.compile failed: {exc}, falling back to eager")
            elif rank == 0:
                print("[model] using eager mode")

            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.95),
            )

            if distributed and world_size > 1:
                if device.type == "cuda":
                    model = DDP(
                        compiled_model,
                        device_ids=[device.index],
                        output_device=device.index,
                    )
                else:
                    model = DDP(compiled_model)
            else:
                model = compiled_model

            is_ddp = isinstance(model, DDP)

            for step in range(1, config.max_iters + 1):
                lr = get_lr(step, config)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                if step % config.eval_interval == 1:
                    metrics = estimate_metrics(
                        model,
                        task,
                        config,
                        device,
                        rank,
                        world_size,
                        use_amp,
                        splits=eval_splits,
                    )
                    if "val" in metrics and metrics["val"]["main_loss"] < best_val_main:
                        best_val_main = metrics["val"]["main_loss"]
                        best_val_step = step
                        if config.save_best_checkpoint and rank == 0:
                            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                            torch.save(
                                {
                                    "run_tag": config.run_tag,
                                    "step": step,
                                    "model_name": config.model_name,
                                    "model_run_label": model_build.run_label,
                                    "dataset_name": task.name,
                                    "val_metrics": metrics["val"],
                                    "config": asdict(config),
                                    "task_summary": task.dataset_summary(),
                                    "model_state_dict": cpu_state_dict(base_model),
                                },
                                checkpoint_path,
                            )
                            print(
                                f"[checkpoint][{model_build.run_label}][{task.name}] "
                                f"saved best val-main checkpoint at step {step} to {checkpoint_path}"
                            )
                    if rank == 0:
                        rendered_splits = " | ".join(
                            f"{split} {format_metric_line(task, metrics[split])}" for split in eval_splits
                        )
                        print(
                            f"[eval][{model_build.run_label}][{task.name}] "
                            f"iter {step:5d} | {rendered_splits} | lr {lr:.6e}"
                        )
                        preview_model = model.module if isinstance(model, DDP) else model
                        preview = task.render_preview(
                            preview_model,
                            device,
                            block_size=config.block_size,
                            use_amp=use_amp,
                            temperature=config.sample_temperature,
                            max_new_tokens=config.sample_new_tokens,
                            prompt=config.sample_prompt,
                        )
                        if preview:
                            print(preview)

                optimizer.zero_grad(set_to_none=True)
                train_sums = {key: 0.0 for key in LOSS_KEYS}

                for micro_step in range(config.grad_accum_steps):
                    batch = task.sample_batch(
                        "train",
                        config.batch_size,
                        config.block_size,
                        device,
                        rank=rank,
                        world_size=world_size,
                    )

                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=use_amp,
                    ):
                        batch_out = model(
                            batch.inputs,
                            targets=batch.targets,
                            loss_mask=batch.loss_mask,
                            return_dict=True,
                        )
                        raw_loss = batch_out["loss"]

                    for key in LOSS_KEYS:
                        train_sums[key] += float(batch_out[key].detach().float().item())

                    scaled_loss = raw_loss / config.grad_accum_steps
                    if is_ddp and config.grad_accum_steps > 1 and micro_step < config.grad_accum_steps - 1:
                        with model.no_sync():
                            scaled_loss.backward()
                    else:
                        scaled_loss.backward()

                if config.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
                optimizer.step()

                if rank == 0 and step % config.log_interval == 0:
                    avg_metrics = {
                        key: value / config.grad_accum_steps for key, value in train_sums.items()
                    }
                    print(
                        f"[train][{model_build.run_label}][{task.name}] iter {step:5d} | "
                        f"{format_metric_line(task, avg_metrics)} | lr {lr:.6e}"
                    )

            checkpoint_kind = "final"
            if config.save_best_checkpoint:
                if distributed and world_size > 1:
                    dist.barrier()
                if checkpoint_path.exists():
                    checkpoint_payload = torch.load(checkpoint_path, map_location=device)
                    base_model.load_state_dict(checkpoint_payload["model_state_dict"])
                    checkpoint_kind = "best_val_main"
                    if rank == 0:
                        step = checkpoint_payload.get("step")
                        val_main = checkpoint_payload.get("val_metrics", {}).get("main_loss")
                        val_main_text = f"{float(val_main):.4f}" if val_main is not None else "n/a"
                        print(
                            f"[checkpoint][{model_build.run_label}][{task.name}] "
                            f"loaded best val-main checkpoint from step {step} (val main={val_main_text})"
                        )
                elif rank == 0 and best_val_step is not None:
                    print(
                        f"[warn] best checkpoint was tracked in memory at step {best_val_step} "
                        f"but no file was found at {checkpoint_path}"
                    )
        else:
            if rank == 0:
                if loaded_external_checkpoint and eval_only_from_checkpoint:
                    reason = "loaded external checkpoint for evaluation-only"
                else:
                    reason = (
                        "baseline is pre-fit on the train split"
                        if not model_build.requires_training
                        else "model exposes no trainable parameters"
                    )
                print(f"[model] skipping SGD loop: {reason}")
            if config.save_best_checkpoint and rank == 0 and not loaded_external_checkpoint:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "run_tag": config.run_tag,
                        "step": 0,
                        "model_name": config.model_name,
                        "model_run_label": model_build.run_label,
                        "dataset_name": task.name,
                        "config": asdict(config),
                        "task_summary": task.dataset_summary(),
                        "model_state_dict": cpu_state_dict(base_model),
                    },
                    checkpoint_path,
                )
                print(
                    f"[checkpoint][{model_build.run_label}][{task.name}] "
                    f"saved fitted-model checkpoint to {checkpoint_path}"
                )

        if rank == 0:
            preview_model = model.module if isinstance(model, DDP) else model
            preview = task.render_preview(
                preview_model,
                device,
                block_size=config.block_size,
                use_amp=use_amp,
                temperature=config.sample_temperature,
                max_new_tokens=config.sample_new_tokens,
                prompt=config.sample_prompt,
            )
            if preview:
                print("\n[preview]\n")
                print(preview)

        if "test" in task.available_splits():
            test_metrics = estimate_metrics(
                model,
                task,
                config,
                device,
                rank,
                world_size,
                use_amp,
                splits=("test",),
            )
            if rank == 0:
                print(f"[test][{model_build.run_label}][{task.name}] {format_metric_line(task, test_metrics['test'])}")

        offline_split = "test" if "test" in task.available_splits() else ("val" if "val" in task.available_splits() else None)
        if rank == 0 and config.run_offline_eval and offline_split is not None:
            try:
                loss_metrics, distribution_metrics, artifact = evaluate_probability_split(
                    base_model,
                    task,
                    split=offline_split,
                    device=device,
                    use_amp=use_amp,
                    batch_size=config.offline_eval_batch_size,
                    include_artifact=config.save_eval_artifact,
                    run_tag=config.run_tag,
                    model_label=model_build.run_label,
                    checkpoint_kind=checkpoint_kind,
                )
                print(
                    f"[offline_eval][{model_build.run_label}][{task.name}][{offline_split}] "
                    f"{format_offline_metric_line(loss_metrics, distribution_metrics)}"
                )
                if artifact is not None:
                    artifact_path, metrics_path = save_evaluation_artifact(
                        artifact,
                        artifact_dir=Path(config.eval_artifact_dir).expanduser().resolve(),
                        run_tag=config.run_tag,
                        split=offline_split,
                    )
                    print(
                        f"[offline_eval][{model_build.run_label}][{task.name}] "
                        f"saved artifact={artifact_path} metrics={metrics_path}"
                    )
            except (NotImplementedError, ValueError, FileNotFoundError) as exc:
                print(f"[offline_eval][{model_build.run_label}][{task.name}] skipped: {exc}")

        rollout_split = offline_split
        if rank == 0 and config.run_rollout_eval and rollout_split is not None:
            rollout_context_steps, rollout_horizon_steps = resolved_rollout_steps(config)
            try:
                rollout_metrics, rollout_curves, rollout_artifact = evaluate_rollout_examples(
                    base_model,
                    task,
                    split=rollout_split,
                    device=device,
                    use_amp=use_amp,
                    batch_size=config.rollout_eval_batch_size,
                    include_artifact=config.save_rollout_artifact,
                    run_tag=config.run_tag,
                    model_label=model_build.run_label,
                    checkpoint_kind=checkpoint_kind,
                    max_examples=config.rollout_max_examples,
                    context_steps=rollout_context_steps,
                    horizon_steps=rollout_horizon_steps,
                    max_context_steps=config.model_max_seq_len,
                )
                print(
                    f"[rollout_eval][{model_build.run_label}][{task.name}][{rollout_split}] "
                    f"{format_rollout_metric_line(rollout_metrics, rollout_curves)}"
                )
                if rollout_artifact is not None:
                    artifact_path, metrics_path = save_rollout_artifact(
                        rollout_artifact,
                        artifact_dir=Path(config.rollout_artifact_dir).expanduser().resolve(),
                        run_tag=config.run_tag,
                        split=rollout_split,
                    )
                    print(
                        f"[rollout_eval][{model_build.run_label}][{task.name}] "
                        f"saved artifact={artifact_path} metrics={metrics_path}"
                    )
            except (NotImplementedError, ValueError, FileNotFoundError) as exc:
                print(
                    f"[rollout_eval][{model_build.run_label}][{task.name}][{rollout_split}] "
                    f"skipped: {exc}"
                )
    finally:
        cleanup_distributed()
