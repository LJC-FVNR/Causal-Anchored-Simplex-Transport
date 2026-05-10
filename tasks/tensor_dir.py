from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

from evaluation.sidecars import build_artifact_labels

from .base import RolloutExample, SequenceBatch, SequenceTask


@dataclass
class TensorSplit:
    inputs: torch.Tensor
    targets: Optional[torch.Tensor]
    loss_mask: Optional[torch.Tensor]
    lengths: torch.Tensor
    sequence_labels: Optional[list[str]] = None
    time_labels: Optional[list[list[str]]] = None
    token_labels: Optional[list[str]] = None


class TensorDirectorySequenceTask(SequenceTask):
    name = "tensor_dir"

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self._splits: Dict[str, TensorSplit] = {}
        self.prob_dim = -1

    def _load_payload(self, path: Path) -> TensorSplit:
        payload = torch.load(path, map_location="cpu")
        if torch.is_tensor(payload):
            inputs = payload
            targets = None
            loss_mask = None
            lengths = None
        elif isinstance(payload, dict):
            inputs = payload.get("inputs", payload.get("sequences"))
            targets = payload.get("targets")
            loss_mask = payload.get("loss_mask", payload.get("mask"))
            lengths = payload.get("lengths")
        else:
            raise TypeError(f"{path} must contain a tensor or dict payload")

        if inputs is None:
            raise ValueError(f"{path} is missing an 'inputs' or 'sequences' tensor")
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)
        if inputs.ndim != 3:
            raise ValueError(f"{path} inputs must have shape (N,T,D), got {tuple(inputs.shape)}")
        if not torch.is_floating_point(inputs):
            raise TypeError(f"{path} inputs must be floating-point simplex vectors")

        inputs = inputs.float().contiguous()

        if targets is not None:
            if targets.ndim == 2:
                if targets.dtype not in (torch.int32, torch.int64, torch.long):
                    raise TypeError(f"{path} integer targets must be int tensors")
                if targets.shape[:2] != inputs.shape[:2]:
                    raise ValueError(
                        f"{path} integer targets must align with inputs on (N,T), got {tuple(targets.shape)}"
                    )
                targets = targets.long().contiguous()
            elif targets.ndim == 3:
                if targets.shape != inputs.shape:
                    raise ValueError(
                        f"{path} distribution targets must match inputs shape {tuple(inputs.shape)}, got {tuple(targets.shape)}"
                    )
                targets = targets.float().contiguous()
            else:
                raise ValueError(f"{path} targets must have shape (N,T) or (N,T,D), got {tuple(targets.shape)}")

        if loss_mask is not None:
            if loss_mask.ndim == 1:
                loss_mask = loss_mask.unsqueeze(0)
            if loss_mask.shape != inputs.shape[:2]:
                raise ValueError(
                    f"{path} loss_mask must have shape {tuple(inputs.shape[:2])}, got {tuple(loss_mask.shape)}"
                )
            loss_mask = loss_mask.to(torch.bool).contiguous()

        if lengths is None:
            if loss_mask is not None:
                lengths = loss_mask.long().sum(dim=-1)
            else:
                lengths = torch.full((inputs.size(0),), inputs.size(1), dtype=torch.long)
        else:
            if lengths.ndim == 0:
                lengths = lengths.unsqueeze(0)
            if lengths.ndim != 1 or lengths.size(0) != inputs.size(0):
                raise ValueError(
                    f"{path} lengths must have shape {(inputs.size(0),)}, got {tuple(lengths.shape)}"
                )
            lengths = lengths.long().contiguous()

        return TensorSplit(inputs=inputs, targets=targets, loss_mask=loss_mask, lengths=lengths)

    def prepare(self, *, rank: int = 0, world_size: int = 1) -> None:
        del world_size
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"dataset_dir does not exist: {self.dataset_dir}")

        required = ["train", "val"]
        optional = ["test"]
        for split in required + optional:
            path = self.dataset_dir / f"{split}.pt"
            if path.exists():
                self._splits[split] = self._load_payload(path)
            elif split in required:
                raise FileNotFoundError(
                    f"missing required split file: {path}. Expected at least train.pt and val.pt."
                )

        self.prob_dim = int(self._splits["train"].inputs.size(-1))
        for split_name, split in self._splits.items():
            if split.inputs.size(-1) != self.prob_dim:
                raise ValueError(
                    f"{split_name}.pt prob_dim mismatch: expected {self.prob_dim}, got {split.inputs.size(-1)}"
                )

        if rank == 0:
            summary = ", ".join(
                f"{name}={tuple(split.inputs.shape)}" for name, split in sorted(self._splits.items())
            )
            print(f"[data] task={self.name} dir={self.dataset_dir}")
            print(f"[data] loaded splits: {summary}")

    def dataset_summary(self) -> Dict[str, object]:
        out = super().dataset_summary()
        out["dataset_dir"] = str(self.dataset_dir)
        max_block_size = self.max_block_size()
        if max_block_size is not None:
            out["max_block_size"] = max_block_size
        for split_name, split in sorted(self._splits.items()):
            out[f"{split_name}_shape"] = tuple(split.inputs.shape)
        return out

    def available_splits(self) -> tuple[str, ...]:
        return tuple(name for name in ("train", "val", "test") if name in self._splits)

    def max_block_size(self) -> Optional[int]:
        if not self._splits:
            return None

        supported = []
        for split in self._splits.values():
            needs_offset = 1 if split.targets is None else 0
            split_max = int(split.lengths.max().item()) - needs_offset
            supported.append(split_max)
        return min(supported) if supported else None

    def _split(self, split: str) -> TensorSplit:
        try:
            return self._splits[split]
        except KeyError as exc:
            raise ValueError(f"split '{split}' is not available for task={self.name}") from exc

    def get_split_payload(self, split: str) -> TensorSplit:
        return self._split(split)

    def get_rollout_examples(
        self,
        split: str,
        *,
        max_examples: int,
        context_steps: int,
        horizon_steps: int,
    ) -> list[RolloutExample]:
        split_data = self._split(split)
        if context_steps < 1:
            raise ValueError("context_steps must be >= 1")
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")

        if split_data.targets is None:
            target_lengths = [max(int(length.item()) - 1, 0) for length in split_data.lengths]
        else:
            target_lengths = [int(length.item()) for length in split_data.lengths]

        labels = build_artifact_labels(
            dataset_dir=self.dataset_dir,
            num_sequences=split_data.inputs.size(0),
            prob_dim=self.prob_dim,
            target_lengths=target_lengths,
        )
        sequence_labels = labels["sequence_labels"]
        time_labels = labels["time_labels"]

        candidates = []
        for seq_idx in range(split_data.inputs.size(0)):
            seq_len = int(split_data.lengths[seq_idx].item())
            if split_data.targets is None:
                if seq_len < 2:
                    continue
                sequence = split_data.inputs[seq_idx, :seq_len].float().contiguous()
                aligned_targets = sequence[1:]
                aligned_mask = (
                    split_data.loss_mask[seq_idx, 1:seq_len].to(torch.bool).contiguous()
                    if split_data.loss_mask is not None
                    else None
                )
            else:
                sequence = split_data.inputs[seq_idx, :seq_len].float().contiguous()
                aligned_targets = split_data.targets[seq_idx, :seq_len].float().contiguous()
                aligned_mask = (
                    split_data.loss_mask[seq_idx, :seq_len].to(torch.bool).contiguous()
                    if split_data.loss_mask is not None
                    else None
                )

            rollout = self._build_rollout_example(
                sequence=sequence,
                aligned_targets=aligned_targets,
                aligned_mask=aligned_mask,
                sequence_label=sequence_labels[seq_idx],
                time_labels=[str(value) for value in time_labels[seq_idx]],
                context_steps=context_steps,
                horizon_steps=horizon_steps,
            )
            if rollout is None:
                continue
            available_horizon = int(rollout.targets.size(0))
            candidates.append((available_horizon, seq_len, rollout))

        if not candidates:
            raise ValueError(
                f"split={split} has no sequences eligible for rollout with context_steps={context_steps} "
                f"and horizon_steps={horizon_steps}"
            )

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in candidates[:max_examples]]

    @staticmethod
    def _build_rollout_example(
        *,
        sequence: torch.Tensor,
        aligned_targets: torch.Tensor,
        aligned_mask: Optional[torch.Tensor],
        sequence_label: str,
        time_labels: list[str],
        context_steps: int,
        horizon_steps: int,
    ) -> Optional[RolloutExample]:
        if aligned_targets.ndim != 2:
            raise ValueError("rollout currently requires distribution-valued aligned targets with shape (T,D)")

        if aligned_mask is not None:
            scored_idx = torch.nonzero(aligned_mask, as_tuple=False).squeeze(-1)
            if scored_idx.numel() == 0:
                return None
            start_target = int(scored_idx[0].item())
            horizon_available = int(aligned_mask[start_target:].long().cumprod(dim=0).sum().item())
            if horizon_available < horizon_steps:
                return None
            context_end = start_target + 1
            if context_end < context_steps:
                return None
            context_start = context_end - context_steps
            context = sequence[context_start:context_end]
            targets = aligned_targets[start_target : start_target + horizon_steps]
            target_time_labels = time_labels[start_target : start_target + horizon_steps]
        else:
            if sequence.size(0) < context_steps + horizon_steps:
                return None
            target_seq_start = sequence.size(0) - horizon_steps
            context_end = target_seq_start
            if context_end < context_steps:
                return None
            context_start = context_end - context_steps
            context = sequence[context_start:context_end]
            targets = sequence[target_seq_start : target_seq_start + horizon_steps]
            time_label_start = max(0, target_seq_start - 1)
            target_time_labels = time_labels[time_label_start : time_label_start + horizon_steps]
            if len(target_time_labels) < horizon_steps:
                target_time_labels = [f"t{idx}" for idx in range(1, horizon_steps + 1)]

        if context.size(0) < 1 or targets.size(0) != horizon_steps:
            return None

        return RolloutExample(
            context=context,
            targets=targets,
            sequence_label=sequence_label,
            time_labels=target_time_labels,
        )

    @staticmethod
    def _shard_indices(indices: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
        if world_size <= 1 or indices.numel() < world_size:
            return indices
        chunk = indices.numel() // world_size
        remainder = indices.numel() % world_size
        start = rank * chunk + min(rank, remainder)
        end = start + chunk + (1 if rank < remainder else 0)
        shard = indices[start:end]
        return shard if shard.numel() > 0 else indices

    def sample_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: torch.device,
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> SequenceBatch:
        split_data = self._split(split)
        needs = block_size + 1 if split_data.targets is None else block_size
        eligible = torch.nonzero(split_data.lengths >= needs, as_tuple=False).squeeze(-1)
        if eligible.numel() == 0:
            raise ValueError(
                f"no sequences in split={split} are long enough for block_size={block_size} (need {needs})"
            )
        eligible = self._shard_indices(eligible, rank=rank, world_size=world_size)

        chosen = eligible[torch.randint(0, eligible.numel(), (batch_size,))]
        inputs_list = []
        targets_list = []
        mask_list = []

        for seq_idx in chosen.tolist():
            seq_len = int(split_data.lengths[seq_idx].item())
            max_start = seq_len - needs
            start = 0 if max_start <= 0 else int(torch.randint(0, max_start + 1, (1,)).item())

            if split_data.targets is None:
                window = split_data.inputs[seq_idx, start : start + block_size + 1]
                inputs = window[:-1]
                targets = window[1:]
                mask = None
                if split_data.loss_mask is not None:
                    mask = split_data.loss_mask[seq_idx, start + 1 : start + block_size + 1]
            else:
                inputs = split_data.inputs[seq_idx, start : start + block_size]
                targets = split_data.targets[seq_idx, start : start + block_size]
                mask = None
                if split_data.loss_mask is not None:
                    mask = split_data.loss_mask[seq_idx, start : start + block_size]

            inputs_list.append(inputs)
            targets_list.append(targets)
            if mask is not None:
                mask_list.append(mask)

        batch_inputs = torch.stack(inputs_list, dim=0).to(device)
        batch_targets = torch.stack(targets_list, dim=0).to(device)
        batch_mask = torch.stack(mask_list, dim=0).to(device) if mask_list else None
        return SequenceBatch(inputs=batch_inputs, targets=batch_targets, loss_mask=batch_mask)
