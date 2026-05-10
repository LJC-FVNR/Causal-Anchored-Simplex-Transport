from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn


@dataclass
class SequenceBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    loss_mask: Optional[torch.Tensor] = None


@dataclass
class RolloutExample:
    context: torch.Tensor
    targets: torch.Tensor
    sequence_label: str
    time_labels: list[str]


@dataclass
class ModelFitSequence:
    inputs: torch.Tensor
    targets: Optional[torch.Tensor] = None
    loss_mask: Optional[torch.Tensor] = None


class SequenceTask(ABC):
    name: str
    prob_dim: int

    def prepare(self, *, rank: int = 0, world_size: int = 1) -> None:
        """Download or load any task assets needed before training."""

    def available_splits(self) -> tuple[str, ...]:
        return ("train", "val")

    def max_block_size(self) -> Optional[int]:
        return None

    @abstractmethod
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
        """Return one training/eval batch aligned for next-step prediction."""

    def dataset_summary(self) -> Dict[str, object]:
        return {"task": self.name, "prob_dim": self.prob_dim}

    def model_config_overrides(
        self,
        train_config: object | None = None,
        *,
        model_name: str | None = None,
    ) -> Dict[str, object]:
        return {}

    def extra_metric_items(self, main_loss: float) -> Dict[str, float]:
        return {}

    def get_split_payload(self, split: str):
        raise NotImplementedError(f"task={self.name} does not expose full split tensors for offline evaluation")

    def iter_model_fit_sequences(
        self,
        split: str,
        *,
        model_name: str | None = None,
    ) -> Iterator[ModelFitSequence]:
        del model_name
        split_payload = self.get_split_payload(split)
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
            yield ModelFitSequence(inputs=seq_inputs, targets=seq_targets, loss_mask=seq_mask)

    def get_rollout_examples(
        self,
        split: str,
        *,
        max_examples: int,
        context_steps: int,
        horizon_steps: int,
    ) -> list[RolloutExample]:
        raise NotImplementedError(f"task={self.name} does not expose rollout examples for multi-step evaluation")

    def ordered_probability_support(self) -> Optional[torch.Tensor]:
        return None

    def render_preview(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        block_size: int,
        use_amp: bool,
        temperature: float = 1.0,
        max_new_tokens: int = 0,
        prompt: Optional[str] = None,
    ) -> Optional[str]:
        return None
