from __future__ import annotations

from typing import List

import torch

from .base import SequenceBatch
from .tensor_dir import TensorDirectorySequenceTask


class MaskedWindowTensorDirectoryTask(TensorDirectorySequenceTask):
    """TensorDirectory task that avoids eval windows with an all-false loss mask."""

    def _valid_window_starts(self, loss_mask: torch.Tensor, start_max: int, block_size: int) -> List[int]:
        valid_starts = []
        for start in range(start_max + 1):
            window_mask = loss_mask[start + 1 : start + block_size + 1]
            if bool(window_mask.any()):
                valid_starts.append(start)
        return valid_starts

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

        inputs_list = []
        targets_list = []
        mask_list = []

        for _ in range(batch_size):
            seq_idx = int(eligible[torch.randint(0, eligible.numel(), (1,))].item())
            seq_len = int(split_data.lengths[seq_idx].item())
            max_start = seq_len - needs
            if max_start < 0:
                raise ValueError(f"sequence length {seq_len} is too short for block_size={block_size}")

            if split_data.targets is None and split_data.loss_mask is not None:
                valid_starts = self._valid_window_starts(
                    split_data.loss_mask[seq_idx],
                    start_max=max_start,
                    block_size=block_size,
                )
                if not valid_starts:
                    raise ValueError(
                        f"split={split} sequence index {seq_idx} has no masked targets inside a block of size {block_size}"
                    )
                start = valid_starts[int(torch.randint(0, len(valid_starts), (1,)).item())]
            else:
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
