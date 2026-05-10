from __future__ import annotations

import csv
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .base import ModelFitSequence, RolloutExample, SequenceBatch, SequenceTask
from .tensor_dir import TensorSplit


class QueueNumInSysTask(SequenceTask):
    name = "queue_num_in_sys"

    def __init__(self, dataset_dir: Optional[str] = None):
        root_dir = Path(__file__).resolve().parents[1]
        default_dir = root_dir / "benchmarks" / "queue_num_in_sys" / "processed"
        self.dataset_dir = Path(dataset_dir or default_dir).expanduser().resolve()
        self.metadata: Dict[str, object] = {}
        self.prob_dim = -1
        self._split_rows: Dict[str, List[Dict[str, object]]] = {}
        self._eligible_cache: Dict[Tuple[str, int, int, int], Tuple[List[Dict[str, object]], torch.Tensor]] = {}
        self._time_label_cache: Dict[str, List[str]] = {}
        self._offline_payload_cache: Dict[Tuple[str, int, int], TensorSplit] = {}

    def prepare(self, *, rank: int = 0, world_size: int = 1) -> None:
        del world_size
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"dataset_dir does not exist: {self.dataset_dir}\n"
                "Run benchmarks/queue_num_in_sys/prepare.py first, or set QUEUE_NUM_IN_SYS_DIR / DATASET_DIR."
            )

        metadata_path = self.dataset_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"missing metadata.json in {self.dataset_dir}\n"
                "Run benchmarks/queue_num_in_sys/prepare.py first."
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.prob_dim = int(self.metadata["prob_dim"])

        available = []
        for split in ("train", "val", "test"):
            split_path = self.dataset_dir / f"{split}_sequences.csv"
            if split_path.exists():
                rows = self._read_manifest(split_path)
                if rows:
                    self._split_rows[split] = rows
                    available.append(split)

        if "train" not in self._split_rows or "val" not in self._split_rows:
            raise FileNotFoundError(
                f"{self.dataset_dir} must contain at least train_sequences.csv and val_sequences.csv"
            )

        if rank == 0:
            summary = ", ".join(f"{name}={len(rows)} systems" for name, rows in self._split_rows.items())
            print(f"[data] task={self.name} dir={self.dataset_dir}")
            print(f"[data] loaded queue manifests: {summary}")

    def available_splits(self) -> tuple[str, ...]:
        return tuple(name for name in ("train", "val", "test") if name in self._split_rows)

    def max_block_size(self) -> Optional[int]:
        if not self._split_rows:
            return None
        min_len = min(
            int(row["sequence_length"])
            for rows in self._split_rows.values()
            for row in rows
        )
        return max(1, min_len - 1)

    def dataset_summary(self) -> Dict[str, object]:
        out = super().dataset_summary()
        out["dataset_dir"] = str(self.dataset_dir)
        out["streamed_from_memmaps"] = True
        for key in ("source_variant", "split_method", "prob_dim", "max_support_width"):
            if key in self.metadata:
                out[key] = self.metadata[key]
        split_counts = self.metadata.get("split_counts")
        if isinstance(split_counts, dict):
            out.update({f"{name}_systems": value for name, value in split_counts.items()})
        return out

    def ordered_probability_support(self) -> Optional[torch.Tensor]:
        return torch.arange(self.prob_dim, dtype=torch.float32)

    def get_split_payload(self, split: str) -> TensorSplit:
        rows = self._split_rows.get(split)
        if rows is None:
            raise ValueError(f"split '{split}' is not available for task={self.name}")

        max_sequences = int(os.environ.get("QUEUE_OFFLINE_NUM_SEQUENCES", "32"))
        if max_sequences < 1:
            raise ValueError("QUEUE_OFFLINE_NUM_SEQUENCES must be >= 1")

        requested_window = int(
            os.environ.get(
                "QUEUE_OFFLINE_WINDOW_LENGTH",
                os.environ.get("MODEL_MAX_SEQ_LEN", os.environ.get("BLOCK_SIZE", "128")),
            )
        )
        if requested_window < 1:
            raise ValueError("QUEUE_OFFLINE_WINDOW_LENGTH must be >= 1")

        cache_key = (split, max_sequences, requested_window)
        cached = self._offline_payload_cache.get(cache_key)
        if cached is not None:
            return cached

        ordered = sorted(
            rows,
            key=lambda row: (
                int(row["support_width"]),
                int(row["sequence_length"]),
                str(row["source"]),
                int(row["dist_id"]),
            ),
        )
        selected = self._select_evenly_spaced_rows(ordered, max_examples=max_sequences)

        window_payloads: List[torch.Tensor] = []
        lengths: List[int] = []
        sequence_labels: List[str] = []
        time_labels: List[List[str]] = []

        for row in selected:
            sequence = self._load_memmap(str(self.dataset_dir / row["npy_relpath"]))
            seq_len = int(row["sequence_length"])
            if seq_len < 2:
                continue

            target_len = min(requested_window, seq_len - 1)
            input_len = target_len + 1
            start = max(0, seq_len - input_len)
            window = np.asarray(sequence[start : start + input_len], dtype=np.float32)

            padded = np.zeros((window.shape[0], self.prob_dim), dtype=np.float32)
            padded[:, : window.shape[1]] = window
            window_payloads.append(torch.from_numpy(padded))
            lengths.append(int(window.shape[0]))
            sequence_labels.append(self._sequence_label_for_row(row))
            full_time_labels = self._time_labels_for_row(row)
            time_labels.append(full_time_labels[start + 1 : start + input_len])

        if not window_payloads:
            raise ValueError(f"split={split} has no queue systems long enough for offline evaluation")

        max_input_len = max(lengths)
        inputs = torch.zeros((len(window_payloads), max_input_len, self.prob_dim), dtype=torch.float32)
        for idx, window in enumerate(window_payloads):
            inputs[idx, : window.size(0)] = window

        payload = TensorSplit(
            inputs=inputs,
            targets=None,
            loss_mask=None,
            lengths=torch.tensor(lengths, dtype=torch.long),
            sequence_labels=sequence_labels,
            time_labels=time_labels,
            token_labels=[f"n={idx}" for idx in range(self.prob_dim)],
        )
        self._offline_payload_cache[cache_key] = payload
        return payload

    def get_rollout_examples(
        self,
        split: str,
        *,
        max_examples: int,
        context_steps: int,
        horizon_steps: int,
    ) -> List[RolloutExample]:
        if context_steps < 1:
            raise ValueError("context_steps must be >= 1")
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")

        rows = self._split_rows.get(split)
        if rows is None:
            raise ValueError(f"split '{split}' is not available for task={self.name}")

        eligible = [
            row
            for row in rows
            if int(row["sequence_length"]) >= context_steps + horizon_steps
        ]
        if not eligible:
            raise ValueError(
                f"split={split} has no queue systems eligible for rollout with "
                f"context_steps={context_steps} and horizon_steps={horizon_steps}"
            )

        ordered = sorted(
            eligible,
            key=lambda row: (
                int(row["support_width"]),
                int(row["sequence_length"]),
                str(row["source"]),
                int(row["dist_id"]),
            ),
        )
        selected = self._select_evenly_spaced_rows(ordered, max_examples=max_examples)

        examples: List[RolloutExample] = []
        for row in selected:
            sequence = self._load_memmap(str(self.dataset_dir / row["npy_relpath"]))
            seq_len = int(row["sequence_length"])
            target_start = seq_len - horizon_steps
            context_end = target_start
            context_start = context_end - context_steps

            context = self._pad_distribution_window(sequence[context_start:context_end])
            targets = self._pad_distribution_window(sequence[target_start : target_start + horizon_steps])
            time_labels = self._time_labels_for_row(row)[target_start : target_start + horizon_steps]

            examples.append(
                RolloutExample(
                    context=context,
                    targets=targets,
                    sequence_label=self._sequence_label_for_row(row),
                    time_labels=time_labels,
                )
            )

        return examples

    def iter_model_fit_sequences(
        self,
        split: str,
        *,
        model_name: str | None = None,
    ):
        normalized_model = None if model_name is None else model_name.strip().lower()
        if normalized_model not in {
            None,
            "bdarma",
            "b-darma",
            "bdarma_map",
            "bdarma-map",
            "bdarma_approx",
            "ilr_var",
            "ilr-var",
            "var_ilr",
            "analog_successor",
            "analog-successor",
            "analog",
            "knn_successor",
            "knn-successor",
        }:
            yield from super().iter_model_fit_sequences(split, model_name=model_name)
            return

        env_prefix = "ILR_VAR"
        fallback_env_prefix = None
        if normalized_model in {"analog_successor", "analog-successor", "analog", "knn_successor", "knn-successor"}:
            env_prefix = "ANALOG"
        elif normalized_model in {"bdarma_map", "bdarma-map", "bdarma_approx"}:
            env_prefix = "BDARMA_MAP"
            fallback_env_prefix = "BDARMA"
        elif normalized_model in {"bdarma", "b-darma"}:
            env_prefix = "BDARMA"

        rows = self._split_rows.get(split)
        if rows is None:
            raise ValueError(f"split '{split}' is not available for task={self.name}")

        max_sequences_raw = os.environ.get(f"{env_prefix}_FIT_MAX_SEQUENCES")
        if (max_sequences_raw is None or not max_sequences_raw.strip()) and fallback_env_prefix is not None:
            max_sequences_raw = os.environ.get(f"{fallback_env_prefix}_FIT_MAX_SEQUENCES")
        max_sequences = None
        if max_sequences_raw is not None and max_sequences_raw.strip():
            max_sequences = int(max_sequences_raw)
            if max_sequences < 1:
                raise ValueError(f"{env_prefix}_FIT_MAX_SEQUENCES must be >= 1")

        fit_window_raw = os.environ.get(f"{env_prefix}_FIT_WINDOW_LENGTH")
        if (fit_window_raw is None or not fit_window_raw.strip()) and fallback_env_prefix is not None:
            fit_window_raw = os.environ.get(f"{fallback_env_prefix}_FIT_WINDOW_LENGTH")
        fit_window = None
        if fit_window_raw is not None and fit_window_raw.strip():
            fit_window = int(fit_window_raw)
            if fit_window < 2:
                raise ValueError(f"{env_prefix}_FIT_WINDOW_LENGTH must be >= 2")

        ordered = sorted(
            rows,
            key=lambda row: (
                int(row["support_width"]),
                int(row["sequence_length"]),
                str(row["source"]),
                int(row["dist_id"]),
            ),
        )
        selected = (
            self._select_evenly_spaced_rows(ordered, max_examples=max_sequences)
            if max_sequences is not None
            else ordered
        )

        for row in selected:
            sequence = self._load_memmap(str(self.dataset_dir / row["npy_relpath"]))
            seq_len = int(row["sequence_length"])
            if seq_len < 2:
                continue

            start = 0
            if fit_window is not None and seq_len > fit_window:
                start = seq_len - fit_window

            yield ModelFitSequence(
                inputs=self._pad_distribution_window(sequence[start:seq_len]),
                targets=None,
                loss_mask=None,
            )

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
        rows, weights = self._eligible_rows(split, block_size, rank=rank, world_size=world_size)
        sampled = torch.multinomial(weights, num_samples=batch_size, replacement=True).tolist()

        inputs_list = []
        targets_list = []
        for row_idx in sampled:
            row = rows[row_idx]
            sequence = self._load_memmap(str(self.dataset_dir / row["npy_relpath"]))
            seq_len = int(row["sequence_length"])
            max_start = seq_len - (block_size + 1)
            start = 0 if max_start <= 0 else int(torch.randint(0, max_start + 1, (1,)).item())
            window = np.asarray(sequence[start : start + block_size + 1], dtype=np.float32)

            padded = np.zeros((window.shape[0], self.prob_dim), dtype=np.float32)
            padded[:, : window.shape[1]] = window

            inputs_list.append(torch.from_numpy(padded[:-1]))
            targets_list.append(torch.from_numpy(padded[1:]))

        batch_inputs = torch.stack(inputs_list, dim=0).to(device)
        batch_targets = torch.stack(targets_list, dim=0).to(device)
        return SequenceBatch(inputs=batch_inputs, targets=batch_targets, loss_mask=None)

    def _eligible_rows(
        self,
        split: str,
        block_size: int,
        *,
        rank: int,
        world_size: int,
    ) -> Tuple[List[Dict[str, object]], torch.Tensor]:
        key = (split, block_size, rank, world_size)
        cached = self._eligible_cache.get(key)
        if cached is not None:
            return cached

        rows = self._split_rows.get(split)
        if rows is None:
            raise ValueError(f"split '{split}' is not available for task={self.name}")

        eligible = [row for row in rows if int(row["sequence_length"]) >= block_size + 1]
        if world_size > 1 and len(eligible) >= world_size:
            sharded = eligible[rank::world_size]
            if sharded:
                eligible = sharded

        if not eligible:
            raise ValueError(
                f"no queue systems in split={split} are long enough for block_size={block_size}"
            )

        weights = torch.tensor(
            [max(1, int(row["sequence_length"]) - block_size) for row in eligible],
            dtype=torch.float32,
        )
        weights = weights / weights.sum()
        cached = (eligible, weights)
        self._eligible_cache[key] = cached
        return cached

    @staticmethod
    @lru_cache(maxsize=128)
    def _load_memmap(path: str):
        return np.load(path, mmap_mode="r")

    def _pad_distribution_window(self, window: np.ndarray) -> torch.Tensor:
        padded = np.zeros((window.shape[0], self.prob_dim), dtype=np.float32)
        padded[:, : window.shape[1]] = np.asarray(window, dtype=np.float32)
        return torch.from_numpy(padded)

    @staticmethod
    def _select_evenly_spaced_rows(rows: Sequence[Dict[str, object]], *, max_examples: int) -> List[Dict[str, object]]:
        if max_examples <= 0 or len(rows) <= max_examples:
            return list(rows)
        positions = np.linspace(0, len(rows) - 1, num=max_examples, dtype=int)
        unique_positions = sorted(set(int(idx) for idx in positions.tolist()))
        return [rows[idx] for idx in unique_positions]

    @staticmethod
    def _sequence_label_for_row(row: Dict[str, object]) -> str:
        return (
            f"{row['source']} | dist={int(row['dist_id'])} | "
            f"T={int(row['sequence_length'])} | D={int(row['support_width'])}"
        )

    def _time_labels_for_row(self, row: Dict[str, object]) -> List[str]:
        cache_key = str(row.get("time_relpath") or row.get("raw_path") or row.get("raw_relpath") or row["dist_id"])
        cached = self._time_label_cache.get(cache_key)
        if cached is not None:
            return cached

        labels = self._load_time_labels_from_sidecar(row)
        if labels is None:
            labels = self._load_time_labels_from_raw(row)
        if labels is None:
            labels = self._synthesized_time_labels(row)

        self._time_label_cache[cache_key] = labels
        return labels

    def _load_time_labels_from_sidecar(self, row: Dict[str, object]) -> Optional[List[str]]:
        time_relpath = row.get("time_relpath")
        if not time_relpath:
            return None
        time_path = self.dataset_dir / str(time_relpath)
        if not time_path.exists():
            return None
        values = np.load(time_path, mmap_mode="r")
        return [self._format_time_value(value, idx) for idx, value in enumerate(values)]

    @staticmethod
    def _load_time_labels_from_raw(row: Dict[str, object]) -> Optional[List[str]]:
        raw_path = row.get("raw_path")
        if not raw_path:
            return None
        path = Path(str(raw_path))
        if not path.exists():
            return None

        labels: List[str] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, item in enumerate(reader):
                labels.append(QueueNumInSysTask._format_time_value(item.get("time"), idx))
        return labels

    @staticmethod
    def _synthesized_time_labels(row: Dict[str, object]) -> List[str]:
        sequence_length = int(row["sequence_length"])
        if sequence_length <= 0:
            return []

        time_start = float(row.get("time_start", 0.0))
        time_step = float(row.get("time_step_mean", 1.0))
        return [
            QueueNumInSysTask._format_time_value(time_start + time_step * idx, idx)
            for idx in range(sequence_length)
        ]

    @staticmethod
    def _format_time_value(raw_value, idx: int) -> str:
        if raw_value is None or raw_value == "":
            return f"t{idx + 1}"
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            return str(raw_value)
        if abs(numeric - round(numeric)) < 1e-9:
            return str(int(round(numeric)))
        return f"{numeric:g}"

    @staticmethod
    def _read_manifest(path: Path) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                casted = dict(row)
                casted["sequence_length"] = int(row["sequence_length"])
                casted["support_width"] = int(row["support_width"])
                casted["dist_id"] = int(row["dist_id"])
                rows.append(casted)
        return rows


def default_queue_num_in_sys_dir() -> str:
    return os.fspath(Path(__file__).resolve().parents[1] / "benchmarks" / "queue_num_in_sys" / "processed")
