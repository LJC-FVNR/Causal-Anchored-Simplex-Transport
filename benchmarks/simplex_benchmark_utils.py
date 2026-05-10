from __future__ import annotations

import csv
import json
import shutil
import urllib.request
from urllib.error import HTTPError, URLError
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch


def download_file(url: str, output_path: Path, *, force: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        print(f"[download] reuse {output_path}")
        return

    tmp_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    try:
        with urllib.request.urlopen(request) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except (HTTPError, URLError):
        if tmp_path.exists():
            tmp_path.unlink()
        try:
            import requests
        except ImportError:
            raise

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        }
        with requests.get(url, headers=headers, stream=True, timeout=60) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as handle:
                shutil.copyfileobj(response.raw, handle)
    tmp_path.replace(output_path)
    print(f"[download] wrote {output_path}")


def parse_float(value: object) -> Optional[float]:
    text = str(value).strip().replace(",", "")
    if text in {"", "NA", "NaN", "nan", ".", "(D)", "-"}:
        return None
    return float(text)


def month_to_index(month_key: str) -> int:
    year = int(month_key[:4])
    month = int(month_key[5:7])
    return year * 12 + month - 1


def index_to_month(index: int) -> str:
    year = index // 12
    month = index % 12 + 1
    return f"{year:04d}-{month:02d}"


def month_sequence(start_month: str, end_month: str) -> List[str]:
    start_idx = month_to_index(start_month)
    end_idx = month_to_index(end_month)
    return [index_to_month(idx) for idx in range(start_idx, end_idx + 1)]


def longest_contiguous_window(
    periods: Iterable[str],
    *,
    to_index: Callable[[str], int],
    from_index: Callable[[int], str],
) -> Tuple[str, str]:
    ordered_idx = sorted({to_index(period) for period in periods})
    if not ordered_idx:
        raise ValueError("cannot compute a contiguous window for an empty period set")

    best_start = best_end = ordered_idx[0]
    cur_start = cur_end = ordered_idx[0]
    for idx in ordered_idx[1:]:
        if idx == cur_end + 1:
            cur_end = idx
            continue
        if (cur_end - cur_start) > (best_end - best_start):
            best_start, best_end = cur_start, cur_end
        cur_start = cur_end = idx

    if (cur_end - cur_start) > (best_end - best_start):
        best_start, best_end = cur_start, cur_end
    return from_index(best_start), from_index(best_end)


def build_tail_masks(length: int, *, val_steps: int, test_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if val_steps <= 0 or test_steps <= 0:
        raise ValueError("val_steps and test_steps must both be positive")

    num_targets = length - 1
    if num_targets <= val_steps + test_steps:
        raise ValueError(
            f"sequence with {num_targets} targets is too short for val_steps={val_steps}, test_steps={test_steps}"
        )

    train_mask = torch.zeros(length, dtype=torch.bool)
    val_mask = torch.zeros(length, dtype=torch.bool)
    test_mask = torch.zeros(length, dtype=torch.bool)
    val_start = length - (val_steps + test_steps)
    test_start = length - test_steps
    train_mask[1:val_start] = True
    val_mask[val_start:test_start] = True
    test_mask[test_start:length] = True
    return train_mask, val_mask, test_mask


def build_fraction_masks(
    length: int,
    *,
    train_fraction: float,
    val_fraction: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if length < 4:
        raise ValueError(f"sequence length must be at least 4, got {length}")
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("expected train_fraction > 0, val_fraction > 0, train_fraction + val_fraction < 1")

    num_targets = length - 1
    train_count = max(1, int(num_targets * train_fraction))
    val_count = max(1, int(num_targets * val_fraction))
    test_count = num_targets - train_count - val_count
    if test_count < 1:
        shift = 1 - test_count
        reducible = min(shift, max(0, train_count - 1))
        train_count -= reducible
        shift -= reducible
        if shift:
            val_count -= min(shift, max(0, val_count - 1))
    test_count = num_targets - train_count - val_count
    if min(train_count, val_count, test_count) < 1:
        raise ValueError(f"invalid split for {num_targets} targets")

    train_mask = torch.zeros(length, dtype=torch.bool)
    val_mask = torch.zeros(length, dtype=torch.bool)
    test_mask = torch.zeros(length, dtype=torch.bool)
    train_end = 1 + train_count
    val_end = train_end + val_count
    train_mask[1:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:length] = True
    return train_mask, val_mask, test_mask


def infer_split_masks(
    length: int,
    *,
    split_mode: str,
    train_fraction: float,
    val_fraction: float,
    val_steps: int,
    test_steps: int,
    auto_tail_min_targets: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if split_mode == "tail":
        return (*build_tail_masks(length, val_steps=val_steps, test_steps=test_steps), "tail")
    if split_mode == "fraction":
        return (*build_fraction_masks(length, train_fraction=train_fraction, val_fraction=val_fraction), "fraction_70_15_15")
    if length - 1 >= auto_tail_min_targets:
        return (*build_tail_masks(length, val_steps=val_steps, test_steps=test_steps), "tail")
    return (*build_fraction_masks(length, train_fraction=train_fraction, val_fraction=val_fraction), "fraction_70_15_15")


def mask_range(mask: torch.Tensor, periods: Sequence[str]) -> Optional[dict[str, object]]:
    indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if indices.numel() == 0:
        return None
    start_idx = int(indices[0].item())
    end_idx = int(indices[-1].item())
    return {
        "start": periods[start_idx],
        "end": periods[end_idx],
        "num_target_steps": int(indices.numel()),
    }


def summarize_lengths(lengths: Sequence[int]) -> dict[str, float]:
    ordered = sorted(int(value) for value in lengths)
    n = len(ordered)
    return {
        "min": float(ordered[0]),
        "median": float(ordered[n // 2]),
        "mean": float(sum(ordered) / n),
        "max": float(ordered[-1]),
    }


def pad_sequences(sequences: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("no sequences to pad")
    prob_dim = int(sequences[0].size(-1))
    max_length = max(int(sequence.size(0)) for sequence in sequences)
    inputs = torch.zeros((len(sequences), max_length, prob_dim), dtype=torch.float32)
    lengths = torch.zeros(len(sequences), dtype=torch.long)
    for idx, sequence in enumerate(sequences):
        if sequence.size(-1) != prob_dim:
            raise ValueError("all sequences must have the same probability dimension")
        length = int(sequence.size(0))
        inputs[idx, :length] = sequence.float()
        lengths[idx] = length
    return inputs, lengths


def save_tensor_splits(
    output_dir: Path,
    *,
    inputs: torch.Tensor,
    lengths: torch.Tensor,
    train_loss_mask: torch.Tensor,
    val_loss_mask: torch.Tensor,
    test_loss_mask: torch.Tensor,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {"inputs": inputs.contiguous(), "lengths": lengths.contiguous()}
    torch.save({**common, "loss_mask": train_loss_mask.contiguous()}, output_dir / "train.pt")
    torch.save({**common, "loss_mask": val_loss_mask.contiguous()}, output_dir / "val.pt")
    torch.save({**common, "loss_mask": test_loss_mask.contiguous()}, output_dir / "test.pt")


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
