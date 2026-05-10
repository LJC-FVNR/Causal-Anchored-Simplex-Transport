from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

from .masked_tensor_dir import MaskedWindowTensorDirectoryTask


class EmberMonthlyTask(MaskedWindowTensorDirectoryTask):
    name = "ember_monthly"

    def __init__(self, dataset_dir: Optional[str] = None):
        root_dir = Path(__file__).resolve().parents[1]
        default_dir = root_dir / "benchmarks" / "ember_monthly" / "processed"
        super().__init__(dataset_dir=dataset_dir or str(default_dir))
        self.metadata: Dict[str, object] = {}

    def prepare(self, *, rank: int = 0, world_size: int = 1) -> None:
        try:
            super().prepare(rank=rank, world_size=world_size)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{exc}\nRun benchmarks/ember_monthly/download.py and benchmarks/ember_monthly/prepare.py first, "
                "or set EMBER_MONTHLY_DIR / DATASET_DIR."
            ) from exc

        metadata_path = self.dataset_dir / "metadata.json"
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    def dataset_summary(self) -> Dict[str, object]:
        out = super().dataset_summary()
        if self.metadata:
            out.update(
                {
                    "benchmark": self.metadata.get("benchmark"),
                    "num_countries": self.metadata.get("num_countries"),
                    "probability_dim": self.metadata.get("probability_dim"),
                    "fuel_vocabulary": self.metadata.get("fuel_vocabulary"),
                    "global_start_month": self.metadata.get("global_start_month"),
                    "global_end_month": self.metadata.get("global_end_month"),
                    "sequence_length_months_summary": self.metadata.get("sequence_length_months_summary"),
                    "split_mode_counts": self.metadata.get("split_mode_counts"),
                    "split_target_totals": self.metadata.get("split_target_totals"),
                }
            )
        return out


def default_ember_monthly_dir() -> str:
    return os.fspath(Path(__file__).resolve().parents[1] / "benchmarks" / "ember_monthly" / "processed")
