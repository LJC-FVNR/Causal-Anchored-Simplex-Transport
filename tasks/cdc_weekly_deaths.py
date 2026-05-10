from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

from .masked_tensor_dir import MaskedWindowTensorDirectoryTask


class CDCWeeklyDeathsTask(MaskedWindowTensorDirectoryTask):
    name = "cdc_weekly_deaths"

    def __init__(self, dataset_dir: Optional[str] = None):
        root_dir = Path(__file__).resolve().parents[1]
        default_dir = root_dir / "benchmarks" / "cdc_weekly_deaths" / "processed"
        super().__init__(dataset_dir=dataset_dir or str(default_dir))
        self.metadata: Dict[str, object] = {}

    def prepare(self, *, rank: int = 0, world_size: int = 1) -> None:
        try:
            super().prepare(rank=rank, world_size=world_size)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{exc}\nRun benchmarks/cdc_weekly_deaths/download.py and benchmarks/cdc_weekly_deaths/prepare.py first, "
                "or set CDC_WEEKLY_DEATHS_DIR / DATASET_DIR."
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
                    "num_states": self.metadata.get("num_states"),
                    "num_weeks": self.metadata.get("num_weeks"),
                    "probability_dim": self.metadata.get("probability_dim"),
                    "resolved_split_mode": self.metadata.get("resolved_split_mode"),
                    "tail_drop_weeks": self.metadata.get("tail_drop_weeks"),
                }
            )
        return out


def default_cdc_weekly_deaths_dir() -> str:
    return os.fspath(Path(__file__).resolve().parents[1] / "benchmarks" / "cdc_weekly_deaths" / "processed")

