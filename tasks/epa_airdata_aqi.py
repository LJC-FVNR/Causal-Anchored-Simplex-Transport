from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch

from .masked_tensor_dir import MaskedWindowTensorDirectoryTask


class EPAAirDataAQITask(MaskedWindowTensorDirectoryTask):
    name = "epa_airdata_aqi"

    def __init__(self, dataset_dir: Optional[str] = None):
        root_dir = Path(__file__).resolve().parents[1]
        default_dir = root_dir / "benchmarks" / "epa_airdata_aqi" / "processed"
        super().__init__(dataset_dir=dataset_dir or str(default_dir))
        self.metadata: Dict[str, object] = {}

    def prepare(self, *, rank: int = 0, world_size: int = 1) -> None:
        try:
            super().prepare(rank=rank, world_size=world_size)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{exc}\nRun benchmarks/epa_airdata_aqi/download.py and benchmarks/epa_airdata_aqi/prepare.py first, "
                "or set EPA_AIRDATA_AQI_DIR / DATASET_DIR."
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
                    "num_sequences": self.metadata.get("num_sequences"),
                    "probability_dim": self.metadata.get("probability_dim"),
                    "aqi_categories": self.metadata.get("aqi_categories"),
                    "sequence_length_months_summary": self.metadata.get("sequence_length_months_summary"),
                    "split_target_totals": self.metadata.get("split_target_totals"),
                }
            )
        return out

    def ordered_probability_support(self) -> Optional[torch.Tensor]:
        return torch.arange(self.prob_dim, dtype=torch.float32)


def default_epa_airdata_aqi_dir() -> str:
    return os.fspath(Path(__file__).resolve().parents[1] / "benchmarks" / "epa_airdata_aqi" / "processed")
