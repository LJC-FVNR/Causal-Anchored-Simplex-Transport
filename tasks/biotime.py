from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

from .masked_tensor_dir import MaskedWindowTensorDirectoryTask


class BioTIMETask(MaskedWindowTensorDirectoryTask):
    name = "biotime"

    def __init__(self, dataset_dir: Optional[str] = None):
        root_dir = Path(__file__).resolve().parents[1]
        default_dir = root_dir / "benchmarks" / "biotime" / "processed"
        super().__init__(dataset_dir=dataset_dir or str(default_dir))
        self.metadata: Dict[str, object] = {}

    def prepare(self, *, rank: int = 0, world_size: int = 1) -> None:
        try:
            super().prepare(rank=rank, world_size=world_size)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{exc}\nRun benchmarks/biotime/download.py and benchmarks/biotime/prepare.py first, "
                "or set BIOTIME_DIR / DATASET_DIR."
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
                    "taxon_filter": self.metadata.get("taxon_filter"),
                    "num_sequences": self.metadata.get("num_sequences"),
                    "num_studies": self.metadata.get("num_studies"),
                    "probability_dim": self.metadata.get("probability_dim"),
                    "top_k": self.metadata.get("top_k"),
                    "min_contiguous_years": self.metadata.get("min_contiguous_years"),
                    "captured_measure_fraction_top_k": self.metadata.get("captured_measure_fraction_top_k"),
                    "split_target_totals": self.metadata.get("split_target_totals"),
                    "sequence_length_years_summary": self.metadata.get("sequence_length_years_summary"),
                }
            )
        return out


def default_biotime_dir() -> str:
    return os.fspath(Path(__file__).resolve().parents[1] / "benchmarks" / "biotime" / "processed")
