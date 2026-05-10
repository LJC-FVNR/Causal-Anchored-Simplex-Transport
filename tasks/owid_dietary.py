from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

from .masked_tensor_dir import MaskedWindowTensorDirectoryTask


class OWIDDietaryTask(MaskedWindowTensorDirectoryTask):
    name = "owid_dietary"

    def __init__(self, dataset_dir: Optional[str] = None):
        root_dir = Path(__file__).resolve().parents[1]
        default_dir = root_dir / "benchmarks" / "owid_dietary" / "processed"
        super().__init__(dataset_dir=dataset_dir or str(default_dir))
        self.metadata: Dict[str, object] = {}

    def prepare(self, *, rank: int = 0, world_size: int = 1) -> None:
        try:
            super().prepare(rank=rank, world_size=world_size)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{exc}\nRun benchmarks/owid_dietary/download.py and benchmarks/owid_dietary/prepare.py first, "
                "or set OWID_DIETARY_DIR / DATASET_DIR."
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
                    "num_entities": self.metadata.get("num_entities"),
                    "num_years": self.metadata.get("num_years"),
                    "probability_dim": self.metadata.get("probability_dim"),
                    "train_target_years": self.metadata.get("train_target_years"),
                    "val_target_years": self.metadata.get("val_target_years"),
                    "test_target_years": self.metadata.get("test_target_years"),
                }
            )
        return out


def default_owid_dietary_dir() -> str:
    return os.fspath(Path(__file__).resolve().parents[1] / "benchmarks" / "owid_dietary" / "processed")

