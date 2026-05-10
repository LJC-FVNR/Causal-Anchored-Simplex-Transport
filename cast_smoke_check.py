from __future__ import annotations

from dataclasses import replace

import torch

from models.cast import CASTModel
from models.registry import build_model
from training.config import TrainingConfig


class TinyDistributionTask:
    name = "cast_smoke"
    prob_dim = 7

    def model_config_overrides(self, train_config: object | None = None, *, model_name: str | None = None):
        del train_config, model_name
        return {}

    def ordered_probability_support(self) -> torch.Tensor:
        return torch.arange(self.prob_dim, dtype=torch.float32)


def _config(ablation: str = "full") -> TrainingConfig:
    base = TrainingConfig.from_env()
    return replace(
        base,
        model_name="cast",
        dataset_name="cast_smoke",
        ablation=ablation,
        run_tag=f"cast_smoke_{ablation}",
        block_size=8,
        model_max_seq_len=8,
        d_model=32,
        n_layer=1,
        n_head=4,
        d_ff=64,
        dropout=0.0,
        attn_dropout=0.0,
        resid_dropout=0.0,
        use_input_stats=True,
        n_simplex_head=None,
        simplex_head_dim=None,
    )


def main() -> None:
    torch.manual_seed(0)
    task = TinyDistributionTask()

    cast = build_model(_config("full"), task)
    no_struct = build_model(_config("without_structural_reg"), task)

    assert cast.name == "cast"
    assert cast.run_label == "cast"
    assert isinstance(cast.model, CASTModel)
    assert cast.config.struct_reg_weight == 5e-4
    assert cast.config.struct_smooth_weight == 1e-4
    assert no_struct.config.struct_reg_weight == 0.0
    assert no_struct.config.struct_smooth_weight == 0.0

    probs = torch.rand(2, 5, task.prob_dim)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    targets = torch.rand(2, 5, task.prob_dim)
    targets = targets / targets.sum(dim=-1, keepdim=True)

    with torch.no_grad():
        output = cast.model(probs, targets=targets, return_dict=True, return_aux=True)

    pred = output["pred"]
    assert tuple(pred.shape) == tuple(probs.shape)
    assert torch.isfinite(output["loss"])
    assert torch.allclose(pred.sum(dim=-1), torch.ones_like(pred[..., 0]), atol=1e-5)
    assert torch.all(pred >= 0)

    print("CAST smoke check passed: MODEL=cast ABLATION=full preserves simplex outputs.")


if __name__ == "__main__":
    main()
