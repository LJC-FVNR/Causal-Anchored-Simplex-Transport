from dataclasses import replace

import torch

from models.cast import CASTModel
from models.registry import build_model
from training.config import TrainingConfig


class TinyTask:
    name = "tiny"
    prob_dim = 5

    def model_config_overrides(self, train_config=None, *, model_name=None):
        return {}

    def ordered_probability_support(self):
        return torch.arange(self.prob_dim, dtype=torch.float32)


def tiny_config(ablation="full"):
    base = TrainingConfig.from_env()
    return replace(
        base,
        model_name="cast",
        dataset_name="tiny",
        ablation=ablation,
        block_size=6,
        model_max_seq_len=6,
        d_model=32,
        n_layer=1,
        n_head=4,
        d_ff=64,
        dropout=0.0,
        attn_dropout=0.0,
        resid_dropout=0.0,
    )


def test_cast_full_uses_structural_regularization():
    result = build_model(tiny_config("full"), TinyTask())
    assert result.name == "cast"
    assert isinstance(result.model, CASTModel)
    assert result.config.struct_reg_weight == 5e-4
    assert result.config.struct_smooth_weight == 1e-4


def test_public_cast_ablation_toggles_structural_regularization():
    result = build_model(tiny_config("without_structural_reg"), TinyTask())
    assert result.config.struct_reg_weight == 0.0
    assert result.config.struct_smooth_weight == 0.0
