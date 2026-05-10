import torch

from tests.test_cast_alias import TinyTask, tiny_config
from models.registry import build_model


def test_cast_outputs_are_simplex_valued():
    torch.manual_seed(0)
    task = TinyTask()
    result = build_model(tiny_config("full"), task)
    probs = torch.rand(2, 4, task.prob_dim)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    targets = torch.rand(2, 4, task.prob_dim)
    targets = targets / targets.sum(dim=-1, keepdim=True)

    out = result.model(probs, targets=targets, return_dict=True)
    pred = out["pred"]
    assert torch.isfinite(out["loss"])
    assert torch.all(pred >= 0)
    assert torch.allclose(pred.sum(dim=-1), torch.ones_like(pred[..., 0]), atol=1e-5)
