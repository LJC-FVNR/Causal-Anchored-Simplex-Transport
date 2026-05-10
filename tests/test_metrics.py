import torch

from evaluation.metrics import bray_curtis_distance, js_divergence, kl_divergence, l1_distance


def test_basic_distribution_metrics_are_finite_and_zero_on_equal_inputs():
    p = torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=torch.float32)
    q = torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=torch.float32)

    assert torch.allclose(kl_divergence(p, q), torch.zeros(2), atol=1e-7)
    assert torch.allclose(js_divergence(p, q), torch.zeros(2), atol=1e-7)
    assert torch.allclose(l1_distance(p, q), torch.zeros(2), atol=1e-7)
    assert torch.allclose(bray_curtis_distance(p, q), torch.zeros(2), atol=1e-7)
