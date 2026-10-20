"""Model tests: dummy-batch forward passes, output shapes and parameter limits."""

import torch

from glimpse.models.sensor_cnn import SensorCNN, count_parameters

SCRATCH_LIMIT = 50_000_000


def test_sensor_cnn_forward_shape_and_budget():
    """(B, 3, 24) sequence + (B, 8) tabular -> (B,) finite logits; far below the 50M from-scratch limit."""
    torch.manual_seed(0)
    model = SensorCNN(n_tabular=8)
    out = model(torch.randn(5, 3, 24), torch.randn(5, 8))
    assert out.shape == (5,) and torch.isfinite(out).all()
    assert count_parameters(model) == 5217 < SCRATCH_LIMIT


def test_sensor_cnn_is_randomly_initialised():
    """Two seeds give different weights: no pretrained state is loaded."""
    torch.manual_seed(0)
    a = SensorCNN().conv[0].weight.detach().clone()
    torch.manual_seed(1)
    assert not torch.equal(a, SensorCNN().conv[0].weight)


def test_parameter_budget_within_limits():
    """Every model (built or planned) is counted programmatically and fits the project limits."""
    from glimpse.models.budget import budget_table

    rows = budget_table()
    assert all(r.within_limit for r in rows)
    assert {r.architecture for r in rows} >= {"convnext_tiny", "vit_small_patch14_dinov2"}
