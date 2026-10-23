"""The sensor model: right output shape, small, random at the start, and within the parameter limits."""

import torch

from glimpse.models.budget import budget_table
from glimpse.models.sensor_cnn import SensorCNN, count_parameters


def test_forward_shape():
    """A dummy batch of 5 meals (3 channels x 24 slots + 8 profile inputs) gives 5 finite scores."""
    out = SensorCNN()(torch.randn(5, 3, 24), torch.randn(5, 8))
    assert out.shape == (5,) and torch.isfinite(out).all()


def test_size_is_small():
    """About 5,000 learnable numbers: far below the 50 million limit for a from-scratch model."""
    n = count_parameters(SensorCNN())
    assert n == 5217 and n < 50_000_000


def test_starts_random():
    """Different seeds give different starting weights, so nothing pretrained is loaded."""
    torch.manual_seed(0)
    a = SensorCNN().conv[0].weight.detach().clone()
    torch.manual_seed(1)
    assert not torch.equal(a, SensorCNN().conv[0].weight)


def test_budget_within_limits():
    """Every built or planned model fits its limit."""
    assert all(r.within_limit for r in budget_table())
