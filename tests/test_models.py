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


def test_photo_model_split_matches_full_and_heads_shapes():
    """lower+upper equals the full backbone; heads give (B,) log-carbs and (B, 4) bin logits; carb bins are right."""
    from glimpse.models.photo_model import PhotoCarbModel, carb_bin

    torch.manual_seed(0)
    m = PhotoCarbModel(pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        feats = m.lower(x)
        assert feats.shape == (2, 384, 14, 14)
        assert torch.allclose(m.upper(feats), m.backbone(x), atol=1e-4)
        reg, bins = m(x)
    assert reg.shape == (2,) and bins.shape == (2, 4)
    assert carb_bin(torch.tensor([0.0, 29.9, 30.0, 59.9, 60.0, 90.0, 200.0])).tolist() == [0, 0, 1, 1, 2, 3, 3]


def test_photo_model_trainable_parts():
    """Phase 1 trains only the heads; phase 2 adds stage 3 + final norm; the saved state is just those parts."""
    from glimpse.models.photo_model import PhotoCarbModel

    m = PhotoCarbModel(pretrained=False)
    m.set_trainable(stage3=False)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 768 * 5 + 5
    m.set_trainable(stage3=True)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n_train == sum(v.numel() for v in m.trainable_state().values())
    assert not any(p.requires_grad for p in m.backbone.stem.parameters())
