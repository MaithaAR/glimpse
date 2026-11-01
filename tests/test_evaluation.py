"""Tests for the evaluation helpers used by the proposal pilot."""

import numpy as np

from glimpse.evaluation import ece_equal_mass, expected_cost, load_locked_set, noninferiority


def test_noninferiority():
    assert noninferiority(-0.01, 0.02) == "non-inferior"
    assert noninferiority(-0.09, -0.04) == "inferior"
    assert noninferiority(-0.05, 0.01) == "inconclusive"


def test_cost():
    costs = {"low_risk": [5, 0], "caution": [0, 1], "abstain": [2, 0.5]}
    y = np.array([1, 0])
    assert expected_cost(np.array([0, 0]), y, costs) == 2.5
    assert expected_cost(np.array([1, 1]), y, costs) == 0.5


def test_ece_perfect():
    p = np.linspace(0.05, 0.95, 1000)
    y = (np.random.default_rng(0).random(1000) < p).astype(int)
    assert ece_equal_mass(p, y) < 0.06


def test_locked_hash():
    assert len(load_locked_set("splits/locked_final_set.json")) == 8


def test_calibration_by_person_finds_rare_spikers_over_predicted():
    """A model that predicts the same risk for everyone over-predicts for rare spikers and under-predicts for others."""
    from glimpse.evaluation import calibration_by_person

    person = np.repeat([1, 2, 3, 4], 10)
    y = np.concatenate([np.zeros(10), np.r_[np.ones(2), np.zeros(8)], np.r_[np.ones(8), np.zeros(2)], np.ones(10)])
    out = calibration_by_person(person, y, np.full(40, 0.5))
    assert out["rare_spikers"]["predicted"] > out["rare_spikers"]["observed"]
    assert out["frequent_spikers"]["predicted"] < out["frequent_spikers"]["observed"]
    assert out["corr_spike_rate_vs_gap"] < -0.9


def test_platt_recalibration_fixes_over_prediction():
    """Over-predicted probabilities get an intercept below 0 and a lower ECE after recalibration."""
    from glimpse.evaluation import apply_platt, fit_platt, reliability

    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.05, 0.6, 600)
    y = (rng.uniform(size=600) < true_p).astype(int)
    over = np.clip(true_p + 0.2, 0, 0.99)  # the model says 20 points too much
    slope, intercept = fit_platt(over, y)
    assert intercept < 0
    assert reliability(apply_platt(over, slope, intercept), y)[1] < reliability(over, y)[1]
    np.testing.assert_allclose(apply_platt(np.array([0.3, 0.7]), 1.0, 0.0), [0.3, 0.7], atol=1e-6)


def test_evaluation_only_measures_it_never_decides():
    """The S3 exit criterion, enforced: the evaluation package has no decision-making function, and no
    module in it imports the training-side thresholds. Evaluation numbers come from Pipeline answers."""
    from pathlib import Path

    import glimpse.evaluation as ev

    for name in ("decide", "choose_thresholds", "cost_only_thresholds"):
        assert not hasattr(ev, name), f"{name} must live in glimpse.training.thresholds, not in evaluation"
    for f in Path("src/glimpse/evaluation").glob("*.py"):
        imports = [ln for ln in f.read_text().splitlines() if ln.startswith(("import ", "from "))]
        assert not any("training.thresholds" in ln for ln in imports), f"{f} imports the decision layer"


def test_per_fold_auroc_platt_equals_raw_within_fold():
    """Platt is monotone, so it cannot change AUROC inside a fold -- the table that defuses the pooled
    platt-below-raw artefact must show auroc_platt == auroc_raw in every row."""
    import pandas as pd

    from glimpse.evaluation.run import _fold_auroc_table

    rng = np.random.default_rng(3)
    rows = []
    for fold in (1, 2):
        raw = rng.uniform(0.05, 0.95, size=40)
        y = (rng.uniform(size=40) < raw).astype(int)
        z = np.log(raw / (1 - raw))
        platt = 1 / (1 + np.exp(-(0.8 * z - 0.3 * fold)))  # fold-specific monotone map
        rows += [{"fold": fold, "spike": int(yy), "risk_raw": rr, "risk_platt": pp, "risk": pp}
                 for yy, rr, pp in zip(y, raw, platt, strict=True)]  # fmt: skip
    table = _fold_auroc_table(pd.DataFrame(rows))
    assert [r["fold"] for r in table] == [1, 2]
    for r in table:
        assert r["n"] == 40
        assert abs(r["auroc_platt"] - r["auroc_raw"]) < 1e-12
