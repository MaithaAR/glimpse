"""S6 ablation helpers: the macro key treats missing values as equal, and channel removal drops exactly
the named columns."""

import numpy as np

from glimpse.studies.ablations import CHANNELS, macro_key


def test_macro_key_rounds_and_handles_missing():
    assert macro_key([85.0, 88.04, 54.2, 12.0, 1170.0]) == (85.0, 88.0, 54.2, 12.0, 1170.0)
    assert macro_key([85.0, float("nan"), 54.2, 12.0, 1170.0]) == macro_key([85.0, float("nan"), 54.2, 12.0, 1170.0])


def test_channels_drop_the_named_columns():
    x = np.arange(2 * 19, dtype=float).reshape(2, 19)  # 15 trend+profile columns + 4 photo columns
    assert np.delete(x, CHANNELS["no_heart_rate"], axis=1).shape == (2, 18)
    assert np.delete(x, CHANNELS["no_profile"], axis=1).shape == (2, 11)
    kept = np.delete(x, CHANNELS["no_profile"], axis=1)[0]
    assert list(kept[:7]) == list(x[0][:7]) and list(kept[-4:]) == list(x[0][-4:])  # photo block untouched


def test_perturbations_change_the_right_things():
    """Dropout keeps the latest reading; the gap removes a 25-minute block; flips classify correctly."""
    from datetime import datetime, timedelta

    import numpy as np

    from glimpse.serve import MealInput
    from glimpse.studies.robustness import _flips, perturb_cgm

    meal = datetime(2024, 6, 1, 12, 0)
    times = [meal - timedelta(minutes=5 * (24 - i)) for i in range(24)]
    m = MealInput(meal_time=meal, glucose_times=times, glucose_mgdl=[100.0] * 24,
                  profile={"age": 40.0, "bmi": 28.0, "a1c_pct": 5.6, "fasting_glucose_mgdl": 95.0})  # fmt: skip
    d = perturb_cgm(m, "cgm_dropout", np.random.default_rng(0))
    assert d.glucose_times[-1] == times[-1] and len(d.glucose_times) < 24
    g = perturb_cgm(m, "cgm_gap", np.random.default_rng(0))
    gone = [t for t in times if t not in g.glucose_times]
    assert gone and all(meal - timedelta(minutes=55) <= t < meal - timedelta(minutes=30) for t in gone)
    f = _flips(["low risk", "caution", "caution", "can't judge"],
               ["caution", "caution", "can't judge", "low risk"])  # fmt: skip
    assert f["dangerous_flips"] == 0.25 and f["newly_cant_judge"] == 0.25 and f["recovered_answers"] == 0.25


def test_probe_counterfactuals_shape_the_cgm_as_registered():
    """Flat replaces every value by the window mean; rising adds 0 at -45 min up to +40 at the meal and
    leaves earlier readings untouched (docs/probes_s13.md settings)."""
    from datetime import datetime, timedelta

    from glimpse.serve import MealInput
    from glimpse.studies.probes import flat_cgm, rising_cgm

    meal = datetime(2024, 6, 1, 12, 0)
    times = [meal - timedelta(minutes=5 * (24 - i)) for i in range(24)]
    vals = [90.0 + i for i in range(24)]
    m = MealInput(meal_time=meal, glucose_times=times, glucose_mgdl=vals,
                  profile={"age": 40.0, "bmi": 28.0, "a1c_pct": 5.6, "fasting_glucose_mgdl": 95.0})  # fmt: skip
    f = flat_cgm(m)
    assert len(set(f.glucose_mgdl)) == 1 and abs(f.glucose_mgdl[0] - sum(vals) / 24) < 1e-9
    r = rising_cgm(m)
    bumps = [rv - v for rv, v in zip(r.glucose_mgdl, vals, strict=True)]
    assert bumps[:15] == [0.0] * 15  # readings before -45 min untouched; the -45 reading itself bumps 0
    assert all(b2 >= b1 for b1, b2 in zip(bumps[15:], bumps[16:], strict=False)) and 0 < max(bumps) <= 40
    assert r.glucose_times == times and r.meal_time == m.meal_time
