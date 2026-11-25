"""Known failures, recorded as tests that are expected to fail (strict: if one starts passing, the suite says so
and the report must change). Added 26 Nov after the third grading round."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.xfail(strict=True, reason="tier 2 (typed carbs) feeds one-hot bins the fusion never saw in training; "
                   "risk is not monotone in typed carbs (report §5, known failure)")  # fmt: skip
def test_typed_carbs_risk_is_monotone():
    import joblib

    from glimpse.serve.pipeline import _bin_onehot
    from glimpse.training.final import FINAL

    pipe = joblib.load(FINAL)["models"]["photo"]
    risks = []
    for grams in (5, 20, 29, 30, 45, 59, 60, 90, 120):
        x = np.full((1, 19), np.nan)  # every non-carb input at the frozen imputer's medians
        x[0, 15] = np.log1p(grams)
        x[0, 16:] = _bin_onehot(grams)[1:]
        risks.append(float(pipe.predict_proba(x)[0, 1]))
    assert all(b >= a for a, b in zip(risks, risks[1:], strict=False)), risks


def test_folder_inputs_without_priors_get_the_training_medians():
    """Not a failure of the code, a disclosed property: a profile.json without onboarding priors (every
    instructor case) is answered with the frozen imputer's medians, learned on training people incl. T2D."""
    import joblib

    from glimpse.training.final import FINAL

    imp = joblib.load(FINAL)["models"]["photo"][0]
    assert imp.statistics_[13] == pytest.approx(0.455, abs=0.001)  # prior spike rate
    assert imp.statistics_[14] == pytest.approx(48.4, abs=0.05)  # prior rise, mg/dL
