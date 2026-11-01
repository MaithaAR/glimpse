"""Decision-making: turning risks into actions and choosing the cut-offs. These are training-time
activities (design D4: selection happens on inner-fold risk), so they live here, not in evaluation --
the evaluation package only measures what Pipeline.answer decided. The vectorised `decide` exists for
searching candidate cut-offs; its output is never a reported number."""

from __future__ import annotations

import numpy as np

from glimpse.evaluation.metrics import expected_cost


def decide(p, lo_thr, hi_thr, abstain_mask=None):
    """low risk if p < lo_thr, caution if p >= hi_thr, abstain in between or where abstain_mask is True."""
    a = np.where(p < lo_thr, 0, np.where(p >= hi_thr, 1, 2))
    if abstain_mask is not None:
        a = np.where(abstain_mask, 2, a)
    return a


def choose_thresholds(p, y, costs, grid=None):
    """(lo, hi) with lo <= hi that give the lowest mean cost: low risk below lo, caution from hi, abstain between.
    Ties go to the first pair found (smallest lo, then smallest hi), so the choice is deterministic."""
    grid = np.round(np.arange(0.0, 1.0001, 0.01), 2) if grid is None else grid
    best = (float("inf"), 0.0, 0.0)
    for lo in grid:
        for hi in grid[grid >= lo]:
            c = expected_cost(decide(p, lo, hi), y, costs)
            if c < best[0] - 1e-12:
                best = (c, float(lo), float(hi))
    return best[1], best[2], best[0]


def cost_only_thresholds(costs):
    """Cut-offs that follow from the costs alone, if the probability p is honest (calibrated): each answer's
    expected cost is a straight line in p; pick the cheapest. Low risk a*p, caution b*(1-p), abstain c*p + d*(1-p)
    with (a, 0), (0, b), (c, d) = low_risk, caution, abstain. Uses no data."""
    a, b = costs["low_risk"][0], costs["caution"][1]
    c, d = costs["abstain"]
    den_lo, den_hi = a - c + d, b - d + c
    if den_lo > 0 and den_hi > 0:
        lo, hi = d / den_lo, (b - d) / den_hi
        if 0 <= lo < hi <= 1:
            return float(lo), float(hi)
    # abstaining is never the cheapest answer: one cut-off where low risk and caution cost the same
    return float(b / (a + b)), float(b / (a + b))
