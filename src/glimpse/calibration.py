"""Fix 1 (design D3, amended 1 Nov): Platt recalibration of the population risk, then a per-person
intercept on the recalibrated logit, fitted only on the person's earlier meals whose CGM label is
already available (two-hour gate). The prior precision m and the decision cut-offs are chosen together
by replaying this protocol on inner-fold out-of-fold risks (D4); m = inf is exactly the no-intercept
system. Pure math lives here; the fold orchestration is in evaluation/run.py. Report: §4 trust."""

from __future__ import annotations

import numpy as np

from glimpse.evaluation.metrics import _logit
from glimpse.training.thresholds import choose_thresholds

def logit(p):
    """log-odds with the same clipping as the metrics module (shared by the Pipeline and the replay)."""
    return _logit(np.asarray(p, dtype=float))


M_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, float("inf")]
LABEL_LAG_MIN = 120  # the CGM labels a meal two hours after it (feedback.py, cgm_auto)


def personal_offset(f: np.ndarray, y: np.ndarray, m: float) -> float:
    """argmin_b  sum_i NLL(y_i, sigmoid(f_i + b)) + (m/2) b^2.  Strictly convex (second derivative >= m),
    solved by damped Newton from b = 0; m = inf or an empty history gives exactly b = 0."""
    if not np.isfinite(m) or len(f) == 0:
        return 0.0
    f, y, b = np.asarray(f, float), np.asarray(y, float), 0.0
    for _ in range(25):
        p = 1.0 / (1.0 + np.exp(-(f + b)))
        step = (float(np.sum(p - y)) + m * b) / (float(np.sum(p * (1.0 - p))) + m)
        step = float(np.clip(step, -4.0, 4.0))  # cheap insurance when every sigmoid saturates
        b -= step
        if abs(step) < 1e-10:
            break
    return float(b)


def replay_offsets(f: np.ndarray, y: np.ndarray, times: np.ndarray, m: float,
                   lag_min: float = LABEL_LAG_MIN) -> np.ndarray:  # fmt: skip
    """b_p(t) for one person's meals, each fitted on the strictly earlier meals whose label was already
    available at t (meal time + two hours <= t) and which have a finite logit. Order of the input is kept;
    the chronology comes from `times` (numpy datetime64)."""
    f, y = np.asarray(f, float), np.asarray(y, float)
    t = np.asarray(times, "datetime64[s]")
    lag = np.timedelta64(int(lag_min * 60), "s")
    out = np.zeros(len(f))
    for i in range(len(f)):
        hist = (t + lag <= t[i]) & np.isfinite(f)
        out[i] = personal_offset(f[hist], y[hist], m)
    return out


def personal_risks(f: np.ndarray, y: np.ndarray, person: np.ndarray, times: np.ndarray, m: float) -> np.ndarray:
    """sigmoid(f + b_p(t)) for every meal, replayed per person in deployment order."""
    f, out = np.asarray(f, float), np.zeros(len(np.asarray(f)))
    person = np.asarray(person)
    for p in np.unique(person):
        i = np.where(person == p)[0]
        out[i] = 1.0 / (1.0 + np.exp(-(f[i] + replay_offsets(f[i], np.asarray(y)[i], np.asarray(times)[i], m))))
    return out


def choose_m_and_cutoffs(p_platt: np.ndarray, y: np.ndarray, person: np.ndarray, times: np.ndarray,
                         costs: dict, m_grid: list[float] = M_GRID) -> dict:  # fmt: skip
    """D3+D4 selection on inner-fold out-of-fold recalibrated risks: for each m, replay the intercept and
    re-choose (lo, hi) by minimum expected cost; exact cost ties break toward the LARGEST m (pre-registered:
    the most-shrunk model is the conservative default). Returns the choice and the full cost-vs-m curve."""
    f = _logit(p_platt)
    best, curve = None, []
    for m in m_grid:
        r = personal_risks(f, y, person, times, m)
        lo, hi, cost = choose_thresholds(r, np.asarray(y), costs)
        curve.append({"m": m, "low_risk_below": lo, "caution_from": hi, "cost": cost})
    for row in curve:  # ascending m; ties (within 1e-12) go to the later, larger m
        if best is None or row["cost"] < best["cost"] - 1e-12 or abs(row["cost"] - best["cost"]) <= 1e-12:
            best = row
    risks = personal_risks(f, y, person, times, best["m"])  # the replayed inner risks at the chosen m
    return {"m": best["m"], "low_risk_below": best["low_risk_below"], "caution_from": best["caution_from"],
            "cost": best["cost"], "curve": curve, "inner_risks": risks}  # fmt: skip


def sweep_cutoffs(risks: np.ndarray, y: np.ndarray, costs: dict) -> tuple[float, float]:
    """(lo, hi) by minimum expected cost on the replayed inner risks for one sweep cost setting (D4
    sensitivity: the main m stays fixed; only the cut-offs move with the costs)."""
    lo, hi, _ = choose_thresholds(np.asarray(risks, float), np.asarray(y, float), costs)
    return lo, hi
