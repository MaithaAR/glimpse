"""S13 explainability probes (pre-registered in docs/probes_s13.md BEFORE this ran): two counterfactuals
through the SAME per-fold Pipelines the evaluation uses, no history. Probe A swaps each meal's photo
estimate for the fold's low-/high-carb reference plate; probe B replays each meal with a flat and a
rising pre-meal CGM. A2 is the committed expected-fail: the photo nudges risk, it rarely flips the
decision -- the S8 gate seen mechanistically. Report: section 5, one table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from glimpse.calibration import choose_m_and_cutoffs
from glimpse.config import GlimpseConfig
from glimpse.data.dataset import MealDataset
from glimpse.data.splits import SplitsV2, meal_rows
from glimpse.evaluation.metrics import apply_platt, fit_platt
from glimpse.evaluation.run import fold_estimates, fold_fusion, inner_oof
from glimpse.serve import MealInput, Pipeline
from glimpse.training.final import cost_dict
from glimpse.training.fusion import photo_table

OUT = Path("outputs/probes")
RAMP_MGDL, RAMP_MIN, REF_QUANTILES = 40.0, 45, (0.10, 0.90)


def flat_cgm(m: MealInput) -> MealInput:
    """Every pre-meal glucose value replaced by the window's own mean (times untouched)."""
    mu = float(np.mean(m.glucose_mgdl))
    return m.model_copy(update={"glucose_mgdl": [mu] * len(m.glucose_mgdl)})


def rising_cgm(m: MealInput) -> MealInput:
    """A linear ramp added over the final RAMP_MIN minutes: 0 at minute -RAMP_MIN, +RAMP_MGDL at the meal."""
    t0, span = m.meal_time - pd.Timedelta(minutes=RAMP_MIN), RAMP_MIN * 60.0
    v = [val + RAMP_MGDL * min(1.0, (pd.Timestamp(x) - t0).total_seconds() / span) if pd.Timestamp(x) >= t0
         else val for x, val in zip(m.glucose_times, m.glucose_mgdl, strict=True)]  # fmt: skip
    return m.model_copy(update={"glucose_mgdl": v})


def reference_estimates(of: pd.DataFrame) -> tuple:
    """The fold's low- and high-carb reference plates: the OOF meals at the 10th/90th grams percentile."""
    ranked, est = of.sort_values("photo_g").reset_index(drop=True), fold_estimates(of)
    lo_row = int(ranked.row.iloc[int(REF_QUANTILES[0] * (len(ranked) - 1))])
    hi_row = int(ranked.row.iloc[int(REF_QUANTILES[1] * (len(ranked) - 1))])
    return est[lo_row], est[hi_row]


def run_probes(ds: MealDataset, sp: SplitsV2, cfg: GlimpseConfig) -> dict:
    """Both probes over every answered development meal; writes outputs/probes/probes.json."""
    from glimpse.demo_cases import meal_input

    log, pcv, y = logger.bind(stage="probes"), Path(cfg.paths.outputs) / "photo_cv", ds.labels()
    oof, nested = pd.read_csv(pcv / "photo_oof.csv"), pd.read_csv(pcv / "photo_nested.csv")
    rows = []
    for k, held in enumerate(sp.dev_folds):
        kk = k + 1
        of, nf = oof[oof.fold == kk], nested[nested.fold == kk]
        pt = photo_table(nf) | photo_table(of)
        lr, c = fold_fusion(ds, sp, cfg, k, pt)
        rows_i, p_i = inner_oof(ds, sp, cfg, k, pt, c)
        a_k, b_k = fit_platt(p_i, y[rows_i])
        if a_k <= 0:
            a_k, b_k = 1.0, 0.0
        times_i = np.array([ds.samples[i].meal_time for i in rows_i], dtype="datetime64[s]")
        person_i = np.array([ds.samples[i].person_id for i in rows_i])
        sel = choose_m_and_cutoffs(apply_platt(p_i, a_k, b_k), y[rows_i], person_i, times_i, cost_dict(cfg))
        cuts, est = (sel["low_risk_below"], sel["caution_from"]), fold_estimates(of)
        ref_lo, ref_hi = reference_estimates(of)
        pipes = {"base": est, "photo_low": {r: ref_lo for r in est}, "photo_high": {r: ref_hi for r in est}}
        pipes = {name: Pipeline(cfg, photo_table=t, fusion=lr, calibration=(a_k, b_k), cutoffs=cuts)
                 for name, t in pipes.items()}  # fmt: skip
        for row in meal_rows(ds, set(held), scored_only=True):
            m = meal_input(cfg, ds, int(row))
            base = pipes["base"].answer(m, key=int(row))
            if base.risk is None:
                continue
            rec = {"fold": kk, "row": int(row), "risk_base": base.risk, "dec_base": base.decision}
            for name in ("photo_low", "photo_high"):
                a = pipes[name].answer(m, key=int(row))
                rec[f"risk_{name}"], rec[f"dec_{name}"] = a.risk, a.decision
            for name, fn in (("flat", flat_cgm), ("rising", rising_cgm)):
                a = pipes["base"].answer(fn(m), key=int(row))
                rec[f"risk_{name}"], rec[f"dec_{name}"] = a.risk, a.decision
            rows.append(rec)
        log.info("fold {} probed", kk)
    df = pd.DataFrame(rows)
    swing_a = (df.risk_photo_high - df.risk_photo_low).to_numpy()
    move_b = (df.risk_rising - df.risk_flat).to_numpy()
    res = {"what": "S13 counterfactual probes through the per-fold evaluation Pipelines, no history; "
                   "settings and expectations pre-registered in docs/probes_s13.md",
           "settings": {"ramp_mgdl": RAMP_MGDL, "ramp_min": RAMP_MIN, "ref_quantiles": REF_QUANTILES},
           "probe_a_photo_swap": {
               "n": int(len(df)),
               "share_direction_ok": float((swing_a >= 0).mean()),
               "share_decision_unchanged": float(((df.dec_photo_low == df.dec_base)
                                                  & (df.dec_photo_high == df.dec_base)).mean()),
               "median_abs_swing": float(np.median(np.abs(swing_a))),
               "a1_direction_pass": bool((swing_a >= 0).mean() >= 0.9),
               "a2_expected_fail_confirmed": bool(((df.dec_photo_low == df.dec_base)
                                                   & (df.dec_photo_high == df.dec_base)).mean() > 0.6)},
           "probe_b_cgm_trend": {
               "n": int(len(df)),
               "share_direction_ok": float((move_b > 0).mean()),
               "median_move": float(np.median(move_b)),
               "b1_direction_pass": bool((move_b > 0).mean() >= 0.9),
               "b2_median_move_vs_a_swing": [float(np.median(np.abs(move_b))),
                                             float(np.median(np.abs(swing_a)))]}}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "probe_answers.csv", index=False)
    (OUT / "probes.json").write_text(json.dumps(res, indent=2))
    pa, pb = res["probe_a_photo_swap"], res["probe_b_cgm_trend"]
    log.info("probe A: direction ok {:.0%} (A1 pass={}), decision unchanged {:.0%} (A2 expected-fail "
             "confirmed={}), median |swing| {:.3f}", pa["share_direction_ok"], pa["a1_direction_pass"],
             pa["share_decision_unchanged"], pa["a2_expected_fail_confirmed"], pa["median_abs_swing"])  # fmt: skip
    log.info("probe B: rising>flat {:.0%} (B1 pass={}), median move {:+.3f}; |B| vs |A| median {:.3f} / "
             "{:.3f}", pb["share_direction_ok"], pb["b1_direction_pass"], pb["median_move"],
             *pb["b2_median_move_vs_a_swing"])  # fmt: skip
    return res
