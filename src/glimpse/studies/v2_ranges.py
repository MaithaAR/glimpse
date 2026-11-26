"""Rule R16 (docs/v2_hardening.md Part C): v2's training input ranges, written once, and what the rule does
(`glimpse v2-input-ranges`). Writes configs/v2_input_ranges.json (refuses to overwrite) from the frozen v2
fusion's training meals, counts development and locked meals outside it (counted, never rescored), and runs
the registered heart-rate sweep, 40 to 200 bpm on one prediabetic profile, with the rule off and on."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from glimpse.serve.rules import V2_INPUTS

OUT = Path("outputs/probes/v2_hr_sweep.json")
PROFILE = {"age": 50.0, "bmi": 29.0, "a1c_pct": 6.2, "fasting_glucose_mgdl": 110.0, "prior_spike_rate": 0.45,
           "prior_rise": 45.0}  # fmt: skip


def write_ranges(cfg, ds) -> dict:
    from glimpse.data.splits import meal_rows
    from glimpse.training.baselines import trend_features

    path = Path(cfg.prototype.v2_input_ranges)
    if path.exists():
        return json.loads(path.read_text())
    people = joblib.load(cfg.prototype.sensor_first_bundle)["train_people"]
    x = trend_features(ds, np.array(sorted(meal_rows(ds, set(people), scored_only=False))))
    rng = {n: [float(np.nanmin(x[:, j])), float(np.nanmax(x[:, j]))] for j, n in enumerate(V2_INPUTS)}
    out = {"what": "rule R16: min and max of each v2 input over the frozen v2 fusion's training meals "
                   "(docs/v2_hardening.md Part C); written once; the rule checks the 7 sensor inputs",
           "training_meals": int(len(x)), "ranges": rng}  # fmt: skip
    path.write_text(json.dumps(out, indent=2))
    return out


def outside_counts(cfg, ds, ranges: dict) -> dict:
    from glimpse.serve.rules import outside_training_range
    from glimpse.training.baselines import trend_features

    res = {}
    for name, f in (("development", "outputs/evaluation_sensor_first/dev_answers.csv"),
                    ("locked", "outputs/locked_eval/locked_answers.csv")):  # fmt: skip
        d = pd.read_csv(f)
        x = trend_features(ds, d.row.to_numpy())
        hit = [outside_training_range(x[i], ranges["ranges"], ranges.get("risk_lowering_side")) is not None
               for i in range(len(d))]  # fmt: skip
        res[name] = {"meals": int(len(d)), "outside": int(sum(hit)),
                     "decisions_of_those": d.decision[hit].value_counts().to_dict()}  # fmt: skip
    return res


def sweep(cfg) -> list[dict]:
    from glimpse.serve import MealInput, Pipeline

    app = Pipeline(cfg, sensor_first=True)
    meal = datetime(2024, 6, 1, 12, 0)
    times = [meal - timedelta(minutes=5 * (24 - i)) for i in range(24)]
    trace = [110.0 + 6.0 * np.sin(i / 3.0) for i in range(24)]  # a calm but real trace (a flat one is out of range)
    rows = []
    for hr in range(40, 201, 10):
        m = MealInput(meal_time=meal, glucose_times=times, glucose_mgdl=trace, hr_times=times,
                      hr_bpm=[float(hr)] * 24, profile=PROFILE)  # fmt: skip
        keep = app.input_ranges
        app.input_ranges = None
        off = app.answer(m)
        app.input_ranges = keep
        on = app.answer(m)
        rows.append({"hr_bpm": hr, "rule_off": off.decision, "risk_off": off.risk, "rule_on": on.decision})
    return rows


def run_v2_ranges(cfg, ds) -> dict:
    ranges = write_ranges(cfg, ds)
    res = {"what": __doc__.splitlines()[0], "heart_rate_range": ranges["ranges"]["heart rate"],
           "outside": outside_counts(cfg, ds, ranges), "sweep": sweep(cfg)}  # fmt: skip
    OUT.write_text(json.dumps(res, indent=2))
    log = logger.bind(stage="v2-input-ranges")
    log.info("outside the training range: {}", res["outside"])
    for r in res["sweep"]:
        log.info("HR {:>3}: off {} ({:.1%}) | on {}", r["hr_bpm"], r["rule_off"], r["risk_off"] or 0, r["rule_on"])
    return res


def run_v1_ranges(cfg, ds) -> dict:
    """R16 for v1 (docs/final_switch.md): ranges from v1's frozen training meals, sides from v1's coefficients,
    counts on v1's development and locked answers, and the same heart-rate sweep with a real meal photo."""
    from glimpse.data.splits import meal_rows
    from glimpse.serve import MealInput, Pipeline
    from glimpse.serve.rules import outside_training_range
    from glimpse.training.baselines import trend_features
    from glimpse.training.final import FINAL

    path = Path(cfg.prototype.v1_input_ranges)
    bundle = joblib.load(FINAL)
    if not path.exists():
        x = trend_features(ds, np.array(sorted(meal_rows(ds, set(bundle["train_people"]), scored_only=False))))
        w = bundle["models"]["photo"][-1].coef_[0]
        path.write_text(json.dumps({
            "what": "rule R16 for v1: min and max of each trend and profile input over the frozen v1 fusion's "
                    "training meals (docs/final_switch.md); the rule checks the 7 sensor inputs, on the side where "
                    "an extreme value lowers v1's risk; written once", "training_meals": int(len(x)),
            "ranges": {n: [float(np.nanmin(x[:, j])), float(np.nanmax(x[:, j]))] for j, n in enumerate(V2_INPUTS)},
            "risk_lowering_side": {V2_INPUTS[j]: ("above" if w[j] < 0 else "below") for j in range(7)}},
            indent=2))  # fmt: skip
    rj = json.loads(path.read_text())
    counts = {}
    for name, f in (("development", "outputs/evaluation_v2/dev_answers.csv"),
                    ("locked", "outputs/locked_eval/locked_answers.csv")):  # fmt: skip
        d = pd.read_csv(f)
        x = trend_features(ds, d.row.to_numpy())
        hit = [outside_training_range(x[i], rj["ranges"], rj["risk_lowering_side"]) is not None for i in range(len(d))]
        counts[name] = {"meals": int(len(d)), "outside": int(sum(hit)),
                        "decisions_of_those": d.decision[hit].value_counts().to_dict()}  # fmt: skip
    app = Pipeline(cfg, sensor_first=False)
    meal = datetime(2024, 6, 1, 12, 0)
    times = [meal - timedelta(minutes=5 * (24 - i)) for i in range(24)]
    trace = [110.0 + 6.0 * np.sin(i / 3.0) for i in range(24)]
    photo = Path("samples/instructor/00_clean/photo.jpg")
    rows = []
    for hr in range(40, 201, 10):
        m = MealInput(meal_time=meal, glucose_times=times, glucose_mgdl=trace, hr_times=times,
                      hr_bpm=[float(hr)] * 24, profile=PROFILE, photo=photo)  # fmt: skip
        keep = app.input_ranges
        app.input_ranges = None
        off = app.answer(m)
        app.input_ranges = keep
        on = app.answer(m)
        rows.append({"hr_bpm": hr, "rule_off": off.decision, "risk_off": off.risk, "rule_on": on.decision})
    res = {"what": "rule R16 for v1 (docs/final_switch.md)", "heart_rate_range": rj["ranges"]["heart rate"],
           "outside": counts, "sweep": rows}  # fmt: skip
    Path("outputs/probes/v1_hr_sweep.json").write_text(json.dumps(res, indent=2))
    logger.bind(stage="v1-input-ranges").info("outside: {} | sweep on: {}", counts,
                                               [(r["hr_bpm"], r["rule_off"], r["rule_on"]) for r in rows])  # fmt: skip
    return res
