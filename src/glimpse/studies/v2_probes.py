"""v2 probes and input monitors (`glimpse v2-probes`; registered in docs/v2_hardening.md, Part B).

B1: probe B on v2, each development meal replayed with a flat and a rising pre-meal CGM through its fold's v2
pipeline, no history (the v1 probe's edit). B2: probe C on v2, each of the 8 profile inputs moved from its
10th to its 90th centile through the frozen v2 fusion, frozen Platt and cut-offs (data-free). B4: monitors
for v2's weak inputs replayed on the development stream with a watch-off block and a label-error block."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from glimpse.serve.rules import decision

OUT = Path("outputs/probes/v2_probes.json")
HR = Path("outputs/v2_explain/v2_hr_present.csv")
PROFILE_Z = ["age", "BMI", "HbA1c", "fasting glucose", "hour (sin)", "hour (cos)", "prior spike rate", "prior rise"]
BLOCK = 20


def probe_b(ds, sp, cfg) -> dict:
    """B1, needs data and weights: rising against flat pre-meal CGM through the fold's v2 pipeline."""
    from glimpse.demo_cases import meal_input
    from glimpse.studies.probes import flat_cgm, rising_cgm
    from glimpse.studies.stress_quality import _fold_pipelines

    pcv = Path(cfg.paths.outputs) / "photo_cv"
    oof, nested = pd.read_csv(pcv / "photo_oof.csv"), pd.read_csv(pcv / "photo_nested.csv")
    rows, hr = [], []
    for k, held in enumerate(sp.dev_folds):
        tab, _ = _fold_pipelines(ds, sp, cfg, k, True, oof, nested)
        for pid in sorted(held):
            for row in meal_rows_of(ds, pid):
                m = meal_input(cfg, ds, int(row))
                hr.append({"row": int(row), "hr_present": bool(m.hr_times)})
                f, r = tab.answer(flat_cgm(m), key=int(row)), tab.answer(rising_cgm(m), key=int(row))
                if f.risk is not None and r.risk is not None:
                    rows.append({"row": int(row), "flat": f.risk, "rising": r.risk})
    pd.DataFrame(hr).sort_values("row").to_csv(HR, index=False)  # B4 reads this without data
    d = pd.DataFrame(rows)
    move = d.rising - d.flat
    return {"meals": int(len(d)), "share_rising_raises_risk": float((move > 0).mean()),
            "median_move": float(move.median()), "expectation_under_half": bool((move > 0).mean() < 0.5)}  # fmt: skip


def meal_rows_of(ds, pid: int) -> list[int]:
    from glimpse.data.splits import meal_rows

    return sorted(meal_rows(ds, {pid}, scored_only=True), key=lambda r: ds.samples[r].meal_time)


def probe_c(cfg) -> dict:
    """B2, data-free: the frozen v2 fusion, frozen Platt and cut-offs on the committed standardised inputs."""
    from glimpse.evaluation.v2_explain import NAMES

    lr = joblib.load(cfg.prototype.sensor_first_bundle)["models"]["sensor"][-1]
    th = json.loads(Path(cfg.prototype.sensor_first_thresholds).read_text())
    a, b = th["platt"]
    lo, hi = th["main"]["low_risk_below"], th["main"]["caution_from"]
    z = pd.read_csv("outputs/v2_explain/v2_features_z.csv")[NAMES].to_numpy()
    w, c = lr.coef_[0], float(lr.intercept_[0])

    def risk(zz):
        return 1 / (1 + np.exp(-(a * (zz @ w + c) + b)))

    def decide(r):
        return np.array([decision(float(x), lo, hi) for x in r])

    out = {}
    for name in PROFILE_Z:
        j = NAMES.index(name)
        p10, p90 = np.quantile(z[:, j], [0.1, 0.9])
        z10, z90 = z.copy(), z.copy()
        z10[:, j], z90[:, j] = p10, p90
        r10, r90 = risk(z10), risk(z90)
        out[name] = {"coefficient_per_sd": float(w[j]), "median_risk_change": float(np.median(r90 - r10)),
                     "share_decision_differs": float(np.mean(decide(r10) != decide(r90)))}  # fmt: skip
    top = sorted(out, key=lambda n: -out[n]["share_decision_differs"])[:3]
    return {"meals": int(len(z)), "cutoffs": [lo, hi], "inputs": out, "top_three": top}


def monitors(cfg) -> dict:
    """B4, data-free once v2_hr_present.csv is committed: blocks of 20 answers in deployment order."""
    v2 = pd.read_csv("outputs/evaluation_sensor_first/dev_answers.csv")
    hr = pd.read_csv(HR).set_index("row").hr_present
    feat = pd.read_csv("outputs/v2_explain/v2_features_z.csv")[["row", "meal_time"]]
    d = v2.merge(feat, on="row").sort_values("meal_time").reset_index(drop=True)
    d["hr_missing"] = ~d.row.map(hr).astype(bool)
    d["sq"] = (d.risk - d.spike) ** 2
    blocks = [d.iloc[i : i + BLOCK] for i in range(0, len(d) - BLOCK + 1, BLOCK)]
    hr_share = np.array([b.hr_missing.mean() for b in blocks])
    brier = np.array([b.sq.mean() for b in blocks])
    line_hr, line_b = float(np.quantile(hr_share, 0.95)), float(np.quantile(brier, 0.95))
    rng = np.random.default_rng(cfg.seed)
    watch_off = 1.0  # every answer in the block without heart rate
    flips = [rng.uniform(size=BLOCK) < 0.25 for _ in range(200)]
    label_err = [float(np.mean((b.risk.to_numpy() - np.where(f, 1 - b.spike, b.spike)) ** 2))
                 for b, f in zip([blocks[i % len(blocks)] for i in range(200)], flips, strict=True)]  # fmt: skip
    return {"block": BLOCK, "clean_blocks": len(blocks), "hr_missing_line": line_hr, "brier_line": line_b,
            "clean_false_alarms_hr": int((hr_share > line_hr).sum()),
            "clean_false_alarms_brier": int((brier > line_b).sum()),
            "watch_off_block_fires": bool(watch_off > line_hr),
            "label_error_block_fire_rate": float(np.mean(np.array(label_err) > line_b)),
            "clean_hr_missing_share": float(d.hr_missing.mean())}  # fmt: skip


def run_v2_probes(cfg, ds=None, sp=None) -> dict:
    res = {"what": __doc__.splitlines()[0]}
    if ds is not None:
        res["probe_b"] = probe_b(ds, sp, cfg)
    elif OUT.exists():  # data-free rerun keeps the committed probe B
        res["probe_b"] = json.loads(OUT.read_text()).get("probe_b")
    res["probe_c"] = probe_c(cfg)
    res["monitors"] = monitors(cfg)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    logger.bind(stage="v2-probes").info("probe B {} | probe C top {} | monitors {}", res.get("probe_b"),
                                        res["probe_c"]["top_three"], res["monitors"])  # fmt: skip
    return res
