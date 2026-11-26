"""Multiplicity check on the report's paired comparisons (26 Nov, after round-21 grading): Holm over the
development contrasts in report §3, with two-sided p-values approximated from each 95% person-bootstrap
interval (normal approximation, standard error = interval width / 3.92). Reads committed outputs only
(`glimpse holm`) and writes outputs/report_metrics/holm.json."""

from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path("outputs/report_metrics/holm.json")


def _p(point: float, lo: float, hi: float) -> float:
    se = (hi - lo) / 3.92
    return math.erfc(abs(point) / se / math.sqrt(2)) if se > 0 else 0.0


def contrasts() -> dict[str, tuple[float, float, float]]:
    r = json.loads(Path("outputs/report_metrics/report_metrics.json").read_text())
    v = json.loads(Path("outputs/report_metrics/v2_metrics.json").read_text())
    pp, fb, wp = r["paired_personal_minus_raw"]["dev"], r["fair_baselines"], r["within_person"]
    cgm = fb["cgm_removal"]
    out = {"v1 intercept minus raw, AUROC": (pp["auroc"]["point"], *pp["auroc"]["ci95"]),
           "v1 intercept minus raw, log-loss": (pp["log_loss"]["point"], *pp["log_loss"]["ci95"]),
           "v2 minus v1, AUROC": (v["paired_v2_minus_v1"]["v2_minus_v1"], *v["paired_v2_minus_v1"]["ci95"]),
           "v1 minus gradient boosting, both with the intercept": (fb["trend_gbm"]["shipped_minus_this"],
                                                                    *fb["trend_gbm"]["ci95"]),
           "v1 minus a running spike rate": (wp["history_only_running_rate"]["shipped_minus_this"],
                                             *wp["history_only_running_rate"]["ci95"]),
           "CGM removal": (cgm["no_photo_minus_profile_only"], *cgm["ci95"]),
           "within-person AUROC, intercept minus raw (v1)": (wp["shipped_minus_raw_within_person"]["point"],
                                                             *wp["shipped_minus_raw_within_person"]["ci95"]),
           }  # fmt: skip
    return out


def run_holm() -> dict:
    c = contrasts()
    ps = sorted(((_p(*v), k) for k, v in c.items()))
    m, res, running = len(ps), {}, 0.0
    for i, (p, k) in enumerate(ps):
        running = max(running, min(1.0, (m - i) * p))
        res[k] = {"point": c[k][0], "ci95": list(c[k][1:]), "p_approx": p, "p_holm": running,
                  "survives_holm_0.05": running < 0.05}  # fmt: skip
    out = {"what": __doc__.splitlines()[0], "family": m, "contrasts": res}
    OUT.write_text(json.dumps(out, indent=2))
    return out
