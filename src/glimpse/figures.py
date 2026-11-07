"""Report figures, rendered ONLY from committed evidence files (`just figures`). Pre-commitment in
docs/final_design.md (7 Nov): figures visualise shipped results; nothing is retuned from their look.
F1 system diagram - F2 per-fold AUROC - F3 reliability + ECE - F4 effect sizes - F5 robustness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from glimpse.evaluation.metrics import ece_equal_mass

OUT = Path("outputs/figures")
BLUE, ORANGE, GREY, RED, GREEN = "#2b6cb0", "#dd6b20", "#718096", "#c53030", "#2f855a"


def _style(plt):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})  # fmt: skip


def f1_system(plt) -> None:
    """The one scoring path: inputs -> rules -> arms -> fusion -> calibration -> decision."""
    fig, ax = plt.subplots(figsize=(11.2, 3.9))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#edf2f7"):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=GREY, lw=1.0, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6, zorder=3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.1})  # fmt: skip

    box(0.0, 2.6, 1.9, 1.0, "meal photo")
    box(0.0, 1.3, 1.9, 1.0, "CGM + HR\n(2 h window)")
    box(0.0, 0.0, 1.9, 1.0, "profile (4 fields)")
    box(2.5, 1.3, 2.4, 1.0, "input rules R1\u2013R10:\nrefuse + name\nthe problem")
    box(5.5, 2.6, 2.8, 1.0, "ConvNeXt-Tiny (pinned;\nlast stage tuned) + 2 heads")
    box(5.5, 1.3, 2.8, 1.0, "engineered trend features")
    box(5.5, 0.0, 2.8, 1.0, "kNN unfamiliar-photo\ncheck (cut 0.614)")
    box(8.9, 1.3, 2.6, 1.0, "logistic fusion (20 par.)\nPlatt (0.96, \u22120.20)\npersonal intercept, m = 1.0")
    box(12.1, 1.3, 2.6, 1.0, "cut-offs (0.15, 0.16)\nlow risk / caution /\ncan't judge + reason")
    for y0 in (3.1, 1.8, 0.5):
        arrow(1.9, y0, 2.5, 1.8)
    arrow(4.9, 1.8, 5.5, 1.8)
    arrow(4.9, 2.05, 5.5, 3.1)
    arrow(4.9, 1.55, 5.5, 0.5)
    arrow(8.3, 3.1, 8.9, 2.05)
    arrow(8.3, 1.8, 8.9, 1.8)
    arrow(8.3, 0.5, 8.9, 1.55)
    arrow(11.5, 1.8, 12.1, 1.8)
    ax.text(7.35, 3.95, "one scoring path: report numbers = demo = app = the grader's folder command",
            ha="center", fontsize=8.8, style="italic", color=GREY)  # fmt: skip
    ax.set_xlim(-0.2, 15.0)
    ax.set_ylim(-0.2, 4.2)
    fig.savefig(OUT / "f1_system.png")
    plt.close(fig)


def f2_per_fold(plt, ev: dict) -> None:
    """Raw vs personalised AUROC per fold, plus the pooled estimates with person-clustered CIs."""
    folds = ev["per_fold_auroc"]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    xs = np.arange(len(folds))
    for i, f in enumerate(folds):
        ax.plot([i, i], [f["auroc_raw"], f["auroc_personal"]], color=GREY, lw=1, zorder=1)
    ax.scatter(xs, [f["auroc_raw"] for f in folds], color=GREY, label="raw fusion", zorder=2)
    ax.scatter(xs, [f["auroc_personal"] for f in folds], color=BLUE, label="personalised (shipped)", zorder=2)
    for name, key, x, c in (("pooled", "risk_personal", 5.0, BLUE),
                            ("pooled\nexcl. fold 1", "risk_personal_excluding_fold1", 5.8, ORANGE)):  # fmt: skip
        a = ev[key]["auroc"]
        ax.errorbar([x], [a["point"]], yerr=[[a["point"] - a["ci95"][0]], [a["ci95"][1] - a["point"]]],
                    fmt="o", color=c, capsize=3)  # fmt: skip
        ax.text(x, a["ci95"][1] + 0.012, name, ha="center", fontsize=7.2, color=c)
    ax.axhline(ev["risk_raw"]["auroc"]["point"], color=GREY, lw=0.8, ls=":")
    ax.text(5.4, ev["risk_raw"]["auroc"]["point"] + 0.004, "pooled raw 0.726", fontsize=7.5, color=GREY)
    ax.set_xticks(xs, [f"fold {f['fold']}\nn={f['n']}" for f in folds], fontsize=7.5)
    ax.set_ylabel("AUROC (held-out people)")
    ax.set_ylim(0.58, 0.9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Personalisation improves every fold (5/5); pooled CIs are person-clustered", fontsize=9)
    fig.savefig(OUT / "f2_per_fold_auroc.png")
    plt.close(fig)


def f3_reliability(plt, da: pd.DataFrame) -> dict:
    """Equal-mass reliability for raw vs shipped risks over the answered development meals, with ECE."""
    d = da.dropna(subset=["risk"])
    y = d.spike.to_numpy().astype(float)
    out = {}
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls="--")
    for col, label, c in (("risk_raw", "raw fusion", GREY), ("risk", "personalised (shipped)", BLUE)):
        p = d[col].to_numpy().astype(float)
        order = np.argsort(p)
        mids, obs = [], []
        for chunk in np.array_split(order, 10):
            mids.append(p[chunk].mean())
            obs.append(y[chunk].mean())
        ece = ece_equal_mass(p, y)
        out[col] = ece
        ax.plot(mids, obs, "o-", color=c, label=f"{label}  (ECE {ece:.3f})", ms=4)
    ax.set_xlabel("predicted spike probability (10 equal-mass bins)")
    ax.set_ylabel("observed spike rate")
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 0.9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Reliability on out-of-fold development meals", fontsize=9)
    fig.savefig(OUT / "f3_reliability.png")
    plt.close(fig)
    return out


def f4_effects(plt, fu: dict, ch: dict, d8: dict) -> None:
    """Every headline effect with its 95% interval on one axis; the zero line tells the story."""
    g = fu["gains_vs_no_meal"]
    rows = [
        ("photo gain (gate, photo meals)", fu["primary_photo_meals_only"]["point"],
         fu["primary_photo_meals_only"]["ci95"], BLUE),
        ("shuffled-photo control", g["photo_shuffled"]["point"], g["photo_shuffled"]["ci95"], GREY),
        ("true logged carbs (RQ1 ceiling)", fu["rq1_photo_vs_logged"]["point"],
         fu["rq1_photo_vs_logged"]["ci95"], GREY),
        ("D8: profile-matched meals removed", d8["gain_photo_meals_only"]["point"],
         d8["gain_photo_meals_only"]["ci95"], RED),
        ("D8 control: random drop", d8["control_random_drop_gain"]["point"],
         d8["control_random_drop_gain"]["ci95"], GREEN),
        ("drop heart rate", ch["delta_vs_full"]["no_heart_rate"]["point"],
         ch["delta_vs_full"]["no_heart_rate"]["ci95"], GREY),
        ("drop profile", ch["delta_vs_full"]["no_profile"]["point"],
         ch["delta_vs_full"]["no_profile"]["ci95"], RED),
    ]  # fmt: skip
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ys = np.arange(len(rows))[::-1]
    for yv, (_label, pt, ci, c) in zip(ys, rows, strict=True):
        ax.plot(ci, [yv, yv], color=c, lw=1.4)
        ax.plot([pt], [yv], "o", color=c, ms=5)
    ax.axvline(0, color=GREY, lw=0.8, ls="--")
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=8)
    ax.set_xlabel("AUROC change (95% person-clustered interval)")
    ax.set_title("Effect sizes: the profile carries the system; the photo is not established", fontsize=9)
    fig.savefig(OUT / "f4_effects.png")
    plt.close(fig)


def f5_robustness(plt, rb: dict) -> None:
    """Decision behaviour under the four pre-registered perturbations: refusal, not silent change."""
    kinds = ["cgm_dropout", "cgm_gap", "photo_blur", "photo_dark"]
    labels = ["CGM dropout 30%", "25-min pre-meal gap", "photo blur r=6", "photo darken ×0.25"]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    xs = np.arange(len(kinds))
    unchanged = [rb[k]["unchanged"] for k in kinds]
    newcj = [rb[k]["newly_cant_judge"] for k in kinds]
    danger = [rb[k]["dangerous_flips"] for k in kinds]
    other = [max(0.0, 1 - u - n - d) for u, n, d in zip(unchanged, newcj, danger, strict=True)]
    bottom = np.zeros(len(kinds))
    for vals, label, c in ((unchanged, "unchanged", "#a0c4e8"), (newcj, "newly can't judge (refusal)", BLUE),
                           (other, "other benign flips", GREY), (danger, "dangerous flips", RED)):  # fmt: skip
        ax.bar(xs, vals, bottom=bottom, label=label, color=c, width=0.62)
        bottom += np.array(vals)
    for i, d in enumerate(danger):
        ax.text(i, 1.03, f"dangerous: {d:.1%}", ha="center", fontsize=7.5,
                color=RED if d else GREEN)  # fmt: skip
    ax.set_xticks(xs, labels, fontsize=7.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("share of meals")
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.set_title("Degraded input triggers refusal, not a silently changed answer", fontsize=9)
    fig.savefig(OUT / "f5_robustness.png")
    plt.close(fig)


def run_figures(cfg=None) -> dict:
    """Render F1-F5 from committed evidence; write the ECE numbers next to them."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    OUT.mkdir(parents=True, exist_ok=True)
    ev = json.loads(Path("outputs/evaluation_v2/dev_summary.json").read_text())
    da = pd.read_csv("outputs/evaluation_v2/dev_answers.csv")
    fu = json.loads(Path("outputs/fusion/fusion_summary.json").read_text())
    ch = json.loads(Path("outputs/ablations/channels.json").read_text())
    d8 = json.loads(Path("outputs/ablations/d8_profile_excluded.json").read_text())
    rb = json.loads(Path("outputs/robustness/robustness.json").read_text())
    f1_system(plt)
    f2_per_fold(plt, ev)
    ece = f3_reliability(plt, da)
    f4_effects(plt, fu, ch, d8)
    f5_robustness(plt, rb)
    report = {"what": "report figures rendered from committed evidence only (pre-commitment in "
                      "docs/final_design.md); ECE = equal-mass, 15 bins, answered dev meals",
              "ece_raw": ece["risk_raw"], "ece_personal": ece["risk"],
              "files": sorted(p.name for p in OUT.glob("*.png"))}  # fmt: skip
    (OUT / "figures.json").write_text(json.dumps(report, indent=2))
    logger.bind(stage="figures").info("5 figures -> {}; ECE raw {:.3f} -> personal {:.3f}",
                                      OUT, ece["risk_raw"], ece["risk"])  # fmt: skip
    return report
