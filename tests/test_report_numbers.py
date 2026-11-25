"""The report's numbers against their evidence files (26 Nov). Runs on a clean checkout: it reads only
committed outputs, so a grader's CI catches any report number that drifts from the file it cites."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPORT = Path("docs/final_report.md")
RM = Path("outputs/report_metrics/report_metrics.json")


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


@pytest.fixture(scope="module")
def text() -> str:
    return REPORT.read_text()


@pytest.fixture(scope="module")
def rm() -> dict:
    return json.loads(RM.read_text())


def _f3(x: float) -> str:
    return f"{x:.3f}"


def test_word_cap_under_the_rubric_rule(text):
    """<= 3,000 words, tables, figures and their captions excluded (the `just wordcount` rule)."""
    kept = [ln for ln in text.splitlines()
            if not (ln.lstrip().startswith(("|", "![")) or (ln.lstrip().startswith("*")
                                                           and not ln.lstrip().startswith("**")))]  # fmt: skip
    assert len(" ".join(kept).split()) <= 3000


def test_no_em_dashes(text):
    assert "—" not in text


def test_dev_and_locked_headlines(text):
    dev = _load("outputs/evaluation_v2/dev_summary.json")
    lk = _load("outputs/locked_eval/locked_summary.json")
    for x in (dev["risk_personal"]["auroc"]["point"], dev["risk_raw"]["auroc"]["point"],
              lk["risk_personal"]["auroc"]["point"], lk["risk_raw"]["auroc"]["point"],
              dev["risk_personal"]["log_loss"], lk["risk_personal"]["log_loss"]):  # fmt: skip
        assert _f3(x) in text, x
    lo, hi = lk["risk_personal"]["auroc"]["ci95"]
    assert f"({lo:.3f}–{hi:.3f})" in text
    assert f"{lk['usefulness']['cost_reduction']:.1%}".replace("-", "−") in text
    d = lk["decisions"]
    assert f"{d['low risk']} low risk,\n{d['caution']} caution, {d["can't judge"]} can't judge" in text


def test_report_metrics_agree_with_the_earlier_evidence(rm):
    """The new descriptive file re-derives the committed numbers exactly before adding its own."""
    fig = _load("outputs/figures/figures.json")
    lk = _load("outputs/locked_eval/locked_summary.json")
    dev = rm["cohorts"]["dev"]
    assert dev["risk"]["ece_equal_mass_15"] == pytest.approx(fig["ece_personal"], abs=1e-12)
    assert dev["risk_raw"]["ece_equal_mass_15"] == pytest.approx(fig["ece_raw"], abs=1e-12)
    assert dev["risk"]["ece_equal_width_15"] == pytest.approx(fig["ece_equal_width_15"]["personal"], abs=1e-12)
    loc = rm["cohorts"]["locked"]
    assert loc["risk"]["auroc"] == pytest.approx(lk["risk_personal"]["auroc"]["point"], abs=1e-12)
    assert loc["risk"]["ece_equal_mass_15"] == pytest.approx(lk["risk_personal"]["ece_equal_mass"]["bins15"])
    assert loc["decisions"]["cost_reduction_vs_best_constant"] == pytest.approx(lk["usefulness"]["cost_reduction"])
    assert sum(b["n"] for b in dev["risk"]["bins_equal_width_15"]) == dev["risk"]["n"] == 460


def test_new_metrics_in_the_report(text, rm):
    for cohort in ("dev", "locked"):
        r = rm["cohorts"][cohort]["risk"]
        for x in (r["auprc"]["point"], r["brier"]["point"], r["ece_equal_width_15"], r["ece_equal_mass_15"],
                  r["base_rate"], r["brier_constant_base_rate"]):  # fmt: skip
            assert _f3(x) in text, (cohort, x)
    rq2 = rm["baselines"]["rq2_partition_v2"]
    assert f"+{rq2['sensor_cnn_minus_trend_lr']:.3f} ({rq2['ci95'][0]:.3f}, +{rq2['ci95'][1]:.3f})".replace(
        "-", "−") in text  # fmt: skip
    for m, x in rm["baselines"]["answered_460"].items():
        if m != "trend_gbm":
            assert _f3(x["auroc"]) in text, m


def test_monitoring_replay_matches_its_registration(text):
    from glimpse.studies import monitoring_replay as mr

    doc = Path("docs/monitoring.md").read_text()
    assert (mr.BLOCK, mr.WINDOW, mr.MIN_CLEARED) == (20, 4, 10)
    assert "block of 20" in doc and "four consecutive blocks (80 answers)" in doc and "at\n  least 10" in doc
    rep = _load("outputs/monitoring/replay.json")["summary"]
    assert f"{rep['dev']['outcome_alerts']} of {rep['dev']['windows']} ordinary dev windows" in text
    assert rep["locked"]["outcome_too_few_cleared"] == rep["locked"]["windows"]


def test_stress_table_and_figure_share_one_run(text):
    """Figure 5 draws levels.json, the same run as the table (the 25 Nov grades found two runs shown as one)."""
    src = Path("src/glimpse/figures.py").read_text()
    assert "outputs/robustness/levels.json" in src and 'robustness.json").read_text' not in src
    lv = _load("outputs/robustness/levels.json")
    assert f"{lv['photo_blur']['6']['dangerous_flips']:.1%}" in text
    worst = max(lv[k][lev]["dangerous_flips"] for k in lv["levels"] for lev in lv[k])
    assert worst < 0.018
    hr = _load("outputs/promised/promised.json")["r6_missing_inputs"]["missing_hr"]["flips"]["dangerous_flips"]
    assert f"{hr:.1%}" in text


def test_every_cited_output_path_exists(text):
    for p in set(re.findall(r"outputs/[\w./-]+[\w/]", text)):
        assert Path(p.rstrip("/")).exists(), p


def _tables(text: str) -> dict[str, list[str]]:
    """Every markdown table row in the report, keyed by its first cell (bold stripped)."""
    rows = {}
    for ln in text.splitlines():
        if ln.startswith("|") and not ln.startswith("|---"):
            cells = [c.strip().replace("**", "") for c in ln.strip().strip("|").split("|")]
            rows[cells[0]] = cells[1:]
    return rows


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%".replace("-", "−")


def test_results_table_cells(text, rm):
    t = _tables(text)
    dev, lk = rm["cohorts"]["dev"]["risk"], rm["cohorts"]["locked"]["risk"]
    row = t["dev + personal intercept (shipped)"]
    a, b = dev["auprc"], dev["brier"]
    assert row[1] == f"{a['point']:.3f} ({a['ci95'][0]:.3f}–{a['ci95'][1]:.3f})"
    assert row[2] == f"{b['point']:.3f} ({b['ci95'][0]:.3f}–{b['ci95'][1]:.3f})"
    assert row[4] == f"{dev['ece_equal_width_15']:.3f} / {dev['ece_equal_mass_15']:.3f}"
    row = t["locked shipped (150 of 153)"]
    assert row[0].startswith(f"{lk['auroc']:.3f}")
    assert row[4] == f"{lk['ece_equal_width_15']:.3f} / {lk['ece_equal_mass_15']:.3f}"
    pg = rm["paired_personal_minus_raw"]
    d = pg["dev"]["auroc"]
    assert t["AUROC"][0].startswith(f"+{d['point']:.3f} ({d['ci95'][0]:.3f}, +{d['ci95'][1]:.3f})".replace("-", "−"))
    assert f"{pg['dev']['folds_improved']} of 5 folds" in t["AUROC"][0]
    ll = pg["dev"]["log_loss"]
    assert t["log-loss"][0] == f"{ll['point']:.3f} ({ll['ci95'][0]:.3f}, {ll['ci95'][1]:.3f})".replace("-", "−")


def test_cost_grid_and_usefulness_cells(text, rm):
    t = _tables(text)
    sweep = {
        (r["missed_spike"], r["abstain_multiplier"]): r
        for r in _load("outputs/evaluation_v2/dev_summary.json")["cost_sweep"]
    }
    for label, ms in (("3×", 3.0), ("5× (declared)", 5.0), ("7×", 7.0), ("10×", 10.0)):
        for cell, am in zip(t[label], (0.5, 1.0, 2.0, 4.0), strict=True):
            r = sweep[(ms, am)]
            assert (
                cell == f"{_pct(r['cost_reduction'])} · {_pct(r['share_low_risk'])} · {_pct(r['missed_among_cleared'])}"
            )
    dec = {k: rm["cohorts"][k]["decisions"] for k in ("dev", "locked")}
    row = t["cost reduction vs the best constant policy (always caution, both cohorts)"]
    ci = rm["band_and_served_calibration"]["cost_reduction_ci95"]

    def withci(x, k):
        lo, hi = ci[k]
        return f"{_pct(x)} ({lo * 100:+.1f}, {hi * 100:+.1f})".replace("-", "−")

    assert row[1:3] == [withci(dec[k]["cost_reduction_vs_best_constant"], k) for k in ("dev", "locked")]
    row = t["mean cost per meal: shipped / always caution / always low risk"]
    for cell, k in zip(row[1:3], ("dev", "locked"), strict=True):
        assert cell == " / ".join(f"{dec[k][p]:.3f}" for p in ("shipped", "always caution", "always low risk"))
    er = rm["errors"]
    assert t["people with a cleared meal"][1] == f"{er['dev']['people_with_a_clear']} of {er['dev']['people']}"
    d, lk = er["dev"], er["locked"]
    want = f"{d['false_cautions']} of {d['cautions']}; {lk['false_cautions']} of {lk['cautions']}"
    assert t["false caution (caution, no spike)"][0] == want


def test_stress_table_cells(text):
    t = _tables(text)
    lv = _load("outputs/robustness/levels.json")

    def u(x):
        return f"{x * 100:.1f}%" if 0.99 < x < 1 else f"{x * 100:.0f}%"

    for label, kind, second in (
        ("CGM dropout 10/30/50% (same decision / low risk↔caution)", "cgm_dropout", "dangerous_flips"),
        ("photo blur r=2/6/10 (same decision / low risk↔caution)", "photo_blur", "dangerous_flips"),
    ):
        for cell, level in zip(t[label], lv["levels"][kind], strict=True):
            c = lv[kind][f"{level:g}"]
            assert cell == f"{u(c['unchanged'])} / {c[second] * 100:.1f}%", (label, cell)


def test_parameter_table_cells(text, rm):
    t = _tables(text)
    ft = rm["fine_tuned_parameters"]["fine_tuned_total"]
    assert t["fine-tuned, deployed (last ConvNeXt stage + heads)"][0] == f"{ft / 1e6:.1f}M"


def test_served_point_and_calibration_cells(text, rm):
    t = _tables(text)
    so = rm["served_operating_point"]
    row = t["cost reduction vs the best constant policy (always caution, both cohorts)"]
    ci = rm["band_and_served_calibration"]["cost_reduction_ci95"]
    cells = [f"{_pct(so[k]['cost_reduction'])} ({ci['served_' + k][0] * 100:+.1f}, {ci['served_' + k][1] * 100:+.1f})"
             for k in ("dev", "locked")]  # fmt: skip
    assert row[3] == " / ".join(cells).replace("-", "−")
    row = t["missed among cleared"]
    d, lk = so["dev"], so["locked"]
    assert row[3] == (f"{_pct(d['missed_among_cleared'])} ({d['n_missed']} of {d['n_cleared']}) / "
                      f"{_pct(lk['missed_among_cleared'])} ({lk['n_missed']} of {lk['n_cleared']})")  # fmt: skip
    fl = rm["ece_noise_floor"]
    row = t["ECE a perfectly calibrated model shows at this size: median, 95th percentile"]
    assert row[:2] == [f"{fl[k]['perfectly_calibrated_median']:.3f}, {fl[k]['perfectly_calibrated_q95']:.3f}"
                       for k in ("dev", "locked")]  # fmt: skip
    assert t["ECE, registered 15 equal-width bins (target ≤ 0.05)"][0] == f"{fl['dev']['observed']:.3f}"
    cil = rm["calibration_in_the_large_pooled"]
    assert t["calibration-in-the-large (mean risk − spike rate)"][:2] == [f"{cil[k]:.3f}".replace("-", "−")
                                                                         for k in ("dev", "locked")]  # fmt: skip
    po = rm["person_offsets"]

    def s(x):
        return f"{x:+.2f}".replace("-", "−")

    assert t["per-person offset b, logit: median (10th, 90th percentile)"][:2] == [
        f"{s(po[k]['median'])} ({s(po[k]['q10'])}, {s(po[k]['q90'])})" for k in ("dev", "locked")]  # fmt: skip


def test_monitoring_and_coverage_claims(text, rm):
    rep = _load("outputs/monitoring/replay.json")
    t = _tables(text)
    assert (
        t["2a, deployed cut-offs on the evaluated (with-history) risk"][1]
        == f"{rep['deployed_operating_point']['summary']['dev']['outcome_alerts']} of 21 dev windows"
    )
    assert rm["heart_rate_r2_from_other_inputs"] == pytest.approx(0.19, abs=0.005)
    cr = {r["abstain_share"]: r for r in rm["coverage_risk"]["dev"]}
    miss = [cr[a]["missed_among_cleared"] for a in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)]
    assert f"{min(miss) * 100:.1f}–{max(miss) * 100:.1f}%" in text
    assert f"from {cr[0.0]['cleared_share'] * 100:.0f}% to {cr[0.3]['cleared_share'] * 100:.0f}%" in text


def test_fair_baseline_and_band_cells(text, rm):
    t = _tables(text)
    fb = rm["fair_baselines"]
    assert fb["reproduces_shipped_max_abs_diff"] < 1e-12

    def d(x):
        return f"{x:+.3f}".replace("-", "−")

    for label, key in (("profile only", "profile_lr"), ("trend + profile LR", "trend_lr"),
                       ("gradient boosting on trend + profile", "trend_gbm"),
                       ("SensorCNN, 3 seeds averaged", "sensor_cnn_3seed_mean")):  # fmt: skip
        r = fb[key]
        assert t[label] == [f"{r['auroc_with_intercept']:.3f}",
                            f"{d(r['shipped_minus_this'])} ({d(r['ci95'][0])}, {d(r['ci95'][1])})"], label  # fmt: skip
    b = rm["band_and_served_calibration"]["band"]
    assert f"{b['cost_without_band']:.3f} against {b['cost_shipped']:.3f}" in text
    assert f"spiked at {b['borderline_spike_rate'] * 100:.0f}%" in text
    sc = rm["band_and_served_calibration"]["served_calibration"]
    row = t["ECE, registered 15 equal-width bins (target ≤ 0.05)"]
    assert row[2] == f"{sc['dev']['ece_equal_width_15']:.3f} / {sc['locked']['ece_equal_width_15']:.3f}"


def test_status_table_timing_cusum_and_what_if(text, rm):
    t = _tables(text)
    lk = _load("outputs/locked_eval/locked_summary.json")
    assert t["AUROC, shipped (95% person-clustered)"][1].startswith(f"{lk['risk_personal']['auroc']['point']:.3f}")
    gb = rm["fair_baselines"]["trend_gbm"]
    assert t["same intercept on gradient boosting (sensors + profile)"][0].startswith(
        f"{gb['auroc_with_intercept']:.3f}"
    )
    st = _load("outputs/efficiency/stage_timing.json")
    row = t["whole photo path / no-photo path (sensors + profile only)"][0]
    assert row == f"{st['photo_path_median_ms']:.1f} / {st['no_photo_path_median_ms']:.2f}"
    assert t["backbone's share of the photo path"] == [f"{st['backbone_share_of_photo_path']:.0%}"]
    assert f"backbone is {st['backbone_share_of_photo_path']:.0%} of the photo path's time" in text
    cu = _load("outputs/monitoring/replay.json")["cusum_served"]
    assert cu["calibration"]["alarms"] == [] and cu["caution_share"]["dev_false_alarms"] == 0
    week = cu["caution_share"]["first_locked_alarm_week"] - cu["caution_share"]["first_locked_week"] + 1
    assert f"one alarm, {week}th locked week" in text
    assert cu["caution_share_deployed_cutoffs"]["alarms"] == [] and cu["caution_share_served"]["alarms"] == []
    np_ = rm["fair_baselines"]["no-photo fusion (the gate's no-meal arm)"]
    assert t["fusion without the photo (the gate's no-meal arm)"] == [
        f"{np_['auroc_with_intercept']:.3f}",
        f"+{np_['shipped_minus_this']:.3f} ({np_['ci95'][0]:+.3f}, {np_['ci95'][1]:+.3f})".replace("-", "−"),
    ]
    sh = rm["share_cis"]
    for row, key in (("cleared as low risk", "cleared"), ("can't judge", "cant_judge")):
        for j, coh in ((1, "dev"), (2, "locked")):
            c = sh[coh][key]
            want = f"{c['point'] * 100:.1f}% ({c['ci95'][0] * 100:.1f}–{c['ci95'][1] * 100:.1f}%)"
            assert t[row][j] == want, (row, coh)
    cs, fw = rm["cold_start_what_if"], rm["falling_now_what_if"]
    d0, d3, l0, l3 = cs["dev"]["0"], cs["dev"]["3"], cs["locked"]["0"], cs["locked"]["3"]

    def cell(c: dict) -> str:
        return f"{c['cleared']}, {c['missed']}, {_pct(c['cost_reduction'])}"

    assert t["as shipped"] == [cell(d0), cell(l0)]
    assert t["cold-start hold: no clear before the person's third labelled meal"] == [cell(d3), cell(l3)]
    fl = {"cleared": fw["cleared"] - fw["would_flip"], "missed": fw["misses_total"] - fw["misses_avoided"],
          "cost_reduction": fw["cost_reduction"]}  # fmt: skip
    assert t['"falling now": never clear a meal in the steepest-falling tenth of all meals'] == [cell(fl), "–"]
    sp = rm["served_imputed_priors"]
    row = t["no history and no onboarding priors (the samples' path; dev, approximate)"][2]
    assert row == (f"AUROC {sp['auroc']:.3f}; {sp['share_low_risk'] * 100:.1f}% cleared, {sp['n_missed']} of "
                   f"{sp['n_cleared']} spike; cost {_pct(sp['cost_reduction'])} / –; with rule R13: 0 cleared, "
                   f"{_pct(sp['with_rule_r13']['cost_reduction'])}")  # fmt: skip
    pc = _load("outputs/probes/probe_c.json")
    assert all(v["expected_direction_met"] for v in pc["inputs"].values())
    ip = pc["imputed_priors"]
    assert f"in-sample clears from {ip['cleared_own_priors']} to {ip['cleared_imputed']}" in text.replace("\n", " ")
    ef = rm["ece_noise_floor"]
    assert t["share of perfectly calibrated draws scoring at or above the observed ECE"] == [
        f"{ef['dev']['share_of_perfectly_calibrated_draws_at_or_above_observed']:.0%}",
        f"{ef['locked']['share_of_perfectly_calibrated_draws_at_or_above_observed']:.0%}",
        "–",
    ]


def test_subgroup_table(text, rm):
    t = _tables(text)
    for g in ("healthy", "prediabetes"):
        r = rm["subgroups"][g]
        lo, hi = r["auroc_ci95"]
        assert t[g] == [
            f"{r['people']}, {r['meals']}, {r['spike_rate'] * 100:.1f}%",
            f"{r['auroc']:.3f} ({lo:.3f}–{hi:.3f})",
            f"{r['mean_risk_minus_spike_rate']:+.3f}".replace("-", "−"),
            f"{r['cleared_share'] * 100:.1f}% ({r['cleared']}), {r['missed']}, "
            f"{r['people_with_a_clear']} of {r['people']}",
        ], g


def test_round8_rows(text, rm):
    t = _tables(text)
    ex = rm["usefulness_excluding_fold_1"]
    row = t[f"dev excluding fold 1 ({ex['meals']} meals): cleared, missed among them, cost saved"]
    assert (
        row[1] == f"{ex['cleared_share'] * 100:.1f}%, {ex['missed']} of {ex['cleared']}, {_pct(ex['cost_reduction'])}"
    )
    cu = _load("outputs/monitoring/replay.json")["cusum_served"]["caution_share_served"]
    assert cu["alarms"] == []
    assert t["the same on the served (no-history) decisions (added after that result)"][1] == (
        f"no alarm; peak {max(cu['cusum']):.1f} of 4"
    )
    pr = _load("outputs/promised/promised.json")
    r3 = pr["r3_label_sensitivity"]
    want = " / ".join(f"{r3[k]['auroc_personal']:.3f} ({r3[k]['spike_rate'] * 100:.0f}%)"
                      for k in ("spike_30", "spike_50", "spike_70"))  # fmt: skip
    assert t["label threshold 30 / 50 / 70 mg/dL: AUROC (spike rate)"] == [want]
    t5 = rm["t5_photo_and_heart_rate"]
    pb = [v["photo_block"] for v in t5["false_clears"].values() if v["class"].startswith("T5")]
    assert max(pb) < 0 and f"(−{-max(pb):.2f} to −{-min(pb):.2f}" in text
    ok = t5["cleared_without_spike"]
    assert f"equally on {ok['share_photo_block_negative']:.0%} of the cleared meals" in text
    assert f"(median −{-ok['median_photo_block']:.2f})" in text


def test_locked_decisions_follow_the_current_cut_off_rule():
    """The committed locked answers (scored once, 25 Nov) re-decided by today's decision function at the frozen
    cut-offs: identical, so the post-tag serving changes (R12, reason wording) did not alter any locked decision
    rule. Nothing is rescored; no metric is recomputed."""
    import pandas as pd

    from glimpse.serve.rules import decision
    from glimpse.training.final import load_thresholds

    th = load_thresholds()["main"]
    la = pd.read_csv("outputs/locked_eval/locked_answers.csv")
    has = la.risk.notna()
    redecided = [decision(float(r), th["low_risk_below"], th["caution_from"]) for r in la.risk[has]]
    assert redecided == la.decision[has].tolist()


def test_r13_cells(text, rm):
    sp = rm["served_imputed_priors"]["with_rule_r13"]
    assert sp["share_low_risk"] == 0
    assert f"{sp['share_abstain'] * 100:.1f}% can't judge" in text
    assert f"with rule R13: 0 cleared, {_pct(sp['cost_reduction'])}" in text


def test_within_person_rows(text, rm):
    t = _tables(text)
    wp = rm["within_person"]
    mw = wp["mean_within_person_auroc"]
    key = f"mean within-person AUROC ({wp['people_with_both_outcomes']} people with both outcomes), raw → shipped"
    assert t[key][0] == f"{mw['risk_raw']:.3f} → {mw['risk']:.3f}"
    assert f"AUROC falls from {mw['risk_raw']:.3f} to {mw['risk']:.3f}" in text
    h = wp["history_only_running_rate"]
    row = t["no model: each person's running spike rate, same gate, shrunk to the other folds' rate"]
    assert row == [f"{h['auroc']:.3f}", f"+{h['shipped_minus_this']:.3f} (+{h['ci95'][0]:.3f}, +{h['ci95'][1]:.3f})"]
    assert f"does {h['shipped_minus_this']:.3f} better than a" in text


def test_cgm_removal_and_ood_rows(text, rm):
    t = _tables(text)
    c = rm["fair_baselines"]["cgm_removal"]
    assert t["CGM removal: the no-photo arm minus profile only (both with the intercept)"][1] == (
        f"+{c['no_photo_minus_profile_only']:.3f} (+{c['ci95'][0]:.3f}, +{c['ci95'][1]:.3f})"
    )
    mo = _load("outputs/monitoring/monitoring.json")
    n, med, above = mo["real_photos_n"], mo["real_photos_median"], mo["real_photos_above_cut"]
    assert t["real phone photos of local meals"] == [str(n), f"{med:.3f}", f"{above} of {n}"]
    assert t["training photos, leave-one-out (the budget)"][1] == f"{mo['training_loo_median']:.3f}"
    sc = {x["photo"]: x["score"] for x in _load("outputs/ood/ood_summary.json")["scores"] if x["set"] == "instructor"}
    assert t["instructor screenshot / dark photo / local dish"][1] == " / ".join(
        f"{sc[k]:.3f}" for k in ("09_screenshot", "08_dark_photo", "11_local_dish")
    )
