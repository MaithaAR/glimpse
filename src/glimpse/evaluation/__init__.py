"""Evaluation: metrics over predictions and decisions. The decision-MAKING functions (decide,
threshold selection) live in glimpse.training.thresholds -- selection is a training-time activity;
this package only measures. Report: §3 evaluation and ablations."""

from glimpse.evaluation.metrics import (
    apply_platt,
    calibration_by_person,
    coverage_report,
    ece_equal_mass,
    expected_cost,
    fit_platt,
    load_locked_set,
    noninferiority,
    paired_person_bootstrap,
    participant_weighted_cqr_quantile,
    person_bootstrap_ci,
    person_folds,
    reliability,
    status,
    usefulness_report,
)

__all__ = [
    "apply_platt",
    "calibration_by_person",
    "coverage_report",
    "ece_equal_mass",
    "expected_cost",
    "fit_platt",
    "load_locked_set",
    "noninferiority",
    "paired_person_bootstrap",
    "participant_weighted_cqr_quantile",
    "person_bootstrap_ci",
    "person_folds",
    "reliability",
    "status",
    "usefulness_report",
]
