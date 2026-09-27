"""Build the meal table under the prespecified rules and print the accounting used in the proposal.
Usage: python scripts/01_build_meals.py [--config configs/default.yaml]"""
import argparse, sys, yaml
sys.path.insert(0, "src")
from glimpse.data import build_meals, add_personal_prior, evaluation_set

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml"); a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
M = build_meals(cfg["data"]["cgmacros_root"], cfg)
M = add_personal_prior(M, cfg)
E = evaluation_set(M)
tgt = E.group.isin(cfg["cohort"]["target_groups"])
print(f"usable meals {int(M.usable.sum())} | onboarding {int((M.usable & M.onboarding).sum())} | post-onboarding {int((M.usable & M.post_onboarding).sum())} | evaluated {len(E)} ({int(tgt.sum())} target, {E[tgt].sid.nunique()} people)")
print(f"spike rate all {E.spike50.mean():.3f} | target {E[tgt].spike50.mean():.3f}")
M.to_parquet("data/meals.parquet") if __import__("os").path.isdir("data") else M.to_csv("meals.csv", index=False)
