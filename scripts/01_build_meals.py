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

# ---- additional counts cited in the proposal ----
O = M[M.post_onboarding & M.usable & M.rise.notna()]
OT = O[O.group.isin(cfg["cohort"]["target_groups"])]
print(f"operational {len(O)} | target operational {len(OT)} meals, {OT.sid.nunique()} people, spike rate {OT.spike50.mean():.3f}, without photo {int(OT.photo.isna().sum())}")
key = M[["carbs_consumed", "protein_consumed", "fat_consumed"]].round(1).astype(str).agg("|".join, axis=1)
iso_t = E[tgt]
k_iso = iso_t[["carbs_consumed", "protein_consumed", "fat_consumed"]].round(1).astype(str).agg("|".join, axis=1)
people_per_key = key.groupby(key).apply(lambda s: M.loc[s.index, "sid"].nunique())
print(f"isolated target meals sharing an exact macro profile with another person: {(k_iso.map(people_per_key) > 1).mean():.3f} (n={len(iso_t)})")
