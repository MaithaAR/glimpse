"""RQ1 pilot with a REAL photo model (person-excluded).

Photo model: ImageNet EfficientNet-B0 (frozen, 1280-d pooled features, flip TTA) + ridge head predicting
log1p(served macros). Fitted NESTED inside each outer person fold: training rows receive inner out-of-fold
photo predictions (4 inner person folds); test rows receive predictions from a head fitted on all outer-training
people. No person's photo, macros or glucose ever informs their own photo predictions.

Downstream: GBM spike classifier (same protocol as pilot/no_meal_baseline.py) with identical sensor/profile/prior
inputs and meal information = none | reconstructed served macros | photo-predicted served macros.
Scored on isolated target-cohort meals that have a photo (matched across the three models).

Usage: python3 scripts/03_photo_experiment.py --photos data/photos --weights data/efficientnet-b0-355c32eb.pth
Outputs: results/photo_features.npz (cache), results/photo_experiment.json, printed log.
"""
import argparse, json, os, sys, warnings
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from glimpse.data import add_personal_prior, evaluation_set
from glimpse.evaluation import paired_person_bootstrap, noninferiority
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

ap = argparse.ArgumentParser()
ap.add_argument("--photos", default="data/photos"); ap.add_argument("--weights", default="data/efficientnet-b0-355c32eb.pth")
ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--alpha", type=float, default=300.0)
ap.add_argument("--profile-excluded", action="store_true", help="photo head never trains on a logged macro profile that occurs among the rows it predicts")
ap.add_argument("--noise-control", action="store_true", help="negative control: replace every photo by seeded random noise")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config)); os.makedirs("results", exist_ok=True)
MAC = ["carbs", "protein", "fat", "fiber", "kcal"]

# ---------------- meals with a usable photo ----------------
M = pd.read_parquet("data/meals.parquet")
M = M if "prior_rate" in M else add_personal_prior(M, cfg)   # 01_build_meals already adds the prior
M["photo_file"] = [os.path.join(a.photos, f"CGMacros/CGMacros-{s:03d}", str(p).strip()) if isinstance(p, str) else None
                   for s, p in zip(M.sid, M.photo)]
M["has_photo"] = [isinstance(f, str) and os.path.exists(f) for f in M.photo_file]
print(f"meal rows {len(M)} | with photo file {int(M.has_photo.sum())}")

# ---------------- frozen EfficientNet-B0 features (cached) ----------------
cache = "results/photo_features_noise.npz" if a.noise_control else "results/photo_features.npz"
files = sorted(set(M.photo_file[M.has_photo]))
if os.path.exists(cache) and list(np.load(cache, allow_pickle=True)["files"]) == files:
    F = np.load(cache, allow_pickle=True)["X"]
else:
    import torch
    from PIL import Image, ImageOps
    from efficientnet_pytorch import EfficientNet
    torch.set_num_threads(os.cpu_count())
    net = EfficientNet.from_name("efficientnet-b0"); net.load_state_dict(torch.load(a.weights)); net.eval()
    MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    def tf(im):  # resize short side to 256, centre-crop 224, ImageNet normalisation
        w, h = im.size; s = 256 / min(w, h); im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
        w, h = im.size; l, t = (w - 224) // 2, (h - 224) // 2; im = im.crop((l, t, l + 224, t + 224))
        return torch.tensor(((np.asarray(im, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1), dtype=torch.float32)
    F = []
    with torch.no_grad():
        for i in range(0, len(files), 32):
            load = (lambda f: Image.fromarray(np.random.default_rng(int(__import__("zlib").crc32(f.encode()))).integers(0, 256, (300, 400, 3), dtype=np.uint8))) if a.noise_control \
                else (lambda f: ImageOps.exif_transpose(Image.open(f)).convert("RGB"))
            x = torch.stack([tf(load(f)) for f in files[i:i + 32]])
            z = [net._avg_pooling(net.extract_features(v)).flatten(1) for v in (x, torch.flip(x, [3]))]
            F.append(((z[0] + z[1]) / 2).numpy())
            print(f"  features {min(i + 32, len(files))}/{len(files)}", flush=True)
    F = np.concatenate(F); np.savez(cache, X=F, files=np.array(files, dtype=object))
fidx = {f: i for i, f in enumerate(files)}

P = M[M.has_photo].reset_index(drop=True)           # photo-model training pool (all groups, onboarding included)
XP = F[[fidx[f] for f in P.photo_file]]
YP = np.log1p(P[[f"{m}_served" for m in MAC]].clip(lower=0).fillna(0).values)

PKEY = P[["carbs_consumed", "protein_consumed", "fat_consumed"]].round(1).astype(str).agg("|".join, axis=1).values

def fit_predict(train_sids, test_rows_mask):
    tr = P.sid.isin(train_sids).values.copy()
    if a.profile_excluded:  # standardised study meals recur across people: block memorising identical meals
        tr &= ~np.isin(PKEY, PKEY[test_rows_mask])
    sc = StandardScaler().fit(XP[tr]); r = Ridge(alpha=a.alpha).fit(sc.transform(XP[tr]), YP[tr])
    return np.expm1(r.predict(sc.transform(XP[test_rows_mask]))).clip(min=0)

def folds(ids, k, seed):
    u = np.array(sorted(set(ids))); np.random.default_rng(seed).shuffle(u); return [u[i::k] for i in range(k)]

# ---------------- downstream protocol ----------------
E = evaluation_set(M)
E = E[E.has_photo].reset_index(drop=True)
Epos = pd.Series(range(len(P)), index=list(zip(P.sid, P.t)))
e2p = np.array([Epos[(s, t)] for s, t in zip(E.sid, E.t)])
LAB = ["a1c", "fglu", "fins", "BMI", "Age", "Triglycerides"]; TRAJ = ["slope60", "slope20", "sd120", "min120", "since_meal", "hr_slope", "mets60"]
REST = ["base", "hour", "hr30", "prior_rate", "prior_rise"] + LAB + TRAJ
REF = [f"{m}_served" for m in MAC]; PH = [f"{m}_photo" for m in MAC]
y = E.spike50.values; sids = E.sid.values; tgt = E.group.isin(cfg["cohort"]["target_groups"]).values
gbm = lambda: HGB(max_depth=3, learning_rate=.05, max_iter=200)

thr = cfg["label"]["spike_threshold_mgdl"]
ON = M[M.onboarding & M.usable]                                      # onboarding meals end before evaluation meals begin
n_on = ON.groupby("sid").size()
FEW = [s_ for s_ in E.sid.unique() if n_on.get(s_, 0) < cfg["onboarding"]["min_meals_for_personal_prior"]]
print(f"population-prior fallback: {len(FEW)} person(s) {FEW}, {int(E.sid.isin(FEW).sum())} of {len(E)} evaluated meals")
preds = {k: np.zeros((3, len(E))) for k in ["none", "reference", "photo"]}
photo_oof = np.zeros((3, len(E), len(MAC)))
for seed in cfg["cv"]["seeds"]:
    for test_people in folds(sids, 5, seed):
        te = np.isin(sids, test_people); train_people = sorted(set(P.sid) - set(test_people))
        ph = np.zeros((len(E), len(MAC)))
        # test rows: head fitted on all outer-training people
        ph[te] = fit_predict(train_people, np.isin(np.arange(len(P)), e2p[te]))[np.argsort(np.argsort(e2p[te]))]
        # training rows: inner person-OOF predictions (inner folds within outer-training people only)
        for inner in folds(train_people, 4, seed + 100):
            rows = np.isin(sids, inner) & ~te
            if rows.any():
                ph[rows] = fit_predict(sorted(set(train_people) - set(inner)), np.isin(np.arange(len(P)), e2p[rows]))[np.argsort(np.argsort(e2p[rows]))]
        D = E.copy(); D[PH] = ph; photo_oof[seed][te] = ph[te]
        # population-prior fallback recomputed from this fold's TRAINING target people only (no held-out outcomes)
        pool = ON[ON.sid.isin(train_people) & ON.group.isin(cfg["cohort"]["target_groups"])]
        D.loc[D.sid.isin(FEW), "prior_rate"] = pool[f"spike{thr}"].mean()
        D.loc[D.sid.isin(FEW), "prior_rise"] = pool.rise.mean()
        for k, cols in [("none", REST), ("reference", REF + REST), ("photo", PH + REST)]:
            preds[k][seed, te] = gbm().fit(D[cols][~te], y[~te]).predict_proba(D[cols][te])[:, 1]
p = {k: v.mean(0) for k, v in preds.items()}
ph_test = photo_oof.mean(0)

# ---------------- photo-model accuracy (outer-test predictions only) ----------------
def err(mask):
    t, q = E.carbs_served.values[mask], ph_test[mask, 0]
    base = np.array([E.carbs_served[~np.isin(sids, [s])].mean() for s in sids[mask]])  # predict the other people's mean
    from scipy.stats import spearmanr
    return dict(n=int(mask.sum()), people=int(len(set(sids[mask]))), mae=float(np.mean(abs(q - t))), bias=float(np.mean(q - t)),
                median_ae=float(np.median(abs(q - t))), mae_constant_baseline=float(np.mean(abs(base - t))),
                spearman=float(spearmanr(q, t).correlation), mean_reference_carbs=float(t.mean()))
acc = {"target": err(tgt), "all": err(np.ones(len(E), bool))}

yt, st = y[tgt], sids[tgt]
auc = {k: float(roc_auc_score(yt, v[tgt])) for k, v in p.items()}
def ci(a_, b_):
    m, lo, hi = paired_person_bootstrap(yt, a_[tgt], b_[tgt], st, B=2000, seed=0)
    return dict(point=float(roc_auc_score(yt, a_[tgt]) - roc_auc_score(yt, b_[tgt])), boot_mean=m, lo=lo, hi=hi, width=hi - lo)
res = dict(n_meals=int(tgt.sum()), n_people=int(len(set(st))), spike_rate=float(yt.mean()), auroc=auc,
           delta_photo_minus_reference=ci(p["photo"], p["reference"]), gain_photo=ci(p["photo"], p["none"]),
           gain_reference=ci(p["reference"], p["none"]), carb_accuracy=acc,
           notes="Exploratory: includes the 8 locked people; population-prior fallback fit on each fold's training people; frozen backbone + ridge head.")
res["rq1_noninferiority"] = noninferiority(res["delta_photo_minus_reference"]["lo"], res["delta_photo_minus_reference"]["hi"], cfg["targets"]["rq1_noninferiority_margin"])
g = res["gain_photo"]; res["rq1_incremental"] = "positive" if g["lo"] > 0 else ("degradation" if g["hi"] < 0 else "inconclusive")
tag = "_noise_control" if a.noise_control else "_profile_excluded" if a.profile_excluded else ("" if a.alpha == 300 else f"_alpha{int(a.alpha)}")
json.dump(res, open(f"results/photo_experiment{tag}.json", "w"), indent=2)

print(f"\nMatched isolated target-cohort meals with photo: n={res['n_meals']}, people={res['n_people']}, spike rate {res['spike_rate']:.3f}")
for k in ["target", "all"]:
    c = acc[k]
    print(f"Carb accuracy ({k}): n={c['n']} MAE {c['mae']:.1f} g (constant baseline {c['mae_constant_baseline']:.1f} g) | bias {c['bias']:+.1f} g | "
          f"median AE {c['median_ae']:.1f} g | Spearman {c['spearman']:.2f} | mean reference {c['mean_reference_carbs']:.1f} g")
for k in ["none", "reference", "photo"]:
    print(f"AUROC {k:10s} {auc[k]:.3f}")
for k in ["gain_reference", "gain_photo", "delta_photo_minus_reference"]:
    c = res[k]; print(f"{k:28s} {c['point']:+.3f}  95% CI [{c['lo']:+.3f}, {c['hi']:+.3f}]  width {c['width']:.3f}")
print(f"RQ1 non-inferiority (margin 0.03): {res['rq1_noninferiority']} | incremental value of photo: {res['rq1_incremental']}")
