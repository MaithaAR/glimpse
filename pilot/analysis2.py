import glob, os, re, numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, HistGradientBoostingRegressor as HGBR
from sklearn.metrics import roc_auc_score, mean_absolute_error
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
R = "/tmp/claude-0/glimpse/cgmacros/CGMacros"
bio = pd.read_csv(f"{R}/bio.csv"); bio.columns = [c.strip() for c in bio.columns]
bio = bio.rename(columns={"subject": "sid", "A1c PDL (Lab)": "a1c", "Fasting GLU - PDL (Lab)": "fglu", "Insulin": "fins"})
bio["group"] = pd.cut(bio.a1c, [0, 5.69, 6.49, 99], labels=["healthy", "prediabetes", "T2D"])
bio = bio[["sid", "a1c", "fglu", "fins", "BMI", "Age", "Triglycerides", "group"]]

def slope(s):
    s = s.dropna()
    if len(s) < 3: return np.nan
    x = (s.index - s.index[0]).total_seconds() / 60
    return np.polyfit(x, s.values, 1)[0]

rows, src = [], {"dexcom": 0, "libre": 0}
for f in sorted(glob.glob(f"{R}/CGMacros-*/CGMacros-*.csv")):
    sid = int(re.findall(r"(\d+)", os.path.basename(f))[-1])
    d = pd.read_csv(f); d["Timestamp"] = pd.to_datetime(d["Timestamp"]); d = d.set_index("Timestamp").sort_index()
    gd = pd.to_numeric(d["Dexcom GL"], errors="coerce").dropna()
    use = "dexcom" if len(gd) > 500 else "libre"
    g = gd if use == "dexcom" else pd.to_numeric(d["Libre GL"], errors="coerce").dropna()
    src[use] += 1
    step = "5min" if use == "dexcom" else "15min"
    g = g.groupby(level=0).mean().resample(step).mean()
    hr = pd.to_numeric(d.get("HR", pd.Series(np.nan, index=d.index)), errors="coerce"); mets = pd.to_numeric(d.get("METs", pd.Series(np.nan, index=d.index)), errors="coerce")
    meals = d[pd.to_numeric(d["Calories"], errors="coerce").fillna(0) > 0]
    mt = list(meals.index)
    for i, t in enumerate(mt):
        pre = g[t - pd.Timedelta("30min"):t].dropna(); post = g[t + pd.Timedelta("1min"):t + pd.Timedelta("120min")]
        if len(pre) == 0 or post.notna().mean() < .7: continue
        r = meals.loc[t]; r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
        prev = [u for u in mt[:i] if u < t]
        rows.append(dict(sid=sid, t=t, sensor=use, base=pre.mean(), rise=post.max() - pre.mean(),
            overlap=any(t < u <= t + pd.Timedelta("120min") for u in mt[i + 1:i + 3]),
            slope60=slope(g[t - pd.Timedelta("60min"):t]), slope20=slope(g[t - pd.Timedelta("20min"):t]),
            sd120=g[t - pd.Timedelta("120min"):t].std(), min120=g[t - pd.Timedelta("120min"):t].min(),
            since_meal=(t - prev[-1]).total_seconds() / 3600 if prev else 12,
            hr30=hr[t - pd.Timedelta("30min"):t].mean(), hr_slope=slope(hr[t - pd.Timedelta("60min"):t]),
            mets60=mets[t - pd.Timedelta("60min"):t].mean(),
            carbs=pd.to_numeric(r["Carbs"], errors="coerce"), protein=pd.to_numeric(r["Protein"], errors="coerce"),
            fat=pd.to_numeric(r["Fat"], errors="coerce"), fiber=pd.to_numeric(r["Fiber"], errors="coerce"),
            kcal=pd.to_numeric(r["Calories"], errors="coerce"),
            eaten=pd.to_numeric(r.get("Amount Consumed"), errors="coerce"),
            has_img=isinstance(r.get("Image path"), str), hour=t.hour + t.minute / 60))
M = pd.DataFrame(rows).merge(bio, on="sid", how="left")
M = M.sort_values(["sid", "t"]); M["rank"] = M.groupby("sid").cumcount() / M.groupby("sid").sid.transform("size")
for thr in [30, 50]: M[f"s{thr}"] = (M.rise >= thr).astype(int)
early = M[M["rank"] < .3]
M = M.join(early.groupby("sid").s50.mean().rename("prior_rate"), on="sid").join(early.groupby("sid").rise.mean().rename("prior_rise"), on="sid")
E = M[(M["rank"] >= .3) & (~M.overlap)].copy()
print(f"sensor used: {src}; meals parsed {len(M)}; eval meals {len(E)} from {E.sid.nunique()} people")
print(f"spike prevalence  >=30: {E.s30.mean():.0%}   >=50: {E.s50.mean():.0%}   median rise {E.rise.median():.0f} mg/dL")
print("eval meals by group:", E.group.value_counts().to_dict(), "| spike50 by group:", E.groupby("group").s50.mean().round(2).to_dict())

MAC = ["carbs", "protein", "fat", "fiber", "kcal"]
LAB = ["a1c", "fglu", "fins", "BMI", "Age", "Triglycerides"]
TRAJ = ["slope60", "slope20", "sd120", "min120", "since_meal", "hr_slope", "mets60"]
SETS = {"B0 carbs only (logistic)": ["carbs"],
        "B1 macros+pre-glucose+hour": MAC + ["base", "hour"],
        "B2 +HR+labs": MAC + ["base", "hour", "hr30"] + LAB,
        "B3 +personal prior (oracle ref)": MAC + ["base", "hour", "hr30", "prior_rate", "prior_rise"] + LAB,
        "B4 B3 + pre-meal trajectory shape": MAC + ["base", "hour", "hr30", "prior_rate", "prior_rise"] + LAB + TRAJ}

def person_folds(sids, k=5, seed=0):
    u = np.array(sorted(set(sids))); r = np.random.default_rng(seed); r.shuffle(u)
    return [np.isin(sids, u[i::k]) for i in range(k)]

def oof(df, fs, y="s50", logistic=False, reg=False, seeds=(0, 1, 2)):
    P = np.zeros((len(seeds), len(df)))
    X = df[fs].astype(float)
    for si, s in enumerate(seeds):
        for te in person_folds(df.sid.values, 5, s):
            tr = ~te
            if logistic: m = LogisticRegression(max_iter=500); m.fit(X[tr].fillna(0), df[y][tr]); P[si, te] = m.predict_proba(X[te].fillna(0))[:, 1]
            elif reg: m = HGBR(max_depth=3, learning_rate=.05, max_iter=200); m.fit(X[tr], df[y][tr]); P[si, te] = m.predict(X[te])
            else: m = HGB(max_depth=3, learning_rate=.05, max_iter=200); m.fit(X[tr], df[y][tr]); P[si, te] = m.predict_proba(X[te])[:, 1]
    return P.mean(0)

def boot_ci(df, p, y="s50", B=1000):
    sids = df.sid.unique(); vals = []
    idx = {s: np.where(df.sid.values == s)[0] for s in sids}
    for _ in range(B):
        pick = np.concatenate([idx[s] for s in rng.choice(sids, len(sids))])
        yy = df[y].values[pick]
        if yy.min() != yy.max(): vals.append(roc_auc_score(yy, p[pick]))
    return np.percentile(vals, [2.5, 97.5])

print("\nSpike >=50 mg/dL, person-level 5-fold x3 seeds, AUROC [95% person-bootstrap CI]")
res = {}
for name, fs in SETS.items():
    p = oof(E, fs, logistic=name.startswith("B0")); res[name] = p
    lo, hi = boot_ci(E, p)
    print(f"  {name:38s} {roc_auc_score(E.s50, p):.3f} [{lo:.2f}-{hi:.2f}]")

print("\nPer-group AUROC (B4) — does it work within each group, not just between groups?")
for gname, gdf in E.assign(p=res["B4 B3 + pre-meal trajectory shape"]).groupby("group"):
    if gdf.s50.nunique() > 1: print(f"  {gname:12s} n={len(gdf):4d}  AUROC {roc_auc_score(gdf.s50, gdf.p):.3f}")

print("\nFair baseline simulation: photo-estimated macros = true macros x lognormal error")
for cv in [0.25, 0.40]:
    N = E.copy()
    for c in MAC: N[c] = N[c] * rng.lognormal(0, cv, len(N))
    for name in ["B3 +personal prior (oracle ref)", "B4 B3 + pre-meal trajectory shape"]:
        p = oof(N, SETS[name]); print(f"  macro error ~{int(cv*100)}%  {name:36s} AUROC {roc_auc_score(N.s50, p):.3f}")

print("\nRise regression (MAE, mg/dL)")
E = E.dropna(subset=["rise"]).reset_index(drop=True)
for name in ["B1 macros+pre-glucose+hour", "B3 +personal prior (oracle ref)", "B4 B3 + pre-meal trajectory shape"]:
    p = oof(E, SETS[name], y="rise", reg=True); print(f"  {name:38s} MAE {mean_absolute_error(E.rise, p):.1f}")
pr = E.prior_rise.fillna(E.rise.median()); print(f"  predict-person-mean (prior_rise)       MAE {mean_absolute_error(E.rise, pr):.1f}")
print(f"\nmeals with photo path in eval set: {E.has_img.mean():.0%}; 'Amount Consumed' non-null: {E.eaten.notna().mean():.0%}, values: {E.eaten.dropna().round(0).value_counts().head(5).to_dict()}")
