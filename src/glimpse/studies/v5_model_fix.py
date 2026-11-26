"""v5 model fix (docs/v5_model_fix.md, registered 26 Nov before any run): variants are tried on the design
people (outer folds 1-3) and logged; the frozen design is scored once on the confirmation people (folds 4-5).
Development people only; nothing served or frozen changes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from glimpse.calibration import personal_offset
from glimpse.data.splits import meal_rows
from glimpse.training.baselines import trend_features
from glimpse.training.fusion import PHOTO_COLS, photo_table

OUT = Path("outputs/v5_model_fix")
DESIGN_FOLDS, CONFIRM_FOLDS = (1, 2, 3), (4, 5)
LAG = np.timedelta64(120 * 60, "s")
CUT = 50.0


@dataclass(frozen=True)
class Variant:
    """One design. Defaults reproduce v1's recipe: label logistic on trend + profile + 4 photo inputs."""

    name: str
    photo: bool = True  # the 4 photo inputs
    centred: bool = False  # 7 trend numbers minus the person's earlier mean
    carbs: str = "none"  # none | confirmed (logged carbs at test) | photo (photo grams in the carb column)
    target: str = "label"  # label | rise (ridge on the rise, mapped to a logit) | soft (label weights by distance)
    drop_near: float = 0.0  # drop training meals with |rise - 50| below this
    model: str = "lr"  # lr | gbm
    C: float | None = None  # None: tuned on the inner folds
    platt: bool = True
    intercept: str = "residual"  # none | residual (v1's) | rate | residual_frozen
    m: float | None = None  # None: v1's per-fold m
    start_after: int = 0  # no offset before this many labelled meals
    freeze_at: int = 0  # residual_frozen: stop updating after this many labelled meals
    extra: dict = field(default_factory=dict)


class Data:
    """Everything precomputed once: features, labels, times, people, photo inputs per fold."""

    def __init__(self, ds, sp, cfg) -> None:
        self.ds, self.sp, self.cfg = ds, sp, cfg
        n = len(ds.samples)
        allr = np.arange(n)
        self.T = trend_features(ds, allr)
        self.y = ds.labels().astype(int)
        self.rise = np.array([s.rise_mgdl for s in ds.samples], float)
        self.pid = np.array([s.person_id for s in ds.samples])
        self.time = np.array([np.datetime64(s.meal_time, "s") for s in ds.samples])
        self.carb = np.array([np.log1p(s.carbs_served_g) if s.carbs_served_g is not None else np.nan
                              for s in ds.samples])  # fmt: skip
        self.centred = np.zeros((n, 7))
        for p in np.unique(self.pid):
            rows = sorted(np.where(self.pid == p)[0], key=lambda i: self.time[i])
            for j, r in enumerate(rows):
                if j:
                    with np.errstate(all="ignore"):
                        self.centred[r] = np.nan_to_num(self.T[r, :7] - np.nanmean(self.T[rows[:j], :7], axis=0))
        pcv = Path(cfg.paths.outputs) / "photo_cv"
        p_oof, p_nested = pd.read_csv(pcv / "photo_oof.csv"), pd.read_csv(pcv / "photo_nested.csv")
        self.photo = {}
        for k in range(1, 6):
            pt = photo_table(p_nested[p_nested.fold == k]) | photo_table(p_oof[p_oof.fold == k])
            a = np.full((n, len(PHOTO_COLS)), np.nan)
            for r, v in pt.items():
                a[r] = v
            self.photo[k] = a
        ev = json.loads(Path("outputs/evaluation_v2/dev_summary.json").read_text())
        self.v1_m = {f["fold"]: f["m"] for f in ev["folds"]}
        self.v1 = pd.read_csv("outputs/evaluation_v2/dev_answers.csv")
        self._hist: dict = {}

    def X(self, v: Variant, fold: int, rows: np.ndarray, test: bool) -> np.ndarray:
        cols = [self.T[rows]]
        if v.photo:
            cols.append(self.photo[fold][rows])
        if v.centred:
            cols.append(self.centred[rows])
        if v.carbs != "none":
            c = self.carb[rows].copy()
            if test and v.carbs == "photo":
                c = self.photo[fold][rows, 0].copy()  # log(1 + photo grams) stands in for the confirmed carbs
            elif test:
                miss = np.isnan(c)
                c[miss] = self.photo[fold][rows, 0][miss]
            cols.append(c[:, None])
        if v.extra.get("static"):
            cols.append(self.static[v.extra["static"]].to_numpy(float)[rows])
        if v.extra.get("photo_macros"):
            cols.append(self.photo_macros[fold - 1][rows])
        if v.extra.get("forecast"):
            cols.append(self.forecast(fold)[rows][:, None])
        if v.extra.get("micro"):
            cols.append(self.micro(fold, int(v.extra["micro"]))[rows])
        for name in v.extra.get("features", []):
            cols.append(self.history_features(fold, v.carbs if v.carbs != "none" else "photo", test)[name][rows])
        return np.hstack(cols)

    @property
    def static(self) -> pd.DataFrame:
        if "static" not in self._hist:
            self._hist["static"] = pd.read_csv(OUT / "v7_static.csv")
        return self._hist["static"]

    @property
    def photo_macros(self) -> np.ndarray:
        if "pm" not in self._hist:
            self._hist["pm"] = np.load(OUT / "v7_photo_macros.npy").astype(float)
        return self._hist["pm"]

    def forecast(self, fold: int) -> np.ndarray:
        """The CGM forecaster's 2-hour rise per sample for outer fold `fold` (outputs/v5_model_fix/)."""
        if "fc" not in self._hist:
            f = OUT / "v7_forecast.npy"
            self._hist["fc"] = np.load(f) if f.is_file() else np.tile(np.load(OUT / "v7_forecast_crossfit.npy"), (5, 1))
        return self._hist["fc"][fold - 1]

    def micro(self, fold: int, n: int) -> np.ndarray:
        """Gut-health principal components per sample, fitted on the fold's training people only."""
        from glimpse.studies.v7_features import microbiome_pcs

        key = ("micro", fold, n)
        if key not in self._hist:
            pcs = microbiome_pcs(set(self.train_people(fold)), n)
            self._hist[key] = pcs.reindex(self.pid).to_numpy()
        return self._hist[key]

    def train_people(self, fold: int) -> list[int]:
        return self._hist.get(("train", fold)) or self.sp.training_people(fold - 1)

    def history_features(self, fold: int, source: str, test: bool) -> dict[str, np.ndarray]:
        """Causal per-person features from earlier meals (labels only from meals whose 2-hour window closed).
        carb_centred: this meal's log carbs minus the person's earlier mean. similar: spike rate of the 3
        earlier labelled meals closest in carbs, minus the person's running rate (0 with under 4 meals).
        prev_rise: mean rise of the earlier labelled meals minus 50, scaled (0 with none). hist_rate: their
        spike rate shrunk to 0.33 with weight 4, as a logit difference. hist_n: log(1 + their number)."""
        key = (fold, source, test)
        if key in self._hist:
            return self._hist[key]
        carb = self.photo[fold][:, 0].copy()
        if source == "confirmed" or (source == "photo" and not test):
            carb = np.where(np.isnan(self.carb), carb, self.carb)
        n = len(self.y)
        out = {k: np.zeros((n, 1)) for k in ("carb_centred", "similar", "prev_rise", "hist_rate", "hist_n")}
        for p in np.unique(self.pid):
            rows = np.array(sorted(np.where(self.pid == p)[0], key=lambda i: self.time[i]))
            c, t, y, r = carb[rows], self.time[rows], self.y[rows].astype(float), self.rise[rows]
            for j, i in enumerate(rows):
                if j and np.isfinite(c[j]) and np.isfinite(c[:j]).any():
                    out["carb_centred"][i] = c[j] - np.nanmean(c[:j])
                h = np.where((t + LAG <= t[j]) & np.isfinite(r))[0]
                if len(h):
                    out["prev_rise"][i] = (np.clip(r[h], -50, 200).mean() - CUT) / 50.0
                    rate = (y[h].sum() + 4 * 0.33) / (len(h) + 4)  # shrunk to a fixed 0.33 with weight 4
                    out["hist_rate"][i] = np.log(rate / (1 - rate)) - np.log(0.33 / 0.67)
                    out["hist_n"][i] = np.log1p(len(h))
                hc = h[np.isfinite(c[h])]
                if len(hc) >= 4 and np.isfinite(c[j]):
                    near = hc[np.argsort(np.abs(c[hc] - c[j]))[:3]]
                    out["similar"][i] = y[near].mean() - y[h].mean()
        self._hist[key] = out
        return out


def _make(v: Variant, c: float, seed: int):
    if v.model == "gbm":
        return HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=int(c), min_samples_leaf=20,
                                              random_state=seed)  # fmt: skip
    if v.target == "rise":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0 / c))
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=c, max_iter=2000, random_state=seed))  # fmt: skip


def _fit_score(v: Variant, D: Data, fold: int, tr: np.ndarray, te: np.ndarray, c: float) -> np.ndarray:
    """Raw logit for the test rows from one model fitted on the training rows."""
    if v.drop_near:
        tr = tr[np.abs(D.rise[tr] - CUT) >= v.drop_near]
    Xtr, Xte = D.X(v, fold, tr, False), D.X(v, fold, te, True)
    if v.model == "blend":  # logistic and small boosted trees, logits averaged
        out = []
        for kind in ("lr", "gbm"):
            mm = _make(Variant("part", model=kind), c if kind == "lr" else 200, D.cfg.seed).fit(Xtr, D.y[tr])
            pp = np.clip(mm.predict_proba(Xte)[:, 1], 1e-6, 1 - 1e-6)
            out.append(np.log(pp / (1 - pp)))
        return (out[0] + out[1]) / 2
    m = _make(v, c, D.cfg.seed)
    if v.target == "rise":
        r = np.clip(D.rise[tr], -50, 200)
        m.fit(Xtr, r)
        resid = float(np.std(r - m.predict(Xtr))) or 1.0
        return (m.predict(Xte) - CUT) / (resid * 0.55)  # logistic approximation to a normal error
    if v.target == "soft":
        w = np.clip(np.abs(D.rise[tr] - CUT) / 20.0, 0.25, 1.0)
        m.fit(Xtr, D.y[tr], **{m.steps[-1][0] + "__sample_weight": w})
    else:
        m.fit(Xtr, D.y[tr])
    p = np.clip(m.predict_proba(Xte)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _grid(v: Variant, D: Data) -> list[float]:
    return [100, 200, 400] if v.model == "gbm" else list(D.cfg.baselines.logreg_c_grid)


def fold_logits(v: Variant, D: Data, k: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Outer fold k (0-based): tune on the inner folds, fit, Platt on the inner out-of-fold logits."""
    sp, ds, fold = D.sp, D.ds, k + 1
    people = sp.training_people(k)
    tr = meal_rows(ds, set(people), scored_only=False)
    te = meal_rows(ds, set(sp.dev_folds[k]), scored_only=True)
    inner = sp.inner_folds[k]
    parts = [(meal_rows(ds, set(people) - set(h), scored_only=False), meal_rows(ds, set(h), scored_only=True))
             for h in inner]  # fmt: skip
    grid = [v.C] if v.C is not None else _grid(v, D)
    best, best_loss, best_oof = grid[0], np.inf, None
    for c in grid:
        oof = [(_fit_score(v, D, fold, a, b, c), b) for a, b in parts]
        f, rows = np.concatenate([o[0] for o in oof]), np.concatenate([o[1] for o in oof])
        loss = log_loss(D.y[rows], 1 / (1 + np.exp(-f)), labels=[0, 1])
        if loss < best_loss - 1e-12:
            best, best_loss, best_oof = c, loss, (f, rows)
    f = _fit_score(v, D, fold, tr, te, best)
    info = {"fold": fold, "C": best}
    if v.platt:
        pl = LogisticRegression(C=1e6, max_iter=1000).fit(best_oof[0][:, None], D.y[best_oof[1]])
        a, b = float(pl.coef_[0, 0]), float(pl.intercept_[0])
        f = a * f + b
        info |= {"platt": [a, b]}
    return te, f, info


def _offsets(v: Variant, D: Data, rows: np.ndarray, f: np.ndarray, m: float, pop: float) -> np.ndarray:
    """Per-person offset for each meal from that person's earlier meals whose 2-hour label has closed."""
    out = np.zeros(len(rows))
    if v.intercept == "none":
        return out
    for p in np.unique(D.pid[rows]):
        ix = np.where(D.pid[rows] == p)[0]
        t, y, fp = D.time[rows[ix]], D.y[rows[ix]].astype(float), f[ix]
        frozen = None
        for j, i in enumerate(ix):
            hist = t + LAG <= t[j]
            n = int(hist.sum())
            if n < max(v.start_after, 1):
                continue
            if v.intercept == "rate":
                r = (y[hist].sum() + m * pop) / (n + m)
                out[i] = np.log(r / (1 - r)) - np.log(pop / (1 - pop))
            elif v.intercept == "residual_frozen" and v.freeze_at and n >= v.freeze_at:
                if frozen is None:
                    first = np.where(hist)[0][: v.freeze_at]
                    frozen = personal_offset(fp[first], y[first], m)
                out[i] = frozen
            else:
                out[i] = personal_offset(fp[hist], y[hist], m)
    return out


def predict(v: Variant, D: Data, folds: tuple[int, ...]) -> pd.DataFrame:
    """Out-of-fold risk for the scored meals of the given outer folds, history replayed in time order."""
    out = []
    for fold in folds:
        te, f, info = fold_logits(v, D, fold - 1)
        m = v.m if v.m is not None else D.v1_m[fold]
        tr_people = set(D.sp.training_people(fold - 1))
        pop = float(D.y[meal_rows(D.ds, tr_people, scored_only=False)].mean())
        off = _offsets(v, D, te, f, m, pop)
        out.append(pd.DataFrame({"fold": fold, "row": te, "person_id": D.pid[te], "spike": D.y[te], "raw": f,
                                 "risk": 1 / (1 + np.exp(-(f + off))), "C": info["C"]}))  # fmt: skip
    return pd.concat(out, ignore_index=True)


def within(d: pd.DataFrame, col: str) -> float:
    return float(np.mean([roc_auc_score(g.spike, g[col]) for _, g in d.groupby("person_id") if g.spike.nunique() == 2]))


def score(d: pd.DataFrame, col: str = "risk") -> dict:
    p, w = float(roc_auc_score(d.spike, d[col])), within(d, col)
    return {"pooled": p, "within": w, "sum": p + w, "brier": float(np.mean((d[col] - d.spike) ** 2)), "meals": len(d)}


def v1_baseline(D: Data, folds: tuple[int, ...]) -> pd.DataFrame:
    return D.v1[D.v1.fold.isin(folds) & D.v1.risk.notna()][["fold", "row", "person_id", "spike", "risk"]].copy()


def try_design(v: Variant, D: Data) -> dict:
    """Score one variant on the design people only and append it to the design log."""
    d = predict(v, D, DESIGN_FOLDS)
    b = v1_baseline(D, DESIGN_FOLDS)
    res = {"variant": asdict(v), "all_meals": score(d), "raw_all_meals": score(d.assign(risk=d.raw)),
           "on_v1_answered": score(d[d.row.isin(b.row)])}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "design_log.json"
    hist = json.loads(log.read_text()) if log.is_file() else []
    hist.append(res)
    log.write_text(json.dumps(hist, indent=1))
    return res


def load():
    from glimpse.config import load_config
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import SplitsV2

    cfg = load_config(Path("configs/default.yaml"))
    return Data(MealDataset.load(Path("data/dataset")), SplitsV2.load(Path("splits/dev_splits_v2.json")), cfg)


FROZEN = {
    "confirmed_carbs": Variant("v5, carbs confirmed", photo=False, centred=True, carbs="confirmed", intercept="none",
                               drop_near=5.0, extra={"features": ["carb_centred", "prev_rise"]}),
    "photo_only": Variant("v5, photo only", centred=True, intercept="none", drop_near=5.0,
                          extra={"features": ["prev_rise"]}),
}  # fmt: skip


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in np.unique(b)))


def _paired(d: pd.DataFrame) -> tuple[dict, dict]:
    """v5 minus v1, pooled AUROC and mean within-person AUROC; bootstrap over people, 2,000 draws, seed 0."""
    rng, ppl = np.random.default_rng(0), d.person_id.unique()
    g = {p: (x.spike.to_numpy(), x.risk.to_numpy(), x.v1.to_numpy()) for p, x in d.groupby("person_id")}
    wd = {p: roc_auc_score(y, a) - roc_auc_score(y, b) for p, (y, a, b) in g.items() if len(set(y)) == 2}

    def pooled(people) -> float:
        y, a, b = (np.concatenate([g[p][i] for p in people]) for i in range(3))
        return float(roc_auc_score(y, a) - roc_auc_score(y, b))

    def inside(people) -> float:
        v = [wd[p] for p in people if p in wd]
        return float(np.mean(v)) if v else float("nan")

    pd_, wd_ = [], []
    for _ in range(2000):
        s = rng.choice(ppl, len(ppl))
        if len(set(np.concatenate([g[p][0] for p in s]))) == 2:
            pd_.append(pooled(s))
        wd_.append(inside(s))
    ci = lambda v: [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]  # noqa: E731
    return ({"point": pooled(ppl), "ci95": ci(pd_)},
            {"point": inside(ppl), "ci95": ci(wd_), "people": len(wd),
             "people_better": int(sum(x > 0 for x in wd.values()))})  # fmt: skip


def confirm(D: Data) -> dict:
    """The one registered scoring of the frozen design on the confirmation people (folds 4 and 5)."""
    out_file = OUT / "confirmation.json"
    if out_file.is_file():
        raise SystemExit("the confirmation has been run; it is scored once (docs/v5_model_fix.md)")
    ev = json.loads(Path("outputs/evaluation_v2/dev_summary.json").read_text())
    cut = {f["fold"]: f["low_risk_below"] for f in ev["folds"]}
    b = v1_baseline(D, CONFIRM_FOLDS).rename(columns={"risk": "v1"})
    res = {"what": "v5 frozen design, confirmation people (outer folds 4 and 5), scored once",
           "people": int(b.person_id.nunique()),
           "v1_shipped": score(b, "v1") | {"ece": _ece(b.spike.to_numpy(), b.v1.to_numpy())}}  # fmt: skip
    for name, v in FROZEN.items():
        d = predict(v, D, CONFIRM_FOLDS)
        d.to_csv(OUT / f"confirm_{name}.csv", index=False)
        j = d.merge(b[["row", "v1"]], on="row")
        clear = j.risk < j.fold.map(cut)
        clear1 = j.v1 < j.fold.map(cut)
        pw = _paired(j)
        res[name] = {
            "all_meals": score(d) | {"ece": _ece(d.spike.to_numpy(), d.risk.to_numpy())},
            "on_v1_answered": score(j) | {"ece": _ece(j.spike.to_numpy(), j.risk.to_numpy())},
            "pooled_auroc_v5_minus_v1": pw[0],
            "within_person_auroc_v5_minus_v1": pw[1],
            "at_v1_fold_cutoffs": {"v5_cleared": int(clear.sum()), "v5_missed": int(j.spike[clear].sum()),
                                   "v1_cleared": int(clear1.sum()), "v1_missed": int(j.spike[clear1].sum())},
            "C_per_fold": d.groupby("fold").C.first().tolist(),
        }  # fmt: skip
        p, w = res[name]["pooled_auroc_v5_minus_v1"], res[name]["within_person_auroc_v5_minus_v1"]
        both = p["point"] > 0 and w["point"] > 0
        res[name]["verdict"] = ("confirmed" if both and (p["ci95"][0] > 0 or w["ci95"][0] > 0)
                                else "direction only" if both else "not confirmed")  # fmt: skip
    out_file.write_text(json.dumps(res, indent=2))
    return res


V7 = {
    "photo_only": Variant("v7, photo only", centred=True, intercept="none", drop_near=5.0,
                          extra={"features": ["prev_rise"], "micro": 3}),
    "meal_confirmed": Variant("v7, meal confirmed", photo=False, centred=True, carbs="confirmed", intercept="none",
                              drop_near=5.0, model="blend",
                              extra={"features": ["carb_centred", "prev_rise"],
                                     "static": ["protein", "fat", "fiber", "kcal", "carb_share", "male", "log_insulin",
                                                "log_trig", "hdl", "log_homa"]}),
}  # fmt: skip
V1_RECIPE = Variant("v1 recipe", m=1.0)


def predict_custom(v: Variant, D: Data, held: list[int], train: list[int], inner: list[list[int]],
                   pfold: int) -> pd.DataFrame:  # fmt: skip
    """predict() for people outside the outer folds: fit on `train`, tune on `inner`, score `held`.
    `pfold` names the committed photo table to read (its estimates never saw the person)."""
    ds = D.ds
    D._hist[("train", pfold)] = list(train)
    for key in [k for k in D._hist if isinstance(k, tuple) and k[0] == "micro"]:
        del D._hist[key]
    tr, te = meal_rows(ds, set(train), scored_only=False), meal_rows(ds, set(held), scored_only=True)
    parts = [(meal_rows(ds, set(train) - set(h), scored_only=False), meal_rows(ds, set(h), scored_only=True))
             for h in inner]  # fmt: skip
    grid = [v.C] if v.C is not None else _grid(v, D)
    best, best_loss, best_oof = grid[0], np.inf, None
    for c in grid:
        oof = [(_fit_score(v, D, pfold, a, b, c), b) for a, b in parts]
        f, rows = np.concatenate([o[0] for o in oof]), np.concatenate([o[1] for o in oof])
        loss = log_loss(D.y[rows], 1 / (1 + np.exp(-f)), labels=[0, 1])
        if loss < best_loss - 1e-12:
            best, best_loss, best_oof = c, loss, (f, rows)
    f = _fit_score(v, D, pfold, tr, te, best)
    if v.platt:
        pl = LogisticRegression(C=1e6, max_iter=1000).fit(best_oof[0][:, None], D.y[best_oof[1]])
        f = float(pl.coef_[0, 0]) * f + float(pl.intercept_[0])
    off = _offsets(v, D, te, f, v.m if v.m is not None else 1.0, float(D.y[tr].mean()))
    del D._hist[("train", pfold)]
    return pd.DataFrame({"row": te, "person_id": D.pid[te], "spike": D.y[te], "risk": 1 / (1 + np.exp(-(f + off)))})


def _boot(d: pd.DataFrame, a: str, b: str) -> dict:
    """a minus b for pooled AUROC, within-person AUROC and Brier; bootstrap over people, 2,000 draws, seed 0."""
    rng, ppl = np.random.default_rng(0), d.person_id.unique()
    g = {p: (x.spike.to_numpy(), x[a].to_numpy(), x[b].to_numpy()) for p, x in d.groupby("person_id")}
    wd = {p: roc_auc_score(y, u) - roc_auc_score(y, w) for p, (y, u, w) in g.items() if len(set(y)) == 2}

    def stats(people) -> tuple[float, float, float]:
        y, u, w = (np.concatenate([g[p][i] for p in people]) for i in range(3))
        ws = [wd[p] for p in people if p in wd]
        return (float(roc_auc_score(y, u) - roc_auc_score(y, w)), float(np.mean(ws)) if ws else float("nan"),
                float(np.mean((u - y) ** 2) - np.mean((w - y) ** 2)))  # fmt: skip

    draws = np.array([stats(rng.choice(ppl, len(ppl))) for _ in range(2000)])
    pt = stats(ppl)
    return {n: {"point": pt[i], "ci95": [float(x) for x in np.nanpercentile(draws[:, i], [2.5, 97.5])]}
            for i, n in enumerate(("pooled_auroc", "within_person_auroc", "brier"))}  # fmt: skip


def _verdict(c: dict) -> str:
    w, p, b = c["within_person_auroc"], c["pooled_auroc"], c["brier"]
    not_worse = p["ci95"][1] >= 0 and b["ci95"][0] <= 0
    if w["point"] > 0 and not_worse:
        return "transfers" if w["ci95"][0] > 0 else "direction only"
    return "does not transfer"


def transfer_check(D: Data) -> dict:
    """The registered one-time check on the 14 type-2 people (never scored before), two halves."""
    out_file = OUT / "transfer_check.json"
    if out_file.is_file():
        raise SystemExit("the transfer check has been run; it is scored once (docs/v5_model_fix.md)")
    t2d, dev = sorted(D.sp.t2d), [p for f in D.sp.dev_folds for p in f]
    halves = [t2d[0::2], t2d[1::2]]
    recipes = {"v1_recipe": V1_RECIPE, "v5_photo_only": FROZEN["photo_only"], "v7_photo_only": V7["photo_only"],
               "v7_meal_confirmed": V7["meal_confirmed"]}  # fmt: skip
    frames = []
    for h, held in enumerate(halves):
        d = None
        for name, v in recipes.items():
            r = predict_custom(v, D, held, dev + halves[1 - h], [list(f) for f in D.sp.dev_folds], 1)
            d = r.rename(columns={"risk": name}) if d is None else d.merge(r[["row", "risk"]].rename(
                columns={"risk": name}), on="row")  # fmt: skip
        frames.append(d.assign(half=h + 1))
    d = pd.concat(frames, ignore_index=True)
    d.to_csv(OUT / "transfer_check.csv", index=False)
    res = {"what": "v7 transfer check on the 14 type-2 people, scored once", "people": int(d.person_id.nunique()),
           "meals": len(d), "spike_rate": float(d.spike.mean()),
           "scores": {n: score(d, n) | {"ece": _ece(d.spike.to_numpy(), d[n].to_numpy())}
                      for n in recipes}}  # fmt: skip
    for n in ("v5_photo_only", "v7_photo_only", "v7_meal_confirmed"):
        c = _boot(d, n, "v1_recipe")
        res[f"{n}_minus_v1_recipe"] = c | {"verdict": _verdict(c)}
    out_file.write_text(json.dumps(res, indent=2))
    return res


def second_look(D: Data) -> dict:
    """v7 on the confirmation people (folds 4 and 5): a second look at people v5 already used once, not clean."""
    b = v1_baseline(D, CONFIRM_FOLDS).rename(columns={"risk": "v1"})
    res = {"what": "v7 on folds 4 and 5, second look (not clean)", "v1_shipped": score(b, "v1")}
    for name, v in V7.items():
        j = predict(v, D, CONFIRM_FOLDS).merge(b[["row", "v1"]], on="row").rename(columns={"risk": name})
        c = _boot(j, name, "v1")
        res[name] = {"on_v1_answered": score(j, name) | {"ece": _ece(j.spike.to_numpy(), j[name].to_numpy())}, **c}
    (OUT / "second_look.json").write_text(json.dumps(res, indent=2))
    return res


def photo_within_person(D: Data) -> dict:
    """Part 4 (registered before the run): v5 photo only with, without and with permuted photo inputs."""
    from dataclasses import replace

    out_file = OUT / "photo_within_person.json"
    if out_file.is_file():
        raise SystemExit("Part 4 has been run; it is scored once (docs/v5_model_fix.md)")
    v = FROZEN["photo_only"]
    arms = {"with_photo": v, "without_photo": replace(v, photo=False)}
    t2d, dev = sorted(D.sp.t2d), [p for f in D.sp.dev_folds for p in f]
    halves, inner = [t2d[0::2], t2d[1::2]], [list(f) for f in D.sp.dev_folds]

    def permuted(fold: int, rows: np.ndarray):
        keep, rng = D.photo[fold].copy(), np.random.default_rng(0)
        for p in np.unique(D.pid[rows]):
            r = rows[D.pid[rows] == p]
            D.photo[fold][r] = keep[rng.permutation(r)]
        return keep

    def frame(run, fold_of) -> pd.DataFrame:
        d = None
        for name, a in arms.items():
            r = run(a).rename(columns={"risk": name})
            d = r if d is None else d.merge(r[["row", name]], on="row")
        keep = {f: permuted(f, d.row.to_numpy()) for f in fold_of}
        r = run(v).rename(columns={"risk": "permuted_photo"})
        for f, k in keep.items():
            D.photo[f] = k
        return d.merge(r[["row", "permuted_photo"]], on="row")

    groups = {
        "type2_14": pd.concat([frame(lambda a, h=h: predict_custom(a, D, halves[h], dev + halves[1 - h], inner, 1), [1])
                               for h in (0, 1)], ignore_index=True),
        "confirmation_8": frame(lambda a: predict(a, D, CONFIRM_FOLDS), CONFIRM_FOLDS),
        "design_15": frame(lambda a: predict(a, D, DESIGN_FOLDS), DESIGN_FOLDS),
    }  # fmt: skip
    res = {"what": "Part 4: the photo within a person under the v5 recipe", "groups": {}}
    for g, d in groups.items():
        c, k = _boot(d, "with_photo", "without_photo"), _boot(d, "permuted_photo", "without_photo")
        w = c["within_person_auroc"]
        verdict = ("the photo adds within a person" if w["ci95"][0] > 0 and k["within_person_auroc"]["ci95"][0] <= 0
                   else "direction only" if w["point"] > 0 and w["ci95"][0] <= 0 else "no")  # fmt: skip
        res["groups"][g] = {"people": int(d.person_id.nunique()), "meals": len(d),
                            "scores": {n: score(d, n) for n in ("with_photo", "without_photo", "permuted_photo")},
                            "with_minus_without": c, "permuted_minus_without": k, "verdict": verdict}  # fmt: skip
        d.to_csv(OUT / f"photo_within_{g}.csv", index=False)
    out_file.write_text(json.dumps(res, indent=2))
    return res


# ------------------------------------------------------------------ Part 5: five more tests of v5 (photo only)
def _cost(dec: np.ndarray, y: np.ndarray, costs: dict) -> float:
    tab = {"low risk": costs["low_risk"], "caution": costs["caution"], "can't judge": costs["abstain"]}
    return float(sum(tab[d][0] if s else tab[d][1] for d, s in zip(dec, y, strict=True)))


def _decide(p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.where(p < lo, "low risk", np.where(p >= hi, "caution", "can't judge"))


def _useful(dec: np.ndarray, y: np.ndarray, costs: dict, pre: np.ndarray) -> dict:
    cl = dec == "low risk"
    always = _cost(np.full(len(y), "caution"), y, costs)
    return {"cost_reduction": 1 - _cost(dec, y, costs) / always, "cleared_share": float(cl.mean()),
            "cant_judge_share": float((dec == "can't judge").mean()), "cleared": int(cl.sum()),
            "missed": int(y[cl].sum()), "prediabetes_cleared": int((cl & pre).sum()),
            "prediabetes_meals": int(pre.sum()), "prediabetes_missed": int(y[cl & pre].sum())}  # fmt: skip


def _selective(y: np.ndarray, p: np.ndarray) -> dict:
    order = np.argsort(-np.abs(p - 0.5))
    out = {}
    for drop in (0.0, 0.1, 0.2, 0.3):
        keep = order[: int(round(len(y) * (1 - drop)))]
        out[f"{int(drop * 100)}%"] = float(np.mean((p[keep] >= 0.5) != y[keep].astype(bool)))
    return out


def _cnn_scores(D: Data) -> np.ndarray:
    """[fold 1..5][sample]: SensorCNN logit, out of person (held people: fit on the fold's training people;
    training people: nested fits that excluded them)."""
    import torch

    from glimpse.data.normalise import Normaliser
    from glimpse.training.photo_cv import fold_plan_v2
    from glimpse.training.scratch import _inputs, fit, predict_proba

    torch.set_num_threads(2)
    ds, cfg = D.ds, D.cfg
    epochs = {t["fold"]: t["cnn_epochs"] for t in json.loads(Path("outputs/cv/cv_summary.json").read_text())["tuning"]}
    out = np.full((5, len(D.y)), np.nan)

    def score_rows(train_people, rows, ep):
        tr = meal_rows(ds, set(train_people), scored_only=False)
        norm = Normaliser.fit(ds.glucose[tr], ds.heart_rate[tr], ds.profile[tr], ds.profile_names)
        model, _ = fit(ds, tr, norm, cfg.training.scratch, cfg, ep, 0, "v5-t4")
        p = np.clip(predict_proba(model, _inputs(ds, rows, norm)), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    for f in fold_plan_v2(D.sp, cfg):
        k, ep = f["fold"] - 1, epochs[f["fold"]]
        held = np.where(np.isin(D.pid, f["held"]))[0]
        out[k, held] = score_rows(f["train"], held, ep)
        for i in f["inner"]:
            rows = np.where(np.isin(D.pid, i["exclude"]))[0]
            out[k, rows] = score_rows([p for p in f["train"] if p not in i["exclude"]], rows, ep)
    return out


def more_tests(D: Data) -> dict:
    from dataclasses import replace

    from glimpse.training.final import cost_dict

    out_file = OUT / "more_tests.json"
    if out_file.is_file():
        raise SystemExit("Part 5 has been run; it is scored once (docs/v5_model_fix.md)")
    v, costs, res = FROZEN["photo_only"], cost_dict(D.cfg), {"what": "Part 5: five more tests of v5 (photo only)"}
    pre = np.array([s.group for s in D.ds.samples]) == "prediabetes"
    male = D.static["male"].to_numpy() == 1
    dev = predict(v, D, DESIGN_FOLDS + CONFIRM_FOLDS)
    v1 = v1_baseline(D, DESIGN_FOLDS + CONFIRM_FOLDS).rename(columns={"risk": "v1"})

    # T1 shuffled labels
    true_y, aucs = D.y.copy(), []
    for s in range(10):
        rng, parts = np.random.default_rng(s), []
        for fold in DESIGN_FOLDS + CONFIRM_FOLDS:
            tr = meal_rows(D.ds, set(D.sp.training_people(fold - 1)), scored_only=False)
            D.y = true_y.copy()
            D.y[tr] = rng.permutation(true_y[tr])
            te, f, _ = fold_logits(v, D, fold - 1)
            parts.append((true_y[te], f))
        D.y = true_y
        aucs.append(float(roc_auc_score(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))))
    res["T1_shuffled_labels"] = {"mean": float(np.mean(aucs)), "min": min(aucs), "max": max(aucs),
                                 "unshuffled": float(roc_auc_score(dev.spike, dev.raw)),
                                 "met": bool(0.45 <= np.mean(aucs) <= 0.55)}  # fmt: skip

    # T2 selective answering
    conf = pd.read_csv(OUT / "confirm_photo_only.csv").merge(v1[["row", "v1"]], on="row")
    t2 = pd.read_csv(OUT / "transfer_check.csv")
    res["T2_selective_answering_error"] = {
        "confirmation": {"v5": _selective(conf.spike.to_numpy(), conf.risk.to_numpy()),
                         "v1": _selective(conf.spike.to_numpy(), conf.v1.to_numpy())},
        "type2": {"v5": _selective(t2.spike.to_numpy(), t2.v5_photo_only.to_numpy()),
                  "v1_recipe": _selective(t2.spike.to_numpy(), t2.v1_recipe.to_numpy())}}  # fmt: skip
    for g in res["T2_selective_answering_error"].values():
        e = list(g["v5"].values())
        g["v5_falls_at_every_step"] = bool(all(b < a for a, b in zip(e, e[1:], strict=False)))

    # T3 usefulness at cut-offs chosen on the design people
    des = dev[dev.fold.isin(DESIGN_FOLDS)]
    grid = np.round(np.arange(0.05, 0.601, 0.01), 2)
    best = min(((lo, hi) for lo in grid for hi in grid if lo <= hi),
               key=lambda c: _cost(_decide(des.risk.to_numpy(), *c), des.spike.to_numpy(), costs))  # fmt: skip
    c8 = dev[dev.fold.isin(CONFIRM_FOLDS)].merge(D.v1[["row", "decision"]], on="row")
    c8 = c8[c8.decision.notna() & c8.row.isin(v1.row)]
    y8, p8 = c8.spike.to_numpy(), pre[c8.row.to_numpy()]
    res["T3_usefulness_confirmation"] = {
        "cutoffs_from_design_people": [float(best[0]), float(best[1])], "meals": len(c8),
        "v5": _useful(_decide(c8.risk.to_numpy(), *best), y8, costs, p8),
        "v1_shipped": _useful(c8.decision.to_numpy(), y8, costs, p8)}  # fmt: skip

    # T4 the from-scratch SensorCNN as a 28th input
    cnn = _cnn_scores(D)
    D._hist["fc"] = cnn
    v_cnn = replace(v, name="v5 + SensorCNN score", extra={**v.extra, "forecast": True})
    dc = predict(v_cnn, D, DESIGN_FOLDS + CONFIRM_FOLDS)
    del D._hist["fc"]
    res["T4_sensor_cnn_input"] = {
        g: {"v5": score(dev[dev.fold.isin(f)]), "v5_plus_cnn": score(dc[dc.fold.isin(f)])}
        for g, f in (("design", DESIGN_FOLDS), ("confirmation", CONFIRM_FOLDS))}  # fmt: skip

    # T5 slices
    j = dev.merge(v1[["row", "v1"]], on="row")
    r = j.row.to_numpy()
    slices = {"women": ~male[r], "men": male[r], "prediabetes": pre[r], "healthy": ~pre[r]}
    res["T5_slices_pooled_auroc"] = {
        n: {"meals": int(m.sum()), "v5": float(roc_auc_score(j.spike[m], j.risk[m])),
            "v1": float(roc_auc_score(j.spike[m], j.v1[m]))} for n, m in slices.items()}  # fmt: skip
    res["T5_slices_pooled_auroc"]["met"] = bool(all(x["v5"] >= x["v1"] - 0.03
                                                    for x in res["T5_slices_pooled_auroc"].values()))  # fmt: skip
    out_file.write_text(json.dumps(res, indent=2))
    return res
