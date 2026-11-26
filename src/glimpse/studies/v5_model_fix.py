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
        for name in v.extra.get("features", []):
            cols.append(self.history_features(fold, v.carbs if v.carbs != "none" else "photo", test)[name][rows])
        return np.hstack(cols)

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
