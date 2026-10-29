"""Spatial evidence: is there spillover between wells, and how far does an intervention reach?

    python -m gwtwin spatial          (~1 min) -> reports/spatial_evidence.json

1. Residual correlation by distance (bore and dug wells): after the water balance has explained
   what it can, are errors of nearby wells in the same round more alike than those of distant
   wells? Pearson on raw residuals and on Huber-clipped residuals (|r| capped at 2 robust SD,
   because the worst 5 % of bore-well rounds carry ~77 % of squared error). 90 % ranges from a
   bootstrap over rounds. This measures CO-MOVEMENT (shared weather, shared timing of pumping),
   not influence of one well on another.
2. Lagged test: does a neighbour's residual in round k−1 predict mine in round k? If heads moved
   laterally on a 3-month time scale, it would. Clustered by well.
3. Engine test: add +500 ha of tank water around the 16 bore wells nearest a fixed point and
   check every other well, with the deployed ML weight (β = 0) and with the ML forced on (β = 1).
4. Hydraulics (physics only, ASSUMED transmissivity from config): diffusion distance in a year.
5. K.C./H.N. Valley difference-in-differences by distance to new dry-season water (0–3, 3–6,
   6–10 km vs 10+ km as the reference), bore and dug wells separately.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .config import ARTIFACTS, PROCESSED, RAW, REPORTS, cfg
from .models.core import prep

BANDS_KM = [0, 5, 10, 20, 40, 80, np.inf]
KC_BANDS = [0, 3, 6, 10, np.inf]
KC_LABELS = ["0–3 km", "3–6 km", "6–10 km", "10+ km"]


def _xy(lat, lon):
    from .areas import well_xy

    return well_xy(lat, lon)


def _residuals(model, aquifer: str, lab: pd.DataFrame) -> pd.DataFrame:
    d = lab[lab.aquifer == aquifer].copy()
    d["r"] = d.dh - model.predict(d)
    s = 1.4826 * np.median(np.abs(d.r - np.median(d.r)))
    d["r_clip"] = np.clip(d.r / s, -2, 2)
    d["x"], d["y"] = _xy(d.lat, d.lon)
    return d


def _pairs(d: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for k, g in d.groupby("k"):
        if len(g) < 2:
            continue
        i, j = np.triu_indices(len(g), 1)
        x, y, r = g.x.to_numpy(), g.y.to_numpy(), g[col].to_numpy()
        rows.append(pd.DataFrame({"k": k, "a": r[i], "b": r[j], "km": np.hypot(x[i] - x[j], y[i] - y[j]) / 1000}))
    p = pd.concat(rows)
    p["band"] = pd.cut(p.km, BANDS_KM, right=False, labels=False)
    return p


def _corr(p):
    a, b = np.r_[p.a, p.b], np.r_[p.b, p.a]          # symmetric pair correlation
    return float(np.corrcoef(a, b)[0, 1]) if len(p) > 10 else None


def residual_correlation(model, lab, aquifer, B=200, seed=0) -> dict:
    d = _residuals(model, aquifer, lab)
    out = {"aquifer": aquifer, "bands_km": [[BANDS_KM[i], None if np.isinf(BANDS_KM[i + 1]) else BANDS_KM[i + 1]]
                                             for i in range(len(BANDS_KM) - 1)], "n_rounds": int(d.k.nunique())}
    rng = np.random.default_rng(seed)
    for col in ("r", "r_clip"):
        p = _pairs(d, col)
        by_k = {k: g for k, g in p.groupby("k")}
        ks = np.array(list(by_k))
        est = [_corr(p[p.band == b]) for b in range(len(BANDS_KM) - 1)]
        boot = []
        for _ in range(B):
            s = pd.concat([by_k[k] for k in rng.choice(ks, len(ks))])
            boot.append([_corr(s[s.band == b]) for b in range(len(BANDS_KM) - 1)])
        boot = np.array(boot, float)
        out["pearson" if col == "r" else "huber_clipped"] = {
            "corr": est, "lo90": np.nanpercentile(boot, 5, 0).tolist(), "hi90": np.nanpercentile(boot, 95, 0).tolist(),
            "pairs": [int((p.band == b).sum()) for b in range(len(BANDS_KM) - 1)]}
    return out


def lagged_neighbour_test(model, lab, aquifer="deep") -> dict:
    import statsmodels.formula.api as smf

    d = _residuals(model, aquifer, lab)
    e = pd.read_csv(PROCESSED / "edges.csv")
    R = d.set_index(["well_id", "k"]).r
    keys = set(R.index)
    rows = []
    for src, g in e.groupby("src"):
        for (w, k) in [(src, k) for k in d.loc[d.well_id == src, "k"]]:
            vals = [(R[(n, k - 1)], ww) for n, ww in zip(g.dst, g.w) if (n, k - 1) in keys]
            if vals and (w, k - 1) in keys:
                rows.append((w, k, R[(w, k)], np.average([v for v, _ in vals], weights=[x for _, x in vals]), R[(w, k - 1)]))
    L = pd.DataFrame(rows, columns=["w", "k", "r", "nb_lag", "own_lag"])
    m = smf.ols("r ~ nb_lag + own_lag + C(k)", L).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(L.w)[0]})
    ci = m.conf_int().loc["nb_lag"].tolist()
    return {"aquifer": aquifer, "coef": float(m.params.nb_lag), "ci95": [float(ci[0]), float(ci[1])], "n": int(len(L)),
            "model": "residual_k ~ weighted mean of graph neighbours' residual_(k−1) + own residual_(k−1) + round FE"}


def engine_spillover_test(center=(13.20, 78.10), n_wells=16, add_ha=500.0, H=4) -> dict:
    from .service import TwinService

    svc = TwinService()
    tw = svc.twin
    st = svc.nowcast()["depths"][-1]
    x, y = _xy(tw.wells.lat, tw.wells.lon)
    cx, cy = _xy([center[0]], [center[1]])
    dist = np.hypot(x - cx[0], y - cy[0]) / 1000
    deep = np.where(tw.aquifer == "deep")[0]
    sel = deep[np.argsort(dist[deep])[:n_wells]]
    orig = tw._water_at
    out = {"center": list(center), "n_treated": n_wells, "add_ha": add_ha, "radius_km": float(dist[sel].max()), "runs": []}
    for beta in (0.0, 1.0):
        saved = dict(tw.model.beta)
        tw.model.beta = {k: beta for k in saved}
        try:
            b = tw.rollout(svc.k_now, H, state0=st, climatology_after=svc.k_now)

            def wa(year):
                a, a1 = orig(year)
                a = a.copy()
                a[sel] += add_ha
                return a, a1
            tw._water_at = wa
            s = tw.rollout(svc.k_now, H, state0=st, climatology_after=svc.k_now)
        finally:
            tw._water_at = orig
            tw.model.beta = saved
        dl = s["depths"][-1] - b["depths"][-1]
        others = np.setdiff1d(np.arange(tw.n), sel)
        moved = others[np.abs(dl[others]) > 1e-4]
        out["runs"].append({"beta": beta, "treated_mean_m": float(np.mean(dl[sel])),
                            "others_n": int(len(others)), "others_moved_n": int(len(moved)),
                            "others_max_abs_m": float(np.nanmax(np.abs(dl[others]))),
                            "moved": [{"km": float(dist[i]), "delta_m": float(dl[i])} for i in moved]})
    return out


def hydraulics() -> dict:
    from .areas import diffusivity_range

    out = {}
    for aq in ("deep", "shallow"):
        D = diffusivity_range(aq)
        out[aq] = {"T_m2_per_day": D["T_m2_per_day"], "S": D["S"],
                   "sqrt_Dt_km_1yr": {k: float(np.sqrt(D[k] * 365) / 1000) for k in ("low", "mid", "high")},
                   "two_sqrt_Dt_km_1yr": {k: float(2 * np.sqrt(D[k] * 365) / 1000) for k in ("low", "mid", "high")}}
    out["note"] = ("ASSUMED transmissivity (config hydraulics). sqrt(D·t) is the distance at which a head change "
                   "has fallen to about half; 2·sqrt(D·t) to about 16 %. Check against NAQUIM aquifer tests.")
    return out


# ----------------------------------------------------------------------------- K.C. Valley by distance
def new_water_distance(min_pixels: int = 5) -> pd.DataFrame:
    """Distance (km) from each well to the nearest pixel that holds dry-season water after the
    project (mean frequency 2020–2025 ≥ 0.5) but did not before (2017–2018 < 0.25)."""
    import rasterio
    from scipy import ndimage

    def mean_of(years):
        arrs = []
        for y in years:
            p = RAW / "water" / f"dry_water_{y}.tif"
            if p.exists():
                with rasterio.open(p) as src:
                    arrs.append(src.read(1))
                    tr, shp = src.transform, src.shape
        return np.nanmean(np.array(arrs), 0), tr, shp

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pre, tr, shp = mean_of([2017, 2018])
        post, _, _ = mean_of(range(2020, 2026))
    new = (np.nan_to_num(post) >= 0.5) & (np.nan_to_num(pre, nan=0) < 0.25)
    lab, n = ndimage.label(new)
    sizes = ndimage.sum(new, lab, range(1, n + 1))
    keep = np.isin(lab, np.where(sizes >= min_pixels)[0] + 1)
    lat0 = 13.35
    dy = abs(tr.e) * 111_320
    dx = tr.a * 111_320 * np.cos(np.radians(lat0))
    edt = ndimage.distance_transform_edt(~keep, sampling=(dy, dx)) / 1000
    w = pd.read_csv(PROCESSED / "wells.csv")
    col = ((w.lon - tr.c) / tr.a).astype(int).clip(0, shp[1] - 1)
    row = ((w.lat - tr.f) / tr.e).astype(int).clip(0, shp[0] - 1)
    w["km_new_water"] = edt[row, col]
    w["new_water_ha_total"] = float(keep.sum() * dx * dy / 1e4)
    return w[["well_id", "km_new_water", "new_water_ha_total"]]


class _Fit:
    def __init__(self, params, bse, pvalues):
        self.params, self.bse, self.pvalues = params, bse, pvalues


def _within_ols(d: pd.DataFrame, y: str, xs: list[str], absorb: str, fe: str, cluster: str) -> _Fit:
    """OLS with `absorb` fixed effects removed by within-demeaning (not dummies) plus `fe` dummies,
    cluster-robust SE. Absorbed effects are nested in clusters, so they do not enter the small-sample
    correction (as in Stata's reghdfe); with dummies they would inflate it for small samples."""
    from scipy import stats

    X = pd.concat([d[xs].astype(float), pd.get_dummies(d[fe], prefix="fe", drop_first=True, dtype=float)], axis=1)
    Y = d[y].astype(float)
    g = d[absorb]
    Xw = X - X.groupby(g).transform("mean")
    Yw = Y - Y.groupby(g).transform("mean")
    keep = Xw.columns[(Xw.abs().sum() > 1e-9)]
    Xw = Xw[keep].to_numpy()
    beta, *_ = np.linalg.lstsq(Xw, Yw.to_numpy(), rcond=None)
    e = Yw.to_numpy() - Xw @ beta
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    cl = pd.factorize(d[cluster])[0]
    G = cl.max() + 1
    meat = np.zeros((Xw.shape[1], Xw.shape[1]))
    for c in range(G):
        sc = Xw[cl == c].T @ e[cl == c]
        meat += np.outer(sc, sc)
    N, K = Xw.shape
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((N - 1) / (N - K))
    se = np.sqrt(np.diag(V))
    names = list(keep)
    params = pd.Series(beta, index=names)
    bse = pd.Series(se, index=names)
    pv = pd.Series(2 * stats.t.sf(np.abs(beta / se), G - 1), index=names)
    return _Fit(params, bse, pv)


def did_by_distance(aquifer: str) -> dict:
    import statsmodels.formula.api as smf

    p = pd.read_parquet(PROCESSED / "panel.parquet")
    dist = new_water_distance()
    d = p[p.depth_m.notna() & (p.aquifer == aquifer) & p.in_study].merge(dist, on="well_id")
    d = d[d.year.between(2010, 2025)]
    span = d.groupby("well_id").year.agg(["min", "max"])
    d = d[d.well_id.isin(span[(span["min"] <= 2017) & (span["max"] >= 2020)].index)].copy()
    d["band"] = pd.cut(d.km_new_water, KC_BANDS, right=False, labels=KC_LABELS)
    d["post"] = (d.date >= pd.Timestamp("2019-06-01")).astype(int)
    d = d[(d.date < pd.Timestamp("2018-06-01")) | (d.post == 1)].copy()
    d["ws"] = d.well_id + "_" + d.month.astype(str)
    d["rain_k"] = d.rain_12m_before.fillna(d.rain_12m_before.mean()) / 100
    terms = []
    for i, lb in enumerate(KC_LABELS[:-1]):
        c = f"post_b{i}"
        d[c] = ((d.band == lb) & (d.post == 1)).astype(int)
        terms.append(c)
    n_by_band = d.groupby("band", observed=False).well_id.nunique().to_dict()
    m = _within_ols(d, "depth_m", ["rain_k"] + terms, absorb="ws", fe="k", cluster="well_id")
    rows = []
    for i, lb in enumerate(KC_LABELS):
        if i == len(KC_LABELS) - 1:
            rows.append({"band": lb, "n_wells": int(n_by_band.get(lb, 0)), "delta_m": 0.0, "ci95": [0.0, 0.0], "reference": True})
            continue
        b, se = m.params[f"post_b{i}"], m.bse[f"post_b{i}"]
        rows.append({"band": lb, "n_wells": int(n_by_band.get(lb, 0)), "delta_m": float(b),
                     "ci95": [float(b - 1.96 * se), float(b + 1.96 * se)], "p": float(m.pvalues[f"post_b{i}"])})
    return {"aquifer": aquifer, "bands": rows, "n_obs": int(len(d)),
            "n_wells": int(d.well_id.nunique()),
            "design": "depth ~ well×season FE (absorbed) + round FE + 12-month rain + post×band (10+ km = reference); "
                      "post = from Jun 2019, Jun 2018–May 2019 dropped; SE clustered by well",
            "note": "negative = shallower water after the project than wells 10+ km from new water"}


def run() -> dict:
    import pickle

    t = time.time()
    lab = prep(pd.read_parquet(PROCESSED / "panel.parquet").dropna(subset=["depth_m", "dh", "rain_int", "et0_int"]))
    dep = pickle.load(open(ARTIFACTS / "deploy_model.pkl", "rb"))["hybrid"]
    ev = pickle.load(open(ARTIFACTS / "eval_models.pkl", "rb"))["hybrid"]
    res = {"residual_correlation": {}, "residual_correlation_out_of_sample": {}}
    print("1) residual correlation by distance")
    for aq in ("deep", "shallow"):
        res["residual_correlation"][aq] = residual_correlation(dep, lab, aq)
        oos = lab[lab.split.isin(["calib", "test"])]
        res["residual_correlation_out_of_sample"][aq] = residual_correlation(ev.phys, oos, aq)
        print("  ", aq, np.round(res["residual_correlation"][aq]["pearson"]["corr"], 3).tolist(),
              "clipped", np.round(res["residual_correlation"][aq]["huber_clipped"]["corr"], 3).tolist())
    print("2) lagged neighbour test")
    res["lagged_neighbour"] = [lagged_neighbour_test(dep, lab, "deep"), lagged_neighbour_test(dep, lab, "shallow")]
    print("  ", [(r["aquifer"], round(r["coef"], 3), np.round(r["ci95"], 3).tolist()) for r in res["lagged_neighbour"]])
    print("3) engine spillover test")
    res["engine_test"] = engine_spillover_test()
    print("  ", [(r["beta"], round(r["treated_mean_m"], 2), r["others_moved_n"], round(r["others_max_abs_m"], 4)) for r in res["engine_test"]["runs"]])
    res["hydraulics"] = hydraulics()
    print("4) K.C. Valley DiD by distance band")
    res["kc_valley_by_distance"] = [did_by_distance("deep"), did_by_distance("shallow")]
    for r in res["kc_valley_by_distance"]:
        print("  ", r["aquifer"], [(b["band"], b["n_wells"], round(b["delta_m"], 2), np.round(b["ci95"], 1).tolist()) for b in r["bands"]])
    kc = REPORTS / "kcvalley.json"
    if kc.exists():
        k = json.loads(kc.read_text())
        res["kc_valley_treated_vs_control"] = [{"aquifer": e["aquifer"], "did_m": e.get("did_post_minus_pre_m"),
                                                "ci95": e.get("did_ci95"), "n_treated": e.get("n_treated"),
                                                "n_control": e.get("n_control")} for e in k["event_study"]]
    mc = REPORTS / "model_comparison.csv"
    if mc.exists():
        m = pd.read_csv(mc)
        m = m[(m.split == "test") & m.model.isin(["Persistence", "Physics (water balance)", "ST-GNN (GAT+GRU)", "XGBoost (no physics)", "Hybrid (physics + monotone ML)"])]
        res["stgnn_comparison"] = m[["model", "aquifer", "MAE", "skill_vs_persistence"]].to_dict("records")
    res["runtime_s"] = round(time.time() - t, 1)
    (REPORTS / "spatial_evidence.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"spatial evidence done in {res['runtime_s']} s")
    return res
