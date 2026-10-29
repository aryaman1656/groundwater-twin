"""Accuracy and uncertainty at area level (region, district, taluk). Writes reports/area_eval.json.

    python -m gwtwin areas            (~2 min; also run by `python -m gwtwin all`)

Two experiments. Neither changes the forecaster; they score it at the scale it is reported.

A. One-round skill on the held-out test years (2023+), same split and models as evaluate.py.
   For every area and round: aggregate the observed and the predicted one-round change over the
   area's wells that were measured, then compare with "no change" (predict 0).
       skill = 1 − MAE(area forecast) / MAE(no change),   corr = Pearson(pred, obs) across area-rounds
   Raw MAE is NOT the headline: averaging many wells lowers MAE by itself, so a smaller MAE at
   taluk level says nothing about skill. A bootstrap over test rounds gives a 90% range.

B. Rolling-origin, multi-round forecasts as the dashboard makes them, then rolling conformal bands.
   For each year Y (2012 … 2025) the deployed model form (physics water balance × learned damping α,
   ML weight β = 0) is refitted on transitions that end before 1 Jan Y. From each round k0 of
   year Y the twin is run from the wells' state L rounds earlier with observed rain (a now-cast with
   no readings for L rounds — the deployed forecast starts from such a now-cast, L ≈ 7), and then
   h = 1…4 rounds ahead with climatological rain (future rain unknown, as in a real forecast).
       predicted area change = median over wells of  sim(k0+h) − sim(k0)
       observed  area change = median over the same wells of  obs(k0+h) − obs(k0)
   Conformal: the 80 % band for year Y uses |error| of all earlier years (expanding window, at least
   5 years). Coverage is then checked on every year that has a band, and on 2023+ alone.
   The band is shown in the dashboard only where coverage is within ±5 points of nominal on at least
   100 area-forecasts (see band_ok); otherwise the dashboard shows the within-area spread (IQR) only.

Caveat: the twin's "pre-2018 baseline level" (used by the mean-reversion term) is computed from all
pre-2018 data, so origins before 2018 see a little of their own future through that one term.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .config import PROCESSED, REPORTS, cfg
from .models.core import Hybrid, Persistence, Physics, SeasonalMean, XGB, prep

LAB_COLS = ["depth_m", "dh", "rain_int", "et0_int"]
NOMINAL = 0.8
LAGS = (0, 7)
H = 4


class DeployedForm:
    """The deployed model's structure: physics × α, no ML (β = 0). α chosen on the last 20 % of the
    training window, exactly as Hybrid.fit does when it switches the ML off."""
    name = "Physics × damping (deployed form)"

    def fit(self, df):
        self.alpha, self.beta = {}, {}
        cut = df.k.quantile(0.8)
        early, late = df[df.k <= cut], df[df.k > cut]
        ph = Physics().fit(early)
        pp, y = ph.predict(late), late.dh.to_numpy()
        for a in df.aquifer.unique():
            m = np.asarray(late.aquifer == a)
            al = np.linspace(0, 1, 21)
            mae = [np.mean(np.abs(y[m] - x * pp[m])) if m.any() else 0 for x in al]
            self.alpha[a], self.beta[a] = float(al[int(np.argmin(mae))]), 0.0
        self.phys = Physics().fit(df)
        return self

    def predict(self, df, return_terms=False):
        p, terms = self.phys.predict(df, return_terms=True)
        a = np.array([self.alpha.get(x, 1.0) for x in np.asarray(df.aquifer, dtype=str)])
        if return_terms:
            t = {k: a * v for k, v in terms.items()}
            t["ml_residual"] = np.zeros(len(df))
            return a * p, t
        return a * p


def _areas_of(df: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    """(level, name, mask) for every area, using study wells only."""
    st = df.in_study.astype(bool) & df.taluk.notna()
    out = [("region", "region", st)]
    for d in sorted(df.loc[st, "district"].dropna().unique()):
        out.append(("district", d, st & (df.district == d)))
    for t in sorted(df.loc[st, "taluk"].dropna().unique()):
        out.append(("taluk", t, st & (df.taluk == t)))
    return out


def _skill(o, p):
    o, p = np.asarray(o, float), np.asarray(p, float)
    mae, mae0 = np.mean(np.abs(o - p)), np.mean(np.abs(o))
    return {"skill": float(1 - mae / mae0) if mae0 > 0 else None,
            "corr": float(np.corrcoef(o, p)[0, 1]) if len(o) > 2 and np.std(p) > 0 else None,
            "MAE_m": float(mae), "MAE_no_change_m": float(mae0), "n": int(len(o))}


def _boot(g: pd.DataFrame, pcol: str, B: int = 400, seed: int = 0):
    """90 % range of skill by resampling whole test rounds (keeps the cross-area correlation)."""
    rng = np.random.default_rng(seed)
    ks = g.k.unique()
    by = {k: x for k, x in g.groupby("k")}
    vals = []
    for _ in range(B):
        s = pd.concat([by[k] for k in rng.choice(ks, len(ks))])
        r = _skill(s.o, s[pcol])
        if r["skill"] is not None:
            vals.append(r["skill"])
    return [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))] if vals else [None, None]


# ----------------------------------------------------------------------------- A. one-round skill
def one_round_skill(lab: pd.DataFrame) -> dict:
    tr, te = lab[lab.split == "train"], lab[lab.split == "test"].copy()
    models = {"Physics": Physics(), "Hybrid": Hybrid(), "XGBoost": XGB(), "Seasonal mean": SeasonalMean(),
              "Deployed form": DeployedForm()}
    for nm, m in models.items():
        m.fit(tr)
        te[f"p_{nm}"] = m.predict(te)
    rows, per_area = [], []
    for aq in ("deep", "shallow"):
        d = te[te.aquifer == aq]
        for level in ("well", "taluk", "district", "region"):
            if level == "well":
                gg = d.assign(o=d.dh, area=d.well_id, n=1)
            else:
                parts = []
                for lv, name, m in _areas_of(d):
                    if lv != level:
                        continue
                    x = d[m.to_numpy()]
                    for agg in ("median", "mean"):
                        a = x.groupby("k").agg(o=("dh", agg), n=("dh", "size"),
                                               **{f"p_{k}": (f"p_{k}", agg) for k in models}).reset_index()
                        a["area"], a["agg"] = name, agg
                        parts.append(a)
                gg = pd.concat(parts)
            for agg in (("median", "mean") if level != "well" else ("well",)):
                g = gg if level == "well" else gg[gg["agg"] == agg]
                for nm in models:
                    r = {"aquifer": aq, "level": level, "agg": agg, "model": nm, **_skill(g.o, g[f"p_{nm}"]),
                         "n_areas": int(g.area.nunique()), "n_rounds": int(g.k.nunique())}
                    if nm in ("Physics", "Deployed form"):
                        r["skill_90"] = _boot(g, f"p_{nm}")
                    rows.append(r)
                if level == "taluk" and agg == "median":
                    for name, x in g.groupby("area"):
                        r = _skill(x.o, x["p_Deployed form"])
                        per_area.append({"aquifer": aq, "taluk": name, **r, "median_wells_per_round": float(x.n.median())})
    return {"rows": rows, "per_taluk": per_area}


# ----------------------------------------------------------------------------- B. rolling multi-round
def rolling_forecasts(p: pd.DataFrame, edges: pd.DataFrame, years=range(2012, 2026), lags=LAGS) -> pd.DataFrame:
    from .data.wells import round_index
    from .twin import Twin

    lab = prep(p.dropna(subset=LAB_COLS))
    twin = None
    rows = []
    for Y in years:
        t0 = time.time()
        model = DeployedForm().fit(lab[lab.target_date < pd.Timestamp(f"{Y}-01-01")])
        if twin is None:
            twin = Twin(p, edges, model)
        twin.model = model
        obs = twin.obs
        st = (twin.wells.in_study.astype(bool) & twin.wells.taluk.notna()).to_numpy()
        tal = np.asarray(twin.wells.taluk.fillna(""), dtype=str)
        dist = np.asarray(twin.wells.district.fillna(""), dtype=str)
        for m_ in (1, 5, 8, 11):
            k0 = round_index(Y, m_)
            if k0 not in obs.columns:
                continue
            o0 = obs[k0].to_numpy(float)
            for L in lags:
                out = twin.rollout(k0 - L, L + H, climatology_after=k0)
                sim0 = out["depths"][L]
                for h in range(1, H + 1):
                    if k0 + h not in obs.columns:
                        continue
                    oh = obs[k0 + h].to_numpy(float)
                    ok = st & np.isfinite(o0) & np.isfinite(oh) & np.isfinite(sim0)
                    pc = out["depths"][L + h] - sim0
                    oc = oh - o0
                    for aq in ("deep", "shallow"):
                        base = ok & (twin.aquifer == aq)
                        areas = [("region", "region", base)]
                        areas += [("district", d, base & (dist == d)) for d in ("Kolar", "Chikkaballapura")]
                        if aq == "deep":
                            areas += [("taluk", t, base & (tal == t)) for t in sorted(set(tal[st]))]
                        for i in np.where(base)[0]:          # single wells, for comparison
                            rows.append({"year": Y, "k0": k0, "lag": L, "h": h, "aquifer": aq, "level": "well",
                                         "area": twin.wid[i], "n_wells": 1, "pred": float(pc[i]), "obs": float(oc[i])})
                        for lv, name, msk in areas:
                            if msk.sum() < 3:
                                continue
                            rows.append({"year": Y, "k0": k0, "lag": L, "h": h, "aquifer": aq, "level": lv,
                                         "area": name, "n_wells": int(msk.sum()),
                                         "pred": float(np.median(pc[msk])), "obs": float(np.median(oc[msk]))})
        print(f"   rolling {Y}: α={ {a: round(v, 2) for a, v in model.alpha.items()} }  {time.time() - t0:4.1f}s")
    return pd.DataFrame(rows)


def rolling_conformal(rf: pd.DataFrame, min_years: int = 5) -> pd.DataFrame:
    """Expanding-window conformal band for each forecast; returns rf with lo/hi/covered."""
    rf = rf.copy()
    rf["err"] = rf.obs - rf.pred
    rf["lo"] = rf["hi"] = rf["q"] = np.nan
    years = sorted(rf.year.unique())
    for Y in years:
        cal = rf[rf.year < Y]
        if cal.year.nunique() < min_years:
            continue
        for key, g in rf[rf.year == Y].groupby(["lag", "h", "aquifer", "level"]):
            c = cal[(cal.lag == key[0]) & (cal.h == key[1]) & (cal.aquifer == key[2]) & (cal.level == key[3])]
            if len(c) < 10:
                continue
            n = len(c)
            q = float(np.quantile(np.abs(c.err), min(1.0, np.ceil((n + 1) * NOMINAL) / n)))
            rf.loc[g.index, "q"] = q
    rf["lo"], rf["hi"] = rf.pred - rf.q, rf.pred + rf.q
    rf["covered"] = np.where(rf.q.notna(), (rf.obs >= rf.lo) & (rf.obs <= rf.hi), np.nan)
    return rf


def coverage_table(rf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    b = rf[rf.q.notna()]
    for (L, h, aq, lv), g in b.groupby(["lag", "h", "aquifer", "level"]):
        for period, x in (("all banded years", g), ("2023+", g[g.year >= 2023])):
            if len(x) == 0:
                continue
            cov = float(x.covered.mean())
            rows.append({"lag": L, "h": h, "aquifer": aq, "level": lv, "period": period, "n": int(len(x)),
                         "n_rounds": int(x.k0.nunique()), "coverage80": cov, "width_m": float(2 * x.q.median()),
                         "skill": _skill(x.obs, x.pred)["skill"], "corr": _skill(x.obs, x.pred)["corr"],
                         "band_ok": bool(period == "all banded years" and len(x) >= 100 and abs(cov - NOMINAL) <= 0.05)})
    return pd.DataFrame(rows)


def deploy_bands(rf: pd.DataFrame, lag: int = 7) -> dict:
    """Band half-widths for the live forecast: calibrated on ALL years (the deployed forecast is
    made after them), one per (aquifer, level, h). Only used where coverage_table says band_ok."""
    out = {}
    g = rf[rf.lag == lag]
    for (aq, lv, h), x in g.groupby(["aquifer", "level", "h"]):
        n = len(x)
        out[f"{aq}|{lv}|{h}"] = {"q80_m": float(np.quantile(np.abs(x.obs - x.pred), min(1.0, np.ceil((n + 1) * NOMINAL) / n))),
                                 "n": int(n), "years": [int(x.year.min()), int(x.year.max())]}
    return out


def run() -> dict:
    t = time.time()
    p = pd.read_parquet(PROCESSED / "panel.parquet")
    edges = pd.read_csv(PROCESSED / "edges.csv")
    lab = prep(p.dropna(subset=LAB_COLS))
    print("A) one-round skill by area level (test 2023+)")
    A = one_round_skill(lab)
    tab = pd.DataFrame(A["rows"])
    print(tab[(tab.model == "Deployed form") | (tab.model == "Physics")]
          .pivot_table(index=["aquifer", "level", "agg"], columns="model", values="skill").round(3).to_string())
    print("B) rolling-origin multi-round forecasts")
    rf = rolling_forecasts(p, edges)
    rf = rolling_conformal(rf)
    cov = coverage_table(rf)
    rf.to_csv(REPORTS / "area_rolling_forecasts.csv", index=False)
    cov.to_csv(REPORTS / "area_conformal_coverage.csv", index=False)
    print(cov[(cov.h == 4)].round(3).to_string(index=False))
    bands = deploy_bands(rf)
    ok = {f"{r.aquifer}|{r.level}|{r.h}": bool(r.band_ok) for r in cov[(cov.lag == 7) & (cov.period == "all banded years")].itertuples()}
    res = {"one_round": A, "rolling_coverage": cov.to_dict("records"), "deploy_bands": bands, "band_ok": ok,
           "nominal": NOMINAL, "lags": list(LAGS), "horizon": H,
           "notes": ["skill = 1 − MAE/MAE(no change) on area medians; raw MAE falls with averaging and is not a skill measure",
                     "taluk results use 10 test rounds; treat single-taluk numbers as indicative",
                     "shallow wells: 0–1 dug wells with test readings in 5 of 11 taluks, so taluk rows for dug wells are not reported in the dashboard"],
           "runtime_s": round(time.time() - t, 1)}
    (REPORTS / "area_eval.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"area evaluation done in {res['runtime_s']} s")
    return res
