"""K.C. Valley / H.N. Valley natural experiment: does filling tanks with treated wastewater raise
groundwater, and does the twin's scenario engine reproduce the effect?

1. Treatment from data: Sentinel-2 dry-season (Feb–Apr) water area within 3 km of each well,
   before (2017–2018) vs after (2020–2025). Dry-season water after 2018 = tanks fed by pipeline.
2. Event study (difference-in-differences) on pre-monsoon (May) depth:
       depth_it = α_i + γ_t + Σ_τ δ_τ · Treated_i · 1[t = τ] + β · rain_it + ε_it
   reference year 2017, standard errors clustered by well.
3. Twin counterfactual: simulate treated wells with tank water held at pre-project levels and
   compare with the simulation using actual tank water. If the scenario engine is right, the
   simulated difference should be close to the observed δ (H4, tested against a real intervention).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import PROCESSED, REPORTS

GAIN_TREATED_HA = 20.0     # dry-season water gain within 3 km that counts as "treated"
GAIN_CONTROL_HA = 3.0


def treatment_table() -> pd.DataFrame:
    w = pd.read_csv(PROCESSED / "well_water_yearly.csv")
    pre = w[w.year <= 2018].groupby("well_id").water_ha_3km.mean().rename("water_pre")
    post = w[w.year >= 2020].groupby("well_id").water_ha_3km.mean().rename("water_post")
    t = pd.concat([pre, post], axis=1)
    t["gain_ha"] = t.water_post - t.water_pre
    t["group"] = np.where(t.gain_ha >= GAIN_TREATED_HA, "treated",
                          np.where(t.gain_ha <= GAIN_CONTROL_HA, "control", "excluded"))
    return t.reset_index()


def event_study(aquifer: str) -> dict:
    """Event study on all four measurement rounds, with well×season and round fixed effects."""
    import statsmodels.formula.api as smf

    p = pd.read_parquet(PROCESSED / "panel.parquet")
    t = treatment_table()
    d = p[p.depth_m.notna() & (p.aquifer == aquifer) & p.in_study].copy()
    d = d.merge(t[["well_id", "group", "gain_ha"]], on="well_id")
    d = d[d.group.isin(["treated", "control"]) & d.year.between(2010, 2025)]
    d["treated"] = (d.group == "treated").astype(int)
    span = d.groupby("well_id").year.agg(["min", "max"])
    keep = span[(span["min"] <= 2017) & (span["max"] >= 2020)].index      # observed before AND after
    d = d[d.well_id.isin(keep)].copy()
    if d[d.treated == 1].well_id.nunique() < 5 or d[d.treated == 0].well_id.nunique() < 5:
        return {"aquifer": aquifer, "error": "not enough treated/control wells with pre and post data"}
    d["ws"] = d.well_id + "_" + d.month.astype(str)
    d["rain_k"] = d.rain_12m_before.fillna(d.rain_12m_before.mean()) / 100
    years = sorted(d.year.unique())
    ref = 2017
    ok = [y for y in years if y != ref and d[(d.year == y) & (d.treated == 1)].well_id.nunique() >= 3
          and d[(d.year == y) & (d.treated == 0)].well_id.nunique() >= 3]
    d = d[d.year.isin(ok + [ref])]
    for y in ok:
        d[f"ev_{y}"] = ((d.year == y) & (d.treated == 1)).astype(int)
    groups = pd.factorize(d.well_id)[0]
    m = smf.ols("depth_m ~ C(ws) + C(k) + rain_k + " + " + ".join(f"ev_{y}" for y in ok), data=d).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    coefs = [{"year": int(ref), "delta_m": 0.0, "lo": 0.0, "hi": 0.0}]
    for y in ok:
        b, se = m.params[f"ev_{y}"], m.bse[f"ev_{y}"]
        coefs.append({"year": int(y), "delta_m": float(b), "lo": float(b - 1.96 * se), "hi": float(b + 1.96 * se)})
    coefs.sort(key=lambda r: r["year"])
    d["post"] = (d.date >= pd.Timestamp("2019-06-01")).astype(int)
    d2 = d[(d.date < pd.Timestamp("2018-06-01")) | (d.post == 1)]
    m2 = smf.ols("depth_m ~ C(ws) + C(k) + rain_k + treated:post", data=d2).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d2.well_id)[0]})
    b, se = m2.params["treated:post"], m2.bse["treated:post"]
    return {"aquifer": aquifer, "n_treated": int(d[d.treated == 1].well_id.nunique()),
            "n_control": int(d[d.treated == 0].well_id.nunique()), "n_obs": int(len(d2)),
            "reference_year": ref, "event_study": coefs,
            "did_post_minus_pre_m": float(b), "did_ci95": [float(b - 1.96 * se), float(b + 1.96 * se)],
            "did_p": float(m2.pvalues["treated:post"]),
            "note": "negative = water table shallower in treated wells (rise) relative to controls"}


def twin_counterfactual(twin, years=(2023, 2024)) -> dict:
    """Simulate May-2018 -> May-<year> with actual vs pre-project tank water for treated wells."""
    from .data.wells import round_index

    t = treatment_table().set_index("well_id")
    treated = [w for w in twin.wid if w in t.index and t.loc[w, "group"] == "treated"]
    if not treated:
        return {"error": "no treated wells"}
    k0 = round_index(2018, 5)
    out = {}
    for y in years:
        H = round_index(y, 5) - k0
        actual = twin.rollout(k0, H)
        cf = twin.rollout(k0, H, water_override={w: float(t.loc[w, "water_pre"]) for w in treated})
        idx = np.array([twin.idx[w] for w in treated])
        diff = actual["depths"][-1][idx] - cf["depths"][-1][idx]   # negative = project made it shallower
        aq = twin.aquifer[idx]
        out[str(y)] = {a: {"mean_effect_m": float(np.nanmean(diff[aq == a])), "n": int((aq == a).sum())}
                       for a in ("shallow", "deep") if (aq == a).any()}
    return out


def run(twin=None) -> dict:
    res = {"treatment_rule": {"treated_gain_ha_3km": GAIN_TREATED_HA, "control_gain_ha_3km": GAIN_CONTROL_HA},
           "event_study": [event_study("deep"), event_study("shallow")]}
    tt = treatment_table()
    res["n_wells_by_group"] = tt.group.value_counts().to_dict()
    tt.to_csv(REPORTS / "kcvalley_treatment.csv", index=False)
    if twin is not None:
        res["twin_counterfactual"] = twin_counterfactual(twin)
    (REPORTS / "kcvalley.json").write_text(json.dumps(res, indent=1, default=float))
    for es in res["event_study"]:
        if "did_post_minus_pre_m" in es:
            print(f"   {es['aquifer']}: DiD = {es['did_post_minus_pre_m']:+.2f} m "
                  f"(95% CI {es['did_ci95'][0]:+.2f} … {es['did_ci95'][1]:+.2f}), "
                  f"{es['n_treated']} treated vs {es['n_control']} control wells")
        else:
            print("  ", es)
    if "twin_counterfactual" in res:
        print("   twin-simulated effect:", res["twin_counterfactual"])
    return res
