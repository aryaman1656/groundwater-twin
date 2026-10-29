"""Deployment-side logic shared by the API and the static export.

  train_deploy()      fit the hybrid model on all data up to the latest complete year
  TwinService         loads the twin once; serves now-cast, forecasts, scenarios, agro-economics
"""
from __future__ import annotations

import json
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import ARTIFACTS, PROCESSED, REPORTS, RAW, cfg
from .data.wells import index_to_round, round_date, round_index
from .models.core import FEATURES, Hybrid, prep
from .twin import INTERVAL_KEY, Scenario, Twin

# Last target date used to fit the deployed model (config time.deploy_train_end). New readings after
# it are still used by the now-cast (each well is reset to its newest reading); they only change the
# fitted coefficients if you move this date and re-run deploy.
DEPLOY_TRAIN_END = str(cfg()["time"].get("deploy_train_end", "2025-06-30"))


def train_deploy():
    p = pd.read_parquet(PROCESSED / "panel.parquet")
    lab = prep(p.dropna(subset=["depth_m", "dh", "rain_int", "et0_int"]))
    lab = lab[lab.target_date <= pd.Timestamp(DEPLOY_TRAIN_END)]
    model = Hybrid().fit(lab)
    ev = pickle.load(open(ARTIFACTS / "eval_models.pkl", "rb"))
    with open(ARTIFACTS / "deploy_model.pkl", "wb") as f:
        pickle.dump({"hybrid": model, "conf_by_h": ev["conf_by_h"], "trained_through": DEPLOY_TRAIN_END,
                     "n_train": int(len(lab))}, f)
    model.phys.summary().to_csv(REPORTS / "physics_coefficients_deploy.csv", index=False)
    print(f"deployed hybrid trained on {len(lab):,} transitions (through {DEPLOY_TRAIN_END})")
    return model


def _now_round() -> int:
    """The most recent CGWB round whose nominal date has passed, plus one (the round being now-cast)."""
    today = pd.Timestamp.today().normalize()
    k = round_index(today.year, 1)
    while round_date(*index_to_round(k + 1)) <= today:
        k += 1
    return k + 1


class TwinService:
    def __init__(self):
        d = pickle.load(open(ARTIFACTS / "deploy_model.pkl", "rb"))
        self.model, self.conf_by_h = d["hybrid"], d["conf_by_h"]
        self.trained_through = d["trained_through"]
        self.panel = pd.read_parquet(PROCESSED / "panel.parquet")
        self.edges = pd.read_csv(PROCESSED / "edges.csv")
        self.twin = Twin(self.panel, self.edges, self.model, self.conf_by_h)
        self.wells = pd.read_csv(PROCESSED / "wells.csv").set_index("well_id").reindex(self.twin.wid)
        obs = self.twin.obs
        self.last_obs_k = obs.apply(lambda r: r.last_valid_index(), axis=1)
        self.k_now = _now_round()
        # start the assimilation run at the earliest "last observation" of active wells (≥ 2023)
        active = self.last_obs_k[self.last_obs_k >= round_index(2023, 1)]
        self.k_start = int(active.min()) if len(active) else int(self.last_obs_k.max())
        self.forc = pd.read_parquet(PROCESSED / "forcing_cells.parquet")
        self.last_forcing_k = int(self.forc.dropna(subset=["rain_int"]).k.max())
        self._nowcast = None

    # ------------------------------------------------------------------ nowcast + forecast
    def nowcast(self):
        """Assimilate every observation from k_start and simulate (observed rain) to the current round."""
        if self._nowcast is None:
            H = self.k_now - self.k_start
            out = self.twin.rollout(self.k_start, H, assimilate=True,
                                    climatology_after=self.last_forcing_k + 1)
            self._nowcast = out
        return self._nowcast

    def forecast(self, H: int = 3, years=None):
        """Forecast H rounds beyond now. Future rain unknown -> ensemble over historical years."""
        now = self.nowcast()
        state = now["depths"][-1]
        since0 = now["since_obs"][-1]
        years = years or list(range(2005, 2026))
        ens = []
        for y in years:
            out = self.twin.rollout(self.k_now, H, state0=state, climatology_after=self.k_now, rng_year=y)
            ens.append(out["depths"][1:])
        ens = np.array(ens)                                   # (n_years, H, n_wells)
        med = np.nanmedian(ens, 0)
        res = {"k": [self.k_now + h for h in range(1, H + 1)], "median": med,
               "ens_p10": np.nanpercentile(ens, 10, 0), "ens_p90": np.nanpercentile(ens, 90, 0),
               "ens_p05": np.nanpercentile(ens, 5, 0), "ens_p95": np.nanpercentile(ens, 95, 0)}
        # combine model error (conformal, by rounds since last real observation) with weather spread
        for cov, lo_k, hi_k in ((0.8, "ens_p10", "ens_p90"), (0.9, "ens_p05", "ens_p95")):
            lo_all, hi_all = [], []
            for h in range(1, H + 1):
                hh = np.clip(since0 + h, 1, max(self.conf_by_h)).astype(int)
                lo = np.zeros(self.twin.n); hi = np.zeros(self.twin.n)
                for hv in np.unique(hh):
                    m = hh == hv
                    l, u = self.twin.intervals(int(hv), med[h - 1], state, cov)
                    lo[m], hi[m] = l[m], u[m]
                c_half_lo, c_half_hi = med[h - 1] - lo, hi - med[h - 1]
                e_half_lo, e_half_hi = med[h - 1] - res[lo_k][h - 1], res[hi_k][h - 1] - med[h - 1]
                lo_all.append(med[h - 1] - np.sqrt(c_half_lo ** 2 + e_half_lo ** 2))
                hi_all.append(med[h - 1] + np.sqrt(c_half_hi ** 2 + e_half_hi ** 2))
            lo_a, hi_a = np.array(lo_all), np.array(hi_all)
            # physical limits: water cannot stand more than ~1 m above ground, and a dug well
            # cannot be emptier than its own depth
            cap = np.where((self.twin.aquifer == "shallow") & np.isfinite(self.twin.well_depth), self.twin.well_depth, np.inf)
            res[f"lo{int(cov * 100)}"] = np.clip(lo_a, -1.0, cap)
            res[f"hi{int(cov * 100)}"] = np.clip(hi_a, -1.0, cap)
        return res

    # ------------------------------------------------------------------ scenarios
    def _membership(self):
        from .areas import membership

        if not hasattr(self, "_mem"):
            w = self.wells.reset_index()
            self._mem = membership(w)
        return self._mem

    def scenario(self, rain_pct=0.0, pump_pct=0.0, tank_pct=0.0, horizon=2, sy_quantile=0.5, agro=True,
                 area=None, lateral=False, weather_range=False):
        """Scenario minus baseline, per well and per area.

        area: None (every well, the original behaviour) or {"level": region|district|taluk|radius, ...}.
        Outputs, in metres of water table (+ = deeper under the scenario):
          wells       per well: delta, lever split, Sy range of the pumping part, zone
          levels      region / district / taluk medians + IQR (study wells only; dug wells not by taluk)
          area_summary  the selected area's wells
          footprint   inside / buffer (tank lever only, 3 km) / outside (no modelled effect)
          pumping     area-level physics-only pumping response with the Sy range, by district
          lateral     OPTIONAL physics-only illustration of lateral spread (never added to delta)
        """
        from .areas import aggregate, area_index, diffusivity_range, lateral_kernel, normalise_area

        lim = cfg()["scenario_limits"]
        rain_pct = float(np.clip(rain_pct, *lim["rain_pct"]))
        pump_pct = float(np.clip(pump_pct, *lim["pump_pct"]))
        tank_pct = float(np.clip(tank_pct, *lim["tank_pct"]))
        horizon = int(np.clip(horizon, 1, 4))
        sy_quantile = float(np.clip(sy_quantile, 0, 1))
        area = normalise_area(area)
        tw = self.twin
        now = self.nowcast()
        state = now["depths"][-1]
        run = lambda sc: tw.rollout(self.k_now, horizon, state0=state, climatology_after=self.k_now, scenario=sc)  # noqa: E731
        base = run(None)
        scn = run(Scenario(rain_pct, pump_pct, tank_pct, sy_quantile, area=area))
        d_end = scn["depths"][-1] - base["depths"][-1]            # + = deeper under the scenario
        # lever split (the engine is linear in each lever apart from the physical depth caps)
        parts = {}
        for name, kw in (("rain", dict(rain_pct=rain_pct)), ("pump", dict(pump_pct=pump_pct)), ("tank", dict(tank_pct=tank_pct))):
            if list(kw.values())[0] == 0:
                parts[name] = np.zeros(tw.n)
            else:
                parts[name] = run(Scenario(sy_quantile=sy_quantile, area=area, **kw))["depths"][-1] - base["depths"][-1]
        # pumping is the only lever that depends on Sy: response ∝ 1/Sy, so the Sy range is exact
        sy_lo, sy_hi = tw.sy_bounds()
        sy_q = tw.sy(sy_quantile)
        pump_at_lo_sy, pump_at_hi_sy = parts["pump"] * sy_q / sy_lo, parts["pump"] * sy_q / sy_hi
        other = d_end - parts["pump"]
        d_lo = other + np.minimum(pump_at_lo_sy, pump_at_hi_sy)
        d_hi = other + np.maximum(pump_at_lo_sy, pump_at_hi_sy)
        lw = tw.lever_weights(area)
        wells = self.wells
        mem = self._membership()
        out_wells = pd.DataFrame({"well_id": tw.wid, "aquifer": tw.aquifer,
                                  "taluk": wells.taluk.to_numpy(), "district": wells.district.to_numpy(),
                                  "in_totals": mem.in_totals.to_numpy(),
                                  "baseline_depth": base["depths"][-1], "scenario_depth": scn["depths"][-1],
                                  "delta_m": d_end, "delta_lo_m": d_lo, "delta_hi_m": d_hi,
                                  "rain_m": parts["rain"], "pump_m": parts["pump"], "tank_m": parts["tank"],
                                  "zone": lw["zone"], "edge_km": lw["edge_km"], "tank_weight": lw["tank"]})
        # ---- areas: taluk table kept in the old shape for the crop chain
        st = out_wells[(out_wells.aquifer == "deep") & out_wells.in_totals]
        tal_rows = []
        for (dist_, tal_), g in st.groupby(["district", "taluk"]):
            share = float((g.zone != "outside").mean())
            inside = g[g.zone != "outside"]
            med = float(inside.delta_m.median()) if len(inside) else 0.0
            # a taluk only partly inside the area: its whole-taluk effect is diluted by the share affected
            tal_rows.append({"district": dist_, "taluk": tal_, "dh_mean": med if share == 1.0 else share * med,
                             "share_wells_affected": share})
        tal = pd.DataFrame(tal_rows)
        tal["dh_sd"] = 0.3 * tal.dh_mean.abs()
        levels = aggregate(mem, {"delta_m": d_end, "delta_lo_m": d_lo, "delta_hi_m": d_hi}, area_index(mem))
        # ---- selected area summary (study wells inside the area)
        inside = out_wells.zone.to_numpy() == "inside"
        area_summary = {}
        for aq in ("deep", "shallow"):
            m = inside & (tw.aquifer == aq) & mem.in_totals.to_numpy()
            mb = inside & (tw.aquifer == aq) & ~mem.in_totals.to_numpy()
            v = d_end[m]
            area_summary[aq] = {"n_wells": int(m.sum()), "n_buffer_wells_inside_not_counted": int(mb.sum()),
                                "median_delta_m": float(np.median(v)) if m.any() else None,
                                "p25": float(np.percentile(v, 25)) if m.any() else None,
                                "p75": float(np.percentile(v, 75)) if m.any() else None,
                                "sy_range_m": [float(np.median(d_lo[m])), float(np.median(d_hi[m]))] if m.any() else None,
                                "indicative": aq == "shallow"}
        zones = {}
        for z in ("inside", "buffer", "outside"):
            m = out_wells.zone.to_numpy() == z
            zones[z] = {"n_wells": int(m.sum()),
                        "max_abs_delta_m": float(np.nanmax(np.abs(d_end[m]))) if m.any() else 0.0,
                        "median_delta_deep_m": float(np.median(d_end[m & (tw.aquifer == "deep")])) if (m & (tw.aquifer == "deep")).any() else None}
        footprint = {"zones": zones, "tank_buffer_km": cfg()["hydraulics"]["tank_buffer_km"],
                     "meaning": {"inside": "rain, pumping and tank levers act",
                                 "buffer": "within 3 km outside the area: only tank water acts, scaled by the share of the well's 3 km disc inside the area",
                                 "outside": "no modelled effect (no spillover is detectable in the data; see Evidence)"}}
        # ---- pumping, area level, physics only
        pumping = None
        if pump_pct != 0:
            pumping = {"physics_only": True, "formula": "Δh = pump% × district draft (CGWB GWRA-2024) ÷ Sy, then the fitted relaxation",
                       "draft_m_per_year": tw.draft, "by_district": []}
            for dist_ in ("Kolar", "Chikkaballapura"):
                for aq in ("deep", "shallow"):
                    m = inside & (tw.dist == dist_) & (tw.aquifer == aq) & mem.in_totals.to_numpy()
                    if not m.any():
                        continue
                    pumping["by_district"].append({
                        "district": dist_, "aquifer": aq, "n_wells": int(m.sum()),
                        "pump_m": float(np.median(parts["pump"][m])),
                        "range_m": [float(np.median(np.minimum(pump_at_lo_sy, pump_at_hi_sy)[m])),
                                    float(np.median(np.maximum(pump_at_lo_sy, pump_at_hi_sy)[m]))],
                        "sy_used": float(np.median(sy_q[m])), "sy_bounds": [float(np.median(sy_lo[m])), float(np.median(sy_hi[m]))],
                        "sy_source": "NAQUIM (config naquim_sy)" if (cfg().get("naquim_sy") or {}) and
                                     any(t in (cfg().get("naquim_sy") or {}) for t in set(wells.taluk.to_numpy()[m])) else "CGWB GEC-2015 norms (config physics.sy_bounds)"})
            pumping["note"] = ("Every well of the same aquifer and district gets the same pumping response: there is no per-well "
                               "consumption data. This is NOT the drawdown inside a pumped bore well.")
        # ---- optional lateral flow (illustration; never added to delta)
        lat_res = None
        if lateral and area is not None:
            t_days = (round_date(*index_to_round(self.k_now + horizon)) - round_date(*index_to_round(self.k_now))).days
            far = {}
            for aq in ("deep", "shallow"):
                m = inside & (tw.aquifer == aq)
                far[aq] = float(np.median((parts["rain"] + parts["pump"])[m])) if m.any() else 0.0
            x_m = np.nan_to_num(lw["edge_km"], nan=-1e6) * 1000
            res_l = {"physics_only": True, "t_days": t_days, "levers": "rain + pumping (tank water already has its fitted 3 km reach)",
                     "far_field_m": far, "D_m2_per_day": {}, "by_distance": []}
            per = {}
            for case in ("low", "mid", "high"):
                lat_ = np.zeros(tw.n)
                for aq in ("deep", "shallow"):
                    D = diffusivity_range(aq)[case]
                    res_l["D_m2_per_day"][f"{aq}_{case}"] = D
                    m = tw.aquifer == aq
                    lat_[m] = far[aq] * lateral_kernel(x_m[m], t_days, D)
                per[case] = lat_
            for lo_, hi_ in ((-1e9, 0), (0, 1), (1, 3), (3, 6), (6, 10), (10, 1e9)):
                m = (tw.aquifer == "deep") & (x_m / 1000 >= lo_) & (x_m / 1000 < hi_)
                if m.any():
                    res_l["by_distance"].append({"from_km": lo_ if lo_ > -1e8 else None, "to_km": hi_ if hi_ < 1e8 else None,
                                                 "n_deep_wells": int(m.sum()),
                                                 **{f"{c}_m": float(np.median(per[c][m])) for c in per}})
            out_wells["lateral_mid_m"] = per["mid"]
            res_l["note"] = ("1-D diffusion from the area edge with ASSUMED transmissivity (config hydraulics) and the Sy bounds. "
                             "Not fitted, not validated, and not part of the forecast or the totals.")
            lat_res = res_l
        # ---- optional weather range for the selected area (rain/tank responses scale with the year's rain)
        wr = None
        if weather_range:
            sel = inside & mem.in_totals.to_numpy()
            vals = {"deep": [], "shallow": []}
            for y in range(2005, 2026):
                b = tw.rollout(self.k_now, horizon, state0=state, climatology_after=self.k_now, rng_year=y)
                s_ = tw.rollout(self.k_now, horizon, state0=state, climatology_after=self.k_now, rng_year=y,
                                scenario=Scenario(rain_pct, pump_pct, tank_pct, sy_quantile, area=area))
                dd = s_["depths"][-1] - b["depths"][-1]
                for aq in vals:
                    m = sel & (tw.aquifer == aq)
                    if m.any():
                        vals[aq].append(float(np.median(dd[m])))
            wr = {aq: {"p10": float(np.percentile(v, 10)), "p50": float(np.median(v)), "p90": float(np.percentile(v, 90)), "n_years": len(v)}
                  for aq, v in vals.items() if v}
        terms = {k: float(np.nanmean(np.sum([s_[k] for s_ in scn["terms"]], 0) - np.sum([b_[k] for b_ in base["terms"]], 0)))
                 for k in scn["terms"][0]}
        res = {"inputs": {"rain_pct": rain_pct, "pump_pct": pump_pct, "tank_pct": tank_pct, "horizon": horizon,
                          "sy_quantile": sy_quantile, "area": area, "lateral": bool(lat_res), "weather_range": bool(weather_range)},
               "from_round": str(round_date(*index_to_round(self.k_now)).date()),
               "to_round": str(round_date(*index_to_round(self.k_now + horizon)).date()),
               "wells": out_wells.replace({np.nan: None}).to_dict("records"),
               "taluks": tal.replace({np.nan: None}).to_dict("records"),
               "levels": levels, "area_summary": area_summary, "footprint": footprint, "pumping": pumping,
               "lateral": lat_res, "weather_range": wr,
               "summary": {a: {"mean_delta_m": float(np.nanmean(d_end[tw.aquifer == a])),
                               "share_wells_deeper_than_0.5m": float(np.nanmean(d_end[tw.aquifer == a] > 0.5))}
                           for a in ("shallow", "deep")},
               "drivers_mean_m": terms,
               "well_note": ("Per-well numbers are the typical response of this aquifer type at this location: the water-balance "
                             "coefficients are shared by every well of the same aquifer, and pumping is the district average ÷ Sy. "
                             "The range shown is from the Sy bounds (pumping) only; it is not a forecast interval.")}
        if agro:
            from .agro import run_chain

            months = [index_to_round(self.k_now + h)[1] for h in range(horizon)]
            share = sum(cfg()["physics"]["draft_share_by_interval"][INTERVAL_KEY[m]] for m in months)
            clim = self.forc.groupby("season_month")[["rain_int", "et0_int"]].mean()
            rain = sum(clim.loc[m, "rain_int"] for m in months) * (1 + rain_pct / 100)
            et0 = sum(clim.loc[m, "et0_int"] for m in months)
            days = (round_date(*index_to_round(self.k_now + horizon)) - round_date(*index_to_round(self.k_now))).days
            res["agro"] = run_chain(tal, share, et0 / days, rain / days * 30.4)
        return res

    def _last_reading_depth(self) -> np.ndarray:
        tw = self.twin
        out = np.full(tw.n, np.nan)
        for i, k in enumerate(self.last_obs_k.reindex(tw.wid).to_numpy(float)):
            if np.isfinite(k):
                out[i] = float(tw.obs.iloc[i][int(k)])
        return out

    # ------------------------------------------------------------------ area forecast (task: taluk aggregation)
    def area_forecast(self, H: int = 4) -> dict:
        """Baseline forecast aggregated to region / district / taluk (study wells only).

        Per area and aquifer: median change from now to each forecast round, the IQR across wells
        (the spread inside the area, NOT an interval), well counts, data age, low-data and
        indicative flags, plus the area-level accuracy record from reports/area_eval.json."""
        from .areas import aggregate, area_index

        now = self.nowcast()
        fc = self.forecast_cached(H)
        tw = self.twin
        mem = self._membership()
        state = now["depths"][-1]
        since = now["since_obs"][-1].astype(float)
        vals = {"change_m": fc["median"][-1] - state, "nowcast_depth_m": state,
                "rounds_since_reading": since,
                "last_reading_depth_m": self._last_reading_depth()}
        for h in range(1, H + 1):
            vals[f"change_h{h}_m"] = fc["median"][h - 1] - state
        areas = area_index(mem)
        rows = aggregate(mem, vals, areas)
        lk = self.last_obs_k.reindex(tw.wid).to_numpy(float)
        ev = json.loads((REPORTS / "area_eval.json").read_text()) if (REPORTS / "area_eval.json").exists() else None
        from .areas import members

        for r in rows:
            a = next(x for x in areas if x["id"] == r["id"])
            m = members(mem, a, r["aquifer"])
            ks = [int(k) for k in lk[m] if np.isfinite(k)]
            r["latest_reading"] = str(round_date(*index_to_round(max(ks))).date()) if ks else None
            r["wells_read_within_4_rounds"] = int(np.sum(since[m] <= 4))
            r["trajectory"] = [{"date": str(round_date(*index_to_round(k)).date()),
                                **{q: r[f"change_h{h}_m"][q] for q in ("median", "p25", "p75")}}
                               for h, k in enumerate(fc["k"], start=1)]
            r["accuracy"] = _area_accuracy(ev, r["level"], r["aquifer"], r["name"]) if ev else None
        return {"now_round": str(round_date(*index_to_round(self.k_now)).date()),
                "target_round": str(round_date(*index_to_round(fc["k"][-1])).date()),
                "rows": rows, "areas": areas,
                "rules": {"deep_levels": cfg()["areas"]["deep_levels"], "shallow_levels": cfg()["areas"]["shallow_levels"],
                          "buffer_wells_excluded": int((~mem.in_totals).sum()),
                          "spread": "IQR = 25th–75th percentile of wells inside the area; it is NOT a forecast interval"}}

    # ------------------------------------------------------------------ per-well detail
    def well_detail(self, well_id: str, H: int = 4):
        tw = self.twin
        i = tw.idx[well_id]
        obs = tw.obs.iloc[i].dropna()
        now = self.nowcast()
        fc = self.forecast_cached(H)
        k_sim = [self.k_start + h for h in range(now["depths"].shape[0])]
        detail = {
            "well": {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                     for k, v in self.wells.iloc[i].to_dict().items()},
            "well_id": well_id,
            "observed": [{"date": str(round_date(*index_to_round(int(k))).date()), "depth": float(v)} for k, v in obs.items()],
            "nowcast": [{"date": str(round_date(*index_to_round(k)).date()), "depth": float(now["depths"][j][i]),
                         "observed": bool(now["since_obs"][j][i] == 0)} for j, k in enumerate(k_sim)],
            "forecast": [{"date": str(round_date(*index_to_round(k)).date()), "median": float(fc["median"][h][i]),
                          "lo80": float(fc["lo80"][h][i]), "hi80": float(fc["hi80"][h][i]),
                          "lo90": float(fc["lo90"][h][i]), "hi90": float(fc["hi90"][h][i])}
                         for h, k in enumerate(fc["k"])],
        }
        detail["explain"] = self.explain(well_id)
        return detail

    @lru_cache(maxsize=4)
    def forecast_cached(self, H=4):
        return self.forecast(H)

    def explain(self, well_id: str):
        """Why the next-round forecast is what it is: physics terms + SHAP of the ML residual."""
        import shap

        tw = self.twin
        i = tw.idx[well_id]
        now = self.nowcast()
        state = now["depths"][-1]
        out = tw.rollout(self.k_now, 1, state0=state, climatology_after=self.k_now)
        terms = {k: float(v[i]) for k, v in out["terms"][0].items()}
        # Show weather terms as departures from a normal season, so large offsetting terms
        # (e.g. evaporative demand vs the seasonal intercept) are folded into "typical change".
        m = index_to_round(self.k_now)[1]
        aq = tw.aquifer[i]
        coef = self.model.phys.coef[aq]
        alpha = self.model.alpha.get(aq, 1.0)
        clim = self.forc[self.forc.season_month == m][["rain_int", "et0_int", "rain_prev_int", "rain_12m_before"]].mean()
        X0 = self._feature_rows(state).iloc[i]
        typical = terms.get("season", 0.0)
        for c in ("rain_int", "et0_int", "rain_prev_int", "rain_12m_before"):
            if c in terms and c in coef:
                normal = alpha * coef[c] * float(clim[c])
                typical += normal
                terms[c] = alpha * coef[c] * float(X0[c]) - normal
        terms["season"] = typical
        # rebuild the feature row the model saw (single step) to explain the ML part
        X = self._feature_rows(state)
        aq = tw.aquifer[i]
        m = self.model.ml.models[aq]
        expl = shap.TreeExplainer(m)
        # SHAP of the raw ML model, scaled by the weight the deployed model actually gives it (α·β)
        w_ml = self.model.alpha.get(aq, 1.0) * self.model.beta.get(aq, 1.0)
        sv = expl.shap_values(X.iloc[[i]][self.model.ml.features])[0] * w_ml
        top = sorted(zip(self.model.ml.features, sv, X.iloc[i][self.model.ml.features]), key=lambda t: -abs(t[1]))[:8]
        return {"ml_weight": w_ml,
                "physics_terms_m": {k: v for k, v in terms.items() if k != "ml_residual"},
                "ml_residual_m": terms.get("ml_residual", 0.0),
                "ml_top_features": [{"feature": f, "shap_m": float(s), "value": None if pd.isna(v) else float(v)} for f, s, v in top],
                "neighbours": self.neighbours(well_id)}

    def _feature_rows(self, state):
        """Feature frame at the current round (same construction as the twin's first step)."""
        tw = self.twin
        cap = {}
        orig = tw.model.predict

        def grab(X, return_terms=False):
            cap["X"] = X
            return orig(X, return_terms=return_terms)

        tw.model.predict = grab
        try:
            tw.rollout(self.k_now, 1, state0=state, climatology_after=self.k_now)
        finally:
            tw.model.predict = orig
        return cap["X"]

    def neighbours(self, well_id: str):
        e = self.edges[self.edges.src == well_id].sort_values("km")
        w = self.wells
        return [{"well_id": r.dst, "km": round(float(r.km), 1), "name": w.loc[r.dst, "name"] if r.dst in w.index else None,
                 "weight": round(float(r.w / e.w.sum()), 3)} for r in e.itertuples()]

    # ------------------------------------------------------------------ map layer
    def map_layer(self, H: int = 4):
        now = self.nowcast()
        fc = self.forecast_cached(H)
        tw = self.twin
        w = self.wells
        last_k = self.last_obs_k
        mem_tot = self._membership().in_totals.to_numpy()
        rows = []
        for i, wid in enumerate(tw.wid):
            lk = last_k.get(wid)
            last_depth = float(tw.obs.iloc[i][lk]) if lk is not None and np.isfinite(tw.obs.iloc[i][lk]) else None
            now_d = float(now["depths"][-1][i])
            f_end = float(fc["median"][-1][i])
            rows.append({
                "well_id": wid, "name": w.iloc[i]["name"], "lat": float(w.iloc[i].lat), "lon": float(w.iloc[i].lon),
                "aquifer": tw.aquifer[i], "agency": w.iloc[i].agency, "well_type": w.iloc[i].well_type,
                "district": w.iloc[i].district if isinstance(w.iloc[i].district, str) else None,
                "taluk": w.iloc[i].taluk if isinstance(w.iloc[i].taluk, str) else None,
                "well_depth": None if pd.isna(w.iloc[i].well_depth) else float(w.iloc[i].well_depth),
                "last_obs_date": str(round_date(*index_to_round(int(lk))).date()) if lk is not None else None,
                "last_obs_depth": last_depth, "nowcast_depth": now_d,
                "rounds_since_obs": int(now["since_obs"][-1][i]),
                "in_totals": bool(mem_tot[i]),
                "forecast": [{"date": str(round_date(*index_to_round(k)).date()), "median": float(fc["median"][h][i]),
                              "lo80": float(fc["lo80"][h][i]), "hi80": float(fc["hi80"][h][i])}
                             for h, k in enumerate(fc["k"])],
                # same season one year later minus now: free of the seasonal up-and-down
                "forecast_change_m": f_end - now_d,
                "active": bool(lk is not None and int(lk) >= round_index(2021, 1)),
                "uncertainty_80_m": float(fc["hi80"][-1][i] - fc["lo80"][-1][i]),
            })
        return rows

    def anomalies(self, z_thresh: float = 6.0, since: str = "2024-01-01"):
        """Wells whose latest observed change (since `since`) is far outside what the model expected.
        z = (observed − expected change) / the well's typical change; |z| ≥ 6 is rare by construction."""
        p = self.panel
        lab = prep(p.dropna(subset=["depth_m", "dh", "rain_int", "et0_int"]))
        lab = lab[lab.target_date >= pd.Timestamp(since)]
        last = lab.sort_values("k").groupby("well_id").tail(1)
        yhat = self.model.predict(last)
        s = np.maximum(last.scale_past.fillna(last.groupby("aquifer").scale_past.transform("median")).to_numpy(float), 0.25)
        z = (last.dh.to_numpy(float) - yhat) / s
        last = last.assign(z=z, expected_dh=yhat)
        flag = last[np.abs(last.z) >= z_thresh]
        return [{"well_id": r.well_id, "date": str(r.target_date.date()), "observed_change_m": float(r.dh),
                 "expected_change_m": float(r.expected_dh), "z": float(r.z)} for r in flag.itertuples()]


def _area_accuracy(ev: dict, level: str, aquifer: str, name: str) -> dict:
    """Pull the area-level accuracy record (reports/area_eval.json) that applies to one area row."""
    out = {}
    one = [r for r in ev["one_round"]["rows"] if r["level"] == level and r["aquifer"] == aquifer
           and r["agg"] == "median" and r["model"] == "Deployed form"]
    if one:
        out["one_round_all_areas"] = {k: one[0].get(k) for k in ("skill", "corr", "skill_90", "n", "n_areas", "n_rounds")}
    if level == "taluk":
        t = [r for r in ev["one_round"]["per_taluk"] if r["taluk"] == name and r["aquifer"] == aquifer]
        if t:
            out["one_round_this_taluk"] = {k: t[0].get(k) for k in ("skill", "corr", "n", "median_wells_per_round")}
    roll = [r for r in ev["rolling_coverage"] if r["level"] == level and r["aquifer"] == aquifer and r["lag"] == 7
            and r["period"] == "all banded years"]
    out["rolling_by_horizon"] = [{k: r.get(k) for k in ("h", "n", "n_rounds", "coverage80", "width_m", "skill", "corr", "band_ok")}
                                 for r in sorted(roll, key=lambda r: r["h"])]
    return out


def meta() -> dict:
    wells = pd.read_csv(PROCESSED / "wells.csv")
    ev = json.loads((REPORTS / "evaluation.json").read_text()) if (REPORTS / "evaluation.json").exists() else {}
    kc = json.loads((REPORTS / "kcvalley.json").read_text()) if (REPORTS / "kcvalley.json").exists() else {}
    return {"region": cfg()["region"]["name"], "n_wells": int(len(wells)), "n_in_study": int(wells.in_study.sum()),
            "wells_by_aquifer": wells.aquifer.value_counts().to_dict(), "evaluation": ev, "kcvalley": kc,
            "config": {k: cfg()[k] for k in ("time", "physics", "scenario_limits", "agro")}}
