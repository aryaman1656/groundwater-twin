"""The digital twin simulator: roll the well network forward round by round.

    state(k0)  --step-->  state(k0+1)  --step-->  ...  state(k0+H)

At each step every well's features are rebuilt from the current (observed or simulated) state,
including neighbour features from the graph, so a change at one well propagates to its
neighbours on the next step. The same engine does three jobs:
  * multi-step forecasts (with conformal intervals per horizon),
  * what-if scenarios (rain / pumping / tank recharge),
  * the "no K.C. Valley" counterfactual used to validate the scenario engine.

Scenario semantics (all physically defined, units in metres of water table):
  rain_pct  : interval rainfall × (1 + rain_pct/100); enters the fitted water balance + monotone ML
  tank_pct  : dry-season tank water area × (1 + tank_pct/100); enters the fitted tank term
  pump_pct  : extra draft ΔD = pump_pct/100 × D_interval (district draft from CGWB GWRA-2024, m of
              water over the district area); water table response = ΔD / Sy  (Sy from config bounds,
              or NAQUIM taluk bounds from config naquim_sy when given)
  area      : optional {"level": region|district|taluk|radius, ...}. Levers then act only on wells
              inside the area (rain, pumping) or whose 3 km tank disc overlaps it (tank water, scaled
              by the overlap share). Wells beyond get NO modelled effect. See gwtwin/areas.py.
              None (default) = every well, the original behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import PROCESSED, RAW, cfg
from .data.wells import index_to_round, round_date
from .models.core import FEATURES, STATIC, prep

INTERVAL_KEY = {1: "Jan_to_May", 5: "May_to_Aug", 8: "Aug_to_Nov", 11: "Nov_to_Jan"}


@dataclass
class Scenario:
    rain_pct: float = 0.0
    pump_pct: float = 0.0
    tank_pct: float = 0.0
    sy_quantile: float = 0.5          # where in the Sy bounds to sit (0 = low Sy → big response)
    name: str = "baseline"
    area: dict | None = None
    extra: dict = field(default_factory=dict)

    def is_baseline(self):
        return self.rain_pct == 0 and self.pump_pct == 0 and self.tank_pct == 0


def district_draft_m_per_year() -> dict:
    """Annual groundwater extraction per district (CGWB GWRA-2024, ham) spread over district area -> m/yr."""
    import geopandas as gpd

    g = pd.read_csv(RAW / "opencity" / "dgwr_karnataka_2024.csv")
    g["d"] = g["Name of District"].str.strip().str.lower()
    tot = {"Kolar": float(g.loc[g.d.str.startswith("kolar"), "Total Annual Extraction"].iloc[0]),
           "Chikkaballapura": float(g.loc[g.d.str.startswith("chikkaballa"), "Total Annual Extraction"].iloc[0])}
    dist = gpd.read_file(RAW / "gis" / "ka_districts.geojson")
    dist = dist[dist.dtname.isin(tot)].to_crs(32643)
    area = dict(zip(dist.dtname, dist.area))
    return {k: tot[k] * 1e4 / area[k] for k in tot}       # ham -> m3 (×1e4) / m2


class Twin:
    def __init__(self, panel: pd.DataFrame, edges: pd.DataFrame, model, conformal_by_h: dict | None = None):
        self.c = cfg()
        self.panel = panel
        self.model = model
        self.conf = conformal_by_h or {}
        self.wells = (panel.sort_values("k").groupby("well_id").last().reset_index())
        self.wid = self.wells.well_id.to_numpy()
        self.idx = {w: i for i, w in enumerate(self.wid)}
        self.n = len(self.wid)
        e = edges[edges.src.isin(self.idx) & edges.dst.isin(self.idx)]
        self.e_src = e.src.map(self.idx).to_numpy()
        self.e_dst = e.dst.map(self.idx).to_numpy()
        self.e_w = e.w.to_numpy()
        self.forcing = pd.read_parquet(PROCESSED / "forcing_cells.parquet")
        self.f_clim = self.forcing.groupby("season_month")[["rain_int", "et0_int", "rain_prev_int"]].mean()
        self.f_idx = self.forcing.set_index(["cell_id", "k"])
        self.cell = self.wells.cell_id.to_numpy()
        self.aquifer = np.asarray(self.wells.aquifer, dtype=str)
        self.well_depth = self.wells.well_depth.to_numpy(float)
        # seasonal past mean change per well & month
        sp = panel.dropna(subset=["dh_season_past"]).sort_values("k").groupby(["well_id", "month"]).dh_season_past.last()
        self.season_past = sp
        water = PROCESSED / "well_water_yearly.csv"
        self.water = pd.read_csv(water) if water.exists() else None
        self.draft = district_draft_m_per_year()
        self.dist = np.asarray(self.wells.district.fillna(""), dtype=str)
        self.obs = panel.pivot_table(index="well_id", columns="k", values="depth_m").reindex(self.wid)
        pre = panel[(panel.date < pd.Timestamp("2018-06-01")) & panel.depth_m.notna()].groupby("well_id").depth_m
        self.base_mean = pre.mean().where(pre.size() >= 4).reindex(self.wid).to_numpy(float)

    # ------------------------------------------------------------------ helpers
    def _forcing_at(self, k: int, use_climatology: bool, rng=None):
        """(n, 3) array of [rain_int, et0_int, rain_12m_before] for round k, evaluated per rainfall cell."""
        m = index_to_round(k)[1]
        cells = np.unique(self.cell)
        vals = {}
        for cid in cells:
            key = (cid, k)
            v = self.f_idx.loc[key, ["rain_int", "et0_int", "rain_12m_before"]].to_numpy(float) \
                if key in self.f_idx.index else np.full(3, np.nan)
            if use_climatology or not np.isfinite(v[:2]).all():
                if rng is not None:   # a historical year's forcing for this season (same year everywhere)
                    v = self._sampled[m][cid]
                else:
                    cl = self.f_clim.loc[m]
                    r12 = v[2] if np.isfinite(v[2]) else self.forcing.rain_12m_before.mean()
                    v = np.array([cl.rain_int, cl.et0_int, r12])
            vals[cid] = v
        return np.array([vals[c] for c in self.cell])

    def _water_at(self, year: int):
        if self.water is None:
            return np.zeros(self.n), np.zeros(self.n)
        y = int(np.clip(year, self.water.year.min(), self.water.year.max()))
        w = self.water[self.water.year == y].set_index("well_id").reindex(self.wid)
        return w.water_ha_3km.fillna(0).to_numpy(copy=True), w.water_ha_1km.fillna(0).to_numpy(copy=True)

    def sy_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-well (low, high) specific yield: NAQUIM taluk values where given, else GEC norms."""
        b = self.c["physics"]["sy_bounds"]
        lo = np.where(self.aquifer == "shallow", b["shallow"][0], b["deep"][0]).astype(float)
        hi = np.where(self.aquifer == "shallow", b["shallow"][1], b["deep"][1]).astype(float)
        naq = self.c.get("naquim_sy") or {}
        if naq:
            tal = np.asarray(self.wells.taluk.fillna(""), dtype=str)
            for t, v in naq.items():
                for aq, (l, h) in v.items():
                    m = (tal == t) & (self.aquifer == aq)
                    lo[m], hi[m] = l, h
        return lo, hi

    def lever_weights(self, area: dict | None) -> dict:
        """Per-well weight of each scenario lever for an area (all ones when area is None)."""
        from .areas import normalise_area

        area = normalise_area(area)
        if area is None:
            one = np.ones(self.n)
            return {"rain": one, "pump": one, "tank": one, "zone": np.full(self.n, "inside"), "edge_km": np.full(self.n, np.nan)}
        key = repr(sorted(area.items()))
        cache = self.__dict__.setdefault("_fp_cache", {})
        if key not in cache:
            from .areas import footprint

            f = footprint(self.wells, area)
            cache[key] = {"rain": f.w_rain.to_numpy(), "pump": f.w_pump.to_numpy(), "tank": f.w_tank.to_numpy(),
                          "zone": f.zone.to_numpy(), "edge_km": f.edge_km.to_numpy()}
        return cache[key]

    def sy(self, q: float) -> np.ndarray:
        lo, hi = self.sy_bounds()
        # log-uniform interpolation (Sy spans an order of magnitude)
        return np.exp(np.log(lo) + q * (np.log(hi) - np.log(lo)))

    def draft_interval(self, month: int) -> np.ndarray:
        share = self.c["physics"]["draft_share_by_interval"][INTERVAL_KEY[month]]
        d_k = self.draft.get("Kolar", 0.0)
        d_c = self.draft.get("Chikkaballapura", 0.0)
        # wells outside the two districts use the mean of the two (buffer zone)
        d = np.where(self.dist == "Kolar", d_k, np.where(self.dist == "Chikkaballapura", d_c, (d_k + d_c) / 2))
        return d * share

    # ------------------------------------------------------------------ rollout
    def rollout(self, k0: int, H: int, scenario: Scenario | None = None, state0: np.ndarray | None = None,
                climatology_after: int | None = None, rng_year: int | None = None,
                water_override: dict | None = None, assimilate: bool = False):
        """Simulate H rounds from round k0. Returns dict with depths (H+1, n) and term breakdown.

        state0: depth at k0 (default: observed at k0, else last observation before k0).
        climatology_after: rounds >= this use climatological forcing (unknown future rain).
        water_override: {well_id: water_ha_3km} to replace tank water (counterfactuals).
        assimilate: if True, whenever a real observation exists for a well at the next round, the
            simulated state is replaced by it (direct-insertion data assimilation). This is the
            observation -> state update loop that makes the model a digital twin, not just a forecaster.
        """
        sc = scenario or Scenario()
        hist = self.obs.loc[:, [c for c in self.obs.columns if c <= k0]]
        if state0 is None:
            state0 = hist.ffill(axis=1).iloc[:, -1].to_numpy(float) if hist.shape[1] else np.full(self.n, np.nan)
        depth = np.array(state0, float)
        hist_vals = hist.to_numpy(float)
        sum_hist = np.nansum(hist_vals, 1)
        n_hist = np.sum(np.isfinite(hist_vals), 1).astype(float)
        prev = hist.iloc[:, -2].to_numpy(float) if hist.shape[1] >= 2 else np.full(self.n, np.nan)
        dh_prev = depth - prev
        sy = self.sy(sc.sy_quantile)
        lw = self.lever_weights(sc.area)
        f_rain, f_tank, f_pump = sc.rain_pct / 100 * lw["rain"], sc.tank_pct / 100 * lw["tank"], sc.pump_pct / 100 * lw["pump"]
        rng = None
        if rng_year is not None:
            rng = rng_year
            self._sampled = {}
            fy = self.forcing.copy()
            fy["yr"] = [index_to_round(k)[0] for k in fy.k]
            for m in (1, 5, 8, 11):
                sub = fy[(fy.season_month == m) & (fy.yr == rng_year)].set_index("cell_id")
                self._sampled[m] = {cid: sub.loc[cid, ["rain_int", "et0_int", "rain_12m_before"]].to_numpy(float)
                                    if cid in sub.index else self.f_clim.loc[m, ["rain_int", "et0_int", "rain_prev_int"]].to_numpy(float)
                                    for cid in set(self.cell)}
        depths = [depth.copy()]
        since = np.zeros(self.n)            # rounds since the state was last set by an observation
        since_all = [since.copy()]
        terms_all = []
        rain_prev_int = None
        static = self.wells.set_index("well_id").reindex(self.wid)
        for step in range(H):
            k = k0 + step
            y, m = index_to_round(k)
            clim = climatology_after is not None and k >= climatology_after
            f = self._forcing_at(k, clim, rng=rng if clim else None)
            rain = f[:, 0] * (1 + f_rain)
            et0 = f[:, 1]
            r12 = f[:, 2] * (1 + f_rain * min(1.0, step / 4))
            if rain_prev_int is None:
                pv = np.array([self.f_idx.loc[(c, k - 1), "rain_int"] if (c, k - 1) in self.f_idx.index else np.nan
                               for c in self.cell], float)
                rain_prev_int = np.where(np.isfinite(pv), pv, self.f_clim.loc[index_to_round(k - 1)[1], "rain_int"])
                rain_prev_int = rain_prev_int * (1 + f_rain) if step > 0 else rain_prev_int
            w3, w1 = self._water_at(index_to_round(k + 1)[0])
            if water_override:
                for wid_, v in water_override.items():
                    if wid_ in self.idx:
                        w3[self.idx[wid_]] = v
            w3 = w3 * (1 + f_tank)
            w1 = w1 * (1 + f_tank)
            past_mean = np.where(n_hist > 0, sum_hist / np.maximum(n_hist, 1), depth)
            anom = depth - past_mean
            base_mean = np.where(np.isfinite(self.base_mean), self.base_mean, past_mean)
            anom_base = depth - base_mean
            # neighbour features
            num_a = np.zeros(self.n); den = np.zeros(self.n); num_d = np.zeros(self.n); cnt = np.zeros(self.n)
            ok = np.isfinite(anom[self.e_dst])
            np.add.at(num_a, self.e_src[ok], self.e_w[ok] * anom[self.e_dst][ok])
            np.add.at(den, self.e_src[ok], self.e_w[ok])
            okd = np.isfinite(dh_prev[self.e_dst])
            np.add.at(num_d, self.e_src[okd], self.e_w[okd] * dh_prev[self.e_dst][okd])
            np.add.at(cnt, self.e_src[okd], self.e_w[okd])
            np.add.at(den, self.e_src[~ok], 0)
            nb_n = np.bincount(self.e_src[ok], minlength=self.n).astype(float)
            sp = np.array([self.season_past.get((w, m), np.nan) for w in self.wid])
            X = pd.DataFrame({
                "well_id": self.wid, "aquifer": self.aquifer, "month": m, "k": k,
                "depth_m": depth, "anom": anom, "anom_base": anom_base, "past_mean": past_mean, "dh_prev": dh_prev,
                "dh_season_past": sp, "frac_depth": depth / np.where(self.well_depth > 0, self.well_depth, np.nan),
                "rounds_since_obs": 0.0,
                "nb_anom": np.where(den > 0, num_a / np.maximum(den, 1e-9), np.nan),
                "nb_dh_prev": np.where(cnt > 0, num_d / np.maximum(cnt, 1e-9), np.nan), "nb_n": nb_n,
                "rain_int": rain, "rain_prev_int": rain_prev_int, "rain_12m_before": r12, "et0_int": et0,
                "water_ha_3km": w3, "water_ha_1km": w1,
            })
            for s in STATIC:
                X[s] = static[s].to_numpy(float) if s in static else np.nan
            X = prep(X)
            dh, terms = self.model.predict(X, return_terms=True)
            pump = f_pump * self.draft_interval(m) / sy
            terms["pumping"] = pump
            dh = dh + pump
            new = depth + dh
            new = np.maximum(new, -1.0)
            cap = np.where((self.aquifer == "shallow") & np.isfinite(self.well_depth), self.well_depth, np.inf)
            new = np.minimum(new, cap)
            since = since + 1
            if assimilate and (k + 1) in self.obs.columns:
                o = self.obs[k + 1].to_numpy(float)
                hit = np.isfinite(o)
                new[hit] = o[hit]
                since[hit] = 0
            since_all.append(since.copy())
            terms_all.append({k_: np.asarray(v_, float) for k_, v_ in terms.items()})
            # update running stats
            ok_new = np.isfinite(new)
            sum_hist = sum_hist + np.where(ok_new, new, 0)
            n_hist = n_hist + ok_new
            dh_prev = new - depth
            depth = new
            rain_prev_int = rain
            depths.append(depth.copy())
        return {"k0": k0, "depths": np.array(depths), "terms": terms_all, "wells": self.wid,
                "since_obs": np.array(since_all)}

    def intervals(self, h: int, yhat_depth: np.ndarray, depth0: np.ndarray, coverage: float):
        """Conformal interval for an h-step-ahead depth (calibrated on recursive errors)."""
        conf = self.conf.get(min(h, max(self.conf) if self.conf else h))
        if conf is None:
            return yhat_depth, yhat_depth
        df = pd.DataFrame({"aquifer": self.aquifer, "scale_past": self.wells.scale_past.to_numpy(float)})
        lo, hi = conf.interval(df, yhat_depth - depth0, coverage)
        return depth0 + lo, depth0 + hi
