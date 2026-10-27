"""Assemble the modelling panel.

One row = one well × one round k (a full quarterly grid, observed or not), with
  * depth_m (NaN if not observed) and the next-round depth (target),
  * forcing over the interval k -> k+1 (rain, ET0) and antecedent rain,
  * dry-season tank water around the well,
  * past-only state features (no information from after round k is ever used),
  * neighbour features from the spatial graph,
  * split label by the date of the TARGET round.

Output: data/processed/panel.parquet, data/processed/edges.csv, data/processed/forcing_cells.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import INTERIM, PROCESSED, cfg
from ..graph import build_edges
from .wells import ROUND_MONTHS, index_to_round, round_date, round_index

SEASON = {1: "Jan→May", 5: "May→Aug", 8: "Aug→Nov", 11: "Nov→Jan"}


def cell_forcing(clim: pd.DataFrame, k0: int, k1: int) -> pd.DataFrame:
    """Rain / ET0 sums per rainfall cell for every interval k -> k+1.
    If an interval is only partly covered by data (the current season), the missing days are
    filled with that cell's day-of-year climatology and the interval is flagged partial."""
    rows = []
    clim = clim.copy()
    clim["doy"] = clim.date.dt.dayofyear
    for cid, g in clim.groupby("cell_id"):
        g = g.set_index("date").sort_index()
        cr = g.rain_mm.cumsum()
        ce = g.et0_mm.cumsum()
        first, last = g.index.min(), g.index.max()
        doy_r = g.groupby("doy").rain_mm.mean()
        doy_e = g.groupby("doy").et0_mm.mean()

        def clim_sum(t_a, t_b, doy_mean):     # climatological sum over (t_a, t_b]
            days = pd.date_range(t_a + pd.Timedelta(days=1), t_b)
            return float(doy_mean.reindex(days.dayofyear).fillna(doy_mean.mean()).sum())

        def interval(series, doy_mean, t0, t1):
            if t0 < first:
                return np.nan, False
            if t1 <= last:
                return series.get(t1, np.nan) - series.get(t0, np.nan), False
            if t0 <= last:                         # partly observed: observed part + climatology
                return series.get(last) - series.get(t0, np.nan) + clim_sum(last, t1, doy_mean), True
            return np.nan, False

        for k in range(k0, k1 + 1):
            t0 = round_date(*index_to_round(k))
            t1 = round_date(*index_to_round(k + 1))
            r, part = interval(cr, doy_r, t0, t1)
            e, _ = interval(ce, doy_e, t0, t1)
            tm12 = t0 - pd.Timedelta(days=365)
            r12 = cr.get(t0, np.nan) - cr.get(tm12, np.nan) if tm12 >= first and t0 <= last else np.nan
            rows.append((cid, k, r, e, r12, part))
    f = pd.DataFrame(rows, columns=["cell_id", "k", "rain_int", "et0_int", "rain_12m_before", "partial"])
    f = f.sort_values(["cell_id", "k"])
    f["rain_prev_int"] = f.groupby("cell_id").rain_int.shift(1)
    f["season_month"] = [index_to_round(k)[1] for k in f.k]
    return f


def build(rounds_override: pd.DataFrame | None = None, save: bool = True, verbose: bool = True):
    c = cfg()
    wells = pd.read_csv(PROCESSED / "wells.csv")
    rounds = rounds_override if rounds_override is not None else pd.read_csv(PROCESSED / "rounds.csv")
    static = pd.read_csv(PROCESSED / "well_static.csv")
    cells = pd.read_csv(INTERIM / "well_cells.csv")
    clim = pd.read_parquet(INTERIM / "climate_daily.parquet")
    water_p = PROCESSED / "well_water_yearly.csv"
    water = pd.read_csv(water_p) if water_p.exists() else pd.DataFrame(columns=["well_id", "year"])

    k0 = round_index(c["time"]["first_year"], 1)
    k_last_clim = round_index(clim.date.max().year, 11)
    k_end = max(rounds.k.max() + 8, k_last_clim)
    fpath = PROCESSED / "forcing_cells.parquet"
    if save or not fpath.exists():
        forc = cell_forcing(clim, k0 - 4, k_end)
        forc.to_parquet(fpath, index=False)
    else:
        forc = pd.read_parquet(fpath)

    # ---- full grid per well
    grid = []
    for w in wells.itertuples():
        r = rounds[rounds.well_id == w.well_id]
        ks = np.arange(max(r.k.min(), k0), k_end + 1)
        grid.append(pd.DataFrame({"well_id": w.well_id, "k": ks}))
    p = pd.concat(grid, ignore_index=True)
    p = p.merge(rounds[["well_id", "k", "depth_m"]], on=["well_id", "k"], how="left")
    p = p.merge(wells[["well_id", "aquifer", "lat", "lon", "well_depth", "in_study", "district", "taluk", "agency"]],
                on="well_id")
    p["year"] = [index_to_round(k)[0] for k in p.k]
    p["month"] = [index_to_round(k)[1] for k in p.k]
    p["date"] = [round_date(y, m) for y, m in zip(p.year, p.month)]
    p["target_date"] = [round_date(*index_to_round(k + 1)) for k in p.k]
    p = p.sort_values(["well_id", "k"]).reset_index(drop=True)
    g = p.groupby("well_id")
    p["depth_next"] = g.depth_m.shift(-1)
    p["dh"] = p.depth_next - p.depth_m
    p["dh_prev"] = p.depth_m - g.depth_m.shift(1)

    # ---- past-only state features
    p["past_mean"] = g.depth_m.transform(lambda s: s.expanding().mean())
    p["past_n"] = g.depth_m.transform(lambda s: s.notna().cumsum())
    p["anom"] = p.depth_m - p.past_mean
    # fixed pre-intervention baseline level (before K.C. Valley started, mid-2018) where available;
    # deviations from it are what the water balance relaxes against, so a lasting rise caused by
    # filled tanks is not silently absorbed into a moving average.
    pre = p[(p.date < pd.Timestamp("2018-06-01")) & p.depth_m.notna()].groupby("well_id").depth_m
    base = pre.mean().where(pre.size() >= 4)
    p["base_mean"] = p.well_id.map(base)
    p["base_mean"] = p.base_mean.fillna(p.past_mean)
    p["anom_base"] = p.depth_m - p.base_mean
    p["last_obs"] = g.depth_m.transform(lambda s: s.ffill())
    p["rounds_since_obs"] = g.depth_m.transform(lambda s: s.isna().astype(int).groupby(s.notna().cumsum()).cumsum())
    # seasonal mean change of this well, using only changes that ended at or before round k
    p["dh_season_past"] = (p.assign(dh_done=g.dh.shift(4))       # change from k-4 -> k-3 is known at k
                           .groupby(["well_id", "month"]).dh_done
                           .transform(lambda s: s.expanding().mean()))
    p["frac_depth"] = p.depth_m / p.well_depth.where(p.well_depth > 0)
    # robust scale of this well's past changes (for conformal intervals), past-only
    p["scale_past"] = g.dh.transform(lambda s: s.shift(1).abs().expanding(min_periods=3).median())

    # ---- forcing
    p = p.merge(cells, on="well_id", how="left")
    p = p.merge(forc.drop(columns=["season_month", "partial"]), on=["cell_id", "k"], how="left")
    clim_int = forc.groupby("season_month").rain_int.mean().rename("rain_int_clim")
    p = p.merge(clim_int, left_on="month", right_index=True, how="left")
    p["rain_anom_int"] = p.rain_int - p.rain_int_clim

    # ---- dry-season tank water (Sentinel-2). Year of the TARGET round's dry season.
    p["water_year"] = [index_to_round(k + 1)[0] for k in p.k]
    if len(water):
        wy0, wy1 = int(water.year.min()), int(water.year.max())
        base = water[water.year <= 2018].groupby("well_id")[["water_ha_1km", "water_ha_3km"]].mean()
        p["water_year_c"] = p.water_year.clip(wy0, wy1)
        p = p.merge(water.rename(columns={"year": "water_year_c"}), on=["well_id", "water_year_c"], how="left")
        # before Sentinel-2 L2A (2017) we assume pre-project conditions = 2017–2018 mean (ASSUMPTION)
        pre = p.water_year < wy0
        for col in ("water_ha_1km", "water_ha_3km"):
            p.loc[pre, col] = p.loc[pre, "well_id"].map(base[col])
            p[col] = p[col].fillna(p.well_id.map(base[col])).fillna(0)
        p = p.drop(columns=["water_year_c"])
        p["water_gain_3km"] = p.water_ha_3km - p.well_id.map(base["water_ha_3km"]).fillna(0)
    else:
        p["water_ha_1km"] = p["water_ha_3km"] = p["water_gain_3km"] = 0.0

    # ---- statics
    p = p.merge(static, on="well_id", how="left")

    # ---- neighbour features (same aquifer, inverse-distance weighted, past-only)
    edges = build_edges(wells)
    if save:
        edges.to_csv(PROCESSED / "edges.csv", index=False)
    st = p[["well_id", "k", "anom", "dh_prev"]]
    e = edges.merge(st, left_on="dst", right_on="well_id").drop(columns="well_id")
    e = e.dropna(subset=["anom"])
    agg = (e.assign(wa=e.w * e.anom, wd=e.w * e.dh_prev.fillna(0), wn=e.w * e.dh_prev.notna())
           .groupby(["src", "k"]).agg(wsum=("w", "sum"), wa=("wa", "sum"), wd=("wd", "sum"),
                                       wn=("wn", "sum"), nb_n=("w", "size")).reset_index())
    agg["nb_anom"] = agg.wa / agg.wsum
    agg["nb_dh_prev"] = np.where(agg.wn > 0, agg.wd / agg.wn.replace(0, np.nan), np.nan)
    p = p.merge(agg[["src", "k", "nb_anom", "nb_dh_prev", "nb_n"]].rename(columns={"src": "well_id"}),
                on=["well_id", "k"], how="left")
    p["nb_n"] = p.nb_n.fillna(0)

    # ---- splits by target date
    t = c["time"]
    p["split"] = np.where(p.target_date <= pd.Timestamp(t["train_end"]), "train",
                          np.where(p.target_date <= pd.Timestamp(t["calib_end"]), "calib", "test"))
    if save:
        p.to_parquet(PROCESSED / "panel.parquet", index=False)
    if verbose:
        lab = p.dropna(subset=["dh", "rain_int"])
        print(f"panel rows: {len(p):,}; labelled transitions: {len(lab):,}")
        print(lab.groupby(["aquifer", "split"]).size().unstack().to_string())
    return p
