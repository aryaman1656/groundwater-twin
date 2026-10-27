"""Clean and unify groundwater observations from all sources into
   wells.csv  (one row per well)  and  rounds.csv  (one row per well × measurement round).

Conventions
-----------
* depth_m  = depth to water below ground level, metres, POSITIVE downwards (bigger = worse).
* A "round" is one of the four CGWB measurement campaigns per year: Jan, May, Aug, Nov.
  Telemetric / monthly records are sampled onto the same rounds, so every well lives on one
  quarterly time grid.
* aquifer = "shallow" for dug wells (phreatic, weathered zone) and "deep" for bore wells /
  piezometers (fractured zone, the one farmers pump). They are modelled separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import INTERIM, PROCESSED, RAW, cfg

ROUND_MONTHS = (1, 5, 8, 11)
NOMINAL_DAY = {1: 5, 5: 15, 8: 20, 11: 5}   # typical CGWB campaign dates


def round_date(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=NOMINAL_DAY[month])


def round_index(year: int, month: int) -> int:
    """Integer index of a round: 4 per year, starting at Jan 2000 = 0."""
    return (year - 2000) * 4 + ROUND_MONTHS.index(month)


def index_to_round(k: int) -> tuple[int, int]:
    return 2000 + k // 4, ROUND_MONTHS[k % 4]


def _aquifer(well_type: str | float, median_depth: float) -> str:
    wt = str(well_type).lower()
    if "dug" in wt:
        return "shallow"
    if any(s in wt for s in ("bore", "piezo", "tube")):
        return "deep"
    return "deep" if median_depth > cfg()["wells"]["shallow_max_depth_m"] else "shallow"


def load_manual() -> pd.DataFrame:
    m = pd.read_csv(RAW / "gw_manual_bbox.csv", parse_dates=["date"])
    m = m[m.data_type_description.str.contains("Water Level", case=False, na=False)]
    out = pd.DataFrame({
        "well_id": "M_" + m.station_code.astype(str),
        "name": m.station_name.str.strip(),
        "agency": m.agency_name,
        "well_type": m.well_type,
        "lat": m.latitude, "lon": m.longitude,
        "well_depth": m.well_depth.where(m.well_depth > 1.5),   # '1' is a placeholder in the source
        "date": m.date,
        "depth_m": m.data_value,
        "source": "manual",
    })
    return out


def load_telemetric() -> pd.DataFrame:
    p = RAW / "gw_telemetric_daily_bbox.csv"
    if not p.exists():
        return pd.DataFrame()
    t = pd.read_csv(p, parse_dates=["date"])
    out = pd.DataFrame({
        "well_id": "T_" + t.station_code.astype(str),
        "name": t.station_name.str.strip(),
        "agency": t.agency_name,
        "well_type": t.well_type.fillna("Bore Well"),
        "lat": t.latitude, "lon": t.longitude,
        "well_depth": t.well_depth,
        "date": t.date,
        # GPRS records store depth as a negative number; the older monthly records of the same
        # stations (2010-2019) store it as positive. Both mean "metres below ground".
        "depth_m": t.data_value.abs(),
        "source": "telemetric",
    })
    return out


def _clean(obs: pd.DataFrame) -> pd.DataFrame:
    obs = obs.dropna(subset=["depth_m", "lat", "lon"])
    obs = obs[(obs.depth_m > -2) & (obs.depth_m < 500)]
    # reject readings deeper than the well itself (+10 m tolerance)
    bad = obs.well_depth.notna() & (obs.depth_m > obs.well_depth + 10)
    obs = obs[~bad]
    # per-well robust outlier filter (median ± 6 MAD) — removes unit/typing errors only
    med = obs.groupby("well_id").depth_m.transform("median")
    mad = obs.groupby("well_id").depth_m.transform(lambda s: (s - s.median()).abs().median())
    obs = obs[(obs.depth_m - med).abs() <= 6 * np.maximum(mad, 1.0) + 5]
    return obs


def to_rounds(obs: pd.DataFrame) -> pd.DataFrame:
    """Map every observation onto the quarterly round grid.
    Manual: the reading taken in a round month.  Daily/monthly series: mean of days 1–21 of the round month."""
    obs = obs[obs.date.dt.month.isin(ROUND_MONTHS)].copy()
    obs = obs[(obs.source == "manual") | (obs.date.dt.day <= 21)]
    obs["year"] = obs.date.dt.year
    obs["month"] = obs.date.dt.month
    # median (not mean) so that pumping spikes in daily borewell records do not bias the round value
    g = obs.groupby(["well_id", "year", "month"], as_index=False).agg(depth_m=("depth_m", "median"),
                                                                       n_raw=("depth_m", "size"))
    g["k"] = [round_index(y, m) for y, m in zip(g.year, g.month)]
    g["date"] = [round_date(y, m) for y, m in zip(g.year, g.month)]
    return g


def drop_spikes(r: pd.DataFrame, min_jump: float = 15.0) -> pd.DataFrame:
    """Remove isolated one-round spikes (x jumps away and straight back), a common data-entry error.
    A reading is a spike if it differs from BOTH neighbouring rounds by > max(min_jump, 4·MAD of the
    well's round-to-round changes) in opposite directions."""
    r = r.sort_values(["well_id", "k"]).copy()
    g = r.groupby("well_id").depth_m
    d_prev = r.depth_m - g.shift(1)
    d_next = g.shift(-1) - r.depth_m
    mad = r.assign(d=d_prev.abs()).groupby("well_id").d.transform("median")
    thr = np.maximum(min_jump, 4 * mad.fillna(0))
    spike = (d_prev.abs() > thr) & (d_next.abs() > thr) & (np.sign(d_prev) != np.sign(d_next))
    print(f"   spike filter removed {int(spike.sum())} readings")
    return r[~spike.fillna(False)]


def build():
    import geopandas as gpd

    obs = pd.concat([load_manual(), load_telemetric()], ignore_index=True)
    obs = _clean(obs)
    obs.to_parquet(INTERIM / "obs_all.parquet", index=False)

    meta = (obs.sort_values("date").groupby("well_id")
            .agg(name=("name", "last"), agency=("agency", "last"), well_type=("well_type", "last"),
                 lat=("lat", "median"), lon=("lon", "median"), well_depth=("well_depth", "max"),
                 source=("source", "first"), first=("date", "min"), last=("date", "max"),
                 median_depth=("depth_m", "median"))
            .reset_index())
    meta["aquifer"] = [_aquifer(w, d) for w, d in zip(meta.well_type, meta.median_depth)]

    # district / taluk membership and distance to the study area
    tal = gpd.read_file(RAW / "gis" / "ka_subdistricts.geojson")
    study = tal[tal.dtname.isin(cfg()["region"]["districts"])][["dtname", "sdtname", "geometry"]]
    pts = gpd.GeoDataFrame(meta, geometry=gpd.points_from_xy(meta.lon, meta.lat), crs=4326)
    j = gpd.sjoin(pts, study, how="left", predicate="within").drop(columns="index_right")
    j = j[~j.index.duplicated()]
    meta["district"] = j.dtname.values
    meta["taluk"] = j.sdtname.values
    study_m = study.to_crs(32643).union_all()
    meta["dist_to_study_km"] = pts.to_crs(32643).distance(study_m).values / 1000
    meta["in_study"] = meta.district.notna()
    meta = meta[meta.dist_to_study_km <= cfg()["region"]["buffer_km"]].reset_index(drop=True)

    rounds = drop_spikes(to_rounds(obs[obs.well_id.isin(meta.well_id)]))
    n = rounds.groupby("well_id").size()
    meta["n_rounds"] = meta.well_id.map(n).fillna(0).astype(int)
    meta = meta[meta.n_rounds >= 4].reset_index(drop=True)
    rounds = rounds[rounds.well_id.isin(meta.well_id)].merge(meta[["well_id", "aquifer"]], on="well_id")

    meta.to_csv(PROCESSED / "wells.csv", index=False)
    rounds.to_csv(PROCESSED / "rounds.csv", index=False)
    print(f"wells: {len(meta)}  ({meta.in_study.sum()} inside the two districts)")
    print(meta.groupby(["aquifer", "agency"]).size().to_string())
    print(f"round observations: {len(rounds):,}  ({rounds.year.min()}–{rounds.year.max()})")
    return meta, rounds
