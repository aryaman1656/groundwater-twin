"""Areas: region -> district -> taluk -> well.

The forecaster is ONE pooled model fitted across both districts. Nothing here fits anything:
this module only (a) says which well belongs to which area, (b) aggregates per-well output,
(c) builds the masks and the spatial footprint used by area-limited scenarios.

Rules (config.yaml -> areas):
  * Only the 312 study wells (inside Kolar / Chikkaballapura, all with a taluk label) count in area
    totals. The 126 buffer wells feed the model but are never in a total.
  * Bore (deep) wells are summarised at region, district and taluk level.
  * Dug (shallow) wells only at region and district level, and always marked indicative.
  * A taluk is "low data" if it is in config low_data_taluks or has fewer than min_deep_wells bore wells.

Footprint of an area-limited scenario (what the engine actually does, nothing more):
  inside   rain, pumping and tank levers all apply to the well
  buffer   0 – tank_buffer_km outside the area: only the tank lever reaches the well, because the
           fitted tank term counts water within 3 km of each well. The share of the well's 3 km
           disc that lies inside the area sets how much of the tank change it sees (this assumes
           water is spread evenly over the disc — an approximation, labelled as such).
  outside  no modelled effect. The data show no detectable spillover (see gwtwin/spatial.py), and
           the deployed engine has none. An OPTIONAL lateral-flow term (lateral_flow()) gives a
           physics-only illustration from assumed transmissivity; it is never part of the forecast.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import erfc

from .config import PROCESSED, RAW, cfg

UTM = 32643
LEVEL_ORDER = {"region": 0, "district": 1, "taluk": 2}


# ----------------------------------------------------------------------------- membership
def membership(wells: pd.DataFrame) -> pd.DataFrame:
    """One row per well: aquifer, district, taluk and whether it counts in area totals."""
    m = wells[["well_id", "aquifer", "district", "taluk", "lat", "lon"]].copy()
    if "in_study" in wells:
        m["in_totals"] = wells["in_study"].astype(bool).to_numpy() & m.taluk.notna().to_numpy()
    else:
        m["in_totals"] = m.taluk.notna()
    m.loc[~m.in_totals, ["district", "taluk"]] = None
    return m.reset_index(drop=True)


def area_index(mem: pd.DataFrame) -> list[dict]:
    """All reportable areas, with bore- and dug-well counts and the low-data flag."""
    a = cfg()["areas"]
    st = mem[mem.in_totals]
    out = [{"level": "region", "id": "region", "name": "Kolar + Chikkaballapura", "district": None,
            "n_deep": int((st.aquifer == "deep").sum()), "n_shallow": int((st.aquifer == "shallow").sum())}]
    for d, g in st.groupby("district"):
        out.append({"level": "district", "id": f"district:{d}", "name": d, "district": d,
                    "n_deep": int((g.aquifer == "deep").sum()), "n_shallow": int((g.aquifer == "shallow").sum())})
    for (d, t), g in st.groupby(["district", "taluk"]):
        out.append({"level": "taluk", "id": f"taluk:{t}", "name": t, "district": d,
                    "n_deep": int((g.aquifer == "deep").sum()), "n_shallow": int((g.aquifer == "shallow").sum())})
    for r in out:
        r["low_data"] = r["level"] == "taluk" and (r["name"] in a["low_data_taluks"] or r["n_deep"] < a["min_deep_wells"])
    return out


def members(mem: pd.DataFrame, area: dict, aquifer: str | None = None) -> np.ndarray:
    """Boolean mask over mem rows: wells counted in this area's totals."""
    m = mem.in_totals.to_numpy().copy()
    if area["level"] == "district":
        m &= (mem.district == area["name"]).to_numpy()
    elif area["level"] == "taluk":
        m &= (mem.taluk == area["name"]).to_numpy()
    if aquifer:
        m &= (mem.aquifer == aquifer).to_numpy()
    return m


def levels_for(aquifer: str) -> list[str]:
    return cfg()["areas"]["deep_levels" if aquifer == "deep" else "shallow_levels"]


# ----------------------------------------------------------------------------- aggregation
def summarise(v: np.ndarray) -> dict:
    """Median, IQR and count of finite values. The IQR is the spread of wells inside the area,
    NOT a forecast interval for the area."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    q = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "median": float(q[1]), "p25": float(q[0]), "p75": float(q[2]),
            "min": float(v.min()), "max": float(v.max())}


def aggregate(mem: pd.DataFrame, values: dict[str, np.ndarray], areas: list[dict] | None = None) -> list[dict]:
    """Aggregate per-well arrays (aligned with mem) for every area and allowed aquifer.

    values: {"name": array}. Each output row: level, id, name, district, aquifer, low_data,
    indicative, and summarise() of each array.
    """
    areas = areas or area_index(mem)
    rows = []
    for a in areas:
        for aq in ("deep", "shallow"):
            if a["level"] not in levels_for(aq):
                continue
            m = members(mem, a, aq)
            r = {k: a[k] for k in ("level", "id", "name", "district", "low_data")}
            r["aquifer"] = aq
            r["indicative"] = aq == "shallow" or a["low_data"]
            r["n_wells"] = int(m.sum())
            for name, arr in values.items():
                r[name] = summarise(np.asarray(arr, float)[m])
            rows.append(r)
    return rows


# ----------------------------------------------------------------------------- geometry
@lru_cache(maxsize=1)
def polygons():
    """Taluk, district and region polygons in UTM 43N (metres)."""
    import geopandas as gpd

    t = gpd.read_file(RAW / "gis" / "ka_subdistricts.geojson")
    t = t[t.dtname.isin(cfg()["region"]["districts"])][["dtname", "sdtname", "geometry"]]
    t = t.rename(columns={"dtname": "district", "sdtname": "taluk"}).to_crs(UTM)
    t["geometry"] = t.geometry.buffer(0)
    out = {("taluk", r.taluk): r.geometry for r in t.itertuples()}
    for d, g in t.groupby("district"):
        out[("district", d)] = g.geometry.union_all()
    out[("region", "region")] = t.geometry.union_all()
    return out


def well_xy(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer

    tr = Transformer.from_crs(4326, UTM, always_xy=True)
    x, y = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x), np.asarray(y)


def normalise_area(area: dict | None) -> dict | None:
    """Accepts None / {"level": "all"} (every well, the old behaviour), region, district, taluk, radius."""
    if not area or area.get("level") in (None, "all"):
        return None
    a = dict(area)
    if a["level"] == "radius":
        a["radius_km"] = float(a.get("radius_km", 10))
        a["lat"], a["lon"] = float(a["lat"]), float(a["lon"])
        a.setdefault("name", f"{a['radius_km']:g} km around {a['lat']:.3f}, {a['lon']:.3f}")
    elif a["level"] == "region":
        a["name"] = "region"
    elif a["level"] in ("district", "taluk"):
        if (a["level"], a.get("name")) not in polygons():
            raise ValueError(f"unknown {a['level']}: {a.get('name')}")
    else:
        raise ValueError(f"unknown area level {a['level']}")
    return a


def area_geometry(area: dict):
    from shapely.geometry import Point

    if area["level"] == "radius":
        x, y = well_xy([area["lat"]], [area["lon"]])
        return Point(x[0], y[0]).buffer(area["radius_km"] * 1000, 128)
    return polygons()[(area["level"], area["name"])]


def footprint(wells: pd.DataFrame, area: dict | None) -> pd.DataFrame:
    """Per well: zone (inside / buffer / outside), distance to the area edge (km; negative inside)
    and the weights each scenario lever gets.

    Inside membership uses the well's district/taluk LABEL for district/taluk areas (the same labels
    the totals use), the polygon for the region, and the distance for a radius.
    """
    n = len(wells)
    if area is None:
        return pd.DataFrame({"well_id": wells.well_id.to_numpy(), "zone": "inside", "edge_km": np.nan,
                             "w_rain": 1.0, "w_pump": 1.0, "w_tank": 1.0})
    import shapely

    buf_km = cfg()["hydraulics"]["tank_buffer_km"]
    x, y = well_xy(wells.lat, wells.lon)
    pts = shapely.points(x, y)
    geom = area_geometry(area)
    if area["level"] == "district":
        inside = (wells.district.fillna("") == area["name"]).to_numpy()
    elif area["level"] == "taluk":
        inside = (wells.taluk.fillna("") == area["name"]).to_numpy()
    else:
        inside = shapely.contains(geom, pts)
    boundary = geom.boundary
    d_edge = shapely.distance(boundary, pts) / 1000.0
    edge_km = np.where(inside, -d_edge, d_edge)
    # labelled-inside wells that sit just outside the polygon (boundary digitising) count as inside, edge 0
    edge_km = np.where(inside & shapely.disjoint(geom, pts), 0.0, edge_km)
    # share of each well's tank disc that lies inside the area
    discs = shapely.buffer(pts, buf_km * 1000, quad_segs=16)
    frac = shapely.area(shapely.intersection(discs, geom)) / shapely.area(discs)
    frac = np.where(inside & (frac < 1e-9), 1.0, frac)            # label-inside but polygon mismatch
    zone = np.where(inside, "inside", np.where(edge_km <= buf_km, "buffer", "outside"))
    w_tank = np.where(zone == "outside", 0.0, frac)
    return pd.DataFrame({"well_id": wells.well_id.to_numpy(), "zone": zone, "edge_km": edge_km,
                         "w_rain": inside.astype(float), "w_pump": inside.astype(float), "w_tank": w_tank})


# ----------------------------------------------------------------------------- optional lateral flow
def lateral_kernel(x_m: np.ndarray, t_days: float, D: float, n: int = 64) -> np.ndarray:
    """1-D, homogeneous, semi-infinite aquifer; a uniform stress switched on at t=0 over x < 0.
    Returns the response at signed distance x (m; x < 0 inside) as a fraction of the far-inside
    response:  g(x) = (1/t) ∫0^t ½ erfc(x / (2√(Dτ))) dτ  outside, 1 − g(|x|) inside.
    Physics only (Theis-type diffusion with D = T/S); no fitted parameter.
    """
    x = np.asarray(x_m, float)
    tau = (np.arange(n) + 0.5) / n * t_days
    a = np.abs(x)[:, None] / (2 * np.sqrt(D * tau[None, :]))
    g = 0.5 * erfc(a).mean(1)
    return np.where(x >= 0, g, 1 - g)


def diffusivity_range(aquifer: str = "deep") -> dict:
    T = cfg()["hydraulics"]["transmissivity_m2_per_day"]
    S = cfg()["physics"]["sy_bounds"][aquifer]
    return {"low": T[0] / S[1], "mid": float(np.sqrt(T[0] * T[1]) / np.sqrt(S[0] * S[1])), "high": T[1] / S[0],
            "T_m2_per_day": T, "S": S}


def spread_km_per_year(aquifer: str = "deep") -> dict:
    """Diffusion length 2√(D·t) after one year — how far a head change has travelled."""
    D = diffusivity_range(aquifer)
    return {k: float(2 * np.sqrt(D[k] * 365) / 1000) for k in ("low", "mid", "high")}
