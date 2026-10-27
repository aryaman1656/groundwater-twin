"""Static (time-invariant) well attributes and yearly dry-season surface water around each well.

Outputs
  data/processed/well_static.csv        elevation, slope, relative elevation, soil texture, land cover
  data/processed/well_water_yearly.csv  dry-season water area (ha) within 1 km and 3 km, per year
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PROCESSED, RAW

WC_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "cropland", 50: "built", 60: "bare", 80: "water"}


def _window_stats(src, lon, lat, radius_km):
    """Return the pixel block within radius_km of (lon, lat) and a circular mask."""
    from rasterio.windows import from_bounds

    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * np.cos(np.deg2rad(lat)))
    win = from_bounds(lon - dlon, lat - dlat, lon + dlon, lat + dlat, src.transform).round_offsets().round_lengths()
    arr = src.read(1, window=win, boundless=True, fill_value=src.nodata if src.nodata is not None else 0)
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((yy - (h - 1) / 2) / (h / 2)) ** 2 + ((xx - (w - 1) / 2) / (w / 2)) ** 2 <= 1
    return arr, mask


def _pixel_area_ha(src, lat):
    a, e = src.transform.a, -src.transform.e
    return (a * 111_320 * np.cos(np.deg2rad(lat))) * (e * 110_574) / 1e4


def build_static():
    import rasterio

    wells = pd.read_csv(PROCESSED / "wells.csv")
    out = wells[["well_id"]].copy()

    dem_p = RAW / "gis" / "dem_90m.tif"
    if dem_p.exists():
        with rasterio.open(dem_p) as src:
            elev, slope, rel = [], [], []
            for lon, lat in zip(wells.lon, wells.lat):
                arr, m = _window_stats(src, lon, lat, 2.0)
                arr = arr.astype(float)
                c = arr[arr.shape[0] // 2, arr.shape[1] // 2]
                elev.append(c)
                rel.append(c - np.nanmean(arr[m]))
                gy, gx = np.gradient(arr, 90.0)
                s = np.degrees(np.arctan(np.hypot(gx, gy)))
                slope.append(np.nanmean(s[m]))
        out["elev_m"], out["slope_deg_2km"], out["rel_elev_2km"] = elev, slope, rel

    wc_p = RAW / "gis" / "worldcover_30m.tif"
    if wc_p.exists():
        with rasterio.open(wc_p) as src:
            rows = []
            for lon, lat in zip(wells.lon, wells.lat):
                arr, m = _window_stats(src, lon, lat, 2.0)
                v = arr[m]
                v = v[v > 0]
                rows.append({f"lc_{n}_2km": float(np.mean(v == k)) if v.size else np.nan for k, n in WC_CLASSES.items()})
        out = pd.concat([out, pd.DataFrame(rows)], axis=1)

    soil_dir = RAW / "soil"
    for prop in ["clay", "sand", "silt", "bdod"]:
        files = [soil_dir / f"{prop}_{d}.tif" for d in ("0-5cm", "5-15cm", "15-30cm")]
        if not all(f.exists() for f in files):
            continue
        vals = np.zeros(len(wells))
        weights = [5, 10, 15]
        for f, wgt in zip(files, weights):
            with rasterio.open(f) as src:
                v = np.array([x[0] for x in src.sample(zip(wells.lon, wells.lat))], dtype=float)
                v[v < 0] = np.nan
                vals += v * wgt
        out[f"soil_{prop}_0_30"] = vals / sum(weights) / 10.0   # g/kg -> % ; bdod cg/cm3 -> 0.1 g/cm3
    out.to_csv(PROCESSED / "well_static.csv", index=False)
    print("static features:", out.shape)
    return out


def build_water_yearly():
    import rasterio

    wells = pd.read_csv(PROCESSED / "wells.csv")
    rows = []
    for p in sorted((RAW / "water").glob("dry_water_*.tif")):
        year = int(p.stem.split("_")[-1])
        with rasterio.open(p) as src:
            for wid, lon, lat in zip(wells.well_id, wells.lon, wells.lat):
                pa = _pixel_area_ha(src, lat)
                rec = {"well_id": wid, "year": year}
                for r in (1.0, 3.0):
                    arr, m = _window_stats(src, lon, lat, r)
                    v = arr[m]
                    valid = np.isfinite(v)
                    # water area = Σ frequency × pixel area, rescaled for cloud-masked pixels
                    area = np.nansum(np.where(valid, v, 0)) * pa
                    cover = valid.mean() if v.size else 0
                    rec[f"water_ha_{int(r)}km"] = area / cover if cover > 0.3 else np.nan
                rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED / "well_water_yearly.csv", index=False)
    print("dry-season water:", df.year.min() if len(df) else None, "–", df.year.max() if len(df) else None)
    return df
