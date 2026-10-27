"""IMD gridded rainfall (0.25°) and temperature (1°) -> daily series per rainfall grid cell,
plus FAO-56 Hargreaves reference evapotranspiration (ET0).

Outputs
  data/interim/climate_daily.parquet   (cell_id, date, rain_mm, tmax, tmin, et0_mm)
  data/interim/well_cells.csv          (well_id, cell_id)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pathlib import Path

from ..config import INTERIM, PROCESSED, RAW, cfg

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _years(var: str) -> list[int]:
    d = RAW / "imd" / var
    return sorted({int(p.stem) for p in list(d.glob("*.grd")) + list(d.glob("*.GRD"))}) if d.exists() else []


def _open(var: str, year: int):
    import imdlib

    ds = imdlib.open_data(var, year, year, "yearwise", str(RAW / "imd")).get_xarray()
    v = list(ds.data_vars)[0]
    da = ds[v].where(ds[v] > -900)
    return da


def hargreaves_et0(tmax, tmin, lat_deg, doy):
    """FAO-56 eq. 52 with extraterrestrial radiation Ra from eq. 21 (mm/day)."""
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)
    delta = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    ra_mm = ra * 0.408
    tmean = (tmax + tmin) / 2
    return 0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(np.clip(tmax - tmin, 0, None))


def _open_realtime(var: str):
    """IMD provisional real-time daily files (data/raw/imd_realtime), for the current year."""
    import glob
    import imdlib

    d = RAW / "imd_realtime"
    prefix = {"rain": "rain_ind0.25", "tmax": "max", "tmin": "min"}[var]
    files = sorted(glob.glob(str(d / f"{prefix}*.grd")))
    if not files:
        return None
    def _date(f):
        stem = Path(f).stem
        if "_" in stem:                       # rain_ind0.25_YY_MM_DD
            yy, mm, dd = stem.split("_")[-3:]
            return pd.Timestamp(year=2000 + int(yy), month=int(mm), day=int(dd))
        s8 = stem[-8:]                        # maxDDMMYYYY / minDDMMYYYY
        return pd.Timestamp(year=int(s8[4:]), month=int(s8[2:4]), day=int(s8[:2]))
    import xarray as xr

    das = []
    for f in files:   # one file per day; open individually so gaps in the IMD archive are harmless
        day = str(_date(f).date())
        try:
            ds = imdlib.open_real_data(var, day, day, str(d)).get_xarray()
        except Exception:  # noqa: BLE001
            continue
        das.append(ds[list(ds.data_vars)[0]])
    if not das:
        return None
    da = xr.concat(das, dim="time")
    return da.where(da > -900)


def build():
    wells = pd.read_csv(PROCESSED / "wells.csv")
    # rainfall grid: 0.25° cells centred on x.00/x.25/...
    wells["cell_lat"] = (np.round(wells.lat * 4) / 4).round(2)
    wells["cell_lon"] = (np.round(wells.lon * 4) / 4).round(2)
    wells["cell_id"] = wells.cell_lat.astype(str) + "_" + wells.cell_lon.astype(str)
    cells = wells[["cell_id", "cell_lat", "cell_lon"]].drop_duplicates().reset_index(drop=True)
    wells[["well_id", "cell_id"]].to_csv(INTERIM / "well_cells.csv", index=False)

    frames = []
    ry, tx, tn = _years("rain"), set(_years("tmax")), set(_years("tmin"))
    print(f"IMD rain years: {ry[0] if ry else None}–{ry[-1] if ry else None} ({len(ry)}); temperature years: {len(tx & tn)}")
    for y in ry:
        rain = _open("rain", y)
        r = rain.sel(lat=xr_index(cells.cell_lat), lon=xr_index(cells.cell_lon), method="nearest")
        rv = r.values  # (time, n) via vectorised indexing
        dates = pd.to_datetime(rain.time.values)
        df = pd.DataFrame(rv, index=dates, columns=cells.cell_id).stack().rename("rain_mm").reset_index()
        df.columns = ["date", "cell_id", "rain_mm"]
        if y in tx and y in tn:
            tmax = _open("tmax", y).interp(lat=xr_index(cells.cell_lat), lon=xr_index(cells.cell_lon))
            tmin = _open("tmin", y).interp(lat=xr_index(cells.cell_lat), lon=xr_index(cells.cell_lon))
            t1 = pd.DataFrame(tmax.values, index=pd.to_datetime(tmax.time.values), columns=cells.cell_id).stack()
            t0 = pd.DataFrame(tmin.values, index=pd.to_datetime(tmin.time.values), columns=cells.cell_id).stack()
            tt = pd.DataFrame({"tmax": t1, "tmin": t0}).reset_index()
            tt.columns = ["date", "cell_id", "tmax", "tmin"]
            df = df.merge(tt, on=["date", "cell_id"], how="left")
        else:
            df["tmax"] = np.nan
            df["tmin"] = np.nan
        frames.append(df)
    # provisional real-time data for the current year (e.g. 2026), if downloaded
    rt = _open_realtime("rain")
    if rt is not None:
        rt = rt.sel(time=rt.time > np.datetime64(f"{ry[-1]}-12-31"))
        if rt.time.size:
            r = rt.sel(lat=xr_index(cells.cell_lat), lon=xr_index(cells.cell_lon), method="nearest")
            df = pd.DataFrame(r.values, index=pd.to_datetime(rt.time.values), columns=cells.cell_id).stack()
            df = df.rename("rain_mm").reset_index()
            df.columns = ["date", "cell_id", "rain_mm"]
            for var in ("tmax", "tmin"):
                t = _open_realtime(var)
                if t is not None:
                    t = t.sel(time=t.time > np.datetime64(f"{ry[-1]}-12-31"))
                    ti = t.interp(lat=xr_index(cells.cell_lat), lon=xr_index(cells.cell_lon))
                    tt = pd.DataFrame(ti.values, index=pd.to_datetime(t.time.values), columns=cells.cell_id)
                    tt = tt.stack().rename(var).reset_index()
                    tt.columns = ["date", "cell_id", var]
                    df = df.merge(tt, on=["date", "cell_id"], how="left")
                else:
                    df[var] = np.nan
            frames.append(df)
            print(f"IMD real-time (provisional): {df.date.min().date()} – {df.date.max().date()}")
    clim = pd.concat(frames, ignore_index=True)
    clim = clim.merge(cells, on="cell_id")
    # fill missing temperature with the day-of-year climatology of that cell (documented gap fill)
    clim["doy"] = clim.date.dt.dayofyear
    for v in ("tmax", "tmin"):
        clim[v] = clim[v].fillna(clim.groupby(["cell_id", "doy"])[v].transform("mean"))
    clim["et0_mm"] = hargreaves_et0(clim.tmax, clim.tmin, clim.cell_lat, clim.doy)
    clim["rain_mm"] = clim.rain_mm.fillna(0)
    clim = clim[["cell_id", "date", "rain_mm", "tmax", "tmin", "et0_mm"]]
    clim.to_parquet(INTERIM / "climate_daily.parquet", index=False)
    print(f"climate: {clim.cell_id.nunique()} cells × {clim.date.nunique()} days")
    return clim


def xr_index(s: pd.Series):
    import xarray as xr

    return xr.DataArray(s.values, dims="pts")
