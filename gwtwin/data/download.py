"""Download every dataset the project uses.

Run:  python -m gwtwin download            (everything)
      python -m gwtwin download --only imd dem

Each function is idempotent: it skips files that already exist.
Sources that need a browser / login are listed in MANUAL_SOURCES and docs/DATA.md.
"""
from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from ..config import RAW, bbox, cfg

IDP_API = "https://ckandev.indiadataportal.com/api/3/action/"
IDP = {  # India Data Portal resource ids (mirror of India-WRIS / DES / Agmarknet tables)
    "gw_manual": "06d6e79c-61ac-46bb-9e03-c7138911c619",
    "gw_telemetric": "f78f7b15-f325-4c39-a291-f44ad4e95334",
    "gw_cgwb_changes": "580a8f6e-3d86-4ca7-ac7d-cd5df12b443c",
    "rain_district_daily": "d97707e6-9309-46a5-ac31-e8482ce8f9eb",
    "apy_des": "f980409d-49a2-42ae-9eb0-182365005c04",
    "apy_upag": "6f70d961-3a4f-4caa-b08f-f72f708791e1",
    "crop_irrigated_area": "0fb99c18-a1f7-46f0-b40d-ddffca021319",
    "sources_of_irrigation": "ef174105-0886-45f2-bd0d-679c45e05845",
    "agcensus_tehsil_crop": "57611aec-0744-495b-81db-ec562eefdf51",
    "apmc_prices": "078fd1ea-dab6-4aca-aed9-3a4b8b2c761d",
    "temperature": "9d4190cc-7528-4086-b44e-b6208e02cde0",
}
DISTRICT_SQL = ("(district_name ILIKE '%kolar%' OR district_name ILIKE '%ballap%' "
                "OR district_name ILIKE '%chikbal%' OR district_name ILIKE '%chickbal%')")

OPENCITY = {
    "dgwr_karnataka_2024.csv": "https://data.opencity.in/dataset/0e2e0a1c-649c-4139-a6ed-a251f7ffca85/resource/a588eb06-1c5c-4db4-9356-18a15496afce/download/39959fc8-34dc-43d4-8e00-fda20fcfc734.csv",
    "ka_taluk_gw_depth_2013_2022.csv": "https://data.opencity.in/dataset/0e211af6-21a2-43a3-88e0-8115c06e8f95/resource/cde2c8ff-7730-45ee-b29f-f17c642adf1e/download/85634e40-d372-436b-861e-46b062d41b2a.csv",
    "ksndmc_rain_all.zip": "https://data.opencity.in/dataset/03e23dd0-8f29-4249-a28a-67bdf8fd07b3/resource/046dc3d1-097b-4a23-9842-d19cdeaf6568/download/199e9862-f637-4c29-a41f-caf8046ac2fa.zip",
}
BOUNDARIES = {
    "ka_districts.geojson": "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/STATES/KARNATAKA/KARNATAKA_DISTRICTS.geojson",
    "ka_subdistricts.geojson": "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/STATES/KARNATAKA/KARNATAKA_SUBDISTRICTS.geojson",
}

MANUAL_SOURCES = """
These sources could not be reached by script (login, captcha or blocked host).
Download them in a browser if you want them; the pipeline runs without them.

  1. India-WRIS groundwater (newest rounds, 2025-26):   https://indiawris.gov.in/wris/#/groundWater
     -> save CSV to data/raw/indiawris/  (same columns as the CGWB 'changes' file)
  2. CGWB NAQUIM aquifer reports (Kolar, Malur, Bangarpet, Mulbagal taluks):
     https://cgwb.gov.in/cgwbpnm/publication-detail/571  (also 572, 568, 579)
     -> read specific yield / transmissivity tables; put values in config/config.yaml physics.sy_bounds
  3. K.C. Valley / H.N. Valley tank list with fill dates: Minor Irrigation Dept. (RTI or project office).
     The pipeline instead detects filled tanks from Sentinel-2 (see surface_water.py).
  4. Atal Jal block reports & piezometers: https://ataljal.mowr.gov.in
  5. ICRISAT district database: https://data.icrisat.org/dld/
"""


# ----------------------------------------------------------------------------- helpers
def _get(url: str, timeout: int = 300, tries: int = 4):
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            print(f"   retry {a + 1}/{tries}: {e}")
            time.sleep(3 * (a + 1))
    raise RuntimeError(f"failed: {url}")


def _sql(q: str):
    d = json.loads(_get(IDP_API + "datastore_search_sql?" + urllib.parse.urlencode({"sql": q})))
    if not d.get("success"):
        raise RuntimeError(d.get("error"))
    return d["result"]["records"]


def _write(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"   (no rows) {path.name}")
        return
    keys = [k for k in rows[0] if k != "_full_text"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"   wrote {len(rows):,} rows -> {path.relative_to(RAW.parent.parent)}")


def _dump(where_sql: str, rid: str, path: Path, page: int = 20000):
    rows, off = [], 0
    while True:
        r = _sql(f'SELECT * FROM "{rid}" WHERE {where_sql} LIMIT {page} OFFSET {off}')
        rows += r
        if len(r) < page:
            break
        off += page
    _write(rows, path)


def _download(url: str, path: Path):
    if path.exists() and path.stat().st_size > 0:
        print(f"   exists {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_get(url, timeout=600))
    print(f"   downloaded {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


# ----------------------------------------------------------------------------- groundwater
def groundwater():
    """Manual groundwater levels (CGWB + state agencies), 1976-2025, with well type & depth."""
    la0, la1, lo0, lo1 = bbox()
    out = RAW / "gw_manual_bbox.csv"
    if not out.exists():
        print("-> groundwater (manual)")
        _dump(f"latitude::float BETWEEN {la0} AND {la1} AND longitude::float BETWEEN {lo0} AND {lo1}",
              IDP["gw_manual"], out, page=10000)
    out = RAW / "idp" / "gw_cgwb_changes.csv"
    if not out.exists():
        print("-> groundwater (CGWB changes table — same as the original project CSV)")
        _dump(f"state_name ILIKE 'karnataka' AND {DISTRICT_SQL}", IDP["gw_cgwb_changes"], out)


def groundwater_telemetric():
    """Telemetric (DWLR) wells. The table has ~29M rows, so it is scanned in id-chunks
    and aggregated to daily means server-side. Takes ~15 minutes."""
    out = RAW / "gw_telemetric_daily_bbox.csv"
    if out.exists():
        print("   exists", out.name)
        return
    la0, la1, lo0, lo1 = bbox()
    rid, step, rows = IDP["gw_telemetric"], 200_000, []
    print("-> groundwater (telemetric, chunked scan)")
    for a in range(0, 29_400_000, step):
        q = (f'SELECT station_code, MAX(station_name) station_name, MAX(agency_name) agency_name, '
             f'MAX(well_type) well_type, MAX(well_aquifer_type) well_aquifer_type, AVG(latitude) latitude, '
             f'AVG(longitude) longitude, MAX(well_depth) well_depth, date, AVG(data_value) data_value, '
             f'COUNT(*) n_obs FROM "{rid}" WHERE _id >= {a} AND _id < {a + step} '
             f'AND latitude BETWEEN {la0} AND {la1} AND longitude BETWEEN {lo0} AND {lo1} '
             f'GROUP BY station_code, date')
        try:
            rows += _sql(q)
        except Exception as e:  # noqa: BLE001
            print("   chunk failed", a, e)
    _write(rows, out)


def idp_tables():
    """District crop, irrigation, price, rainfall and temperature tables for the two districts."""
    print("-> India Data Portal tables")
    for name in ["rain_district_daily", "apy_des", "apy_upag", "crop_irrigated_area",
                 "sources_of_irrigation", "agcensus_tehsil_crop", "apmc_prices"]:
        p = RAW / "idp" / f"{name}.csv"
        if p.exists():
            print("   exists", p.name)
            continue
        _dump(f"state_name ILIKE 'karnataka' AND {DISTRICT_SQL}", IDP[name], p)


# ----------------------------------------------------------------------------- climate
def imd():
    """IMD gridded daily rainfall (0.25°) and Tmax/Tmin (1°) via imdlib."""
    import imdlib  # noqa: WPS433

    first = cfg()["time"]["first_year"]
    last = time.localtime().tm_year - 1
    d = RAW / "imd"
    d.mkdir(parents=True, exist_ok=True)
    for var in ["rain", "tmax", "tmin"]:
        have = {p.stem for p in (d / var).glob("*.grd")} if (d / var).exists() else set()
        years = [y for y in range(first, last + 1) if str(y) not in have]
        for y in years:
            try:
                print(f"-> IMD {var} {y}")
                imdlib.get_data(var, y, y, fn_format="yearwise", file_dir=str(d))
            except Exception as e:  # noqa: BLE001
                print(f"   IMD {var} {y} not available: {e}")


# ----------------------------------------------------------------------------- GIS
def boundaries():
    print("-> boundaries")
    for name, url in BOUNDARIES.items():
        _download(url, RAW / "gis" / name)


def opencity():
    print("-> OpenCity (taluk groundwater depth, groundwater resources 2024, KSNDMC rainfall)")
    for name, url in OPENCITY.items():
        _download(url, RAW / "opencity" / name)


def dem():
    """Copernicus GLO-30 DEM tiles covering the bbox, mosaicked to a 90 m GeoTIFF."""
    import numpy as np
    import rasterio
    from rasterio.merge import merge
    from rasterio.enums import Resampling

    out = RAW / "gis" / "dem_90m.tif"
    if out.exists():
        print("   exists", out.name)
        return
    print("-> Copernicus DEM")
    la0, la1, lo0, lo1 = bbox()
    srcs = []
    for lat in range(int(np.floor(la0)), int(np.floor(la1)) + 1):
        for lon in range(int(np.floor(lo0)), int(np.floor(lo1)) + 1):
            name = f"Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM"
            srcs.append(rasterio.open(f"/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"))
    arr, tr = merge(srcs, bounds=(lo0, la0, lo1, la1), res=(1 / 1200, 1 / 1200),
                    resampling=Resampling.average)
    prof = srcs[0].profile | {"driver": "GTiff", "height": arr.shape[1], "width": arr.shape[2],
                              "transform": tr, "compress": "deflate", "tiled": True}
    prof.pop("blockxsize", None); prof.pop("blockysize", None)
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(arr)
    print("   wrote", out.name, arr.shape)


def worldcover():
    """ESA WorldCover 2021 (10 m) resampled to 30 m (mode) for the bbox."""
    import numpy as np
    import rasterio
    from rasterio.merge import merge
    from rasterio.enums import Resampling

    out = RAW / "gis" / "worldcover_30m.tif"
    if out.exists():
        print("   exists", out.name)
        return
    print("-> ESA WorldCover 2021")
    la0, la1, lo0, lo1 = bbox()
    tiles = sorted({(int(np.floor(lat / 3) * 3), int(np.floor(lon / 3) * 3))
                    for lat in (la0, la1) for lon in (lo0, lo1)})
    srcs = [rasterio.open("/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                          f"ESA_WorldCover_10m_2021_v200_N{a:02d}E{o:03d}_Map.tif") for a, o in tiles]
    arr, tr = merge(srcs, bounds=(lo0, la0, lo1, la1), res=(1 / 3600, 1 / 3600), resampling=Resampling.mode)
    prof = {"driver": "GTiff", "dtype": "uint8", "count": 1, "height": arr.shape[1], "width": arr.shape[2],
            "crs": "EPSG:4326", "transform": tr, "compress": "deflate", "tiled": True, "nodata": 0}
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(arr)
    print("   wrote", out.name, arr.shape)


def soilgrids():
    """SoilGrids 2.0 clay / sand / silt (0-30 cm) clipped to the bbox at ~250 m."""
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds

    la0, la1, lo0, lo1 = bbox()
    w, h = int((lo1 - lo0) * 400), int((la1 - la0) * 400)  # 0.0025° ≈ 270 m
    tr = from_bounds(lo0, la0, lo1, la1, w, h)
    for prop in ["clay", "sand", "silt", "bdod"]:
        for depth in ["0-5cm", "5-15cm", "15-30cm"]:
            out = RAW / "soil" / f"{prop}_{depth}.tif"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            url = f"/vsicurl/https://files.isric.org/soilgrids/latest/data/{prop}/{prop}_{depth}_mean.vrt"
            print("-> SoilGrids", prop, depth)
            with rasterio.open(url) as src, WarpedVRT(src, crs="EPSG:4326", transform=tr, width=w, height=h,
                                                      resampling=Resampling.average) as vrt:
                arr = vrt.read(1)
            with rasterio.open(out, "w", driver="GTiff", dtype=arr.dtype, count=1, width=w, height=h,
                               crs="EPSG:4326", transform=tr, compress="deflate", nodata=-32768) as dst:
                dst.write(arr, 1)


def sentinel2_water():
    from .surface_water import download_dry_season_water
    download_dry_season_water()


STEPS = {
    "groundwater": groundwater, "telemetric": groundwater_telemetric, "idp": idp_tables, "imd": imd,
    "boundaries": boundaries, "opencity": opencity, "dem": dem, "worldcover": worldcover,
    "soil": soilgrids, "water": sentinel2_water,
}


def run(only: list[str] | None = None):
    for name, fn in STEPS.items():
        if only and name not in only:
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"!! step '{name}' failed: {e}")
    print(MANUAL_SOURCES)
