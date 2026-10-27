"""Dry-season surface water from Sentinel-2 (detects tanks kept full by K.C./H.N. Valley water).

Why: in Feb–Apr, rain-fed tanks in Kolar/Chikkaballapura are normally dry. A tank holding water
in the dry season after 2018 is almost certainly receiving treated wastewater. This gives a
data-derived "recharge intervention" layer without needing the government's tank list.

Method: for each year, take up to N clear Sentinel-2 L2A scenes per tile in Feb–Apr, read the
Scene Classification Layer (SCL; class 6 = water) at ~60 m, and compute
    water_frequency = (#scenes classified water) / (#scenes with a valid, cloud-free pixel)
Output: data/raw/water/dry_water_<year>.tif  (EPSG:4326, ~65 m)
"""
from __future__ import annotations

import numpy as np

from ..config import RAW, bbox

STAC = "https://earth-search.aws.element84.com/v1"
VALID = {4, 5, 6, 7, 11}      # vegetation, bare, water, unclassified, snow -> usable observation
WATER = 6


def _grid():
    from rasterio.transform import from_bounds

    la0, la1, lo0, lo1 = bbox()
    res = 0.0006
    w, h = int(round((lo1 - lo0) / res)), int(round((la1 - la0) / res))
    return from_bounds(lo0, la0, lo1, la1, w, h), w, h


def _read_scl(href: str, transform, w: int, h: int):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    with rasterio.open(href) as src:
        with WarpedVRT(src, crs="EPSG:4326", transform=transform, width=w, height=h,
                       resampling=Resampling.nearest, nodata=0) as vrt:
            return vrt.read(1)


def download_dry_season_water(years=range(2017, 2027), scenes_per_tile: int = 4, max_cloud: float = 15):
    import rasterio
    from pystac_client import Client

    out_dir = RAW / "water"
    out_dir.mkdir(parents=True, exist_ok=True)
    la0, la1, lo0, lo1 = bbox()
    transform, w, h = _grid()
    client = Client.open(STAC)
    for year in years:
        out = out_dir / f"dry_water_{year}.tif"
        if out.exists():
            print("   exists", out.name)
            continue
        items = list(client.search(collections=["sentinel-2-l2a"], bbox=[lo0, la0, lo1, la1],
                                   datetime=f"{year}-02-01/{year}-04-30",
                                   query={"eo:cloud_cover": {"lt": max_cloud}}).items())
        by_tile: dict[str, list] = {}
        for it in items:
            by_tile.setdefault(it.properties.get("grid:code", it.id[4:9]), []).append(it)
        n_water = np.zeros((h, w), np.uint8)
        n_valid = np.zeros((h, w), np.uint8)
        used = 0
        for tile, its in by_tile.items():
            its.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
            for it in its[:scenes_per_tile]:
                href = it.assets["scl"].href
                try:
                    scl = _read_scl(href, transform, w, h)
                except Exception as e:  # noqa: BLE001
                    print("   skip", it.id, e)
                    continue
                valid = np.isin(scl, list(VALID))
                n_valid += valid
                n_water += (scl == WATER)
                used += 1
        freq = np.where(n_valid > 0, n_water / np.maximum(n_valid, 1), np.nan).astype("float32")
        with rasterio.open(out, "w", driver="GTiff", dtype="float32", count=1, width=w, height=h,
                           crs="EPSG:4326", transform=transform, compress="deflate", nodata=np.nan) as dst:
            dst.write(freq, 1)
        print(f"   {year}: {len(items)} scenes found, {used} used -> {out.name}")
