"""Export everything the dashboard needs as static JSON (artifacts/web/).

The API serves the same data live; the static export lets the dashboard run with no backend
(e.g. on GitHub Pages) using a precomputed grid of scenarios.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from .config import ARTIFACTS, PROCESSED, RAW, REPORTS, cfg

WEB = ARTIFACTS / "web"
SCENARIO_GRID = {"rain_pct": [-40, -20, 0, 20], "pump_pct": [-20, 0, 20, 40], "tank_pct": [-100, 0, 100], "horizon": [2]}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def dump(name, obj):
    WEB.mkdir(parents=True, exist_ok=True)
    (WEB / name).parent.mkdir(parents=True, exist_ok=True)
    (WEB / name).write_text(json.dumps(_clean(obj), separators=(",", ":")))


def taluks_geojson():
    import geopandas as gpd

    t = gpd.read_file(RAW / "gis" / "ka_subdistricts.geojson")
    t = t[t.dtname.isin(cfg()["region"]["districts"])][["dtname", "sdtname", "geometry"]]
    t = t.rename(columns={"dtname": "district", "sdtname": "taluk"})
    t["geometry"] = t.geometry.simplify(0.002, preserve_topology=True)
    return json.loads(t.to_json())


def copy_to_dist():
    """Put the static export inside web/dist/data so the built dashboard works with no backend."""
    import shutil

    from .config import ROOT

    dist = ROOT / "web" / "dist"
    if dist.exists():
        tgt = dist / "data"
        if tgt.exists():
            shutil.rmtree(tgt)
        shutil.copytree(WEB, tgt)
        print("   copied static data to web/dist/data")


def export_all(scenario_grid: bool = True):
    from .service import TwinService, meta

    svc = TwinService()
    m = meta()
    m["now_round"] = str(pd.Timestamp(svc.twin.obs.columns.max()))  # placeholder, replaced below
    from .data.wells import index_to_round, round_date

    m["now_round"] = str(round_date(*index_to_round(svc.k_now)).date())
    m["assimilation_start"] = str(round_date(*index_to_round(svc.k_start)).date())
    m["forcing_observed_until"] = str(pd.read_parquet(PROCESSED.parent / "interim" / "climate_daily.parquet").date.max().date())
    m["trained_through"] = svc.trained_through
    phys = pd.read_csv(REPORTS / "physics_coefficients_deploy.csv").to_dict("records")
    m["physics_deploy"] = phys
    m["damping"] = {"alpha": svc.model.alpha, "beta": svc.model.beta}
    dump("meta.json", m)
    dump("taluks.json", taluks_geojson())
    layer = svc.map_layer()
    dump("wells.json", layer)
    dump("anomalies.json", svc.anomalies())
    print(f"   wells: {len(layer)}")
    dump("wells_detail.json", {w: svc.well_detail(w) for w in svc.twin.wid})
    print("   per-well detail written (one bundle)")
    if scenario_grid:
        grid = []
        keys = list(SCENARIO_GRID)
        for combo in itertools.product(*SCENARIO_GRID.values()):
            kw = dict(zip(keys, combo))
            r = svc.scenario(**kw)
            r["wells"] = [{"well_id": w["well_id"], "delta_m": w["delta_m"]} for w in r["wells"]]
            grid.append(r)
        dump("scenarios.json", {"grid": SCENARIO_GRID, "results": grid})
        print(f"   scenarios: {len(grid)} precomputed")
    copy_to_dist()
    return m
