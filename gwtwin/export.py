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
from .data.wells import index_to_round, round_date

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


def scenario_units(svc, H: int = 4, unit: float = 10.0) -> dict:
    """Per-well response to +10 % of each lever, every horizon, for the dashboard's offline mode.

    The engine is linear in each lever (the water balance is linear and the ML weight is 0), so any
    slider setting and any area can be composed in the browser:
        Δh = rain%/10 · u_rain · w_inside + pump%/10 · u_pump · w_inside · Sy(0.5)/Sy(q) + tank%/10 · u_tank · w_tank
    The only non-linearity is the physical cap (water ≤ 1 m above ground, dug well ≤ its depth);
    max_abs_error_check reports the error of this composition against the live engine.
    Weather range (p10/p90 over 21 historical years) is given for rain and tank; pumping does not
    depend on the weather.
    """
    from .areas import area_index, footprint, normalise_area
    from .twin import Scenario

    tw = svc.twin
    st = svc.nowcast()["depths"][-1]
    k = svc.k_now

    def resp(sc, y=None):
        kw = dict(state0=st, climatology_after=k)
        if y is not None:
            kw["rng_year"] = y
        b = tw.rollout(k, H, **kw)
        s = tw.rollout(k, H, scenario=sc, **kw)
        return (s["depths"][1:] - b["depths"][1:])            # (H, n)

    u = {"rain": resp(Scenario(rain_pct=unit)), "pump": resp(Scenario(pump_pct=unit)), "tank": resp(Scenario(tank_pct=unit))}
    wr = {"rain": [], "tank": []}
    for y in range(2005, 2026):
        wr["rain"].append(resp(Scenario(rain_pct=unit), y)[-1])
        wr["tank"].append(resp(Scenario(tank_pct=unit), y)[-1])
    sy_lo, sy_hi = tw.sy_bounds()
    # error check: a combined, area-limited scenario composed from units vs the live engine
    area = normalise_area({"level": "taluk", "name": "Kolar"})
    lw = tw.lever_weights(area)
    live = resp(Scenario(rain_pct=-30, pump_pct=30, tank_pct=-100, area=area))
    comp = (-3 * u["rain"] * lw["rain"] + 3 * u["pump"] * lw["pump"] - 10 * u["tank"] * lw["tank"])
    err = float(np.nanmax(np.abs(live - comp)))
    mem = svc._membership()
    areas = []
    wells_df = tw.wells
    for a in area_index(mem):
        spec = {"level": a["level"], "name": a["name"]} if a["level"] != "region" else {"level": "region"}
        f = footprint(wells_df, normalise_area(spec))
        areas.append({"id": a["id"], "level": a["level"], "name": a["name"],
                      "inside": [int(i) for i in np.where(f.zone == "inside")[0]],
                      "tank": [[int(i), round(float(w), 4)] for i, w in enumerate(f.w_tank) if w > 0],
                      "buffer": [int(i) for i in np.where(f.zone == "buffer")[0]]})
    return {"unit_pct": unit, "horizon": H, "well_ids": list(tw.wid),
            "dates": [str(round_date(*index_to_round(k + h)).date()) for h in range(1, H + 1)],
            "u": {k_: np.round(v, 5).tolist() for k_, v in u.items()},
            "weather_p10": {k_: np.round(np.nanpercentile(np.array(v), 10, 0), 5).tolist() for k_, v in wr.items()},
            "weather_p90": {k_: np.round(np.nanpercentile(np.array(v), 90, 0), 5).tolist() for k_, v in wr.items()},
            "sy_lo": sy_lo.tolist(), "sy_hi": sy_hi.tolist(), "sy_mid": tw.sy(0.5).tolist(),
            "areas": areas, "tank_buffer_km": cfg()["hydraulics"]["tank_buffer_km"],
            "max_abs_error_check_m": err}


def evidence() -> dict:
    """Area-level accuracy and spatial evidence for the dashboard (reports/*.json, trimmed)."""
    out = {}
    for name in ("area_eval", "spatial_evidence"):
        p = REPORTS / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    if "area_eval" in out:
        ae = out["area_eval"]
        ae["one_round"]["rows"] = [r for r in ae["one_round"]["rows"] if r["agg"] in ("median", "well")]
    from .areas import spread_km_per_year

    out["spread_km_per_year"] = {"deep": spread_km_per_year("deep"), "shallow": spread_km_per_year("shallow")}
    return out


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
    dump("areas.json", svc.area_forecast())
    print("   area forecast (region / district / taluk)")
    dump("evidence.json", evidence())
    su = scenario_units(svc)
    dump("scenario_units.json", su)
    print(f"   scenario unit responses (offline composition error {su['max_abs_error_check_m']:.4f} m)")
    if scenario_grid:
        grid = []
        keys = list(SCENARIO_GRID)
        for combo in itertools.product(*SCENARIO_GRID.values()):
            kw = dict(zip(keys, combo))
            r = svc.scenario(**kw)
            r["wells"] = [{"well_id": w["well_id"], "delta_m": w["delta_m"]} for w in r["wells"]]
            r.pop("levels", None)
            grid.append(r)
        dump("scenarios.json", {"grid": SCENARIO_GRID, "results": grid})
        print(f"   scenarios: {len(grid)} precomputed")
    copy_to_dist()
    return m
