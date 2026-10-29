"""FastAPI backend for the groundwater digital twin dashboard.

Run:  python -m gwtwin serve      (or: uvicorn api.main:app --port 8000)
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
import math
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal, Optional

from pydantic import BaseModel, Field

from gwtwin.config import ARTIFACTS, REPORTS, ROOT

STATE: dict = {}
WEB_DIST = ROOT / "web" / "dist"


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else round(o, 4)
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return _clean(o.item())
    except ImportError:
        pass
    return o


@asynccontextmanager
async def lifespan(app: FastAPI):
    from gwtwin.service import TwinService

    svc = TwinService()
    STATE["svc"] = svc
    STATE["wells"] = _clean(svc.map_layer())        # now-cast + 3-round forecast, computed once
    import pandas as pd
    from gwtwin.config import INTERIM

    STATE["forcing_until"] = str(pd.read_parquet(INTERIM / "climate_daily.parquet", columns=["date"]).date.max().date())
    yield


app = FastAPI(title="Groundwater Digital Twin — Kolar & Chikkaballapura", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AreaIn(BaseModel):
    level: Literal["all", "region", "district", "taluk", "radius"] = "all"
    name: Optional[str] = Field(None, description="district or taluk name")
    lat: Optional[float] = None
    lon: Optional[float] = None
    radius_km: Optional[float] = Field(None, gt=0, le=60)


class ScenarioIn(BaseModel):
    rain_pct: float = Field(0, ge=-50, le=50, description="change in rainfall, %")
    pump_pct: float = Field(0, ge=-50, le=50, description="change in groundwater extraction, %")
    tank_pct: float = Field(0, ge=-100, le=100, description="change in dry-season tank water, %")
    horizon: int = Field(2, ge=1, le=4, description="rounds ahead (1 round ≈ 3 months)")
    sy_quantile: float = Field(0.5, ge=0, le=1, description="position in the specific-yield range")
    area: Optional[AreaIn] = Field(None, description="limit the levers to an area; omit for every well")
    lateral: bool = Field(False, description="add the physics-only lateral-flow illustration (never part of the totals)")
    weather_range: bool = Field(False, description="p10–p90 of the area response over 21 weather years (~15 s)")


@app.get("/api/health")
def health():
    return {"ok": "svc" in STATE}


@app.get("/api/meta")
def meta():
    from gwtwin.data.wells import index_to_round, round_date
    from gwtwin.service import meta as m

    svc = STATE["svc"]
    out = m()
    out["now_round"] = str(round_date(*index_to_round(svc.k_now)).date())
    out["assimilation_start"] = str(round_date(*index_to_round(svc.k_start)).date())
    out["trained_through"] = svc.trained_through
    out["forcing_observed_until"] = STATE.get("forcing_until")
    out["damping"] = {"alpha": svc.model.alpha, "beta": svc.model.beta}
    p = REPORTS / "physics_coefficients_deploy.csv"
    if p.exists():
        import pandas as pd

        out["physics_deploy"] = pd.read_csv(p).to_dict("records")
    return JSONResponse(_clean(out))


@app.get("/api/wells")
def wells():
    return JSONResponse(STATE["wells"])


@app.get("/api/wells/{well_id}")
def well(well_id: str):
    svc = STATE["svc"]
    if well_id not in svc.twin.idx:
        raise HTTPException(404, "unknown well")
    return JSONResponse(_clean(svc.well_detail(well_id)))


@app.get("/api/taluks")
def taluks():
    from gwtwin.export import taluks_geojson

    if "taluks" not in STATE:
        STATE["taluks"] = taluks_geojson()
    return JSONResponse(STATE["taluks"])


@app.post("/api/scenario")
def scenario(s: ScenarioIn):
    kw = s.model_dump()
    a = kw.pop("area")
    if a and a.get("level") != "all":
        kw["area"] = {k: v for k, v in a.items() if v is not None}
    try:
        return JSONResponse(_clean(STATE["svc"].scenario(**kw)))
    except (ValueError, KeyError) as e:
        raise HTTPException(422, str(e))


@app.get("/api/areas")
def areas():
    """Baseline forecast by region / district / taluk (study wells only), with data age and accuracy record."""
    if "areas" not in STATE:
        STATE["areas"] = _clean(STATE["svc"].area_forecast())
    return JSONResponse(STATE["areas"])


@app.get("/api/evidence")
def evidence():
    from gwtwin.export import evidence as ev

    return JSONResponse(_clean(ev()))


@app.get("/api/anomalies")
def anomalies():
    return JSONResponse(_clean(STATE["svc"].anomalies()))


@app.get("/api/report/{name}")
def report(name: str):
    allowed = {"evaluation": "evaluation.json", "kcvalley": "kcvalley.json", "area_eval": "area_eval.json",
               "spatial_evidence": "spatial_evidence.json"}
    if name not in allowed or not (REPORTS / allowed[name]).exists():
        raise HTTPException(404, "report not found — run `python -m gwtwin evaluate` / `deploy`")
    return JSONResponse(json.loads((REPORTS / allowed[name]).read_text()))


# ---- dashboard (built React app). In development run `npm run dev` inside web/ instead.
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = WEB_DIST / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(WEB_DIST / "index.html")
