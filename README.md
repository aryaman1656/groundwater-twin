# Groundwater digital twin: Kolar & Chikkaballapura

A working, end-to-end system that:

1. tracks **438 monitoring wells** (312 inside the two districts) from CGWB, the Karnataka Ground Water Directorate and the Andhra Pradesh board,
2. keeps each well's state current by **resetting it to every new reading** and simulating between readings with observed IMD rainfall (to Oct 2026),
3. **forecasts** water levels one year ahead with **calibrated uncertainty bands**,
4. runs **what-if scenarios** (rainfall, pumping, K.C./H.N. Valley tank water) with physically defined responses,
5. turns a scenario into **crop yield and income at risk** (tomato, potato, maize, rice), with Monte Carlo ranges,
6. checks the what-if engine against a **real intervention**, the K.C. Valley treated-wastewater project,
7. serves all of it in a **map dashboard** (React + Leaflet) backed by a **FastAPI** service.

The design follows the project brief but changes four things that would not hold up. See [docs/APPROACH.md](docs/APPROACH.md) for the reasons.

---

## Quick start (about 5 minutes; the processed data is included)

You need Python 3.10 or newer. You don't need Node unless you want to edit the dashboard.

```bash
# 1. create an environment
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate

# 2. install
pip install -r requirements.txt

# 3. start the dashboard + API
python -m gwtwin serve
```

Open **http://127.0.0.1:8000**. The API docs are at http://127.0.0.1:8000/docs.

**No Python at hand?** The built dashboard also runs from precomputed results:

```bash
cd web/dist && python -m http.server 8080      # then open http://127.0.0.1:8080
```

(Any static file server works. Opening `index.html` directly from disk will not work because browsers block `fetch` on `file://`. In this mode the scenario sliders snap to a precomputed grid.)

---

## Rebuilding everything from source data

| Step | Command | Time | What it does |
|---|---|---|---|
| 1 | `python -m gwtwin download` | 40–90 min | Downloads groundwater, IMD rain/temperature, DEM, land cover, soil, Sentinel-2 water, crop/price tables. Skips files that exist. |
| 2 | `python -m gwtwin build` | ~2 min | Cleans wells, maps readings to the Jan/May/Aug/Nov rounds, builds climate per well, terrain/soil/land-cover features, tank-water series, the spatial graph and the modelling panel. |
| 3 | `python -m gwtwin evaluate` | ~4 min | Trains and compares 6 models on held-out years, ablation, sparse-data test, conformal coverage, multi-step forecasts. Writes `reports/`. Add `--no-gnn --no-sparse` for a 20-second run. |
| 4 | `python -m gwtwin deploy` | ~6 min | Trains the deployed model on all data, runs the K.C. Valley natural experiment, exports the dashboard data. |
| 5 | `python -m gwtwin serve` | instant | API + dashboard on port 8000. |

`python -m gwtwin all` runs steps 2–4. The zip ships without the 600 MB of raw IMD grid files; `build` reuses the extracted climate table (`data/interim/climate_daily.parquet`) when they are absent.

To download only some sources: `python -m gwtwin download --only groundwater imd`.
Step names: `groundwater telemetric idp imd boundaries opencity dem worldcover soil water`.

### Updating the twin when new readings arrive

This loop is what makes it a twin rather than a one-off model:

1. Delete `data/raw/gw_manual_bbox.csv` and `data/raw/gw_telemetric_daily_bbox.csv`, then `python -m gwtwin download --only groundwater telemetric`.
   Or put an India-WRIS export in the same format into `data/raw/`.
2. For new rainfall: `python -m gwtwin download --only imd`. For the current year, real-time IMD files go in `data/raw/imd_realtime/` (see `gwtwin/data/climate.py`).
3. `python -m gwtwin all`, then restart `serve`. Every well is reset to its newest reading and simulated forward from there.

---

## Editing the dashboard

```bash
cd web
npm install
npm run dev          # http://localhost:5173 , proxies /api to the Python server on :8000
npm run build        # writes web/dist (served by `python -m gwtwin serve`)
```

After `npm run build`, run `python -c "from gwtwin.export import copy_to_dist; copy_to_dist()"` to copy the static data back into `web/dist/data`.

---

## Project layout

```
config/config.yaml        every tunable number, with its source or an ASSUMPTION tag
gwtwin/
  data/download.py        all downloads (reproducible) + list of sources that need a browser
  data/wells.py           cleaning, well type → shallow/deep aquifer, quarterly rounds, spike filter
  data/climate.py         IMD grids → per-cell daily rain/temperature, FAO-56 Hargreaves ET0
  data/static.py          elevation, slope, valley position, soil, land cover, tank water per well
  data/surface_water.py   Sentinel-2 dry-season water (detects tanks fed by K.C./H.N. Valley)
  data/panel.py           modelling table: forcing per interval, past-only features, graph features, splits
  graph.py                k-nearest-neighbour well graph within each aquifer
  models/core.py          persistence, seasonal mean, XGBoost, physics water balance, hybrid
  models/gnn.py           ST-GNN (GAT + GRU) comparison model, PyTorch
  models/conformal.py     normalised split-conformal intervals
  twin.py                 the simulator: round-by-round rollout, scenarios, data assimilation
  evaluate.py             all experiments → reports/
  kcvalley.py             difference-in-differences + twin counterfactual
  agro.py                 groundwater → water deficit → FAO-33 yield loss → ₹, Monte Carlo
  service.py              now-cast, forecasts, scenarios, explanations for the API
  export.py               static JSON for the dashboard
api/main.py               FastAPI app
web/                      React + Leaflet + Recharts dashboard
reports/                  metrics, coverage, ablation, sparse test, K.C. Valley results
tests/                    unit tests (python -m pytest -q)
docs/APPROACH.md          design decisions, results against hypotheses H1–H6, paper outline
docs/DATA.md              every dataset: source, URL, coverage, processing, caveats
```

---

## Results on held-out years (test = 2023 onward)

Mean absolute error of the change in water level over one round (~3 months). "vs no-change" compares each model with simply repeating the last reading.

| Model | Bore wells MAE | vs no-change | Dug wells MAE | vs no-change |
|---|---|---|---|---|
| No-change (persistence) | 5.26 m | – | 0.81 m | – |
| Seasonal mean | 5.74 m | −9% | 1.01 m | −25% |
| XGBoost (no physics) | 5.01 m | +5% | 0.84 m | −3% |
| **Physics water balance** | **4.95 m** | **+6%** | 0.88 m | −9% |
| Hybrid (physics + monotone ML) | 5.08 m | +3% | 0.81 m | 0% |
| ST-GNN (GAT + GRU) | 5.63 m | −7% | 0.95 m | −17% |

- **Multi-step (the twin rolled forward):** for bore wells the twin beats no-change by 7–9% at 1–4 rounds ahead; for dug wells no-change stays hard to beat.
- **Uncertainty:** the 80% / 90% conformal bands covered 81% / 91% of bore-well test readings and 83% / 94% of dug-well readings.
- **Sparse data:** hiding up to 30% of training readings changes bore-well MAE by less than 0.1 m.
- **K.C. Valley:** wells near newly filled tanks show no statistically clear difference from other wells after 2018 (bore wells +4.0 m, 95% CI −4.8 to +12.8; dug wells +0.5 m, CI −0.7 to +1.6). The twin's simulated project effect (about 0.4 m shallower) lies inside those intervals.

The deployed model chose, on recent held-out data, to switch the ML correction off and rely on the water balance (bore wells undamped; dug wells shrunk 55% toward no-change). That choice is learned, not hard-coded (`Hybrid.fit`).

Full numbers: `reports/evaluation.json`, `reports/*.csv`.

---

## Splitting the work across four people

| Person | Owns | Files |
|---|---|---|
| Data & GIS | downloads, cleaning, NAQUIM specific-yield values, Sentinel-2 tanks | `gwtwin/data/*`, `config.yaml` |
| Forecasting ML | models, evaluation, ablation, sparse test | `gwtwin/models/*`, `evaluate.py` |
| Scenarios & agro-economics | twin rollout, K.C. Valley test, crop chain | `twin.py`, `kcvalley.py`, `agro.py` |
| Product | API, dashboard, deployment | `api/`, `web/`, `service.py`, `export.py` |

## Troubleshooting

- **SSL errors when downloading rasters behind a corporate proxy:** set `GDAL_CURL_CA_BUNDLE` (and `REQUESTS_CA_BUNDLE`) to your proxy's CA file.
- **IMD downloads are slow** (about 1 minute per year-file). They resume where they stopped.
- **No PyTorch?** Run `python -m gwtwin evaluate --no-gnn`; everything else works without it.
- **India-WRIS / CGWB / data.gov.in pages don't load from scripts.** They block automated access in some networks. The pipeline uses the India Data Portal mirror; manual steps are printed at the end of `download`.
