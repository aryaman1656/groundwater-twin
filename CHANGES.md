# Changes, 30 Oct 2026 — area reporting and spatial evidence

The core model (gwtwin/models/*) is unchanged; the numbers it rests on were re-run first (docs/VERIFICATION.md).

1. **Frozen forecast** `forecasts/frozen/2026-10-30/` (wells, meta, detail + SHA-256 MANIFEST). `python -m gwtwin freeze` / `score` (gwtwin/freeze.py).
2. **Area aggregation** (gwtwin/areas.py, TwinService.area_forecast, artifacts/web/areas.json, GET /api/areas): bore wells by region / district / taluk, dug wells by region / district only (indicative); median, IQR, counts, rounds since last reading, low-data flag (Gudibanda, Bagepalli, Malur); buffer wells excluded.
3. **Dashboard**: Areas tab (region view, taluk panel with forecast path, accuracy record, well list), taluk choropleth (click for panel), "rounds since the last real reading" on the top bar, every well and every taluk, and as a map layer; Evidence tab; area skill in Model check. README update loop rewritten.
4. **Area-limited scenarios**: Scenario(area=…) for region / district / taluk / circle (Twin.lever_weights); results at region, taluk and well level; per-well label "typical response of this aquifer type at this location" with the Sy range. Works offline too (scenario_units.json; composition error 0.0000 m).
5. **Pumping**: area-level, physics-only block with the Sy range by district; `naquim_sy` config hook; Theis drawdown only as a user-input illustration.
6. **Uncertainty**: rolling multi-year conformal at area level (gwtwin/area_eval.py). Taluk coverage 77–85 % but bands are wide; four-round bands have too few cases, so the dashboard shows the IQR and the rolling record, not an interval.
7. **Headline accuracy** = skill vs no-change and correlation at taluk / district / region level.
8. **Spatial**: footprint (inside / 3 km tank buffer / no modelled effect), optional physics-only lateral-flow illustration, evidence figures (residual correlation by distance; K.C. Valley DiD by distance band for bore and dug wells, flagged underpowered), ST-GNN as comparison only (gwtwin/spatial.py).

Also: `time.deploy_train_end` moved to config.yaml (same value), new tests (tests/test_areas.py; 13 pass).
