// Data access. Uses the live FastAPI backend when it is running; otherwise falls back to the
// static export in ./data (built by `python -m gwtwin deploy`). Offline, scenarios are composed in
// the browser from per-well unit responses (exact for this linear engine apart from depth caps);
// crop and income results exist offline only for region-wide scenarios on the precomputed grid.
import { composeScenario } from "./areas.js";

let mode = null; // "live" | "static"
let scenarioGrid = null;

async function json(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

export async function detectMode() {
  if (mode) return mode;
  try {
    const h = await json("api/health");
    mode = h.ok ? "live" : "static";
  } catch {
    mode = "static";
  }
  return mode;
}

export const getMode = () => mode;

export async function getMeta() {
  return (await detectMode()) === "live" ? json("api/meta") : json("data/meta.json");
}

export async function getWells() {
  return (await detectMode()) === "live" ? json("api/wells") : json("data/wells.json");
}

export async function getTaluks() {
  return (await detectMode()) === "live" ? json("api/taluks") : json("data/taluks.json");
}

let detailBundle = null;
export async function getWell(id) {
  if ((await detectMode()) === "live") return json(`api/wells/${encodeURIComponent(id)}`);
  if (!detailBundle) detailBundle = await json("data/wells_detail.json");
  if (!detailBundle[id]) throw new Error(`no data for well ${id}`);
  return detailBundle[id];
}

export async function getAreas() {
  return (await detectMode()) === "live" ? json("api/areas") : json("data/areas.json");
}

export async function getEvidence() {
  return (await detectMode()) === "live" ? json("api/evidence") : json("data/evidence.json");
}

let units = null;
export async function getUnits() {
  if (!units) units = await json("data/scenario_units.json");
  return units;
}

export async function getAnomalies() {
  return (await detectMode()) === "live" ? json("api/anomalies") : json("data/anomalies.json");
}

function nearest(values, v) {
  return values.reduce((a, b) => (Math.abs(b - v) < Math.abs(a - v) ? b : a));
}

export async function getScenarioGrid() {
  if (!scenarioGrid) scenarioGrid = await json("data/scenarios.json");
  return scenarioGrid.grid;
}

export async function runScenario(s, wells) {
  if ((await detectMode()) === "live") {
    return json("api/scenario", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(s),
    });
  }
  const u = await getUnits();
  const res = composeScenario(u, wells, s);
  if (!scenarioGrid) scenarioGrid = await json("data/scenarios.json");
  const g = scenarioGrid.grid;
  const regionWide = !s.area || s.area.level === "all";
  const onGrid = regionWide && g.rain_pct.includes(s.rain_pct) && g.pump_pct.includes(s.pump_pct) &&
    g.tank_pct.includes(s.tank_pct) && g.horizon.includes(s.horizon) && (s.sy_quantile ?? 0.5) === 0.5;
  if (onGrid) {
    const hit = scenarioGrid.results.find((r) => r.inputs.rain_pct === s.rain_pct && r.inputs.pump_pct === s.pump_pct &&
      r.inputs.tank_pct === s.tank_pct && r.inputs.horizon === s.horizon);
    if (hit) { res.agro = hit.agro; res.drivers_mean_m = hit.drivers_mean_m; res.from_round = hit.from_round; }
  }
  res.agro_note = onGrid ? null : "Crop and income results need the live backend for this setting (offline they exist only for region-wide scenarios on the precomputed grid).";
  return res;
}

// kept for reference: the old snapping lookup
export async function runScenarioGrid(s) {
  if (!scenarioGrid) scenarioGrid = await json("data/scenarios.json");
  const g = scenarioGrid.grid;
  const want = {
    rain_pct: nearest(g.rain_pct, s.rain_pct),
    pump_pct: nearest(g.pump_pct, s.pump_pct),
    tank_pct: nearest(g.tank_pct, s.tank_pct),
    horizon: nearest(g.horizon, s.horizon),
  };
  const hit = scenarioGrid.results.find(
    (r) =>
      r.inputs.rain_pct === want.rain_pct &&
      r.inputs.pump_pct === want.pump_pct &&
      r.inputs.tank_pct === want.tank_pct &&
      r.inputs.horizon === want.horizon
  );
  return { ...hit, snapped: want };
}
