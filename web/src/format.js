export const COLORS = {
  ink: "#1d2a33",
  ink2: "#4b5a64",
  stone: "#8c949a",
  rule: "#d5dbd8",
  water: "#1f6f8b",
  water2: "#9cc6d4",
  umber: "#9a6a1f",
  umber2: "#dcc195",
  alert: "#a3263a",
};

export const fmt = (v, d = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d);

export const signed = (v, d = 1) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  const r = Number(Math.abs(v).toFixed(d));
  return `${r === 0 ? "" : v > 0 ? "+" : "−"}${r.toFixed(d)}`;
};

// rupee amounts given in lakh; shows crore above 100 lakh. Negative values get a minus sign.
export const lakh = (v) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  const a = Math.abs(v);
  const s = a >= 100 ? `₹${(a / 100).toFixed(1)} crore` : `₹${a.toFixed(1)} lakh`;
  return v < 0 ? `−${s}` : s;
};
// a loss figure (positive = loss) written for people: "₹x crore lost" / "₹x crore gained"
export const lossText = (v) => (v === null || v === undefined ? "–" : v < 0 ? `${lakh(-v)} gained` : `${lakh(v)} lost`);

export const dateLabel = (s) => {
  if (!s) return "–";
  const d = new Date(s);
  return d.toLocaleDateString("en-IN", { month: "short", year: "numeric" });
};

function mix(a, b, t) {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  const c = pa.map((x, i) => Math.round(x + (pb[i] - x) * t));
  return `#${c.map((x) => x.toString(16).padStart(2, "0")).join("")}`;
}

// change in depth (m, + = water table falls): blue (rise) – grey – umber (fall)
export function changeColor(v, span = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "#b9c0bd";
  const t = Math.max(-1, Math.min(1, v / span));
  return t < 0 ? mix("#c9cfcc", COLORS.water, -t) : mix("#c9cfcc", COLORS.umber, t);
}

// depth to water (m): pale (shallow) to ink (deep)
export function depthColor(v, max = 60) {
  if (v === null || v === undefined || Number.isNaN(v)) return "#b9c0bd";
  const t = Math.max(0, Math.min(1, Math.sqrt(v / max)));
  return mix("#cfe3ea", "#14303d", t);
}

export function uncertaintyColor(v, max = 40) {
  if (v === null || v === undefined || Number.isNaN(v)) return "#b9c0bd";
  const t = Math.max(0, Math.min(1, v / max));
  return mix("#e7ebe9", "#4b5a64", t);
}

export const TERM_LABELS = {
  season: "Typical change this season",
  rain_int: "Rain this season",
  rain_prev_int: "Rain last season",
  rain_12m_before: "Rain over past year",
  et0_int: "Evaporative demand",
  water_ha_3km: "Tank water nearby",
  anom_base: "Return toward normal level",
  ml_residual: "ML correction",
  pumping: "Change in pumping",
};

export const FEATURE_LABELS = {
  depth_m: "Current depth",
  anom: "Depth vs recent average",
  anom_base: "Depth vs pre-2018 level",
  past_mean: "Long-run average depth",
  dh_prev: "Last change",
  dh_season_past: "Usual change this season",
  frac_depth: "Water level / well depth",
  rounds_since_obs: "Rounds since last reading",
  nb_anom: "Neighbour wells vs normal",
  nb_dh_prev: "Neighbours' last change",
  nb_n: "Number of neighbours",
  rain_int: "Rain this season",
  rain_prev_int: "Rain last season",
  rain_12m_before: "Rain over past year",
  et0_int: "Evaporative demand",
  water_ha_3km: "Tank water within 3 km",
  water_ha_1km: "Tank water within 1 km",
  elev_m: "Elevation",
  slope_deg_2km: "Slope",
  rel_elev_2km: "Height above valley",
  lc_cropland_2km: "Cropland share",
  lc_built_2km: "Built-up share",
  lc_tree_2km: "Tree cover share",
  lc_water_2km: "Water share",
  soil_clay_0_30: "Soil clay %",
  soil_sand_0_30: "Soil sand %",
  well_depth: "Well depth",
  m_jan: "Season: Jan–May",
  m_may: "Season: May–Aug",
  m_aug: "Season: Aug–Nov",
  m_nov: "Season: Nov–Jan",
};
