// Area helpers used by the dashboard: statistics, area specs, and the offline scenario composition.
// Rules mirror gwtwin/areas.py: only study wells (in_totals) count in totals; dug wells are
// summarised only at region and district level.

export const quantile = (xs, q) => {
  const v = xs.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  const pos = (v.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return v[lo] + (v[hi] - v[lo]) * (pos - lo);
};

export const summarise = (xs) => {
  const v = xs.filter((x) => Number.isFinite(x));
  return { n: v.length, median: quantile(v, 0.5), p25: quantile(v, 0.25), p75: quantile(v, 0.75) };
};

export const areaLabel = (a) => {
  if (!a || a.level === "all") return "every well in the model (as before)";
  if (a.level === "region") return "both districts";
  if (a.level === "radius") return `${a.radius_km} km around the picked point`;
  return `${a.name} ${a.level}`;
};

export const LEVEL_FOR = { deep: ["region", "district", "taluk"], shallow: ["region", "district"] };

export function inArea(w, a) {
  if (!a || a.level === "all") return true;
  if (a.level === "region") return !!w.in_totals;
  if (a.level === "district") return w.district === a.name;
  if (a.level === "taluk") return w.taluk === a.name;
  return false;
}

// great-circle distance, km
export function haversine(lat1, lon1, lat2, lon2) {
  const R = 6371.0088, r = Math.PI / 180;
  const dlat = (lat2 - lat1) * r, dlon = (lon2 - lon1) * r;
  const s = Math.sin(dlat / 2) ** 2 + Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(dlon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
}

// share of a circle of radius a (centre at distance d) that lies inside a circle of radius R
export function circleOverlapShare(a, R, d) {
  if (d >= a + R) return 0;
  if (d <= Math.abs(R - a)) return R >= a ? 1 : (R * R) / (a * a);
  const a2 = a * a, R2 = R * R;
  const A = a2 * Math.acos((d * d + a2 - R2) / (2 * d * a)) + R2 * Math.acos((d * d + R2 - a2) / (2 * d * R))
    - 0.5 * Math.sqrt((-d + a + R) * (d + a - R) * (d - a + R) * (d + a + R));
  return A / (Math.PI * a2);
}

// per-well lever weights for an area, offline (same definitions as gwtwin/areas.footprint)
export function weightsFor(units, wells, area) {
  const n = units.well_ids.length;
  const inside = new Float64Array(n), tank = new Float64Array(n);
  const zone = new Array(n).fill("outside");
  if (!area || area.level === "all") {
    inside.fill(1); tank.fill(1); zone.fill("inside");
    return { inside, tank, zone };
  }
  if (area.level === "radius") {
    const byId = new Map(wells.map((w) => [w.well_id, w]));
    const buf = units.tank_buffer_km;
    units.well_ids.forEach((id, i) => {
      const w = byId.get(id);
      const d = haversine(area.lat, area.lon, w.lat, w.lon);
      if (d <= area.radius_km) { inside[i] = 1; zone[i] = "inside"; }
      else if (d <= area.radius_km + buf) zone[i] = "buffer";
      tank[i] = zone[i] === "outside" ? 0 : circleOverlapShare(buf, area.radius_km, d);
    });
    return { inside, tank, zone };
  }
  const id = area.level === "region" ? "region" : `${area.level}:${area.name}`;
  const a = units.areas.find((x) => x.id === id);
  if (!a) throw new Error(`no precomputed footprint for ${id}`);
  a.inside.forEach((i) => { inside[i] = 1; zone[i] = "inside"; });
  a.buffer.forEach((i) => { zone[i] = "buffer"; });
  a.tank.forEach(([i, w]) => { tank[i] = w; });
  return { inside, tank, zone };
}

// Offline scenario: compose the per-well unit responses (exact for this linear engine apart from
// the physical depth caps; the export reports the composition error).
export function composeScenario(units, wells, inp) {
  const h = Math.max(1, Math.min(units.horizon, inp.horizon)) - 1;
  const { inside, tank, zone } = weightsFor(units, wells, inp.area);
  const byId = new Map(wells.map((w) => [w.well_id, w]));
  const syq = (i, q) => Math.exp(Math.log(units.sy_lo[i]) + q * (Math.log(units.sy_hi[i]) - Math.log(units.sy_lo[i])));
  const q = inp.sy_quantile ?? 0.5;
  const out = units.well_ids.map((id, i) => {
    const w = byId.get(id) || {};
    const rain = (inp.rain_pct / units.unit_pct) * units.u.rain[h][i] * inside[i];
    const pumpMid = (inp.pump_pct / units.unit_pct) * units.u.pump[h][i] * inside[i];
    const pump = pumpMid * units.sy_mid[i] / syq(i, q);
    const tk = (inp.tank_pct / units.unit_pct) * units.u.tank[h][i] * tank[i];
    const pLo = pumpMid * units.sy_mid[i] / units.sy_lo[i], pHi = pumpMid * units.sy_mid[i] / units.sy_hi[i];
    // weather range (end of horizon only; approximate offline)
    const wr = (k, pct, wgt) => [(pct / units.unit_pct) * units.weather_p10[k][i] * wgt, (pct / units.unit_pct) * units.weather_p90[k][i] * wgt];
    const r10 = wr("rain", inp.rain_pct, inside[i]), t10 = wr("tank", inp.tank_pct, tank[i]);
    return {
      well_id: id, aquifer: w.aquifer, district: w.district, taluk: w.taluk, in_totals: !!w.in_totals,
      delta_m: rain + pump + tk, rain_m: rain, pump_m: pump, tank_m: tk,
      delta_lo_m: rain + tk + Math.min(pLo, pHi), delta_hi_m: rain + tk + Math.max(pLo, pHi),
      weather_lo_m: Math.min(r10[0], r10[1]) + Math.min(t10[0], t10[1]) + pump,
      weather_hi_m: Math.max(r10[0], r10[1]) + Math.max(t10[0], t10[1]) + pump,
      zone: zone[i],
    };
  });
  return finishScenario(out, inp, units.dates[h]);
}

// Area tables from per-well rows (used offline, and to fill gaps in live results)
export function finishScenario(rows, inp, toDate) {
  const st = rows.filter((r) => r.in_totals);
  const levels = [];
  const push = (level, name, district, sel) => {
    ["deep", "shallow"].forEach((aq) => {
      if (!LEVEL_FOR[aq].includes(level)) return;
      const x = sel.filter((r) => r.aquifer === aq);
      levels.push({ level, name, district, aquifer: aq, n_wells: x.length, indicative: aq === "shallow",
                    delta_m: summarise(x.map((r) => r.delta_m)), delta_lo_m: summarise(x.map((r) => r.delta_lo_m)),
                    delta_hi_m: summarise(x.map((r) => r.delta_hi_m)) });
    });
  };
  push("region", "Kolar + Chikkaballapura", null, st);
  [...new Set(st.map((r) => r.district))].sort().forEach((d) => push("district", d, d, st.filter((r) => r.district === d)));
  [...new Set(st.map((r) => r.taluk))].sort().forEach((t) => push("taluk", t, st.find((r) => r.taluk === t).district, st.filter((r) => r.taluk === t)));
  const insideStudy = st.filter((r) => r.zone === "inside");
  const area_summary = {};
  ["deep", "shallow"].forEach((aq) => {
    const x = insideStudy.filter((r) => r.aquifer === aq);
    const s = summarise(x.map((r) => r.delta_m));
    area_summary[aq] = { n_wells: x.length, median_delta_m: s.median, p25: s.p25, p75: s.p75, indicative: aq === "shallow",
                         sy_range_m: x.length ? [quantile(x.map((r) => r.delta_lo_m), 0.5), quantile(x.map((r) => r.delta_hi_m), 0.5)] : null,
                         weather_range_m: x.length && x[0].weather_lo_m !== undefined
                           ? [quantile(x.map((r) => r.weather_lo_m), 0.5), quantile(x.map((r) => r.weather_hi_m), 0.5)] : null };
  });
  const zones = {};
  ["inside", "buffer", "outside"].forEach((z) => {
    const x = rows.filter((r) => r.zone === z);
    zones[z] = { n_wells: x.length, max_abs_delta_m: x.reduce((m, r) => Math.max(m, Math.abs(r.delta_m || 0)), 0) };
  });
  const taluks = levels.filter((l) => l.level === "taluk" && l.aquifer === "deep").map((l) => ({ taluk: l.name, district: l.district, dh_mean: l.delta_m.median }));
  let pumping = null;
  if (inp.pump_pct) {
    pumping = { physics_only: true, by_district: [] };
    ["Kolar", "Chikkaballapura"].forEach((d) => ["deep", "shallow"].forEach((aq) => {
      const x = insideStudy.filter((r) => r.district === d && r.aquifer === aq);
      if (!x.length) return;
      pumping.by_district.push({ district: d, aquifer: aq, n_wells: x.length, pump_m: quantile(x.map((r) => r.pump_m), 0.5),
        range_m: [quantile(x.map((r) => Math.min(r.delta_lo_m - r.rain_m - r.tank_m, r.delta_hi_m - r.rain_m - r.tank_m)), 0.5),
                  quantile(x.map((r) => Math.max(r.delta_lo_m - r.rain_m - r.tank_m, r.delta_hi_m - r.rain_m - r.tank_m)), 0.5)] });
    }));
  }
  return { inputs: inp, to_round: toDate, wells: rows, levels, area_summary, footprint: { zones, tank_buffer_km: 3 }, taluks, pumping, offline: true };
}

// Theis (1935) drawdown s = Q/(4πT)·W(u), u = r²S/(4Tt). W(u) = E1(u), the exponential integral.
export function expint1(u) {
  if (u <= 0) return Infinity;
  if (u < 1) {
    let sum = 0, term = 1;
    for (let k = 1; k < 60; k++) { term *= -u / k; sum += -term / k; }
    return -0.5772156649015329 - Math.log(u) + sum;
  }
  // continued fraction (Lentz), converges for u >= 1
  let b = u + 1, c = 1 / 1e-300, d = 1 / b, h = d;
  for (let i = 1; i < 200; i++) {
    const an = -i * i;
    b += 2;
    d = 1 / (an * d + b);
    c = b + an / c;
    const del = c * d;
    h *= del;
    if (Math.abs(del - 1) < 1e-12) break;
  }
  return h * Math.exp(-u);
}

export function theis({ Q, T, S, r, t }) {
  const u = (r * r * S) / (4 * T * t);
  return { u, W: expint1(u), s: (Q / (4 * Math.PI * T)) * expint1(u) };
}
