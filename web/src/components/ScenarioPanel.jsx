import React, { useEffect, useMemo, useState } from "react";
import { runScenario, getMode, getScenarioGrid } from "../api.js";
import { COLORS, fmt, signed, dateLabel, TERM_LABELS } from "../format.js";
import { areaLabel, theis } from "../areas.js";

function Slider({ label, value, setValue, min, max, step, unit, hint, marks }) {
  return (
    <div className="control">
      <div className="row">
        <label htmlFor={label}>{label}</label>
        <span className="val">{value > 0 ? "+" : ""}{value}{unit}</span>
      </div>
      <input id={label} type="range" min={min} max={max} step={step} value={value}
             onChange={(e) => setValue(Number(e.target.value))} list={marks ? `${label}-marks` : undefined} />
      {marks && <datalist id={`${label}-marks`}>{marks.map((m) => <option key={m} value={m} />)}</datalist>}
      {hint && <div className="hint">{hint}</div>}
    </div>
  );
}

const PRESETS = [
  { name: "Drought year", rain_pct: -40, pump_pct: 20, tank_pct: 0 },
  { name: "Tanks run dry", rain_pct: 0, pump_pct: 0, tank_pct: -100 },
  { name: "Tank water doubled", rain_pct: 0, pump_pct: 0, tank_pct: 100 },
  { name: "Pumping cut 20%", rain_pct: 0, pump_pct: -20, tank_pct: 0 },
];

function AreaPicker({ area, setArea, districts, taluks, picked, setPicking, picking }) {
  const level = area?.level || "all";
  return (
    <div className="control">
      <div className="row"><label htmlFor="area-level">Where the change applies</label></div>
      <div className="arearow">
        <select id="area-level" value={level} onChange={(e) => {
          const l = e.target.value;
          if (l === "all") setArea(null);
          else if (l === "region") setArea({ level: "region" });
          else if (l === "district") setArea({ level: "district", name: districts[0] });
          else if (l === "taluk") setArea({ level: "taluk", name: taluks[0] });
          else setArea({ level: "radius", lat: picked?.lat ?? 13.14, lon: picked?.lon ?? 78.13, radius_km: area?.radius_km ?? 10 });
        }}>
          <option value="all">Every well in the model (as before)</option>
          <option value="region">Both districts</option>
          <option value="district">One district</option>
          <option value="taluk">One taluk</option>
          <option value="radius">A circle on the map</option>
        </select>
        {level === "district" && (
          <select aria-label="District" value={area.name} onChange={(e) => setArea({ level: "district", name: e.target.value })}>
            {districts.map((d) => <option key={d}>{d}</option>)}
          </select>
        )}
        {level === "taluk" && (
          <select aria-label="Taluk" value={area.name} onChange={(e) => setArea({ level: "taluk", name: e.target.value })}>
            {taluks.map((d) => <option key={d}>{d}</option>)}
          </select>
        )}
      </div>
      {level === "radius" && (
        <>
          <div className="btnrow">
            <button className={`btn ghost ${picking ? "on" : ""}`} onClick={() => setPicking(!picking)}>{picking ? "Click the map…" : "Pick the centre on the map"}</button>
            <span className="small muted" style={{ alignSelf: "center" }}>{fmt(area.lat, 3)}°N, {fmt(area.lon, 3)}°E</span>
          </div>
          <Slider label="Radius" value={area.radius_km} setValue={(v) => setArea({ ...area, radius_km: v })} min={3} max={40} step={1} unit=" km" />
        </>
      )}
      <div className="hint">
        Inside the area all three levers act. Within 3 km outside it only tank water reaches a well (the fitted tank term counts water within 3 km).
        Beyond that the model has no effect: the data show no detectable spillover between wells (see Evidence).
      </div>
    </div>
  );
}

function TheisCard() {
  const [v, setV] = useState({ Q: "", T: "", S: "", r: "", t: "" });
  const num = Object.fromEntries(Object.entries(v).map(([k, x]) => [k, Number(x)]));
  const ok = Object.values(v).every((x) => x !== "" && Number(x) > 0);
  const out = ok ? theis(num) : null;
  const field = (k, label, ph) => (
    <label className="tfield" key={k}><span>{label}</span>
      <input type="number" min="0" step="any" value={v[k]} placeholder={ph} onChange={(e) => setV({ ...v, [k]: e.target.value })} />
    </label>
  );
  return (
    <details className="card">
      <summary>Drawdown next to one pumping well — an illustration with your own numbers</summary>
      <p className="small">
        The twin cannot give the drawdown inside a particular bore well: it has no pumping rate or aquifer test for any well. If you have them,
        the Theis formula gives a textbook estimate. <strong>This uses only the numbers you type; nothing comes from the monitoring data.</strong>{" "}
        Theis assumes a uniform, confined aquifer; fractured granite is neither, so real drawdown can differ a lot.
      </p>
      <div className="tgrid">
        {field("Q", "Pumping rate Q (m³/day)", "e.g. from a meter")}
        {field("T", "Transmissivity T (m²/day)", "NAQUIM / pump test")}
        {field("S", "Storativity S (–)", "pump test")}
        {field("r", "Distance from the well r (m)", "≥ well radius")}
        {field("t", "Time since pumping started (days)", "")}
      </div>
      {out && (
        <p className="small"><strong>Illustrative drawdown: {fmt(out.s, 2)} m</strong> at {num.r} m after {num.t} days (u = {out.u.toExponential(2)}, W(u) = {fmt(out.W, 3)}).
          s = Q/(4πT)·W(u), u = r²S/(4Tt).</p>
      )}
      {!ok && <p className="small muted">Fill in all five values to see the estimate.</p>}
    </details>
  );
}

export default function ScenarioPanel({ result, setResult, onShowMap, wells, areaIndex, area, setArea, picking, setPicking, selectedWell }) {
  const [rain, setRain] = useState(result?.inputs?.rain_pct ?? -20);
  const [pump, setPump] = useState(result?.inputs?.pump_pct ?? 20);
  const [tank, setTank] = useState(result?.inputs?.tank_pct ?? 0);
  const [horizon, setHorizon] = useState(result?.inputs?.horizon ?? 4);
  const [lateral, setLateral] = useState(false);
  const [weather, setWeather] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [grid, setGrid] = useState(null);
  const isStatic = getMode() === "static";
  useEffect(() => { if (isStatic) getScenarioGrid().then(setGrid).catch(() => {}); }, [isStatic]);
  const districts = useMemo(() => (areaIndex || []).filter((a) => a.level === "district").map((a) => a.name), [areaIndex]);
  const taluks = useMemo(() => (areaIndex || []).filter((a) => a.level === "taluk").map((a) => a.name).sort(), [areaIndex]);

  const run = async (s) => {
    setBusy(true); setErr(null);
    try {
      const r = await runScenario(s, wells);
      setResult(r);
      onShowMap();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };
  const inputs = (o = {}) => ({ rain_pct: rain, pump_pct: pump, tank_pct: tank, horizon, area: area || undefined,
                                lateral: !!(lateral && area), weather_range: weather, ...o });

  const drivers = result?.drivers_mean_m || {};
  const span = Math.max(0.3, ...Object.values(drivers).map((v) => Math.abs(v)));
  const S = result?.area_summary;
  const sel = selectedWell && result?.wells?.find((w) => w.well_id === selectedWell);
  const selMeta = selectedWell && wells.find((w) => w.well_id === selectedWell);
  const resArea = result?.inputs?.area;
  const affected = (l) => {
    const xs = (result?.wells || []).filter((w) => w.in_totals && w.aquifer === l.aquifer &&
      (l.level === "region" || (l.level === "district" ? w.district === l.name : w.taluk === l.name)));
    return xs.length ? xs.filter((w) => w.zone !== "outside").length / xs.length : null;
  };

  return (
    <div>
      <h2>What if…</h2>
      <p className="muted">
        Change the conditions for the coming seasons and compare with the baseline forecast. Both start from the twin's estimate of today's
        water levels and use typical rainfall unless you change it.
      </p>
      {isStatic && (
        <div className="note">Running without the Python backend: results are composed in the browser from precomputed per-well responses
          (exact for this model). Crop and income results, the lateral-flow illustration and the weather range need the live backend
          {grid ? `; offline, crop results exist for region-wide scenarios on the grid (rain ${grid.rain_pct.join(", ")}%; pumping ${grid.pump_pct.join(", ")}%; tanks ${grid.tank_pct.join(", ")}%; 2 rounds)` : ""}.</div>
      )}
      <AreaPicker area={area} setArea={setArea} districts={districts} taluks={taluks} picking={picking} setPicking={setPicking} />
      <Slider label="Rainfall" value={rain} setValue={setRain} min={-50} max={50} step={5} unit="%" marks={grid?.rain_pct}
              hint="Scales seasonal rainfall in the fitted water balance." />
      <Slider label="Groundwater pumping" value={pump} setValue={setPump} min={-50} max={50} step={5} unit="%" marks={grid?.pump_pct}
              hint="Physics only: pumping never varied in the data. Extra water removed (CGWB 2024 district draft) ÷ specific yield." />
      <Slider label="Dry-season tank water" value={tank} setValue={setTank} min={-100} max={100} step={10} unit="%" marks={grid?.tank_pct}
              hint="Scales the water held in tanks within 3 km of each well (Sentinel-2). −100% means no K.C./H.N. Valley water." />
      <Slider label="Look ahead" value={horizon} setValue={setHorizon} min={1} max={4} step={1} unit=" rounds"
              hint="One round is about 3 months (CGWB measures in Jan, May, Aug and Nov)." />
      {!isStatic && (
        <div className="control small">
          <label className="check"><input type="checkbox" checked={weather} onChange={(e) => setWeather(e.target.checked)} /> Show how the response varies with the weather year (21 years, ~15 s)</label>
          <label className="check"><input type="checkbox" checked={lateral} disabled={!area} onChange={(e) => setLateral(e.target.checked)} /> Add a lateral-flow illustration beyond the area (physics only, assumed transmissivity{!area ? "; pick an area first" : ""})</label>
        </div>
      )}
      <div className="btnrow">
        <button className="btn" disabled={busy} onClick={() => run(inputs())}>{busy ? "Running…" : "Run scenario"}</button>
        {PRESETS.map((p) => (
          <button key={p.name} className="btn ghost" disabled={busy}
                  onClick={() => { setRain(p.rain_pct); setPump(p.pump_pct); setTank(p.tank_pct); run(inputs(p)); }}>
            {p.name}
          </button>
        ))}
      </div>
      {err && <p className="alert">Scenario failed: {err}</p>}

      {result && S && (
        <div className="result">
          <h3 style={{ marginTop: 0 }}>
            Rain {signed(result.inputs.rain_pct, 0)}%, pumping {signed(result.inputs.pump_pct, 0)}%, tank water {signed(result.inputs.tank_pct, 0)}% — {areaLabel(resArea)}
          </h3>
          <p className="small muted">{result.from_round ? `${dateLabel(result.from_round)} to ` : "To "}{dateLabel(result.to_round)}, scenario minus baseline. + = deeper water.</p>
          <div className="pair">
            {["deep", "shallow"].map((a) => (
              <div key={a}>
                <div className="big num" style={{ color: (S[a].median_delta_m ?? 0) > 0 ? COLORS.umber : COLORS.water }}>{signed(S[a].median_delta_m, 2)} m</div>
                <div className="small">
                  median {a === "deep" ? "bore well" : "dug well"} inside the area ({S[a].n_wells} wells){a === "shallow" && " — indicative"}.
                  {result.inputs.pump_pct !== 0 && S[a].sy_range_m && <> Range from the specific-yield bounds: {signed(S[a].sy_range_m[0], 1)} to {signed(S[a].sy_range_m[1], 1)} m.</>}
                  {result.weather_range?.[a] && <> Across 21 weather years: {signed(result.weather_range[a].p10, 1)} to {signed(result.weather_range[a].p90, 1)} m.</>}
                  {!result.weather_range && S[a].weather_range_m && (result.inputs.rain_pct || result.inputs.tank_pct) ? <> Across weather years (approx.): {signed(S[a].weather_range_m[0], 1)} to {signed(S[a].weather_range_m[1], 1)} m.</> : null}
                </div>
              </div>
            ))}
          </div>

          {result.footprint && (
            <>
              <h3>Footprint</h3>
              <div className="zones">
                <div className="zone inside"><b className="num">{result.footprint.zones.inside.n_wells}</b> wells inside: all levers act</div>
                <div className="zone buffer"><b className="num">{result.footprint.zones.buffer.n_wells}</b> within 3 km outside: tank water only{result.inputs.tank_pct ? `, up to ${signed(result.footprint.zones.buffer.max_abs_delta_m, 2)} m` : ""}</div>
                <div className="zone outside"><b className="num">{result.footprint.zones.outside.n_wells}</b> beyond: no modelled effect</div>
              </div>
            </>
          )}

          {sel && (
            <div className="note">
              <strong>{selMeta?.name || sel.well_id}</strong>: {signed(sel.delta_m, 2)} m{sel.zone !== "inside" && ` (${sel.zone} the area)`}.{" "}
              This is the typical response of this aquifer type at this location, not this well's own behaviour: every
              {sel.aquifer === "deep" ? " bore" : " dug"} well shares the same water-balance coefficients, and pumping is the district average.
              {result.inputs.pump_pct !== 0 && <> Range from the specific-yield bounds: {signed(sel.delta_lo_m, 2)} to {signed(sel.delta_hi_m, 2)} m.</>}
            </div>
          )}

          {result.pumping && (
            <>
              <h3>Pumping — physics only</h3>
              <table>
                <thead><tr><th>District</th><th className="r">Response</th><th className="r">Range over Sy bounds</th><th className="r">Wells</th></tr></thead>
                <tbody>
                  {result.pumping.by_district.map((p) => (
                    <tr key={p.district + p.aquifer}><td>{p.district}, {p.aquifer === "deep" ? "bore" : "dug"} wells</td>
                      <td className="r">{signed(p.pump_m, 2)} m</td><td className="r">{signed(p.range_m[0], 1)} to {signed(p.range_m[1], 1)} m</td><td className="r">{p.n_wells}</td></tr>
                  ))}
                </tbody>
              </table>
              <p className="small muted">Extra draft = pumping change × district draft (CGWB GWRA-2024: Kolar 0.18 m/yr, Chikkaballapura 0.12 m/yr over the district area) ÷ specific yield,
                then the fitted return toward normal. Every well of one aquifer and district gets the same number: there is no per-well consumption data.
                The range spans the specific-yield bounds in config.yaml (bore wells 0.005–0.03, GEC norms); NAQUIM taluk values will narrow it.
                This is the regional water table, not the drawdown inside a pumped well.</p>
            </>
          )}

          {result.lateral && (
            <>
              <h3>Lateral flow beyond the area — illustration only</h3>
              <table>
                <thead><tr><th>Distance from the edge</th><th className="r">Bore wells</th><th className="r">Low T/S</th><th className="r">Mid</th><th className="r">High</th></tr></thead>
                <tbody>
                  {result.lateral.by_distance.map((b, i) => (
                    <tr key={i}><td>{b.from_km === null ? "inside" : `${b.from_km}–${b.to_km ?? "…"} km`}</td><td className="r">{b.n_deep_wells}</td>
                      <td className="r">{signed(b.low_m, 2)}</td><td className="r">{signed(b.mid_m, 2)}</td><td className="r">{signed(b.high_m, 2)}</td></tr>
                  ))}
                </tbody>
              </table>
              <p className="small muted">{result.lateral.note} Rain and pumping only; over {result.lateral.t_days} days.</p>
            </>
          )}

          {Object.keys(drivers).length > 0 && (
            <>
              <h3>Where the difference comes from</h3>
              {Object.entries(drivers).filter(([, v]) => Math.abs(v) > 1e-4).map(([k, v]) => (
                <div className="termbar" key={k}>
                  <span>{TERM_LABELS[k] || k}</span>
                  <span className="track"><span className="zero" />
                    <span className="fill" style={{ left: `${v < 0 ? 50 - Math.min(50, (Math.abs(v) / span) * 50) : 50}%`, width: `${Math.min(50, (Math.abs(v) / span) * 50)}%`, background: v < 0 ? COLORS.water : COLORS.umber }} />
                  </span>
                  <span className="num" style={{ textAlign: "right" }}>{signed(v, 2)} m</span>
                </div>
              ))}
              <p className="small muted">Average over all {result.wells.length} wells, including those outside the area.</p>
            </>
          )}

          <h3>By area (bore wells)</h3>
          <table>
            <thead><tr><th>Area</th><th className="r">Median</th><th className="r">Middle half</th><th className="r">Wells</th><th className="r">Affected</th></tr></thead>
            <tbody>
              {(result.levels || []).filter((l) => l.aquifer === "deep").map((l) => (
                <tr key={l.level + l.name} className={l.level === "taluk" ? "" : "best"}>
                  <td>{l.level === "taluk" ? " " : ""}{l.name}{l.level === "district" ? " district" : ""}</td>
                  <td className="r">{signed(l.delta_m.median, 2)} m</td>
                  <td className="r">{signed(l.delta_m.p25, 2)} to {signed(l.delta_m.p75, 2)}</td><td className="r">{l.n_wells}</td>
                  <td className="r">{affected(l) === null ? "–" : `${fmt(affected(l) * 100, 0)}%`}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="small muted">"Affected" = share of the area's study wells inside the selected area or its 3 km tank buffer. Where it is below 100%, the median and middle half include unaffected wells (0 m).</p>
          {result.agro_note && <p className="small muted">{result.agro_note}</p>}
        </div>
      )}
      <TheisCard />
    </div>
  );
}
