import React, { useEffect, useState } from "react";
import { runScenario, getMode, getScenarioGrid } from "../api.js";
import { COLORS, fmt, signed, dateLabel, TERM_LABELS } from "../format.js";

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
      <div className="hint">{hint}</div>
    </div>
  );
}

const PRESETS = [
  { name: "Drought year", rain_pct: -40, pump_pct: 20, tank_pct: 0 },
  { name: "Tanks run dry", rain_pct: 0, pump_pct: 0, tank_pct: -100 },
  { name: "Tank water doubled", rain_pct: 0, pump_pct: 0, tank_pct: 100 },
  { name: "Pumping cut 20%", rain_pct: 0, pump_pct: -20, tank_pct: 0 },
];

export default function ScenarioPanel({ result, setResult, onShowMap }) {
  const [rain, setRain] = useState(result?.inputs?.rain_pct ?? -20);
  const [pump, setPump] = useState(result?.inputs?.pump_pct ?? 20);
  const [tank, setTank] = useState(result?.inputs?.tank_pct ?? 0);
  const [horizon, setHorizon] = useState(result?.inputs?.horizon ?? 2);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [grid, setGrid] = useState(null);
  const isStatic = getMode() === "static";
  useEffect(() => { if (isStatic) getScenarioGrid().then(setGrid).catch(() => {}); }, [isStatic]);

  const run = async (s) => {
    setBusy(true); setErr(null);
    try {
      const r = await runScenario(s);
      setResult(r);
      onShowMap();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const drivers = result?.drivers_mean_m || {};
  const span = Math.max(0.3, ...Object.values(drivers).map((v) => Math.abs(v)));

  return (
    <div>
      <h2>What if…</h2>
      <p className="muted">
        Change the conditions for the coming seasons and compare with the baseline forecast. Both start from the
        twin's estimate of today's water levels and use typical rainfall unless you change it.
      </p>
      {isStatic && (
        <div className="note">Running without the Python backend: sliders snap to precomputed values
          {grid ? ` (rain ${grid.rain_pct.join(", ")}%; pumping ${grid.pump_pct.join(", ")}%; tank water ${grid.tank_pct.join(", ")}%)` : ""}.
          Start the API for any combination.</div>
      )}
      <Slider label="Rainfall" value={rain} setValue={setRain} min={-50} max={50} step={5} unit="%" marks={grid?.rain_pct}
              hint="Scales seasonal rainfall; enters the fitted water balance and the ML correction (which can only raise levels with more rain)." />
      <Slider label="Groundwater pumping" value={pump} setValue={setPump} min={-50} max={50} step={5} unit="%" marks={grid?.pump_pct}
              hint="Extra extraction on top of the CGWB 2024 estimate. Pumping never varied in the data, so its effect is physics only: extra water removed ÷ specific yield." />
      <Slider label="Dry-season tank water" value={tank} setValue={setTank} min={-100} max={100} step={10} unit="%" marks={grid?.tank_pct}
              hint="Scales the water held in tanks near each well (Sentinel-2). −100% means no K.C./H.N. Valley water." />
      <Slider label="Look ahead" value={horizon} setValue={setHorizon} min={1} max={4} step={1} unit=" rounds"
              hint="One round is about 3 months (CGWB measures in Jan, May, Aug and Nov)." />
      <div className="btnrow">
        <button className="btn" disabled={busy} onClick={() => run({ rain_pct: rain, pump_pct: pump, tank_pct: tank, horizon })}>
          {busy ? "Running…" : "Run scenario"}
        </button>
        {PRESETS.map((p) => (
          <button key={p.name} className="btn ghost" disabled={busy}
                  onClick={() => { setRain(p.rain_pct); setPump(p.pump_pct); setTank(p.tank_pct); run({ ...p, horizon }); }}>
            {p.name}
          </button>
        ))}
      </div>
      {err && <p className="alert">Scenario failed: {err}</p>}

      {result && (
        <div className="result">
          <h3 style={{ marginTop: 0 }}>
            {result.snapped ? "Nearest precomputed scenario" : "Result"}: rain {signed(result.inputs.rain_pct, 0)}%,
            pumping {signed(result.inputs.pump_pct, 0)}%, tank water {signed(result.inputs.tank_pct, 0)}%
          </h3>
          <p className="small muted">{dateLabel(result.from_round)} to {dateLabel(result.to_round)}, compared with the baseline forecast.</p>
          <div className="pair">
            {["deep", "shallow"].map((a) => (
              <div key={a}>
                <div className="big num" style={{ color: result.summary[a].mean_delta_m > 0 ? COLORS.umber : COLORS.water }}>
                  {signed(result.summary[a].mean_delta_m, 2)} m
                </div>
                <div className="small">{a === "deep" ? "bore wells" : "dug wells"} on average; {fmt(result.summary[a]["share_wells_deeper_than_0.5m"] * 100, 0)}% fall more than 0.5 m extra</div>
              </div>
            ))}
          </div>
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
          <p className="small muted">Average over all wells; positive = deeper water. The map now shows scenario minus baseline; taluks are shaded by their bore wells.</p>
          <table>
            <thead><tr><th>Taluk</th><th className="r">Bore wells, median change</th></tr></thead>
            <tbody>
              {[...result.taluks].sort((a, b) => b.dh_mean - a.dh_mean).map((t) => (
                <tr key={t.taluk}><td>{t.taluk} <span className="muted small">({t.district})</span></td><td className="r">{signed(t.dh_mean, 2)} m</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
