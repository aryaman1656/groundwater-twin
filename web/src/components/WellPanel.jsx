import React, { useEffect, useState } from "react";
import { getWell } from "../api.js";
import Borehole from "./Borehole.jsx";
import Hydrograph from "./Hydrograph.jsx";
import { COLORS, fmt, signed, dateLabel, TERM_LABELS, FEATURE_LABELS } from "../format.js";

function TermBar({ label, value, span }) {
  const w = Math.min(50, (Math.abs(value) / span) * 50);
  const left = value < 0 ? 50 - w : 50;
  return (
    <div className="termbar">
      <span>{label}</span>
      <span className="track">
        <span className="zero" />
        <span className="fill" style={{ left: `${left}%`, width: `${w}%`, background: value < 0 ? COLORS.water : COLORS.umber }} />
      </span>
      <span className="num" style={{ textAlign: "right" }}>{signed(value, 2)} m</span>
    </div>
  );
}

function RegionSummary({ wells, meta }) {
  const inStudy = wells.filter((w) => w.district && w.active);
  const deep = inStudy.filter((w) => w.aquifer === "deep");
  const falling = deep.filter((w) => w.forecast_change_m > 1).length;
  const rising = deep.filter((w) => w.forecast_change_m < -1).length;
  return (
    <div className="empty">
      <strong>Pick a well on the map</strong>
      You'll see its measured water levels, what the twin estimates for the time since the last reading,
      a forecast with uncertainty bands, and what is driving it.
      <h3>Across the two districts</h3>
      <div className="pair">
        <div><div className="big num">{falling}</div><div className="small">bore wells forecast to be more than 1 m deeper in {dateLabel(wells[0]?.forecast?.at(-1)?.date)} than now</div></div>
        <div><div className="big num">{rising}</div><div className="small">bore wells forecast to be more than 1 m shallower</div></div>
      </div>
      <p className="small" style={{ marginTop: 14 }}>
        {inStudy.length} wells inside Kolar and Chikkaballapura have readings since 2021 ({deep.length} bore wells, {inStudy.length - deep.length} dug wells).
        Older wells and {wells.filter((w) => !w.district).length} wells within 15 km of the border also feed the model. Comparing the same season a year apart removes the normal dry-season fall.
        The twin resets each well to every new reading since {dateLabel(meta?.assimilation_start)}, and uses IMD rainfall up to {dateLabel(meta?.forcing_observed_until)}.
      </p>
    </div>
  );
}

export default function WellPanel({ wellId, wells, meta, onTaluk }) {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    if (!wellId) return;
    setD(null); setErr(null);
    getWell(wellId).then(setD).catch((e) => setErr(String(e)));
  }, [wellId]);

  if (!wellId) return <RegionSummary wells={wells} meta={meta} />;
  if (err) return <div className="empty"><strong>Couldn't load this well</strong>{err}. Check that the API is running (python -m gwtwin serve).</div>;
  if (!d) return <div className="empty">Loading well…</div>;

  const w = d.well;
  const obs = d.observed.map((o) => o.depth);
  const now = d.nowcast.at(-1);
  const last = d.forecast.at(-1);
  const terms = d.explain?.physics_terms_m || {};
  const allTerms = { ...terms, ml_residual: d.explain?.ml_residual_m || 0 };
  const span = Math.max(0.5, ...Object.values(allTerms).map((v) => Math.abs(v)));
  const lastObs = d.observed.at(-1);
  const lw = wells.find((x) => x.well_id === wellId);
  const since = lw?.rounds_since_obs;

  return (
    <div>
      <div className="wellhead">
        <div>
          <h2>{w.name}</h2>
          <div className="muted small">
            {w.well_type || (w.aquifer === "deep" ? "Bore well" : "Dug well")}, {w.agency}
            {w.taluk ? <>, <button className="link" onClick={() => onTaluk?.(w.taluk)}>{w.taluk} taluk</button></> : " (outside the study districts: feeds the model, not counted in any total)"}
          </div>
        </div>
      </div>
      <div className={`agebar ${since > 4 ? "stale" : ""}`}>
        <div className="agebig num">{since ?? "–"}</div>
        <div><strong>rounds since the last real reading</strong> ({dateLabel(lastObs?.date)}). {since > 4
          ? "Everything after that date is the twin's simulation with observed rain, not a measurement, and the forecast starts from it."
          : "The forecast starts close to a real measurement."}</div>
      </div>

      <div className="facts">
        <div className="fact"><div className="label">Last reading</div><div className="value">{fmt(lastObs?.depth)} m</div><div className="small muted">{dateLabel(lastObs?.date)}</div></div>
        <div className="fact"><div className="label">Twin estimate now</div><div className="value">{fmt(now?.depth)} m</div><div className="small muted">{dateLabel(now?.date)}</div></div>
        <div className="fact"><div className="label">Forecast {dateLabel(last?.date)}</div><div className="value">{fmt(last?.median)} m</div><div className="small muted">80%: {fmt(last?.lo80)}–{fmt(last?.hi80)} m</div></div>
      </div>

      <div className="duo">
        <Borehole wellDepth={w.well_depth} now={now?.depth} lo={last?.lo80} hi={last?.hi80}
                  histMin={Math.min(...obs)} histMax={Math.max(...obs)} />
        <Hydrograph detail={d} />
      </div>
      <p className="small muted">
        Blue dots are measurements. The dashed line is the twin between readings (it resets to every new measurement).
        The brown line and bands are the forecast with 80% and 90% intervals. Future rain is unknown, so the bands
        combine model error with 21 years of possible weather.
      </p>

      <h3>What drives the next round (about 3 months)</h3>
      {Object.entries(allTerms).map(([k, v]) => (
        <TermBar key={k} label={TERM_LABELS[k] || k} value={v} span={span} />
      ))}
      <p className="small muted">Positive = water table falls. Weather is shown as its departure from a normal season (future rain is assumed normal, so only the past year's rain differs); the rest is the usual change for this time of year.</p>

      {d.explain?.ml_weight === 0 && (
        <p className="small muted">For {w.aquifer === "deep" ? "bore" : "dug"} wells the ML correction is switched off: on recent held-out data the water balance alone did better, so the deployed model uses physics only.</p>
      )}
      {d.explain?.ml_weight > 0 && d.explain?.ml_top_features?.length > 0 && (
        <>
          <h3>Inside the ML correction</h3>
          <table>
            <thead><tr><th>Input</th><th className="r">Value</th><th className="r">Effect (m)</th></tr></thead>
            <tbody>
              {d.explain.ml_top_features.map((f) => (
                <tr key={f.feature}><td>{FEATURE_LABELS[f.feature] || f.feature}</td><td className="r">{fmt(f.value, 2)}</td><td className="r">{signed(f.shap_m, 3)}</td></tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {d.explain?.neighbours?.length > 0 && (
        <>
          <h3>Linked wells in the spatial graph</h3>
          <table>
            <thead><tr><th>Well</th><th className="r">Distance</th><th className="r">Weight</th></tr></thead>
            <tbody>
              {d.explain.neighbours.map((n) => (
                <tr key={n.well_id}><td>{n.name || n.well_id}</td><td className="r">{fmt(n.km)} km</td><td className="r">{fmt(n.weight * 100, 0)}%</td></tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
