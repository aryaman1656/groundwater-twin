import React, { useState } from "react";
import { fmt, lakh, lossText, signed, dateLabel } from "../format.js";

export default function AgroPanel({ result, goScenario }) {
  const [showAll, setShowAll] = useState(false);
  if (!result?.agro)
    return (
      <div className="empty">
        <strong>Run a scenario first</strong>
        Crop and income effects are worked out from a scenario's change in the deep aquifer, the one farmers pump.
        <div className="btnrow" style={{ marginTop: 12 }}><button className="btn" onClick={goScenario}>Set up a scenario</button></div>
      </div>
    );
  const a = result.agro;
  const t = a.total_rs_lakh;
  const gain = t.p50 < 0;
  const A = a.assumptions;
  const rows = [...a.rows].sort((x, y) => y.loss_rs_lakh_p50 - x.loss_rs_lakh_p50);

  return (
    <div>
      <h2>Crops and income at risk</h2>
      <p className="muted">
        Scenario rain {signed(result.inputs.rain_pct, 0)}%, pumping {signed(result.inputs.pump_pct, 0)}%, tank water{" "}
        {signed(result.inputs.tank_pct, 0)}%, {dateLabel(result.from_round)} to {dateLabel(result.to_round)}. Groundwater-irrigated crops only.
      </p>
      <div className="pair" style={{ marginTop: 12 }}>
        <div>
          <div className="big num">{lakh(Math.abs(t.p50))}</div>
          <div className="small">{gain ? "income gained" : "income lost"} in the median simulation</div>
        </div>
        <div>
          <div className="big num" style={{ fontSize: 22 }}>{lossText(t.p10)}</div>
          <div className="big num" style={{ fontSize: 22 }}>to {lossText(t.p90)}</div>
          <div className="small">80% of simulations fall in this range</div>
        </div>
      </div>

      <h3>By crop</h3>
      <table>
        <thead><tr><th>Crop</th><th className="r">Area at risk</th><th className="r">Water from irrigation</th><th className="r">Median</th><th className="r">80% range</th></tr></thead>
        <tbody>
          {Object.entries(a.by_crop).map(([c, v]) => (
            <tr key={c}>
              <td>{c}{v.ky_source?.startsWith("ASSUMPTION") ? " *" : ""}</td>
              <td className="r">{fmt(v.area_at_risk_ha_p50, 0)} ha</td>
              <td className="r">{fmt(v.irrigation_share_of_water_need * 100, 0)}%</td>
              <td className="r">{lossText(v.p50)}</td>
              <td className="r">{lakh(v.p10)} to {lakh(v.p90)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="small muted">Area at risk = irrigated area where the median yield loss exceeds 5%. The range is the loss; a minus sign means a gain. * Rice has no FAO-33 seasonal Ky; 1.05 is an assumption to verify.</p>

      <h3>By taluk</h3>
      <table>
        <thead><tr><th>Taluk</th><th>Crop</th><th className="r">Irrigated area</th><th className="r">Supply cut</th><th className="r">Yield loss</th><th className="r">Loss, median</th></tr></thead>
        <tbody>
          {(showAll ? rows : rows.slice(0, 10)).map((r) => (
            <tr key={r.taluk + r.crop}>
              <td>{r.taluk}</td><td>{r.crop}</td><td className="r">{fmt(r.irr_area_ha, 0)} ha</td>
              <td className="r">{fmt(r.supply_cut_pct_p50, 1)}%</td>
              <td className="r">{fmt(r.yield_loss_pct_p10, 1)}–{fmt(r.yield_loss_pct_p90, 1)}%</td>
              <td className="r">{lossText(r.loss_rs_lakh_p50)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > 10 && (
        <div className="btnrow"><button className="btn ghost" onClick={() => setShowAll(!showAll)}>{showAll ? "Show top 10" : `Show all ${rows.length} rows`}</button></div>
      )}

      <h3>How the number is built</h3>
      <ul className="plain small">
        <li>Change in bore-well level (scenario minus baseline) × specific yield = extra water missing from storage.</li>
        <li>That deficit ÷ the irrigation water pumped over the same months = share of supply lost, times θ (how much of the deficit actually reaches farms).</li>
        <li>Supply lost × share of the crop's water need met by irrigation = shortfall in crop evapotranspiration (FAO-56 Kc, IMD rainfall and temperature).</li>
        <li>Shortfall × Ky (FAO-33) = yield loss; × yield × irrigated area × mandi price = rupees.</li>
        <li>Specific yield, θ, yields, prices and the size of the groundwater response are all sampled, so the result is a range.</li>
      </ul>
      {A && (
        <>
          <h3>Inputs and assumptions</h3>
          <table>
            <tbody>
              <tr><td>Specific yield, bore wells</td><td className="r">{A.Sy_deep_range.join(" to ")}</td></tr>
              <tr><td>θ, deficit reaching farms</td><td className="r">{A.theta_range.join(" to ")}</td></tr>
              {Object.entries(A.yields_t_per_ha).map(([c, y]) => (
                <tr key={c}><td>{c} yield (t/ha)</td><td className="r">{fmt(y.p50)} to {fmt(y.p90)} <span className="muted small">{y.source}</span></td></tr>
              ))}
              {Object.entries(A.prices_rs_per_tonne).map(([c, p]) => (
                <tr key={c}><td>{c} price (₹/t)</td><td className="r">{fmt(p.p10, 0)} to {fmt(p.p90, 0)} <span className="muted small">{p.n} APMC records</span></td></tr>
              ))}
              <tr><td>Crop areas</td><td className="r">{A.crop_areas}</td></tr>
            </tbody>
          </table>
          <p className="small muted">Not modelled: {A.not_modelled.join("; ")}. This is a model-based estimate for comparing scenarios, not an audited loss figure.</p>
        </>
      )}
    </div>
  );
}
