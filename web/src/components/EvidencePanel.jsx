import React from "react";
import { ComposedChart, Line, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer, ReferenceLine, ErrorBar } from "recharts";
import { COLORS, fmt, signed } from "../format.js";

const bandLabel = (b) => (b[1] === null ? `${b[0]}+ km` : `${b[0]}–${b[1]} km`);

function CorrFigure({ rc }) {
  const data = rc.bands_km.map((b, i) => ({
    band: bandLabel(b),
    raw: rc.pearson.corr[i], rawErr: [rc.pearson.corr[i] - rc.pearson.lo90[i], rc.pearson.hi90[i] - rc.pearson.corr[i]],
    clip: rc.huber_clipped.corr[i], clipErr: [rc.huber_clipped.corr[i] - rc.huber_clipped.lo90[i], rc.huber_clipped.hi90[i] - rc.huber_clipped.corr[i]],
    pairs: rc.pearson.pairs[i],
  }));
  return (
    <div style={{ height: 210 }}>
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 8, right: 10, left: -8, bottom: 0 }}>
          <CartesianGrid stroke="#e3e8e5" vertical={false} />
          <XAxis dataKey="band" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} domain={[-0.05, "auto"]} />
          <ReferenceLine y={0} stroke={COLORS.stone} />
          <Tooltip formatter={(v, n) => (Array.isArray(v) ? null : [fmt(v, 3), n])} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line dataKey="raw" name="Residual correlation" stroke={COLORS.ink} dot={{ r: 3 }} isAnimationActive={false}>
            <ErrorBar dataKey="rawErr" width={3} stroke={COLORS.stone} />
          </Line>
          <Line dataKey="clip" name="Same, outliers clipped" stroke={COLORS.water} strokeDasharray="4 3" dot={{ r: 3 }} isAnimationActive={false}>
            <ErrorBar dataKey="clipErr" width={3} stroke={COLORS.water2} />
          </Line>
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function KCFigure({ r }) {
  // plotted as rise (= −Δdepth), so up = water table rose relative to wells 10+ km away
  const data = r.bands.map((b) => ({ band: b.band, d: -b.delta_m, err: [b.ci95[1] - b.delta_m, b.delta_m - b.ci95[0]], n: b.n_wells, ref: !!b.reference }));
  return (
    <div style={{ height: 200 }}>
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 8, right: 10, left: -8, bottom: 0 }}>
          <CartesianGrid stroke="#e3e8e5" vertical={false} />
          <XAxis dataKey="band" tick={{ fontSize: 11 }} tickFormatter={(b) => `${b} (${data.find((x) => x.band === b)?.n})`} />
          <YAxis tick={{ fontSize: 11 }} label={{ value: "rise, m", angle: -90, position: "insideLeft", fontSize: 10, dx: 14, dy: 20 }} />
          <ReferenceLine y={0} stroke={COLORS.stone} />
          <Tooltip formatter={(v) => (Array.isArray(v) ? null : `${signed(v, 2)} m rise`)} />
          <Bar dataKey="d" name="Rise after the project, vs 10+ km" fill={COLORS.water2} isAnimationActive={false}>
            <ErrorBar dataKey="err" width={4} stroke={COLORS.ink} />
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function EvidencePanel({ evidence }) {
  const sp = evidence?.spatial_evidence;
  if (!sp) return <div className="empty"><strong>No spatial evidence yet</strong>Run <code>python -m gwtwin spatial</code> and re-export.</div>;
  const lag = sp.lagged_neighbour || [];
  const eng = sp.engine_test;
  const hyd = sp.hydraulics;
  const kc = sp.kc_valley_by_distance || [];
  const kc0 = sp.kc_valley_treated_vs_control || [];
  const gnn = sp.stgnn_comparison || [];
  return (
    <div>
      <h2>Does a change in one place reach the next?</h2>
      <p className="muted">
        The scenario footprint stops 3 km outside the selected area because that is all the data and the model support. These are the tests behind that choice.
      </p>

      <h3>1. Residual correlation by distance</h3>
      {["deep", "shallow"].map((aq) => sp.residual_correlation?.[aq] && (
        <div key={aq}>
          <div className="small"><strong>{aq === "deep" ? "Bore wells" : "Dug wells"}</strong></div>
          <CorrFigure rc={sp.residual_correlation[aq]} />
        </div>
      ))}
      <p className="small">
        After the water balance, what is left of two wells' changes in the same round is only weakly alike (bore wells about 0.1 within 5 km),
        and it fades slowly with distance rather than dropping off near a well. That shape points to shared weather and shared timing of
        pumping across the region, not water moving from well to well. This is co-movement, not influence. Error bars: 90% bootstrap over rounds.
      </p>

      <h3>2. Does a neighbour's surprise come to me next round?</h3>
      <ul className="plain small">
        {lag.map((r) => (
          <li key={r.aquifer}>{r.aquifer === "deep" ? "Bore" : "Dug"} wells: coefficient {signed(r.coef, 3)} (95% CI {signed(r.ci95[0], 3)} to {signed(r.ci95[1], 3)}), {r.n.toLocaleString("en-IN")} well-rounds. No detectable effect.</li>
        ))}
      </ul>

      <h3>3. Does the engine spread a local change?</h3>
      {eng && (
        <p className="small">
          Adding {eng.add_ha} ha of tank water around {eng.n_treated} bore wells (within {fmt(eng.radius_km, 0)} km) raised them by{" "}
          {signed(-eng.runs[0].treated_mean_m, 2)} m over a year. The other {eng.runs[0].others_n} wells moved by exactly 0 with the deployed model.
          With the ML correction forced on, {eng.runs[1].others_moved_n} of them moved, by at most {fmt(eng.runs[1].others_max_abs_m, 2)} m and not all in the
          same direction: tree-model noise through the neighbour features, not a physical spillover.
        </p>
      )}

      <h3>4. How far can heads travel? (physics, assumed values)</h3>
      {hyd && (
        <p className="small">
          With transmissivity {hyd.deep.T_m2_per_day.join("–")} m²/day (an ASSUMPTION for fractured granite; check NAQUIM) and storage
          {" "}{hyd.deep.S.join("–")}, a change in head spreads about {fmt(hyd.deep.sqrt_Dt_km_1yr.low, 1)}–{fmt(hyd.deep.sqrt_Dt_km_1yr.high, 1)} km in a year
          (to half its size; {fmt(hyd.deep.two_sqrt_Dt_km_1yr.low, 1)}–{fmt(hyd.deep.two_sqrt_Dt_km_1yr.high, 1)} km to about a sixth). The nearest monitored
          bore well is usually 2–4 km away (median 3 km), so only at the high end of these values would a neighbour feel part of a change within a year,
          and the lagged test above finds no sign of it.
          The optional lateral-flow illustration in What if uses this physics; it is never part of the forecast.
        </p>
      )}

      <h3>5. K.C./H.N. Valley tanks, by distance to new dry-season water</h3>
      <p className="small">
        Difference-in-differences: change after the project (from mid-2019) relative to wells 10+ km from any new dry-season water, with fixed effects for each
        well and season and each round, and a rainfall control. Bars: rise of the water table relative to the 10+ km wells, with 95% intervals; number of wells in brackets.
      </p>
      {kc.map((r) => (
        <div key={r.aquifer}>
          <div className="small"><strong>{r.aquifer === "deep" ? "Bore wells" : "Dug wells"}</strong> ({r.n_wells} wells with readings before and after)</div>
          <KCFigure r={r} />
        </div>
      ))}
      <div className="note">
        Underpowered. The simple treated-vs-control comparison already had a bore-well interval of about ±8.8 m
        {kc0.find((k) => k.aquifer === "deep") && ` (${signed(kc0.find((k) => k.aquifer === "deep").did_m, 1)} m, 95% CI ${signed(kc0.find((k) => k.aquifer === "deep").ci95[0], 1)} to ${signed(kc0.find((k) => k.aquifer === "deep").ci95[1], 1)})`};
        splitting the wells into distance bands leaves 7–19 wells per band. No band shows a clear effect for bore wells, and there is no
        steady fall-off with distance for either aquifer. The single dug-well band below zero (6–10 km) sits farther out than the 0–3 km band,
        so it is not evidence of a distance gradient; with six tests one such result is expected by chance.
      </div>

      <h3>6. The graph network (ST-GNN), for comparison only</h3>
      <table>
        <thead><tr><th>Model</th><th className="r">Bore wells MAE</th><th className="r">vs no-change</th><th className="r">Dug wells MAE</th><th className="r">vs no-change</th></tr></thead>
        <tbody>
          {[...new Set(gnn.map((g) => g.model))].map((m) => (
            <tr key={m}><td>{m}</td>
              {["deep", "shallow"].map((aq) => {
                const g = gnn.find((x) => x.model === m && x.aquifer === aq);
                return <React.Fragment key={aq}><td className="r">{fmt(g?.MAE, 2)} m</td><td className="r">{m === "Persistence" ? "–" : `${signed((g?.skill_vs_persistence ?? 0) * 100, 0)}%`}</td></React.Fragment>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="small muted">
        Learning from neighbours gave no gain: the ST-GNN is behind no-change for both aquifers on held-out years. Rain is smooth across the region,
        so a network cannot tell "my rain" from "my neighbour's rain", and attention weights describe which wells move together, not which
        well affects which. No attention map is shown; if one is added, it must be read as co-movement, not spillover.
      </p>
    </div>
  );
}
