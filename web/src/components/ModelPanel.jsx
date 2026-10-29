import React from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, ComposedChart, Line, ErrorBar, ReferenceLine, CartesianGrid } from "recharts";
import { COLORS, fmt, signed } from "../format.js";

const AQ = { deep: "Bore wells", shallow: "Dug wells" };
const eff = (e) => (e === undefined || e === null ? "–" : `${fmt(Math.abs(e), 2)} m ${e < 0 ? "shallower" : "deeper"}`);

function Comparison({ ev }) {
  const rows = ev.test_metrics || [];
  const models = [...new Set(rows.map((r) => r.model))];
  const get = (m, a, k) => rows.find((r) => r.model === m && r.aquifer === a)?.[k];
  const best = {};
  ["deep", "shallow"].forEach((a) => {
    best[a] = models.reduce((b, m) => (get(m, a, "MAE") < get(b, a, "MAE") ? m : b), models[0]);
  });
  return (
    <table>
      <thead><tr><th>Model</th><th className="r">Bore wells MAE</th><th className="r">vs no-change</th><th className="r">Dug wells MAE</th><th className="r">vs no-change</th></tr></thead>
      <tbody>
        {models.map((m) => (
          <tr key={m}>
            <td>{m}</td>
            {["deep", "shallow"].map((a) => (
              <React.Fragment key={a}>
                <td className="r" style={{ fontWeight: best[a] === m ? 700 : 400 }}>{fmt(get(m, a, "MAE"), 2)} m</td>
                <td className="r">{m === "Persistence" ? "–" : `${signed(get(m, a, "skill_vs_persistence") * 100, 0)}%`}</td>
              </React.Fragment>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Horizon({ ev }) {
  const rows = (ev.horizon || []).filter((r) => r.nominal === 0.8);
  const data = [1, 2, 3, 4].map((h) => {
    const o = { h: `${h} round${h > 1 ? "s" : ""}` };
    rows.filter((r) => r.horizon_rounds === h).forEach((r) => {
      o[`twin_${r.aquifer}`] = r.MAE_twin;
      o[`pers_${r.aquifer}`] = r.MAE_persistence;
      o[`cov_${r.aquifer}`] = r.empirical;
    });
    return o;
  });
  return (
    <>
      <div style={{ height: 200 }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 6, right: 6, left: -10, bottom: 0 }}>
            <CartesianGrid stroke="#e3e8e5" vertical={false} />
            <XAxis dataKey="h" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v) => `${fmt(v, 2)} m`} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="twin_deep" name="Twin, bore wells" fill={COLORS.ink} />
            <Bar dataKey="pers_deep" name="No-change, bore wells" fill={COLORS.stone} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <table>
        <thead><tr><th>Ahead</th><th className="r">Bore wells: twin / no-change</th><th className="r">Dug wells: twin / no-change</th><th className="r">80% band covers</th></tr></thead>
        <tbody>
          {data.map((d) => (
            <tr key={d.h}><td>{d.h}</td>
              <td className="r">{fmt(d.twin_deep, 2)} / {fmt(d.pers_deep, 2)} m</td>
              <td className="r">{fmt(d.twin_shallow, 2)} / {fmt(d.pers_shallow, 2)} m</td>
              <td className="r">{fmt(d.cov_deep * 100, 0)}% / {fmt(d.cov_shallow * 100, 0)}%</td></tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function EventStudy({ es }) {
  if (!es || es.error) return <p className="small muted">{es?.aquifer}: {es?.error || "not available"}</p>;
  const data = es.event_study.map((r) => ({ year: r.year, d: r.delta_m, err: [r.delta_m - r.lo, r.hi - r.delta_m] }))
    .filter((r) => Number.isFinite(r.err[0]) && Math.abs(r.d) < 60);
  return (
    <div>
      <div className="small"><strong>{AQ[es.aquifer]}</strong>: {es.n_treated} wells near newly filled tanks vs {es.n_control} wells without.
        Difference after vs before: <strong>{signed(es.did_post_minus_pre_m, 2)} m</strong> (95% CI {signed(es.did_ci95[0], 2)} to {signed(es.did_ci95[1], 2)} m).</div>
      <div style={{ height: 170 }}>
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 8, right: 6, left: -10, bottom: 0 }}>
            <CartesianGrid stroke="#e3e8e5" vertical={false} />
            <XAxis dataKey="year" tick={{ fontSize: 11 }} />
            <YAxis reversed tick={{ fontSize: 11 }} label={{ value: "m, down = deeper", angle: -90, position: "insideLeft", fontSize: 10, dx: 14, dy: 40 }} />
            <ReferenceLine y={0} stroke={COLORS.stone} />
            <ReferenceLine x={2018} stroke={COLORS.water} strokeDasharray="3 3" label={{ value: "K.C. Valley starts", fontSize: 10, fill: COLORS.water, position: "insideTopRight" }} />
            <Tooltip formatter={(v) => (Array.isArray(v) ? "" : `${signed(v, 2)} m`)} />
            <Line dataKey="d" name="Treated minus control" stroke={COLORS.ink} dot={{ r: 3 }} isAnimationActive={false}>
              <ErrorBar dataKey="err" width={3} stroke={COLORS.stone} />
            </Line>
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

const pctS = (v) => (v === null || v === undefined ? "–" : `${signed(v * 100, 0)}%`);

function AreaSkill({ ae }) {
  if (!ae) return <p className="small muted">Run <code>python -m gwtwin areas</code> to score the model by area.</p>;
  const rows = ae.one_round.rows.filter((r) => r.model === "Deployed form" && (r.agg === "median" || r.level === "well"));
  const roll = ae.rolling_coverage.filter((r) => r.lag === 7 && r.period === "all banded years");
  const levels = ["well", "taluk", "district", "region"];
  const name = { well: "Single well", taluk: "Taluk (11)", district: "District (2)", region: "Both districts" };
  return (
    <>
      <table>
        <thead><tr><th>Scale</th><th className="r">Bore: skill</th><th className="r">corr.</th><th className="r">Dug: skill</th><th className="r">corr.</th></tr></thead>
        <tbody>
          {levels.map((lv) => {
            const d = rows.find((r) => r.level === lv && r.aquifer === "deep");
            const s = rows.find((r) => r.level === lv && r.aquifer === "shallow");
            return <tr key={lv}><td>{name[lv]}</td><td className="r">{pctS(d?.skill)}{d?.skill_90 ? <span className="muted small"> ({pctS(d.skill_90[0])} to {pctS(d.skill_90[1])})</span> : null}</td><td className="r">{fmt(d?.corr, 2)}</td>
              <td className="r">{lv === "taluk" ? <span className="muted">not shown</span> : pctS(s?.skill)}</td><td className="r">{lv === "taluk" ? "" : fmt(s?.corr, 2)}</td></tr>;
          })}
        </tbody>
      </table>
      <p className="small muted">One round ahead with observed rain, test years 2023+ (10 rounds for bore wells, 5 for dug wells), area value = median of its wells.
        Skill = 1 − error / error of "no change"; 90% range from resampling rounds. Bore wells gain skill with area size; dug wells do not show reliable skill at any scale.</p>
      <h4 className="small" style={{ margin: "12px 0 4px" }}>Forecasts made like the live one (rain unknown, 7 rounds after the last reading), 2017–2025</h4>
      <table>
        <thead><tr><th>Rounds ahead</th>{["taluk", "district", "well"].map((l) => <th className="r" key={l}>{name[l].split(" ")[0]} skill / 80% covered / width</th>)}</tr></thead>
        <tbody>
          {[1, 2, 3, 4].map((h) => (
            <tr key={h}><td>{h}</td>
              {["taluk", "district", "well"].map((l) => {
                const r = roll.find((x) => x.aquifer === "deep" && x.level === l && x.h === h);
                return <td className="r" key={l}>{pctS(r?.skill)} / {fmt((r?.coverage80 ?? NaN) * 100, 0)}% / {fmt(r?.width_m, 1)} m</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="small muted">Bore wells. Each year's model was refitted on earlier data only; each year's 80% band was calibrated on earlier years only.
        Area bands cover roughly as promised but are wide, and four rounds ahead the area forecast does no better than no change: next year's rain decides most of it.</p>
    </>
  );
}

export default function ModelPanel({ meta, evidence }) {
  const ev = meta?.evaluation || {};
  const kc = meta?.kcvalley || {};
  const phys = meta?.physics_deploy || ev.physics || [];
  const cov = (ev.coverage_1step || []).filter((r) => r.model?.startsWith("Hybrid"));
  const twinKC = kc.twin_counterfactual || {};
  return (
    <div>
      <h2>Does the model hold up?</h2>
      <p className="muted">All scores are on years the models never saw: trained on readings up to 2021, intervals calibrated on 2022, tested on 2023 onward.
        "No-change" (next reading = last reading) is the bar every model has to clear.</p>

      <h3>One round ahead (about 3 months)</h3>
      <Comparison ev={ev} />
      <p className="small muted">MAE = average error in metres. Deep bore-well levels jump around by several metres with pumping, so their errors are large for every model.
        The graph network (ST-GNN) is included as the spec asked; with roughly 9,000 readings it does not beat simpler models.</p>

      <h3>At the scale it is reported: taluk, district, region</h3>
      <AreaSkill ae={evidence?.area_eval} />

      <h3>Several rounds ahead (single wells)</h3>
      <Horizon ev={ev} />

      <h3>Are the uncertainty bands honest?</h3>
      <table>
        <thead><tr><th>Wells</th><th className="r">Promised</th><th className="r">Achieved on test years</th><th className="r">Average width</th></tr></thead>
        <tbody>
          {cov.map((r) => (
            <tr key={r.aquifer + r.nominal}><td>{AQ[r.aquifer]}</td><td className="r">{fmt(r.nominal * 100, 0)}%</td><td className="r">{fmt(r.empirical * 100, 0)}%</td><td className="r">{fmt(r.mean_width_m, 1)} m</td></tr>
          ))}
        </tbody>
      </table>

      <h3>Which information matters</h3>
      <table>
        <thead><tr><th>Variant</th><th className="r">Bore wells</th><th className="r">Dug wells</th></tr></thead>
        <tbody>
          {[...new Set((ev.ablation || []).map((r) => r.variant))].map((v) => (
            <tr key={v}><td>{v}</td>
              {["deep", "shallow"].map((a) => <td className="r" key={a}>{fmt(ev.ablation.find((r) => r.variant === v && r.aquifer === a)?.MAE, 3)} m</td>)}
            </tr>
          ))}
        </tbody>
      </table>

      {ev.sparse?.length > 0 && (
        <>
          <h3>Fewer readings</h3>
          <table>
            <thead><tr><th>Training readings hidden</th><th className="r">Hybrid with graph</th><th className="r">Hybrid, no graph</th><th className="r">XGBoost</th></tr></thead>
            <tbody>
              {[...new Set(ev.sparse.map((r) => r.hidden_frac))].map((f) => {
                const m = (name) => {
                  const xs = ev.sparse.filter((r) => r.hidden_frac === f && r.model === name && r.aquifer === "deep");
                  return xs.reduce((s, r) => s + r.MAE, 0) / Math.max(1, xs.length);
                };
                return <tr key={f}><td>{fmt(f * 100, 0)}%</td><td className="r">{fmt(m("Hybrid (with graph)"), 2)}</td><td className="r">{fmt(m("Hybrid (no graph)"), 2)}</td><td className="r">{fmt(m("XGBoost (no physics)"), 2)}</td></tr>;
              })}
            </tbody>
          </table>
          <p className="small muted">Bore-well MAE in metres, averaged over repeats. Accuracy barely moves when up to 30% of readings are removed.</p>
        </>
      )}

      <h3>Fitted water balance</h3>
      <table>
        <thead><tr><th></th><th className="r">Bore wells</th><th className="r">Dug wells</th></tr></thead>
        <tbody>
          {[["m_per_100mm_rain", "Rise per 100 mm of rain in the season (m)"], ["m_per_100mm_et0", "Fall per 100 mm of evaporative demand (m)"],
            ["relaxation_per_round", "Share of an anomaly that fades each round"], ["equilibrium_rise_m_per_100ha", "Long-run rise per 100 ha of tank water within 3 km (m)"]].map(([k, l]) => (
            <tr key={k}><td>{l}</td>{["deep", "shallow"].map((a) => <td className="r" key={a}>{fmt(phys.find((p) => p.aquifer === a)?.[k], 3)}</td>)}</tr>
          ))}
        </tbody>
      </table>

      <h3>Checking the what-if engine against K.C. Valley</h3>
      <p className="small">Since 2018, treated Bengaluru wastewater has been pumped into tanks in Kolar and Chikkaballapura. Sentinel-2 shows which tanks now hold water in the dry season,
        so wells near them are compared with wells far from them, before and after, with fixed effects for each well, season and year, and a rainfall control.</p>
      {(kc.event_study || []).map((es) => <EventStudy key={es.aquifer} es={es} />)}
      <p className="small">
        The twin, asked "what if the tanks had stayed at their pre-project level?", says the project left bore wells near filled tanks{" "}
        <strong>{eff(twinKC["2024"]?.deep?.mean_effect_m)}</strong> by 2024, and dug wells {eff(twinKC["2024"]?.shallow?.mean_effect_m)}.
        Both fall inside the confidence intervals above, so the monitoring data cannot reject the twin, but it also cannot confirm it:
        only a handful of long-record wells sit near the filled tanks.
      </p>
    </div>
  );
}
