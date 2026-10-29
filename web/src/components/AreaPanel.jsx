import React, { useMemo, useState } from "react";
import { ComposedChart, Area, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, ReferenceLine } from "recharts";
import { COLORS, fmt, signed, dateLabel, changeColor } from "../format.js";

const pct = (v, d = 0) => (v === null || v === undefined ? "–" : `${signed(v * 100, d)}%`);
const AQ = { deep: "bore wells", shallow: "dug wells" };

function Swatch({ v }) {
  return <span className="swatch" style={{ background: changeColor(v) }} aria-hidden="true" />;
}

function DataAge({ row, wellsRead }) {
  const s = row.rounds_since_reading;
  const stale = (s?.min ?? 0) > 4;
  return (
    <div className={`agebar ${stale ? "stale" : ""}`}>
      <div className="agebig num">{fmt(s?.median, 0)}</div>
      <div>
        <strong>rounds since the last real reading</strong> (median of {row.n_wells} {AQ[row.aquifer]}; range {fmt(s?.min, 0)}–{fmt(s?.max, 0)}).
        {" "}Newest reading in this area: <strong>{dateLabel(row.latest_reading)}</strong>.
        {stale && <> No well here has been read in the last 4 rounds, so this forecast starts from the twin's now-cast, not from recent data.</>}
        {!stale && wellsRead !== undefined && <> {wellsRead} wells read in the last 4 rounds.</>}
      </div>
    </div>
  );
}

function Trajectory({ row, now }) {
  const data = [{ t: new Date(now).getTime(), median: 0, band: [0, 0] },
    ...row.trajectory.map((p) => ({ t: new Date(p.date).getTime(), median: p.median, band: [p.p25, p.p75] }))];
  return (
    <div style={{ height: 190 }}>
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 8, right: 10, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#e3e8e5" vertical={false} />
          <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} ticks={data.map((d) => d.t)}
                 tickFormatter={(t) => dateLabel(t)} tick={{ fontSize: 11, fill: COLORS.ink2 }} />
          <YAxis reversed tick={{ fontSize: 11, fill: COLORS.ink2 }} width={40}
                 label={{ value: "m change, down = deeper", angle: -90, position: "insideLeft", fontSize: 10, fill: COLORS.ink2, dy: 60 }} />
          <ReferenceLine y={0} stroke={COLORS.stone} />
          <Tooltip labelFormatter={(t) => dateLabel(t)}
                   formatter={(v, n) => [Array.isArray(v) ? `${signed(v[0], 2)} to ${signed(v[1], 2)} m` : `${signed(v, 2)} m`, n === "band" ? "middle half of wells (IQR)" : "median well"]} />
          <Area dataKey="band" stroke="none" fill={COLORS.umber2} fillOpacity={0.6} isAnimationActive={false} />
          <Line dataKey="median" stroke={COLORS.umber} strokeWidth={2} dot={{ r: 2.5 }} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function AccuracyBox({ acc, level, isTaluk }) {
  if (!acc) return <p className="small muted">Area accuracy not computed yet (run <code>python -m gwtwin areas</code>).</p>;
  const one = acc.one_round_all_areas;
  const mine = acc.one_round_this_taluk;
  const roll = acc.rolling_by_horizon || [];
  const h4 = roll.find((r) => r.h === 4);
  return (
    <>
      <p className="small">
        <strong>One round ahead, with the rain that actually fell</strong>: skill against "no change" across all {level}s{" "}
        <strong>{pct(one?.skill)}</strong>{one?.skill_90 && <> (90% range {pct(one.skill_90[0])} to {pct(one.skill_90[1])})</>},
        correlation {fmt(one?.corr, 2)}, over {one?.n_rounds} test rounds.
        {isTaluk && mine && <> This taluk alone: {pct(mine.skill)}, correlation {fmt(mine.corr, 2)} ({mine.n} rounds, about {fmt(mine.median_wells_per_round, 0)} wells per round).</>}
      </p>
      <p className="small">
        <strong>As this forecast is made</strong> (rain unknown, starting from a now-cast 7 rounds after the last reading), past {level} forecasts
        four rounds ahead had skill <strong>{pct(h4?.skill)}</strong> against no change. Beyond about two rounds the forecast is a typical-year
        projection; it should not be read as knowing next year's rain.
      </p>
      <table>
        <thead><tr><th>Rounds ahead</th><th className="r">Skill</th><th className="r">Corr.</th><th className="r">80% band width</th><th className="r">Covered</th><th className="r">Cases</th></tr></thead>
        <tbody>
          {roll.map((r) => (
            <tr key={r.h}><td>{r.h}</td><td className="r">{pct(r.skill)}</td><td className="r">{fmt(r.corr, 2)}</td>
              <td className="r">{fmt(r.width_m, 1)} m</td><td className="r">{fmt(r.coverage80 * 100, 0)}%</td><td className="r">{r.n}</td></tr>
          ))}
        </tbody>
      </table>
      <p className="small muted">
        Rolling check, 2017–2025: the model was refitted each year on earlier data only, and the band was calibrated on earlier years only.
        {h4 && !h4.band_ok ? " At four rounds there are too few past cases to call the band calibrated, so no interval is drawn: the shading on the chart is the spread of wells inside the area (IQR), not an interval." : " The shading on the chart is the spread of wells inside the area (IQR), not an interval."}
      </p>
    </>
  );
}

function RegionView({ data, onTaluk, evidence }) {
  const rows = data.rows;
  const reg = rows.find((r) => r.level === "region" && r.aquifer === "deep");
  const dists = rows.filter((r) => r.level === "district");
  const tal = rows.filter((r) => r.level === "taluk").sort((a, b) => b.change_m.median - a.change_m.median);
  return (
    <div>
      <h2>Kolar and Chikkaballapura, by area</h2>
      <p className="muted">
        Baseline forecast of the change in depth to water from {dateLabel(data.now_round)} to {dateLabel(data.target_round)} (same season a year
        later). One model is fitted across both districts; areas only add up its well-by-well output. Positive = water deeper.
      </p>
      <DataAge row={reg} />
      <div className="pair" style={{ marginTop: 14 }}>
        <div>
          <div className="big num" style={{ color: reg.change_m.median > 0 ? COLORS.umber : COLORS.water }}>{signed(reg.change_m.median, 2)} m</div>
          <div className="small">median bore well, both districts ({reg.n_wells} wells). Middle half of wells: {signed(reg.change_m.p25, 1)} to {signed(reg.change_m.p75, 1)} m.</div>
        </div>
        <div>
          {dists.filter((d) => d.aquifer === "deep").map((d) => (
            <div key={d.id} className="small" style={{ marginBottom: 6 }}>
              <strong>{d.name}</strong>: {signed(d.change_m.median, 2)} m <span className="muted">(IQR {signed(d.change_m.p25, 1)} to {signed(d.change_m.p75, 1)}, {d.n_wells} wells)</span>
            </div>
          ))}
        </div>
      </div>

      <h3>Taluks (bore wells)</h3>
      <p className="small muted">Click a taluk here or on the map. "Low data": few bore wells, so one unusual well moves the median.</p>
      <table className="clickable">
        <thead><tr><th>Taluk</th><th className="r">Median change</th><th className="r">Middle half of wells</th><th className="r">Wells</th><th className="r">Rounds since reading</th></tr></thead>
        <tbody>
          {tal.map((t) => (
            <tr key={t.id} onClick={() => onTaluk(t.name)} tabIndex={0} onKeyDown={(e) => e.key === "Enter" && onTaluk(t.name)}>
              <td><Swatch v={t.change_m.median} />{t.name} <span className="muted small">{t.district}</span>{t.low_data && <span className="tag">low data</span>}</td>
              <td className="r">{signed(t.change_m.median, 2)} m</td>
              <td className="r">{signed(t.change_m.p25, 1)} to {signed(t.change_m.p75, 1)}</td>
              <td className="r">{t.n_wells}</td>
              <td className="r">{fmt(t.rounds_since_reading.median, 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Dug wells (shallow), district level only — indicative</h3>
      <table>
        <thead><tr><th>Area</th><th className="r">Median change</th><th className="r">Middle half</th><th className="r">Wells</th><th className="r">Rounds since reading</th></tr></thead>
        <tbody>
          {rows.filter((r) => r.aquifer === "shallow").map((r) => (
            <tr key={r.id}><td>{r.name}</td><td className="r">{signed(r.change_m.median, 2)} m</td><td className="r">{signed(r.change_m.p25, 2)} to {signed(r.change_m.p75, 2)}</td>
              <td className="r">{r.n_wells}</td><td className="r">{fmt(r.rounds_since_reading.median, 0)}</td></tr>
          ))}
        </tbody>
      </table>
      <p className="small muted">Most taluks have 1–8 dug wells (5 of 11 had 0–1 with test readings), so dug wells are not shown by taluk. Even at district level the
        model shows no reliable skill for dug wells on held-out years.</p>

      <h3>How much to trust area numbers</h3>
      <AccuracyBox acc={tal[0]?.accuracy} level="taluk" />
      <p className="small muted">
        Headline accuracy is skill against no change and correlation, not raw error in metres: averaging many wells makes the error smaller by
        itself. Single bore wells are mostly noise (skill about +6%); areas are where the signal is.
      </p>
      <p className="small muted">{data.rules.buffer_wells_excluded} wells within 15 km of the districts feed the model but are left out of every total.</p>
    </div>
  );
}

function TalukView({ data, name, wells, onBack, onSelectWell }) {
  const row = data.rows.find((r) => r.level === "taluk" && r.name === name && r.aquifer === "deep");
  const dist = data.rows.find((r) => r.level === "district" && r.name === row?.district && r.aquifer === "deep");
  const [sort, setSort] = useState("change");
  const list = useMemo(() => {
    const xs = wells.filter((w) => w.taluk === name && w.in_totals !== false);
    const key = { change: (w) => -(w.forecast_change_m ?? -99), age: (w) => w.rounds_since_obs, name: (w) => w.name };
    return [...xs].sort((a, b) => (key[sort](a) < key[sort](b) ? -1 : key[sort](a) > key[sort](b) ? 1 : 0));
  }, [wells, name, sort]);
  if (!row) return <div className="empty"><strong>{name}</strong>No bore-well data for this taluk.</div>;
  const shallowN = wells.filter((w) => w.taluk === name && w.aquifer === "shallow").length;
  return (
    <div>
      <button className="link" onClick={onBack}>← All areas</button>
      <h2>{row.name} taluk {row.low_data && <span className="tag">low data</span>}</h2>
      <div className="muted small">{row.district} district · {row.n_wells} bore wells{shallowN ? ` · ${shallowN} dug wells (not summarised by taluk)` : ""}</div>
      {row.low_data && <div className="note">Only {row.n_wells} bore wells (and fewer with recent readings): treat this taluk's numbers as indicative.</div>}
      <DataAge row={row} wellsRead={row.wells_read_within_4_rounds} />
      <div className="facts">
        <div className="fact"><div className="label">Median change to {dateLabel(data.target_round)}</div><div className="value">{signed(row.change_m.median, 2)} m</div><div className="small muted">+ = deeper</div></div>
        <div className="fact"><div className="label">Middle half of wells</div><div className="value">{signed(row.change_m.p25, 1)} to {signed(row.change_m.p75, 1)}</div><div className="small muted">spread, not an interval</div></div>
        <div className="fact"><div className="label">District median</div><div className="value">{signed(dist?.change_m?.median, 2)} m</div><div className="small muted">{row.district}</div></div>
      </div>
      <h3>Forecast path (median bore well, change from now)</h3>
      <Trajectory row={row} now={data.now_round} />
      <h3>Accuracy at taluk level</h3>
      <AccuracyBox acc={row.accuracy} level="taluk" isTaluk />
      <h3>Wells in {row.name}</h3>
      <div className="small muted" style={{ marginBottom: 6 }}>
        Sort by <button className="link" onClick={() => setSort("change")}>change</button> · <button className="link" onClick={() => setSort("age")}>rounds since reading</button> · <button className="link" onClick={() => setSort("name")}>name</button>
      </div>
      <table className="clickable">
        <thead><tr><th>Well</th><th className="r">Change</th><th className="r">Since reading</th><th className="r">Last reading</th></tr></thead>
        <tbody>
          {list.map((w) => (
            <tr key={w.well_id} onClick={() => onSelectWell(w.well_id)} tabIndex={0} onKeyDown={(e) => e.key === "Enter" && onSelectWell(w.well_id)}>
              <td><Swatch v={w.forecast_change_m} />{w.name} <span className="muted small">{w.aquifer === "deep" ? "bore" : "dug"}</span></td>
              <td className="r">{signed(w.forecast_change_m, 2)} m</td>
              <td className="r"><span className={w.rounds_since_obs > 4 ? "stale-num" : ""}>{w.rounds_since_obs}</span> rounds</td>
              <td className="r">{dateLabel(w.last_obs_date)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function AreaPanel({ data, wells, taluk, setTaluk, onSelectWell, evidence }) {
  if (!data) return <div className="empty">Loading area summaries…</div>;
  if (taluk) return <TalukView data={data} name={taluk} wells={wells} onBack={() => setTaluk(null)} onSelectWell={onSelectWell} />;
  return <RegionView data={data} onTaluk={setTaluk} evidence={evidence} />;
}
