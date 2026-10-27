import React, { useMemo } from "react";
import { ComposedChart, Area, Line, Scatter, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, ReferenceLine } from "recharts";
import { COLORS, fmt, dateLabel } from "../format.js";

const ts = (s) => new Date(s).getTime();

export default function Hydrograph({ detail, since = "2013-01-01" }) {
  const { data, nowT, ticks } = useMemo(() => {
    const m = new Map();
    const put = (d, k, v) => {
      const t = ts(d);
      if (t < ts(since)) return;
      m.set(t, { ...(m.get(t) || { t }), [k]: v });
    };
    detail.observed.forEach((o) => put(o.date, "obs", o.depth));
    detail.nowcast.forEach((n) => put(n.date, "sim", n.depth));
    const last = detail.nowcast[detail.nowcast.length - 1];
    if (last) {
      put(last.date, "med", last.depth);
      put(last.date, "b80", [last.depth, last.depth]);
      put(last.date, "b90", [last.depth, last.depth]);
    }
    detail.forecast.forEach((f) => {
      put(f.date, "med", f.median);
      put(f.date, "b80", [f.lo80, f.hi80]);
      put(f.date, "b90", [f.lo90, f.hi90]);
    });
    const data = [...m.values()].sort((a, b) => a.t - b.t);
    const y0 = new Date(data[0]?.t ?? Date.now()).getFullYear() + 1;
    const y1 = new Date(data.at(-1)?.t ?? Date.now()).getFullYear();
    const step = y1 - y0 > 8 ? 2 : 1;
    const ticks = [];
    for (let y = y0; y <= y1; y += step) ticks.push(new Date(`${y}-01-01`).getTime());
    return { data, nowT: last ? ts(last.date) : null, ticks };
  }, [detail, since]);

  return (
    <div className="chartbox">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="#e3e8e5" vertical={false} />
          <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} ticks={ticks}
                 tickFormatter={(t) => new Date(t).getFullYear()} tick={{ fontSize: 11, fill: COLORS.ink2 }} />
          <YAxis reversed domain={[(lo) => Math.min(0, Math.floor(lo)), "auto"]} allowDataOverflow={false} tick={{ fontSize: 11, fill: COLORS.ink2 }} width={38}
                 label={{ value: "m below ground", angle: -90, position: "insideLeft", fontSize: 11, fill: COLORS.ink2, dy: 40 }} />
          <Tooltip
            labelFormatter={(t) => dateLabel(t)}
            formatter={(v, name) => {
              const names = { obs: "Measured", sim: "Twin (between readings)", med: "Forecast", b80: "80% band", b90: "90% band" };
              return [Array.isArray(v) ? `${fmt(v[0])} – ${fmt(v[1])} m` : `${fmt(v)} m`, names[name] || name];
            }}
          />
          {nowT && <ReferenceLine x={nowT} stroke={COLORS.stone} strokeDasharray="2 3" label={{ value: "now", fontSize: 10, fill: COLORS.ink2, position: "top" }} />}
          <Area dataKey="b90" stroke="none" fill={COLORS.umber2} fillOpacity={0.35} isAnimationActive={false} connectNulls />
          <Area dataKey="b80" stroke="none" fill={COLORS.umber2} fillOpacity={0.7} isAnimationActive={false} connectNulls />
          <Line dataKey="sim" stroke={COLORS.stone} strokeWidth={1.4} strokeDasharray="4 3" dot={false} isAnimationActive={false} connectNulls />
          <Line dataKey="med" stroke={COLORS.umber} strokeWidth={2} dot={{ r: 2.5 }} isAnimationActive={false} connectNulls />
          <Line dataKey="obs" stroke={COLORS.water} strokeWidth={1.2} dot={{ r: 2.6, fill: COLORS.water }} isAnimationActive={false} connectNulls />
          <Scatter dataKey="obs" fill={COLORS.water} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
