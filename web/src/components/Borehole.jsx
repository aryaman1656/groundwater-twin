import React from "react";
import { COLORS, fmt } from "../format.js";

// To-scale cross-section of the well: ground at the top, depth increases downward.
export default function Borehole({ wellDepth, now, lo, hi, histMin, histMax }) {
  const deepest = Math.max(wellDepth || 0, hi || 0, histMax || 0, now || 0, 5);
  const H = 250, top = 18, bottom = H - 10;
  const y = (d) => top + (Math.max(0, d) / (deepest * 1.05)) * (bottom - top);
  const cx = 48, w = 26;
  const wellBottom = wellDepth ? y(wellDepth) : bottom;
  return (
    <svg viewBox={`0 0 112 ${H}`} width="100%" height={H} role="img"
         aria-label={`Water level ${fmt(now)} m below ground; forecast range ${fmt(lo)} to ${fmt(hi)} m`}>
      {/* ground */}
      <line x1="4" x2="108" y1={top} y2={top} stroke={COLORS.ink} strokeWidth="1.5" />
      {[...Array(9)].map((_, i) => (
        <line key={i} x1={6 + i * 12} x2={12 + i * 12} y1={top} y2={top - 6} stroke={COLORS.stone} strokeWidth="1" />
      ))}
      <text x="108" y={top - 8} textAnchor="end" fontSize="9" fill={COLORS.ink2}>ground</text>
      {/* casing */}
      <rect x={cx - w / 2} y={top} width={w} height={wellBottom - top} fill="#f1f4f2" stroke={COLORS.stone} />
      {/* historic range */}
      {histMin !== undefined && histMax !== undefined && (
        <g>
          <line x1={cx + w / 2 + 10} x2={cx + w / 2 + 10} y1={y(histMin)} y2={y(histMax)} stroke={COLORS.stone} strokeWidth="3" />
          <text x={cx + w / 2 + 14} y={y(histMax) + 3} fontSize="9" fill={COLORS.ink2}>past</text>
          <text x={cx + w / 2 + 14} y={y(histMax) + 13} fontSize="9" fill={COLORS.ink2}>range</text>
        </g>
      )}
      {/* water column */}
      {now !== undefined && now !== null && (
        <rect x={cx - w / 2 + 1} y={y(now)} width={w - 2} height={Math.max(0, wellBottom - y(now) - 1)} fill={COLORS.water2} />
      )}
      {/* forecast band */}
      {lo !== undefined && hi !== undefined && (
        <rect x={cx - w / 2 - 6} y={y(lo)} width={w + 12} height={Math.max(2, y(hi) - y(lo))}
              fill="none" stroke={COLORS.umber} strokeDasharray="3 2" />
      )}
      {now !== undefined && now !== null && (
        <g>
          <line x1={cx - w / 2 - 10} x2={cx + w / 2} y1={y(now)} y2={y(now)} stroke={COLORS.water} strokeWidth="2.5" />
          <text x="2" y={y(now) - 4} fontSize="10" fontWeight="650" fill={COLORS.water}>{fmt(now)} m</text>
        </g>
      )}
      {wellDepth ? (
        <text x={cx} y={Math.min(H - 1, wellBottom + 9)} textAnchor="middle" fontSize="9" fill={COLORS.ink2}>
          {fmt(wellDepth, 0)} m deep
        </text>
      ) : null}
    </svg>
  );
}
