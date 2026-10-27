import React, { useMemo } from "react";
import { MapContainer, TileLayer, CircleMarker, GeoJSON, Tooltip, LayersControl } from "react-leaflet";
import { changeColor, depthColor, uncertaintyColor, fmt, signed, COLORS } from "../format.js";

const LAYERS = {
  change: { label: "Forecast change over the next year (same season)", unit: "m", ramp: ["#1f6f8b", "#c9cfcc", "#9a6a1f"], ends: ["rise 3 m", "no change", "fall 3 m"] },
  depth: { label: "Water level now (depth below ground)", unit: "m", ramp: ["#cfe3ea", "#14303d"], ends: ["shallow", "60 m deep"] },
  uncertainty: { label: "Forecast uncertainty (80% band width)", unit: "m", ramp: ["#e7ebe9", "#4b5a64"], ends: ["narrow", "40 m wide"] },
  scenario: { label: "Scenario minus baseline", unit: "m", ramp: ["#1f6f8b", "#c9cfcc", "#9a6a1f"], ends: ["rise 3 m", "same", "fall 3 m"] },
};

export default function MapView({ wells, taluks, layer, setLayer, aquifers, setAquifers, selected, onSelect, scenario, anomalies }) {
  const scenDelta = useMemo(() => {
    const m = new Map();
    (scenario?.wells || []).forEach((w) => m.set(w.well_id, w.delta_m));
    return m;
  }, [scenario]);
  const talukDelta = useMemo(() => {
    const m = new Map();
    (scenario?.taluks || []).forEach((t) => m.set(t.taluk, t.dh_mean));
    return m;
  }, [scenario]);
  const anomalous = useMemo(() => new Set((anomalies || []).map((a) => a.well_id)), [anomalies]);

  const value = (w) => {
    if (layer === "change") return w.forecast_change_m;
    if (layer === "depth") return w.nowcast_depth;
    if (layer === "uncertainty") return w.uncertainty_80_m;
    return scenDelta.get(w.well_id);
  };
  const color = (w) => {
    const v = value(w);
    if (layer === "depth") return depthColor(v);
    if (layer === "uncertainty") return uncertaintyColor(v);
    return changeColor(v);
  };
  const L = LAYERS[layer];
  const shown = wells.filter((w) => aquifers[w.aquifer]);
  const talukKey = `${layer}-${scenario ? JSON.stringify(scenario.inputs) : "none"}`;

  return (
    <div className="mapwrap">
      <MapContainer center={[13.33, 78.0]} zoom={9} preferCanvas scrollWheelZoom>
        <LayersControl position="topright">
          <LayersControl.BaseLayer checked name="Light grey">
            <TileLayer
              url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
              attribution="Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors"
              maxZoom={16}
            />
          </LayersControl.BaseLayer>
          <LayersControl.BaseLayer name="Satellite (see the tanks)">
            <TileLayer
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              attribution="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics"
              maxZoom={18}
            />
          </LayersControl.BaseLayer>
          <LayersControl.BaseLayer name="OpenStreetMap">
            <TileLayer url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
                       attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' />
          </LayersControl.BaseLayer>
        </LayersControl>
        {taluks && (
          <GeoJSON
            key={talukKey}
            data={taluks}
            style={(f) => {
              const d = layer === "scenario" ? talukDelta.get(f.properties.taluk) : undefined;
              return {
                color: COLORS.ink2, weight: 1, opacity: 0.6,
                fillColor: d === undefined ? "#ffffff" : changeColor(d),
                fillOpacity: d === undefined ? 0.05 : 0.35,
              };
            }}
            onEachFeature={(f, l) => {
              const d = talukDelta.get(f.properties.taluk);
              l.bindTooltip(
                `${f.properties.taluk} taluk${layer === "scenario" && d !== undefined ? `: deep wells ${signed(d, 2)} m` : ""}`,
                { sticky: true }
              );
            }}
          />
        )}
        {shown.map((w) => {
          const isSel = selected === w.well_id;
          const v = value(w);
          const inactive = !w.active && layer !== "scenario";
          return (
            <CircleMarker
              key={w.well_id}
              center={[w.lat, w.lon]}
              radius={isSel ? 9 : inactive ? 3 : w.aquifer === "shallow" ? 5.5 : 4.5}
              pathOptions={{
                color: isSel ? COLORS.ink : anomalous.has(w.well_id) ? COLORS.alert : inactive ? COLORS.stone : "#ffffff",
                weight: isSel ? 3 : anomalous.has(w.well_id) ? 2 : 0.8,
                fillColor: inactive ? "#ffffff" : color(w),
                fillOpacity: inactive ? 0.6 : w.district ? 0.95 : 0.55,
              }}
              eventHandlers={{ click: () => onSelect(w.well_id) }}
            >
              <Tooltip direction="top" offset={[0, -6]}>
                <strong>{w.name}</strong> ({w.aquifer === "deep" ? "bore well" : "dug well"}{w.active ? "" : ", no reading since 2021"})<br />
                {layer === "depth" && `${fmt(v)} m below ground`}
                {(layer === "change" || layer === "scenario") && `${signed(v, 2)} m ${v > 0 ? "deeper" : v < 0 ? "shallower" : ""}`}
                {layer === "uncertainty" && `80% band ${fmt(v)} m wide`}
              </Tooltip>
            </CircleMarker>
          );
        })}
      </MapContainer>
      <div className="legend">
        <select value={layer} onChange={(e) => setLayer(e.target.value)} aria-label="Map layer">
          {Object.entries(LAYERS).map(([k, l]) => (
            <option key={k} value={k} disabled={k === "scenario" && !scenario}>
              {l.label}
            </option>
          ))}
        </select>
        <div className="ramp" style={{ background: `linear-gradient(90deg, ${L.ramp.join(",")})` }} />
        <div className="ramp-labels">
          {L.ends.map((e) => (
            <span key={e}>{e}</span>
          ))}
        </div>
        <label>
          <input type="checkbox" checked={aquifers.deep} onChange={(e) => setAquifers({ ...aquifers, deep: e.target.checked })} />
          Bore wells and piezometers (deep, {wells.filter((w) => w.aquifer === "deep").length})
        </label>
        <label>
          <input type="checkbox" checked={aquifers.shallow} onChange={(e) => setAquifers({ ...aquifers, shallow: e.target.checked })} />
          Dug wells (shallow, {wells.filter((w) => w.aquifer === "shallow").length})
        </label>
        <div className="hint" style={{ marginTop: 8 }}>
          Faded points are outside the two districts; small hollow points have no reading since 2021.{" "}
          <span className="alert">Red ring</span>: a reading since 2024 far from what the model expected.
        </div>
      </div>
    </div>
  );
}
