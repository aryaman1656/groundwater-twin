import React, { useMemo } from "react";
import { MapContainer, TileLayer, CircleMarker, Circle, GeoJSON, Tooltip, LayersControl, useMapEvents } from "react-leaflet";
import { changeColor, depthColor, uncertaintyColor, fmt, signed, dateLabel, COLORS } from "../format.js";

const LAYERS = {
  change: { label: "Forecast change over the next year (same season)", ramp: ["#1f6f8b", "#c9cfcc", "#9a6a1f"], ends: ["rise 3 m", "no change", "fall 3 m"] },
  depth: { label: "Water level now (depth below ground)", ramp: ["#cfe3ea", "#14303d"], ends: ["shallow", "60 m deep"] },
  age: { label: "Rounds since the last real reading", ramp: ["#e7ebe9", "#a3263a"], ends: ["0", "12+ rounds"] },
  uncertainty: { label: "Forecast uncertainty, single wells (80% band width)", ramp: ["#e7ebe9", "#4b5a64"], ends: ["narrow", "40 m wide"] },
  scenario: { label: "Scenario minus baseline", ramp: ["#1f6f8b", "#c9cfcc", "#9a6a1f"], ends: ["rise 3 m", "same", "fall 3 m"] },
};

function ageColor(v) {
  if (v === null || v === undefined) return "#b9c0bd";
  const t = Math.max(0, Math.min(1, v / 12));
  const a = [0xe7, 0xeb, 0xe9], b = [0xa3, 0x26, 0x3a];
  return `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * t)).join(",")})`;
}

function Picker({ active, onPick }) {
  useMapEvents({ click: (e) => { if (active) onPick({ lat: e.latlng.lat, lon: e.latlng.lng }); } });
  return null;
}

export default function MapView({ wells, taluks, areas, layer, setLayer, aquifers, setAquifers, selected, onSelect, selectedTaluk, onTaluk,
                                  scenario, anomalies, scenarioArea, previewArea, picking, onPick }) {
  const scen = useMemo(() => {
    const m = new Map();
    (scenario?.wells || []).forEach((w) => m.set(w.well_id, w));
    return m;
  }, [scenario]);
  const talukScen = useMemo(() => {
    const m = new Map();
    (scenario?.levels || []).filter((l) => l.level === "taluk" && l.aquifer === "deep").forEach((l) => m.set(l.name, l.delta_m.median));
    if (!m.size) (scenario?.taluks || []).forEach((t) => m.set(t.taluk, t.dh_mean));
    return m;
  }, [scenario]);
  const talukBase = useMemo(() => {
    const m = new Map();
    (areas?.rows || []).filter((r) => r.level === "taluk" && r.aquifer === "deep").forEach((r) => m.set(r.name, r));
    return m;
  }, [areas]);
  const anomalous = useMemo(() => new Set((anomalies || []).map((a) => a.well_id)), [anomalies]);

  const value = (w) => {
    if (layer === "change") return w.forecast_change_m;
    if (layer === "depth") return w.nowcast_depth;
    if (layer === "uncertainty") return w.uncertainty_80_m;
    if (layer === "age") return w.rounds_since_obs;
    return scen.get(w.well_id)?.delta_m;
  };
  const color = (w) => {
    const v = value(w);
    if (layer === "depth") return depthColor(v);
    if (layer === "uncertainty") return uncertaintyColor(v);
    if (layer === "age") return ageColor(v);
    return changeColor(v);
  };
  const L = LAYERS[layer];
  const shown = wells.filter((w) => aquifers[w.aquifer]);
  const inScenArea = (p) => {
    const a = scenarioArea;
    if (!a || layer !== "scenario") return false;
    if (a.level === "region") return true;
    if (a.level === "district") return p.district === a.name;
    if (a.level === "taluk") return p.taluk === a.name;
    return false;
  };
  const talukKey = `${layer}-${selectedTaluk}-${picking ? "p" : ""}-${scenario ? JSON.stringify(scenario.inputs) : "none"}-${areas ? 1 : 0}`;

  const talukStyle = (f) => {
    const p = f.properties;
    const base = talukBase.get(p.taluk);
    const sel = selectedTaluk === p.taluk && layer !== "scenario";
    let fill, op = 0.45;
    if (layer === "change") fill = base ? changeColor(base.change_m.median) : "#ffffff";
    else if (layer === "age") fill = base ? ageColor(base.rounds_since_reading.median) : "#ffffff";
    else if (layer === "scenario") fill = talukScen.has(p.taluk) ? changeColor(talukScen.get(p.taluk)) : "#ffffff";
    else { fill = "#ffffff"; op = 0.05; }
    if (base?.low_data && (layer === "change" || layer === "age")) op = 0.22;
    return {
      color: sel || inScenArea(p) ? COLORS.ink : COLORS.ink2,
      weight: sel ? 3 : inScenArea(p) ? 2.5 : 1,
      opacity: 0.8,
      dashArray: base?.low_data ? "4 3" : null,
      fillColor: fill, fillOpacity: op,
    };
  };

  return (
    <div className={`mapwrap ${picking ? "picking" : ""}`}>
      <MapContainer center={[13.33, 78.0]} zoom={9} preferCanvas scrollWheelZoom>
        <Picker active={picking} onPick={onPick} />
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
            style={talukStyle}
            onEachFeature={(f, l) => {
              const p = f.properties;
              const b = talukBase.get(p.taluk);
              const s = talukScen.get(p.taluk);
              const txt = layer === "scenario" && s !== undefined
                ? `<b>${p.taluk}</b>: bore wells ${signed(s, 2)} m under the scenario`
                : b ? `<b>${p.taluk}</b> taluk${b.low_data ? " (low data)" : ""}<br/>Bore wells, median change ${signed(b.change_m.median, 2)} m (${b.n_wells} wells)<br/>Last real reading ${fmt(b.rounds_since_reading.median, 0)} rounds ago`
                : `<b>${p.taluk}</b> taluk`;
              l.bindTooltip(txt, { sticky: true });
              l.on("click", () => { if (!picking) onTaluk(p.taluk); });
            }}
          />
        )}
        {layer === "scenario" && scenarioArea?.level === "radius" && (
          <>
            <Circle center={[scenarioArea.lat, scenarioArea.lon]} radius={scenarioArea.radius_km * 1000} pathOptions={{ color: COLORS.ink, weight: 2, fill: false }} />
            <Circle center={[scenarioArea.lat, scenarioArea.lon]} radius={(scenarioArea.radius_km + 3) * 1000} pathOptions={{ color: COLORS.ink2, weight: 1, dashArray: "5 4", fill: false }} />
          </>
        )}
        {previewArea?.level === "radius" && (
          <Circle center={[previewArea.lat, previewArea.lon]} radius={previewArea.radius_km * 1000} pathOptions={{ color: COLORS.alert, weight: 1.5, dashArray: "3 3", fill: false }} />
        )}
        {shown.map((w) => {
          const isSel = selected === w.well_id;
          const v = value(w);
          const inactive = !w.active && layer !== "scenario";
          const z = scen.get(w.well_id)?.zone;
          const buffer = layer === "scenario" && z === "buffer";
          return (
            <CircleMarker
              key={w.well_id}
              center={[w.lat, w.lon]}
              radius={isSel ? 9 : inactive ? 3 : w.aquifer === "shallow" ? 5.5 : 4.5}
              pathOptions={{
                color: isSel ? COLORS.ink : anomalous.has(w.well_id) ? COLORS.alert : buffer ? COLORS.ink : inactive ? COLORS.stone : "#ffffff",
                weight: isSel ? 3 : anomalous.has(w.well_id) || buffer ? 2 : 0.8,
                dashArray: buffer ? "2 2" : null,
                fillColor: inactive ? "#ffffff" : color(w),
                fillOpacity: inactive ? 0.6 : w.district ? 0.95 : 0.55,
              }}
              eventHandlers={{ click: () => { if (!picking) onSelect(w.well_id); } }}
            >
              <Tooltip direction="top" offset={[0, -6]}>
                <strong>{w.name}</strong> ({w.aquifer === "deep" ? "bore well" : "dug well"}{w.district ? `, ${w.taluk}` : ", outside the districts"})<br />
                {layer === "depth" && `${fmt(v)} m below ground (twin estimate)`}
                {(layer === "change" || layer === "scenario") && `${signed(v, 2)} m ${v > 0 ? "deeper" : v < 0 ? "shallower" : ""}`}
                {layer === "uncertainty" && `80% band ${fmt(v)} m wide`}
                {layer === "scenario" && z && z !== "inside" && <><br />{z === "buffer" ? "within 3 km of the area: tank water only" : "outside the area: no modelled effect"}</>}
                <br /><span className={w.rounds_since_obs > 4 ? "stale-num" : ""}>{w.rounds_since_obs} rounds since the last real reading</span> ({dateLabel(w.last_obs_date)})
              </Tooltip>
            </CircleMarker>
          );
        })}
      </MapContainer>
      {picking && <div className="pickhint">Click the map to place the centre of the circle</div>}
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
          {(layer === "change" || layer === "age") && <>Taluks are shaded by their bore wells' median; dashed outline = low data. Click a taluk for its panel. </>}
          {layer === "scenario" && <>Dashed ring = well within 3 km outside the area (tank water only). </>}
          Faded points are outside the two districts (not in any total); small hollow points have no reading since 2021.{" "}
          <span className="alert">Red ring</span>: a reading since 2024 far from what the model expected.
        </div>
      </div>
    </div>
  );
}
