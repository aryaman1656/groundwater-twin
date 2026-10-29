import React, { useEffect, useState } from "react";
import MapView from "./components/MapView.jsx";
import AreaPanel from "./components/AreaPanel.jsx";
import WellPanel from "./components/WellPanel.jsx";
import ScenarioPanel from "./components/ScenarioPanel.jsx";
import AgroPanel from "./components/AgroPanel.jsx";
import ModelPanel from "./components/ModelPanel.jsx";
import EvidencePanel from "./components/EvidencePanel.jsx";
import DataPanel from "./components/DataPanel.jsx";
import { detectMode, getMeta, getWells, getTaluks, getAnomalies, getAreas, getEvidence } from "./api.js";
import { dateLabel } from "./format.js";

const TABS = [
  ["area", "Areas"],
  ["well", "Well"],
  ["scenario", "What if"],
  ["agro", "Crops and income"],
  ["model", "Model check"],
  ["evidence", "Evidence"],
  ["data", "About the data"],
];

export default function App() {
  const [mode, setMode] = useState(null);
  const [meta, setMeta] = useState(null);
  const [wells, setWells] = useState([]);
  const [taluks, setTaluks] = useState(null);
  const [areas, setAreas] = useState(null);
  const [evidence, setEvidence] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const [selected, setSelected] = useState(null);
  const [taluk, setTaluk] = useState(null);
  const [tab, setTab] = useState("area");
  const [layer, setLayer] = useState("change");
  const [aquifers, setAquifers] = useState({ deep: true, shallow: true });
  const [scenario, setScenario] = useState(null);
  const [scenArea, setScenArea] = useState(null);
  const [picking, setPicking] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        setMode(await detectMode());
        const [m, w, t, a] = await Promise.all([getMeta(), getWells(), getTaluks(), getAnomalies().catch(() => [])]);
        setMeta(m); setWells(w); setTaluks(t); setAnomalies(a);
        getAreas().then(setAreas).catch(() => {});
        getEvidence().then(setEvidence).catch(() => {});
      } catch (e) {
        setError(String(e));
      }
    })();
  }, []);

  const select = (id) => { setSelected(id); if (tab !== "scenario") setTab("well"); };
  const pickTaluk = (t) => { setTaluk(t); setTab("area"); };
  const onPick = (p) => {
    setScenArea({ level: "radius", radius_km: scenArea?.radius_km ?? 10, ...p });
    setPicking(false);
  };

  if (error)
    return (
      <div className="empty" style={{ padding: 40 }}>
        <strong>The dashboard couldn't load its data</strong>
        {error}. Start the backend with <code>python -m gwtwin serve</code>, or build the static export with <code>python -m gwtwin deploy</code>.
      </div>
    );

  const inStudy = wells.filter((w) => w.district).length;
  const reg = areas?.rows?.find((r) => r.level === "region" && r.aquifer === "deep");
  return (
    <div className="app">
      <header className="topbar">
        <h1 className="title">Kolar–Chikkaballapura groundwater twin</h1>
        <div className="status">
          <span>Now-cast for <b>{dateLabel(meta?.now_round)}</b></span>
          {reg && <span className="stale-chip">Newest well reading <b>{dateLabel(reg.latest_reading)}</b> · median <b>{reg.rounds_since_reading.median}</b> rounds old</span>}
          <span><b className="num">{inStudy}</b> wells in the districts, <b className="num">{wells.length}</b> in the model</span>
          <span>Rain data to <b>{dateLabel(meta?.forcing_observed_until)}</b></span>
          {mode && <span className={`badge ${mode}`}>{mode === "live" ? "Live model" : "Precomputed (no backend)"}</span>}
        </div>
      </header>
      <div className="main">
        <MapView wells={wells} taluks={taluks} areas={areas} layer={layer} setLayer={setLayer} aquifers={aquifers} setAquifers={setAquifers}
                 selected={selected} onSelect={select} selectedTaluk={taluk} onTaluk={pickTaluk} scenario={scenario} anomalies={anomalies}
                 scenarioArea={scenario?.inputs?.area ?? null} previewArea={tab === "scenario" ? scenArea : null}
                 picking={picking} onPick={onPick} />
        <section className="panel" aria-label="Details">
          <div className="tabs" role="tablist">
            {TABS.map(([k, label]) => (
              <button key={k} role="tab" className="tab" aria-selected={tab === k} onClick={() => setTab(k)}>{label}</button>
            ))}
          </div>
          <div className="panelbody">
            {tab === "area" && <AreaPanel data={areas} wells={wells} taluk={taluk} setTaluk={setTaluk} onSelectWell={(id) => { setSelected(id); setTab("well"); }} evidence={evidence} />}
            {tab === "well" && <WellPanel wellId={selected} wells={wells} meta={meta} onTaluk={pickTaluk} />}
            {tab === "scenario" && <ScenarioPanel result={scenario} setResult={setScenario} onShowMap={() => setLayer("scenario")} wells={wells}
                                                  areaIndex={areas?.areas} area={scenArea} setArea={setScenArea} picking={picking} setPicking={setPicking}
                                                  selectedWell={selected} />}
            {tab === "agro" && <AgroPanel result={scenario} goScenario={() => setTab("scenario")} />}
            {tab === "model" && <ModelPanel meta={meta} evidence={evidence} />}
            {tab === "evidence" && <EvidencePanel evidence={evidence} />}
            {tab === "data" && <DataPanel meta={meta} />}
          </div>
        </section>
      </div>
    </div>
  );
}
