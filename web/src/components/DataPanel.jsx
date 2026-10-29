import React from "react";
import { dateLabel, fmt } from "../format.js";

const SOURCES = [
  ["Groundwater levels, manual (CGWB and state boards), 1976–2025, with well type and depth", "India-WRIS via India Data Portal", "https://indiawris.gov.in/wris/#/groundWater"],
  ["Groundwater levels, telemetric and monthly (Karnataka Ground Water Directorate, CGWB)", "India-WRIS via India Data Portal", "https://indiawris.gov.in/wris/#/timeseriesdata"],
  ["Daily rainfall 0.25° and temperature 1°, 2005–2025; provisional real-time 2026", "India Meteorological Department (imdlib)", "https://imdlib.readthedocs.io"],
  ["Dry-season surface water around each well, 2017–2026", "Sentinel-2 L2A scene classification (AWS Earth Search)", "https://earth-search.aws.element84.com/v1"],
  ["Groundwater extraction and recharge by district, 2024", "CGWB Dynamic Ground Water Resources (via OpenCity)", "https://data.opencity.in/dataset/national-compilation-on-dynamic-ground-water-resources-of-india-2024"],
  ["Elevation (Copernicus GLO-30), land cover (ESA WorldCover 2021), soil texture (SoilGrids 2.0)", "Copernicus / ESA / ISRIC", "https://www.esa-worldcover.org"],
  ["Irrigated crop area by taluk (Agricultural Census 2015-16), crop yields (DES), mandi prices (Agmarknet)", "India Data Portal", "https://indiadataportal.com"],
  ["Taluk boundaries", "datta07/INDIAN-SHAPEFILES (GitHub)", "https://github.com/datta07/INDIAN-SHAPEFILES"],
];

export default function DataPanel({ meta }) {
  return (
    <div>
      <h2>About this twin</h2>
      <p>
        A model of {meta?.n_wells ?? "–"} monitored wells in and around Kolar and Chikkaballapura. Dug wells (shallow) and bore wells (deep)
        are kept apart because they tap different layers of rock. Farmers here pump from the deep layer.
      </p>
      <ul className="plain small">
        <li>Each round (Jan, May, Aug, Nov) every well's water level moves by a fitted water balance: rain recharges, evaporative demand and pumping draw down, nearby tank water raises the long-run level, and anomalies fade back. A small machine-learning correction handles what the balance misses; it is constrained so more water in can never mean deeper water.</li>
        <li>Whenever a real reading arrives the twin resets that well to it, so the map always starts from the latest evidence (readings since {dateLabel(meta?.assimilation_start)}).</li>
        <li>Forecast bands combine the model's error on held-out years (conformal intervals) with 21 years of possible weather.</li>
        <li>One model is fitted across both districts. Areas (both districts, each district, 11 taluks) only add up its well-by-well output; the 126 wells outside the districts feed the model but are in no total. Dug wells are summarised by district only.</li>
        <li>Pumping never varied in the data, so the pumping slider uses physics only: extra water removed divided by specific yield ({meta?.config?.physics?.sy_bounds?.deep?.join("–")} for bore wells, from CGWB norms).</li>
      </ul>

      <h3>Where the data comes from</h3>
      <table>
        <tbody>
          {SOURCES.map(([what, who, url]) => (
            <tr key={what}><td>{what}</td><td><a href={url} target="_blank" rel="noreferrer">{who}</a></td></tr>
          ))}
        </tbody>
      </table>

      <h3>Limits to keep in mind</h3>
      <ul className="plain small">
        <li>Scenario results are model-based "what-ifs", not proven cause and effect. The K.C. Valley check is consistent with the twin but too small a sample to confirm it.</li>
        <li>Deep bore-well levels are noisy (pumping nearby), so forecast bands for them are wide. That width is real uncertainty, not a display choice.</li>
        <li>Income estimates use district or state yields, 2015-16 crop areas and recent mandi prices. Use them to compare scenarios, not as audited losses.</li>
        <li>Specific yield is a range from CGWB norms until values from the taluk aquifer-mapping (NAQUIM) reports are entered in config/config.yaml.</li>
        <li>2026 rainfall is IMD's provisional real-time data.</li>
      </ul>

      <h3>Model version</h3>
      <table>
        <tbody>
          <tr><td>Trained on readings through</td><td className="r">{meta?.trained_through}</td></tr>
          <tr><td>Rainfall observed until</td><td className="r">{meta?.forcing_observed_until}</td></tr>
          <tr><td>Current round (now-cast)</td><td className="r">{dateLabel(meta?.now_round)}</td></tr>
          <tr><td>Wells (deep / shallow)</td><td className="r">{meta?.wells_by_aquifer?.deep} / {meta?.wells_by_aquifer?.shallow}</td></tr>
          <tr><td>Weight on ML correction, bore / dug wells</td><td className="r">{fmt(meta?.damping?.beta?.deep, 2)} / {fmt(meta?.damping?.beta?.shallow, 2)}</td></tr>
        </tbody>
      </table>
    </div>
  );
}
