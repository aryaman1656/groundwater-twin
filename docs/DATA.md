# Data sources and processing

Retrieved 27 Oct 2026 unless noted. Everything marked *script* is fetched by `python -m gwtwin download`.

## Groundwater (target)

| Dataset | Source / URL | Coverage | Notes |
|---|---|---|---|
| Groundwater level, manual (*script*) | India-WRIS via India Data Portal, resource `06d6e79c-…` · https://indiawris.gov.in/wris/#/groundWater | 1976–2025, 314 stations in the bbox (CGWB + Andhra Pradesh board) | Has `well_type`, `well_aquifer_type`, `well_depth`, which the original project CSV lacked. Depth below ground, positive. |
| Groundwater level, telemetric + monthly (*script*, ~15 min chunked scan) | India Data Portal resource `f78f7b15-…` (29 M rows) | Karnataka GW Directorate monthly 2010–2019, daily 2021–2025; CGWB and AP DWLRs | Stored as negative numbers (GPRS) or positive (older monthly); both mean metres below ground. Round value = median of days 1–21 of the round month. |
| CGWB "changes in depth" (*script*) | India Data Portal `580a8f6e-…` | 2013–2021 | Identical to the original `kolar_chikkaballapura_water_level.csv`; kept for traceability. |
| State taluk average static water level | Karnataka Ground Water Directorate via OpenCity | 2013–2022, annual | Not used in training; useful for sanity checks of the deep aquifer (Kolar taluk ~48 m → 86 m (2019) → 15 m (2022)). |

Cleaning (`gwtwin/data/wells.py`): readings deeper than the well + 10 m dropped; per-well 6×MAD outlier filter;
isolated one-round spikes removed (both neighbours differ by > max(15 m, 4×MAD) in opposite directions).
Aquifer class = shallow for dug wells, deep for bore wells/piezometers/tube wells.

## Weather

| Dataset | Source | Notes |
|---|---|---|
| Daily rainfall 0.25°, Tmax/Tmin 1° (*script*) | IMD Pune gridded data via `imdlib` | 2005–2025 archive. |
| 2026 real-time rain (0.25°) and temperature (0.5°) (*script*) | IMD real-time via `imdlib.get_real_data` | Provisional; 1 Jan – 26 Oct 2026. |
| ET0 | Computed: FAO-56 Hargreaves (eq. 52) from Tmax/Tmin and latitude | 2026 temperature gaps filled with day-of-year climatology. |
| KSNDMC Gram-Panchayat/hobli rainfall | OpenCity | Annual 2020–2024; included for checks, not used in models. |

## Static layers

| Layer | Source | Use |
|---|---|---|
| Copernicus GLO-30 DEM (*script*) | AWS open data | Elevation, slope, height above the 2-km valley floor |
| ESA WorldCover 2021 (*script*) | AWS open data | Cropland / built-up / tree / water share within 2 km |
| SoilGrids 2.0 clay, sand, silt, bulk density 0–30 cm (*script*) | ISRIC | Soil texture at each well |
| Taluk and district boundaries (*script*) | datta07/INDIAN-SHAPEFILES (GitHub) | Membership, maps |
| Dry-season surface water 2017–2026 (*script*) | Sentinel-2 L2A SCL via AWS Earth Search STAC | Water area within 1 and 3 km of each well, Feb–Apr each year (the tanks K.C./H.N. Valley keeps full) |

## Pumping and recharge

| Dataset | Source | Use |
|---|---|---|
| Dynamic Ground Water Resources 2024, Karnataka by district (*script*) | CGWB via OpenCity | Annual extraction (Kolar 71,709 ham; Chikkaballapura 50,222 ham), irrigation share. Pumping scenarios and the agro chain. |
| Specific yield bounds | CGWB GEC-2015 norms for weathered/fractured granite–gneiss | `config.yaml` → `physics.sy_bounds`. Replace with NAQUIM taluk values (manual). |

## Agriculture and prices

| Dataset | Source | Use |
|---|---|---|
| Agricultural Census 2015-16, tehsil × crop irrigated area (*script*) | India Data Portal `57611aec-…` | Irrigated area of tomato, potato, maize, paddy per taluk |
| Crop APY (DES) (*script*) | India Data Portal `f980409d-…` | Potato, maize, rice yields (district, last 10 years) |
| Horticulture area and production, Karnataka (*script*) | India Data Portal `1015a755-…` | Tomato yield (state final estimates) |
| Land Use Statistics: sources of irrigation, crop-wise irrigated area (*script*) | India Data Portal | Kolar irrigation is 100% tubewell |
| APMC arrivals and prices (*script*) | Agmarknet via India Data Portal `078fd1ea-…` | Modal prices, Oct 2025 – May 2026, nine APMCs in the two districts |
| FAO-56 Kc and stage lengths, FAO-33 Ky | FAO Irrigation & Drainage Papers 56 and 33 | `config.yaml` → `agro.crops` |

## Need a browser (not scriptable from all networks)

1. India-WRIS portal exports for rounds after the India Data Portal snapshot.
2. CGWB NAQUIM taluk reports (Kolar, Malur, Bangarpet, Mulbagal): specific yield and transmissivity → narrow `sy_bounds`.
3. K.C./H.N. Valley tank list with fill dates (Minor Irrigation Dept.); the pipeline detects tanks from Sentinel-2 instead.
4. Atal Bhujal Yojana block reports and piezometers (ataljal.mowr.gov.in).
5. Karnataka drought-loss memoranda for 2023 and 2026 (for H6), Revenue Dept.

## Licences

India Data Portal mirrors Government of India open data (NDSAP / ODbL-style attribution); IMD data via
imdlib for research use; Copernicus DEM and Sentinel-2 under Copernicus free and open licence;
ESA WorldCover CC-BY 4.0; SoilGrids CC-BY 4.0. Cite each source in the report.
