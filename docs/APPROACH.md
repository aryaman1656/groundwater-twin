# Approach: what changed from the project spec, and why

The spec's research question and five-part integration (spatial-temporal forecasting, sparse data,
counterfactual scenarios, calibrated uncertainty, hydro-economic translation) are kept. Several of
the methods it prescribed would not survive an evaluator who checks the data, so they were changed.
Each change below states the problem, the fix, and the evidence.

## 1. The core model: a water balance with an ML correction, not an ST-GNN

**Problem.** After cleaning there are ~8,800 labelled quarterly transitions across 438 wells. In the
training window only ~40–70 shallow wells have long records. A GAT+GRU network has tens of
thousands of parameters and no built-in physics.

**Fix.** The core model is a per-aquifer **water-table-fluctuation balance** fitted by sign-constrained,
Huber-robust least squares:

    Δh = θ_season + b_rain·P + b_prev·P_prev + b_12m·P_12m + b_et0·ET0 + b_tank·W + b_rev·(h − h_baseline)

with b_rain, b_prev, b_12m, b_tank, b_rev ≤ 0 and b_et0 ≥ 0. Here h_baseline is the well's mean level
before mid-2018, so a lasting rise from filled tanks is not silently absorbed. On top sits an XGBoost
correction with **monotone constraints** (more rain or tank water can never predict deeper water).
Weights α (overall damping toward no-change) and β (trust in the ML part) are chosen per aquifer
on the last 20% of the training window. The ST-GNN (GAT+GRU) is still built and evaluated, as the
spec asked.

**Evidence (test years 2023+).** The physics balance has the lowest bore-well error (4.95 m vs 5.26 m
for no-change). The ST-GNN is worst of the learned models (5.63 m). The deployed model's learned
weights switch the ML correction off (β = 0) for both aquifers, so the data itself says the physics
carries the signal.

## 2. Scenarios defined physically, not by perturbing inputs

**Problem.** The spec simulates "extraction × 1.2" by perturbing a model input. There is no pumping
time series in any dataset, so a model trained on these data has never seen pumping vary and its
response to a pumping slider is undefined. Tree models simply return a flat line.

**Fix.** Each slider enters through a defined physical term:

| Slider | How it enters | Identified from |
|---|---|---|
| Rainfall | P in the fitted balance (+ monotone ML) | 20 years of IMD rain × water levels |
| Tank water | W (dry-season water within 3 km, Sentinel-2) in the fitted balance | the K.C./H.N. Valley build-out after 2018 |
| Pumping | Δh = ΔD / Sy, with D = district draft from CGWB GWRA-2024 and Sy from GEC norms | physics only, stated as such |

Direction is guaranteed by construction (H4). Magnitude is tested against K.C. Valley (§6).

## 3. Volume-based stress instead of the depth-ratio index

**Problem.** The spec's GWS = (GWL_base − GWL_pred)/GWL_base divides by depth. A 0.5 m drop in a
well 1 m deep scores 50% stress; a 2 m drop in a well 20 m deep scores 10%. That ranks the
smaller physical change as the bigger stress.

**Fix.** Storage deficit ΔS = Sy·Δh (metres of water) is compared with the irrigation water pumped
over the same months: supply cut f = θ·ΔS/D_season. Then FAO-56 crop water need, FAO-33 Ky,
yields, areas and prices give rupees, all inside a Monte Carlo over Sy, θ, yields, prices and the
response size.

## 4. Two aquifers, and only groundwater-linked crops

- Dug wells (median 2–7 m) and bore wells (tens to hundreds of metres) sit in different layers of
  hard rock and behave differently. They get separate models, graphs and conformal intervals.
- Farmers here pump from bore wells, and Kolar's irrigation is 100% tubewell (Land Use Statistics
  2023-24). The economic layer therefore uses **bore-well** responses.
- Ragi is ~96% rainfed (Agricultural Census 2015-16: 5.6k ha irrigated vs 151.6k ha unirrigated),
  so it is **not** linked to groundwater. Tomato (13.6k ha irrigated), potato (10.0k ha), maize
  (8.8k ha) and paddy (3.8k ha) are.

## 5. Conformal intervals instead of MC dropout

MC dropout gives no coverage guarantee and is often over-confident on small data. Normalised
split-conformal intervals (scaled by each well's past variability) are calibrated on 2022 and
**checked** on 2023+ (H5), both one round ahead and for recursive multi-round forecasts. For true
forecasts, the bands also include weather uncertainty from 21 historical years.

## 6. Two real-world validation tests the spec did not have

- **K.C. Valley natural experiment.** Sentinel-2 identifies wells near tanks that hold water in the
  dry season after 2018 (115 treated, 149 control wells). A difference-in-differences with
  well×season and round fixed effects plus a rainfall control asks whether treated wells rose
  relative to controls. The twin is then asked the counterfactual ("tanks at pre-project level").
- **Live 2026 drought.** The now-cast and forecast use IMD real-time rain to Oct 2026. Freeze
  `artifacts/web/wells.json` with a date before CGWB publishes the Nov 2026 and Jan/May 2027 rounds,
  then score it: an out-of-sample test nobody can call leaky.

## 7. Data assimilation: the "twin" part

The handbook says a dashboard is not a digital twin unless observations update the state. Here
`Twin.rollout(assimilate=True)` resets each well to every new reading and simulates between
readings with observed forcing. The map always starts from the latest evidence.

---

## Results against the spec's hypotheses

| | Hypothesis | Result |
|---|---|---|
| H1 | Spatial-temporal modelling beats per-well models | **Weak at best.** Removing the graph features worsens test MAE by 0.035 m (dug wells) and 0.005 m (bore wells); the ST-GNN is worse than XGBoost and the physics balance. |
| H2 | Environmental covariates help | **Partly.** Removing rain/ET/tank inputs from the hybrid barely changes 1-round MAE (their effect is already in the physics term). The physics balance alone, which is all covariates, is the best bore-well model. |
| H3 | Graph helps most when data are sparse | **Not supported.** With 10–30% of training readings hidden, models with and without graph features stay within 0.04 m of each other, and the gap does not grow with sparsity. |
| H4 | Scenarios are directionally right and spatially coherent | **Holds by construction** (sign constraints, monotone ML, physical pumping term). K.C. Valley: the twin's effect (bore wells ~0.4 m shallower) lies inside the observed DiD interval (−4.8 to +12.8 m), which is too wide to confirm it. |
| H5 | Intervals reach nominal coverage | **Supported.** 80%/90% bands: 81%/91% (bore wells), 83%/94% (dug wells) on 2023+. |
| H6 | Yield/income estimates move in the right direction for stress years | Implemented with Monte Carlo ranges. Compare with the state's 2023 and 2026 drought-loss memoranda for Kolar/Chikkaballapura (not machine-readable; enter them manually). |

Negative results on H1 and H3 are findings, not failures: with India's sparse monitoring, a
well-constrained water balance with honest uncertainty beats a deep spatial model. Write that up
plainly; it is the more useful message for groundwater agencies.

## Suggested paper outline (IEEE conference)

1. Introduction: sparse monitoring, hard-rock aquifers, why decision-makers need ₹ not metres.
2. Study area and data: two districts, 438 wells, two aquifers, K.C./H.N. Valley.
3. Methods: physics-constrained hybrid, conformal intervals, data assimilation, scenario semantics, agro-economic chain.
4. Experiments: model comparison (incl. ST-GNN), ablation, sparse-data test, multi-step coverage, K.C. Valley DiD and counterfactual.
5. Results and discussion: when deep models don't help; what the intervals say about bore-well noise.
6. Limitations: Sy from norms until NAQUIM values are entered; pumping response is physics only; crop areas from 2015-16; prices from one season of APMC data; rice Ky assumed.
7. Conclusion and the live 2026 forecast check.

## Limitations to state in the report

- Pumping was never observed over time; its scenario response is physics (ΔD/Sy), not learned.
- Specific yield is a range from GEC norms; read the NAQUIM taluk reports and narrow `config.yaml`.
- Sentinel-2 L2A starts in 2017, so tank water before 2017 is set to the 2017–18 level.
- Bore-well "water levels" in daily telemetric records are affected by pumping; round values use the median of days 1–21.
- Crop areas are from the 2015-16 Agricultural Census (the latest tehsil-level release); Chikkaballapura taluk is missing and filled with its district's mean.
- Tomato yields are state-level final estimates; district tomato yields are not published after 2004.
- Scenario outputs are model-based "what-ifs" and the ₹ figures are for comparing scenarios, not audited losses.
