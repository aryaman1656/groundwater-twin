# Re-verification of the numbers behind the area plan (30 Oct 2026)

Re-run on the shipped data with the existing train / calib / test split (test = target date 2023+).
"Reproduced" means the same code and data give the same number. Where a number did not reproduce
exactly, the reason (usually a method detail not written down) and the value now used are given.
None of the differences changes a decision, so the core model was **not** changed.

Scripts used: `gwtwin/evaluate.py` (model comparison), `gwtwin/area_eval.py` (scale and partition skill),
`gwtwin/spatial.py` (spillover tests), plus the service for the what-if numbers.

## Model comparison (one-round change, MAE, test 2023+)

| Model | Bore wells | Dug wells | Status |
|---|---|---|---|
| Persistence | 5.258 | 0.810 | reproduced |
| Physics | 4.946 | 0.881 | reproduced |
| XGBoost | 5.006 | 0.835 | reproduced |
| Hybrid (eval fit) | 5.075 | 0.813 | reproduced |
| Seasonal mean | 5.738 | 1.014 | reproduced |
| ST-GNN | 5.58 (reported 5.63) | 0.98 (0.95) | close; torch 2.15 CPU run, training is not bit-for-bit deterministic. Still worse than no-change for both aquifers. |

Deployed weights: deep α 1.0, β 0; shallow α 0.45, β 0 — reproduced. (The *evaluation* hybrid, fitted on
data to 2021, has β = 1 for deep wells; only the deployed fit switches the ML off.)

## Noise and scale (deep wells, physics model, test)

| Claim | Now | Status |
|---|---|---|
| Deep-well R² ≈ 0.08 | 0.076 (physics), 0.081 (hybrid) | reproduced |
| True regional mean explains ~11 % of a well's variance | 11.9 % leave-one-out, 12.8 % including the well itself | reproduced (leave-one-out) |
| Worst 5 % of well-rounds carry 77 % of squared error | 77.2 % | reproduced |
| Single well +6 % (corr 0.35) | +5.9 % (0.35) | reproduced |
| ~25 km cell +25 % (0.64) | +21 % (0.55); range 18–23 % over 16 grid origins | **origin-dependent**, lower on average |
| ~50 km cell +28 % (0.72) | +27 % (0.65); range 22–32 % | origin-dependent |
| Whole region +34 % (0.93, 10 rounds) | +34.3 % (0.93, 10 rounds) | reproduced |

## Partitions of the 312 study wells (deep, physics, area mean)

| Partition | Claimed | Now | Status |
|---|---|---|---|
| District (2) | 31 % | 30.6 % (corr 0.85) | reproduced |
| Taluk (11) | 24 % (corr 0.67) | 24.1 % (0.67) | reproduced |
| 0.25° grid | 23 % | 15–23 % over grid origins (mean 19 %) | upper end of the range |
| 0.15° grid | 18 % | 12–14 % | not reproduced; lower |
| ~28 km grid (for comparison) | ≈ taluk | 16–23 % (mean 19 %) | taluks are **at least** as good as a same-size grid |

Conclusion unchanged: taluks are a sound reporting unit (no worse than an equal-size grid, and they
match the agro layer). With area **medians** (what the dashboard shows) taluk skill is 31 % (90 % range
19–40 %), corr 0.61; with means 22 % / 0.67 for the deployed form.

Shallow wells: taluk −0.9 % (claimed −1.5 %), district +13.5 % (reproduced); 5 of 11 taluks have 0–1
dug wells with test readings (reproduced: Bagepalli 0, Sidlaghatta 0, Chikkaballapura 1, Malur 1, Srinivaspur 1).

Taluk bore-well counts (10–38; Gudibanda 10, Bagepalli 12, Malur 12) and all 312 study wells having a
taluk label: reproduced. Labels also match the taluk polygons (0 wells outside their own taluk + 500 m).

## Area-level uncertainty

The earlier check (2022 calibration only) was replaced by a rolling one; see `reports/area_eval.json`
and the Model check tab. It is not comparable to the old 9.5 m / 9.3 m numbers, which used a different
target.

## Spillover

| Claim | Now | Status |
|---|---|---|
| +500 ha tank water around 16 bore wells moves them ≈ −3.14 m | −3.21 m (16 wells nearest 13.20 N, 78.10 E) | reproduced in size (well set not recorded) |
| Other wells move exactly 0, β = 0 | 0.0000 m for all 422 | reproduced |
| …and exactly 0 with β = 1 | **4 of 422 move, by −0.03 to +0.17 m** | not reproduced. With the ML on, neighbour features carry a tiny, sign-inconsistent change. The conclusion (no physical spillover in the engine) still holds; the deployed engine has none. |
| Residual correlation 0.135 / 0.125 / 0.110 / 0.074 / 0.048 / 0.035 | Pearson 0.092 / 0.085 / 0.073 / 0.058 / 0.038 / 0.048; outlier-clipped 0.119 / 0.106 / 0.097 / 0.064 / 0.047 / 0.039 | **method-dependent**; the exact earlier method was not recorded. Same shape: weak, slow decay, no short-range peak. |
| Neighbour residual at k−1 → mine at k: +0.021 (−0.026, +0.078) | +0.029 (−0.031, +0.090) with own lag and round effects; +0.024 (−0.034, +0.082) without own lag | reproduced in substance |
| Heads travel ~1–2 km per year (T 10–50 m²/day) | √(D·t): 0.35–1.9 km; 2√(D·t): 0.7–3.8 km (S = 0.005–0.03) | consistent; T still unverified |

Nearest-neighbour spacing of study bore wells: median 3 km (IQR 2.1–4.3 km).

## What-if magnitudes (4 rounds, from the Nov 2026 now-cast)

| Claim | Now | Status |
|---|---|---|
| Rain −20 %: deep +1.07 m (1.02–1.15), shallow +0.19 m | deep +1.32 m with climatological rain, +1.20 m median over 21 weather years (wells 1.06–1.51; years 0.95–1.88); shallow +0.27 m | **not reproduced**: ~20 % larger. Same order and same conclusion (≈1 m per −20 % rain over a year, nearly identical for every well). |
| Pumping +20 %: Kolar deep +2.8 m (1.1–6.9) | +2.75 m (1.12–6.73) | reproduced |
| Pumping +20 %: Chikkaballapura deep +1.84 m (0.75–4.5) | +1.81 m (0.74–4.43) | reproduced |
| District draft 0.18 / 0.118 m/yr | 0.180 / 0.118 | reproduced |
| Absolute 4-round 80 % band for a deep well ≈ 48 m wide | study bore wells: mean 44.5 m, median 23 m (p10 8.6, p90 105 m); all 354 bore wells mean 40 m | roughly; the mean is pulled up by a heavy tail, the typical well's band is about half that |

## Taluk prototype (deep, median forecast change to 2027-11-05)

All eleven taluk medians reproduced to 0.01 m (Gauribidanur 0.52 … Gudibanda 2.57), region 1.29 m
(IQR 0.49–2.16), Sidlaghatta IQR −0.38 to 2.64. Every study bore well's last reading is 6–15 rounds
old (median 7; newest reading May 2025). The shipped forecast is reproduced by the code to 5×10⁻⁵ m.
