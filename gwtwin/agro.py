"""Groundwater scenario -> irrigation water -> crop yield -> income, with Monte Carlo uncertainty.

Chain (per taluk, per crop, per Monte Carlo draw):
  1. Δh      scenario minus baseline depth change of DEEP wells (the aquifer farmers pump), taluk median,
             perturbed by the twin's forecast uncertainty.
  2. ΔS      = Sy · Δh                     extra storage deficit, metres of water over the taluk
  3. f       = θ · ΔS / D_season           fraction of the season's irrigation supply that is lost
             (D_season = irrigation draft over the forecast window, CGWB GWRA-2024; θ ∈ config range)
  4. 1−ETa/ETm = f · s_irr                 s_irr = share of crop water need met by irrigation
             = 1 − Peff/ETc,  ETc = Kc·ET0 (FAO-56), Peff = USDA-SCS effective rainfall
  5. 1−Ya/Ym = Ky · (1 − ETa/ETm)          FAO-33 (Doorenbos & Kassam)
  6. ₹       = (1−Ya/Ym) · Ym · area · price
Negative values mean the scenario helps (e.g. more recharge).

Why not the spec's GWS = (GWL_base − GWL_pred)/GWL_base?  It divides by depth, so a 0.5 m drop at a
1 m-deep well scores 50 % "stress" while a 2 m drop at a 20 m-deep well scores 10 %. Storage change
(Sy·Δh) is the physically meaningful quantity, and it is compared with how much water is pumped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW, cfg

AGCENSUS_NAME = {"Tomato": "TOMATO", "Potato": "POTATO", "Maize": "MAIZE", "Rice": "PADDY"}
PRICE_FILTER = {"Tomato": ("Tomato", None), "Potato": ("Potato", None), "Maize": ("Maize", ["Local", "Hybrid/Local", "Yellow"]),
                "Rice": ("Paddy(Common)", None)}
TALUK_ALIAS = {"Bangarapet": "Bangarapet", "Srinivaspur": "Srinivaspur", "Sidlaghatta": "Sidlaghatta",
               "Gudibanda": "Gudibanda", "Gauribidanur": "Gauribidanur", "Bagepalli": "Bagepalli",
               "Chintamani": "Chintamani", "Kolar": "Kolar", "Malur": "Malur", "Mulbagal": "Mulbagal"}


def crop_areas() -> pd.DataFrame:
    """Irrigated area (ha) per taluk and crop, Agricultural Census 2015-16 (latest tehsil-level release)."""
    t = pd.read_csv(RAW / "idp" / "agcensus_tehsil_crop.csv")
    t = t[t.year == "2015-16"]
    rows = []
    for crop, name in AGCENSUS_NAME.items():
        s = t[t.crop_name == name].groupby(["district_name", "tehsil_name"]).irr_ar.sum()
        for (dist, tal), v in s.items():
            rows.append({"district": dist, "taluk": TALUK_ALIAS.get(tal, tal), "crop": crop, "irr_area_ha": float(v)})
    a = pd.DataFrame(rows)
    # Chikkaballapura taluk is missing from the tehsil release: use the mean of its district's other taluks
    # (ASSUMPTION, flagged in the UI)
    chik = a[a.district.str.contains("ballap", case=False)]
    if not (a.taluk == "Chikkaballapura").any() and len(chik):
        m = chik.groupby("crop").irr_area_ha.mean().reset_index()
        m["district"], m["taluk"], m["assumed"] = "Chikkaballapura", "Chikkaballapura", True
        a = pd.concat([a, m], ignore_index=True)
    a["assumed"] = a.get("assumed", False)
    a["assumed"] = a["assumed"].fillna(False).astype(bool)
    return a


def yields() -> dict:
    """Max-attainable yield proxy Ym (t/ha): (P50, P90) of recent district (or state) yields."""
    out = {}
    a = pd.read_csv(RAW / "idp" / "apy_des.csv")
    for crop in ("Potato", "Maize", "Rice"):
        x = a[(a.crop_name == crop) & (a.season != "Total") & (a.season != "Whole Year")]
        y = x.groupby("year").apply(lambda g: g.production.sum() / g.area.sum() if g.area.sum() > 0 else np.nan,
                                    include_groups=False).dropna()
        y = y.sort_index().tail(10)
        out[crop] = {"p50": float(y.quantile(0.5)), "p90": float(y.quantile(0.9)), "source": "DES APY, Kolar+Chikkaballapura, last 10 yrs"}
    h = pd.read_csv(RAW / "idp" / "horticulture_karnataka.csv")
    t = h[(h.crop_name == "Tomato") & (h.estimate_type == "Final Estimates")].copy()
    t["y"] = t.production / t.area
    t = t.sort_values("year").tail(6)
    out["Tomato"] = {"p50": float(t.y.median()), "p90": float(t.y.quantile(0.9)),
                     "source": "Karnataka state final estimates (district tomato yields not published after 2004)"}
    return out


def prices() -> dict:
    """₹/tonne from APMC modal prices in the two districts' markets (P10, P50, P90)."""
    p = pd.read_csv(RAW / "idp" / "apmc_prices.csv")
    out = {}
    for crop, (name, varieties) in PRICE_FILTER.items():
        x = p[p.commodity_name == name]
        if varieties:
            x = x[x.variety.isin(varieties)]
        q = x.modal_price.quantile([0.1, 0.5, 0.9]) * 10      # Rs/quintal -> Rs/tonne
        out[crop] = {"p10": float(q.iloc[0]), "p50": float(q.iloc[1]), "p90": float(q.iloc[2]),
                     "n": int(len(x)), "period": f"{p.date.min()} to {p.date.max()}"}
    return out


def crop_water_share(crop: str, et0_mm_day: float, rain_mm_month: float) -> float:
    """s_irr = 1 − Peff/ETc for the crop's season (FAO-56 Kc, USDA-SCS effective rain)."""
    c = cfg()["agro"]["crops"][crop]
    kc_ini, kc_mid, kc_end = c["kc"]
    d1, d2, d3, d4 = c["days"]
    days = d1 + d2 + d3 + d4
    kc_avg = (kc_ini * d1 + (kc_ini + kc_mid) / 2 * d2 + kc_mid * d3 + (kc_mid + kc_end) / 2 * d4) / days
    etc = kc_avg * et0_mm_day * days
    p_month = max(rain_mm_month, 0.0)
    peff_month = p_month * (125 - 0.2 * p_month) / 125 if p_month < 250 else 125 + 0.1 * p_month
    peff = peff_month * days / 30.4
    return float(np.clip(1 - peff / max(etc, 1e-6), 0, 1))


def run_chain(delta_h_by_taluk: pd.DataFrame, horizon_shares: float, et0_mm_day: float, rain_mm_month: float,
              n_draws: int | None = None, seed: int = 0) -> dict:
    """delta_h_by_taluk: columns taluk, district, dh_mean (m, + = deeper vs baseline), dh_sd (m).
    horizon_shares: fraction of the annual draft that falls inside the forecast window."""
    from .twin import district_draft_m_per_year

    c = cfg()
    n = n_draws or c["agro"]["monte_carlo_draws"]
    rng = np.random.default_rng(seed)
    areas = crop_areas()
    ym, pr = yields(), prices()
    g = pd.read_csv(RAW / "opencity" / "dgwr_karnataka_2024.csv")
    g["d"] = g["Name of District"].str.lower()
    irr_share = {}
    for dist, key in (("Kolar", "kolar"), ("Chikkaballapura", "chikkaballa")):
        row = g[g.d.str.startswith(key)].iloc[0]
        irr_share[dist] = float(row["Irrigation - Annual extraction"] / row["Total Annual Extraction"])
    draft = district_draft_m_per_year()
    sy_lo, sy_hi = c["physics"]["sy_bounds"]["deep"]
    th_lo, th_hi = c["agro"]["theta_range"]

    crops = list(c["agro"]["crops"])
    s_irr = {k: crop_water_share(k, et0_mm_day, rain_mm_month) for k in crops}
    res_rows = []
    totals = np.zeros(n)
    by_crop_tot = {k: np.zeros(n) for k in crops}
    area_at_risk = {k: np.zeros(n) for k in crops}
    for r in delta_h_by_taluk.itertuples():
        dist = "Kolar" if str(r.district).lower().startswith("kolar") else "Chikkaballapura"
        d_season = draft[dist] * irr_share[dist] * horizon_shares            # m of water pumped for irrigation
        sy = np.exp(rng.uniform(np.log(sy_lo), np.log(sy_hi), n))
        theta = rng.uniform(th_lo, th_hi, n)
        dh = rng.normal(r.dh_mean, max(r.dh_sd, 1e-6), n)
        f = np.clip(theta * sy * dh / max(d_season, 1e-6), -0.5, 1.0)          # supply cut (− = gain)
        for crop in crops:
            a = areas[(areas.taluk == r.taluk) & (areas.crop == crop)].irr_area_ha.sum()
            if a <= 0:
                continue
            ky = c["agro"]["crops"][crop]["ky"]
            yl = np.clip(ky * f * s_irr[crop], -0.5, 1.0)                     # relative yield loss
            ymax = rng.uniform(ym[crop]["p50"], ym[crop]["p90"], n)
            price = rng.triangular(pr[crop]["p10"], pr[crop]["p50"], max(pr[crop]["p90"], pr[crop]["p50"] + 1), n)
            loss_rs = yl * ymax * a * price
            totals += loss_rs
            by_crop_tot[crop] += loss_rs
            area_at_risk[crop] += a * (yl > 0.05)
            res_rows.append({"taluk": r.taluk, "district": dist, "crop": crop, "irr_area_ha": round(a, 1),
                             "supply_cut_pct_p50": float(np.median(f) * 100),
                             "yield_loss_pct_p10": float(np.percentile(yl, 10) * 100),
                             "yield_loss_pct_p50": float(np.median(yl) * 100),
                             "yield_loss_pct_p90": float(np.percentile(yl, 90) * 100),
                             "loss_rs_lakh_p10": float(np.percentile(loss_rs, 10) / 1e5),
                             "loss_rs_lakh_p50": float(np.median(loss_rs) / 1e5),
                             "loss_rs_lakh_p90": float(np.percentile(loss_rs, 90) / 1e5)})
    q = lambda v: {"p10": float(np.percentile(v, 10) / 1e5), "p50": float(np.median(v) / 1e5),   # noqa: E731
                   "p90": float(np.percentile(v, 90) / 1e5)}
    return {
        "rows": res_rows,
        "total_rs_lakh": q(totals),
        "by_crop": {k: {**q(v), "area_at_risk_ha_p50": float(np.median(area_at_risk[k])),
                        "irrigation_share_of_water_need": s_irr[k], "ky": c["agro"]["crops"][k]["ky"],
                        "ky_source": c["agro"]["crops"][k]["ky_source"]} for k, v in by_crop_tot.items()},
        "assumptions": {
            "Sy_deep_range": [sy_lo, sy_hi], "theta_range": [th_lo, th_hi],
            "yields_t_per_ha": ym, "prices_rs_per_tonne": pr, "irrigation_share_of_draft": irr_share,
            "draft_m_per_year": draft, "horizon_draft_share": horizon_shares,
            "crop_areas": "Agricultural Census 2015-16, irrigated area by tehsil",
            "not_modelled": ["Ragi (≈96% rainfed here, not groundwater-linked)", "Mulberry (no FAO Kc/Ky)",
                             "Mango and other perennials"],
        },
    }
