"""Forecasting models for the change in depth-to-water over one round (≈3 months).

All models predict  dh = depth(k+1) − depth(k)   (metres, + = water table falling).

Models
------
Persistence        dh = 0                                        (must-beat baseline)
SeasonalMean       dh = this well's past mean change for the season (else aquifer-season mean)
XGBoost            gradient boosting on all features, no physics, no constraints (spec Stage 1)
Physics            constrained linear water balance (signs fixed by physics, see below)
Hybrid  (core)     Physics + monotone-constrained XGBoost on the physics residual

Physics term (per aquifer class), from the water-table-fluctuation balance −Sy·Δh = R − D − Q:
    dh = θ_season + b_rain·rain_int + b_prev·rain_prev_int + b_12m·rain_12m_before
         + b_et0·et0_int + b_tank·water_ha_3km + b_rev·anom
with sign constraints b_rain, b_prev, b_12m, b_tank ≤ 0 (more water in → table rises),
b_et0 ≥ 0 (more evaporative demand → falls), b_rev ≤ 0 (anomalies relax back: lateral flow).
Pumping is NOT fitted (it never varies in the data) — it enters scenarios through D/Sy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

STATIC = ["elev_m", "slope_deg_2km", "rel_elev_2km", "lc_cropland_2km", "lc_built_2km", "lc_tree_2km",
          "lc_water_2km", "soil_clay_0_30", "soil_sand_0_30", "well_depth"]
STATE = ["depth_m", "anom", "anom_base", "past_mean", "dh_prev", "dh_season_past", "frac_depth", "rounds_since_obs"]
GRAPH = ["nb_anom", "nb_dh_prev", "nb_n"]
FORCING = ["rain_int", "rain_prev_int", "rain_12m_before", "et0_int", "water_ha_3km", "water_ha_1km"]
SEASON = ["m_jan", "m_may", "m_aug", "m_nov"]
FEATURES = STATE + GRAPH + FORCING + STATIC + SEASON

# monotone direction w.r.t. dh (−1: more of x → smaller dh → shallower water)
MONO = {"rain_int": -1, "rain_prev_int": -1, "rain_12m_before": -1, "water_ha_3km": -1, "water_ha_1km": -1,
        "et0_int": 1}

PHYS_TERMS = ["rain_int", "rain_prev_int", "rain_12m_before", "et0_int", "water_ha_3km", "anom_base"]
PHYS_SIGN = {"rain_int": -1, "rain_prev_int": -1, "rain_12m_before": -1, "et0_int": 1, "water_ha_3km": -1,
             "anom_base": -1}


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    for m, c in zip((1, 5, 8, 11), SEASON):
        df[c] = (df.month == m).astype(float)
    return df


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = add_season(df.copy())
    for c in ("dh_prev", "nb_dh_prev", "dh_season_past", "frac_depth", "anom", "nb_anom"):
        if c in df:
            df[c] = df[c].astype(float)
    return df


# ----------------------------------------------------------------------------- baselines
class Persistence:
    name = "Persistence"

    def fit(self, df):
        return self

    def predict(self, df):
        return np.zeros(len(df))


class SeasonalMean:
    name = "Seasonal mean"

    def fit(self, df):
        self.fallback = df.groupby(["aquifer", "month"]).dh.median()
        return self

    def predict(self, df):
        fb = np.array([self.fallback.get((a, m), 0.0) for a, m in zip(df.aquifer, df.month)])
        v = df.dh_season_past.to_numpy(dtype=float)
        return np.where(np.isfinite(v), v, fb)


class XGB:
    name = "XGBoost (no physics)"

    def __init__(self, features=None, monotone=False, seed=0):
        self.features = features or FEATURES
        self.monotone = monotone
        self.seed = seed

    def _model(self):
        import xgboost as xgb

        params = dict(n_estimators=600, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=20, reg_lambda=5.0, objective="reg:pseudohubererror", huber_slope=2.0,
                      random_state=self.seed, n_jobs=2, tree_method="hist", early_stopping_rounds=40)
        if self.monotone:
            params["monotone_constraints"] = tuple(MONO.get(f, 0) for f in self.features)
        return xgb.XGBRegressor(**params)

    def fit(self, df, y=None):
        self.models = {}
        for aq, g in df.groupby("aquifer"):
            m = self._model()
            yy = g.dh if y is None else y.loc[g.index]
            # time-ordered early stopping: last 20% of the training rounds act as validation
            cut = g.k.quantile(0.8)
            tr, va = g.k <= cut, g.k > cut
            m.fit(g.loc[tr, self.features], yy[tr], eval_set=[(g.loc[va, self.features], yy[va])], verbose=False)
            self.models[aq] = m
        return self

    def predict(self, df):
        out = np.zeros(len(df))
        aq = np.asarray(df.aquifer, dtype=str)
        for a, m in self.models.items():
            idx = aq == a
            if idx.any():
                out[idx] = m.predict(df.loc[idx, self.features])
        return out


# ----------------------------------------------------------------------------- physics
class Physics:
    """Sign-constrained, Huber-robust linear water balance, one per aquifer class."""
    name = "Physics (water balance)"

    def fit(self, df):
        self.coef, self.scale = {}, {}
        for aq, g in df.groupby("aquifer"):
            X, cols = self._design(g)
            y = g.dh.to_numpy(float)
            lb = np.full(X.shape[1], -np.inf)
            ub = np.full(X.shape[1], np.inf)
            for i, c in enumerate(cols):
                s = PHYS_SIGN.get(c)
                if s == -1:
                    ub[i] = 0
                elif s == 1:
                    lb[i] = 0
            w = np.ones(len(y))
            for _ in range(6):  # IRLS with Huber weights
                sw = np.sqrt(w)
                res = lsq_linear(X * sw[:, None], y * sw, bounds=(lb, ub), lsmr_tol="auto")
                r = y - X @ res.x
                s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
                w = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-9))
            self.coef[aq] = pd.Series(res.x, index=cols)
        return self

    def _design(self, g):
        cols = SEASON + PHYS_TERMS
        X = g[cols].to_numpy(float)
        X = np.nan_to_num(X)
        return X, cols

    def predict(self, df, return_terms=False):
        out = np.zeros(len(df))
        terms = {c: np.zeros(len(df)) for c in PHYS_TERMS + ["season"]}
        aq = np.asarray(df.aquifer, dtype=str)
        for a, coef in self.coef.items():
            idx = aq == a
            if not idx.any():
                continue
            X, cols = self._design(df.loc[idx])
            contrib = X * coef.to_numpy()[None, :]
            out[idx] = contrib.sum(1)
            for i, c in enumerate(cols):
                key = "season" if c in SEASON else c
                terms[key][idx] += contrib[:, i]
        return (out, terms) if return_terms else out

    def summary(self) -> pd.DataFrame:
        rows = []
        for aq, c in self.coef.items():
            rows.append({"aquifer": aq,
                         "m_per_100mm_rain": round(-c["rain_int"] * 100, 3),
                         "m_per_100mm_prev_rain": round(-c["rain_prev_int"] * 100, 3),
                         "m_per_100mm_12m_rain": round(-c["rain_12m_before"] * 100, 4),
                         "m_per_100mm_et0": round(c["et0_int"] * 100, 3),
                         "m_per_100ha_tank_water": round(-c["water_ha_3km"] * 100, 3),
                         "relaxation_per_round": round(-c["anom_base"], 3),
                         # equilibrium shift: metres of rise per 100 ha of dry-season tank water
                         "equilibrium_rise_m_per_100ha": round(c["water_ha_3km"] / c["anom_base"] * 100, 3)
                         if c["anom_base"] < 0 else None})
        return pd.DataFrame(rows)


class Hybrid:
    """Physics + monotone XGBoost on the physics residual (the core model).

    Per-aquifer weights are chosen on the last 20 % (in time) of the training window:
    prediction = α·(physics + β·ML).  β ≤ 1 limits how much the ML correction is trusted; α < 1 means "trust the model less, lean towards
    persistence" — it is a forecast combination with the no-change baseline, learned from data,
    so the model is never systematically worse than persistence on recent data.
    """
    name = "Hybrid (physics + monotone ML)"

    def __init__(self, features=None, seed=0, damping=True):
        self.features, self.seed, self.damping = features, seed, damping
        self.phys = Physics()
        self.ml = XGB(features=features, monotone=True, seed=seed)
        self.alpha, self.beta = {}, {}

    def _fit_core(self, df):
        phys, ml = Physics().fit(df), XGB(features=self.features, monotone=True, seed=self.seed)
        ml.fit(df, y=df.dh - phys.predict(df))
        return phys, ml

    def fit(self, df):
        self.alpha = {a: 1.0 for a in df.aquifer.unique()}
        self.beta = {a: 1.0 for a in df.aquifer.unique()}
        if self.damping:
            cut = df.k.quantile(0.8)
            early, late = df[df.k <= cut], df[df.k > cut]
            if len(late) > 50:
                phys, ml = self._fit_core(early)
                pp, pm = phys.predict(late), ml.predict(late)
                y = late.dh.to_numpy()
                for a in late.aquifer.unique():
                    m = np.asarray(late.aquifer == a)
                    best = (np.inf, 1.0, 1.0)
                    for al in np.linspace(0, 1, 21):
                        for be in (0.0, 0.25, 0.5, 0.75, 1.0):
                            mae = np.mean(np.abs(y[m] - al * (pp[m] + be * pm[m])))
                            if mae < best[0] - 1e-9:
                                best = (mae, al, be)
                    self.alpha[a], self.beta[a] = float(best[1]), float(best[2])
        self.phys, self.ml = self._fit_core(df)
        return self

    def _alpha_vec(self, df):
        return np.array([self.alpha.get(a, 1.0) for a in np.asarray(df.aquifer, dtype=str)])

    def predict(self, df, return_terms=False):
        p, terms = self.phys.predict(df, return_terms=True)
        b = np.array([self.beta.get(x, 1.0) for x in np.asarray(df.aquifer, dtype=str)])
        r = b * self.ml.predict(df)
        a = self._alpha_vec(df)
        if return_terms:
            terms = {k: a * v for k, v in terms.items()}
            terms["ml_residual"] = a * r
            return a * (p + r), terms
        return a * (p + r)


def metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    e = yhat - y
    ss = np.sum((y - y.mean()) ** 2)
    return {"n": int(len(y)), "MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "R2": float(1 - np.sum(e ** 2) / ss) if ss > 0 else np.nan}
