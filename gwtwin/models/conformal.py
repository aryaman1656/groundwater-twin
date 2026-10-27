"""Normalized split-conformal prediction intervals.

score_i = |y_i − ŷ_i| / σ_i,   σ_i = robust scale of this well's past round-to-round changes
(falls back to the aquifer median for wells without history).  The (1−α) quantile q of the
calibration scores gives the interval  ŷ ± q·σ, which covers the truth with probability ≥ 1−α
if calibration and test errors are exchangeable. We check that on the held-out test years (H5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def well_scale(df: pd.DataFrame, floor: float = 0.25) -> np.ndarray:
    """Past-only robust scale: median |dh| of the well's earlier transitions."""
    s = df["scale_past"].to_numpy(float) if "scale_past" in df else np.full(len(df), np.nan)
    aq_med = df.groupby("aquifer")["scale_past"].transform("median").to_numpy(float) if "scale_past" in df else 1.0
    s = np.where(np.isfinite(s), s, aq_med)
    s = np.where(np.isfinite(s), s, 1.0)
    return np.maximum(s, floor)


class Conformal:
    def __init__(self, coverages=(0.8, 0.9)):
        self.coverages = coverages
        self.q: dict = {}

    def fit(self, df: pd.DataFrame, yhat: np.ndarray, key_cols=("aquifer",)):
        sc = np.abs(df.dh.to_numpy(float) - yhat) / well_scale(df)
        d = df[list(key_cols)].copy()
        d["score"] = sc
        for key, g in d.groupby(list(key_cols)):
            n = len(g)
            for c in self.coverages:
                level = min(1.0, np.ceil((n + 1) * c) / n)
                self.q[(key if isinstance(key, tuple) else (key,), c)] = float(np.quantile(g.score, level))
        self.key_cols = key_cols
        return self

    def interval(self, df: pd.DataFrame, yhat: np.ndarray, coverage: float):
        s = well_scale(df)
        keys = [tuple(r) for r in df[list(self.key_cols)].itertuples(index=False)]
        q = np.array([self.q.get((k, coverage), np.nan) for k in keys])
        return yhat - q * s, yhat + q * s

    def coverage_report(self, df, yhat) -> pd.DataFrame:
        rows = []
        for c in self.coverages:
            lo, hi = self.interval(df, yhat, c)
            y = df.dh.to_numpy(float)
            inside = (y >= lo) & (y <= hi)
            for aq in sorted(df.aquifer.unique()):
                m = np.asarray(df.aquifer == aq)
                rows.append({"aquifer": aq, "nominal": c, "empirical": float(inside[m].mean()),
                             "mean_width_m": float(np.mean(hi[m] - lo[m])), "n": int(m.sum())})
        return pd.DataFrame(rows)
