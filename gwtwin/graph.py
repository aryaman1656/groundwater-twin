"""Spatial graph of monitoring wells.

Nodes are wells. Edges connect each well to its k nearest wells of the SAME aquifer class
(shallow↔shallow, deep↔deep) within max_edge_km. Edge weight = inverse distance.
Alternative graphs (all-aquifer, distance-threshold) are built for the ablation study.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import cfg


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def build_edges(wells: pd.DataFrame, k: int | None = None, max_km: float | None = None,
                same_aquifer: bool = True) -> pd.DataFrame:
    k = k or cfg()["graph"]["k_neighbours"]
    max_km = max_km or cfg()["graph"]["max_edge_km"]
    lat, lon = wells.lat.values, wells.lon.values
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(d, np.inf)
    if same_aquifer:
        aq = np.asarray(wells.aquifer, dtype=str)
        d = np.where(aq[:, None] == aq[None, :], d, np.inf)
    rows = []
    for i in range(len(wells)):
        nn = np.argsort(d[i])[:k]
        for j in nn:
            if d[i, j] <= max_km:
                rows.append((wells.well_id.iat[i], wells.well_id.iat[j], d[i, j]))
    e = pd.DataFrame(rows, columns=["src", "dst", "km"])
    e["w"] = 1.0 / np.maximum(e.km, 0.5)
    return e
