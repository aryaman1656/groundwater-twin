"""Project paths and configuration loading."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

for _p in (RAW, INTERIM, PROCESSED, ARTIFACTS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def cfg() -> dict:
    with open(ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def bbox() -> tuple[float, float, float, float]:
    b = cfg()["region"]["bbox"]
    return b["lat_min"], b["lat_max"], b["lon_min"], b["lon_max"]
