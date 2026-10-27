"""Fast checks of the pieces most likely to break silently.  Run:  python -m pytest -q"""
import numpy as np
import pandas as pd

from gwtwin.data.climate import hargreaves_et0
from gwtwin.data.wells import index_to_round, round_index
from gwtwin.models.conformal import Conformal
from gwtwin.models.core import Physics, prep


def test_round_index_roundtrip():
    for y in (2005, 2018, 2026):
        for m in (1, 5, 8, 11):
            assert index_to_round(round_index(y, m)) == (y, m)


def test_hargreaves_reasonable():
    # Kolar (13°N), mid-April, hot dry day: ET0 should be ~5–7 mm/day
    et0 = hargreaves_et0(np.array([35.0]), np.array([22.0]), np.array([13.1]), np.array([105]))
    assert 4.5 < et0[0] < 7.5


def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    month = rng.choice([1, 5, 8, 11], n)
    rain = rng.gamma(2, 80, n)
    et0 = rng.normal(400, 60, n)
    anom = rng.normal(0, 2, n)
    dh = 0.5 - 0.006 * rain + 0.002 * et0 - 0.2 * anom + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"aquifer": "shallow", "month": month, "rain_int": rain, "rain_prev_int": 0.0,
                       "rain_12m_before": 0.0, "et0_int": et0, "water_ha_3km": 0.0, "anom_base": anom, "dh": dh})
    return prep(df)


def test_physics_recovers_signs_and_scale():
    df = _synthetic()
    m = Physics().fit(df)
    c = m.coef["shallow"]
    assert c["rain_int"] < 0 and abs(c["rain_int"] + 0.006) < 0.002
    assert c["et0_int"] >= 0
    assert c["anom_base"] < 0


def test_physics_sign_constraint_holds_even_if_data_disagree():
    df = _synthetic()
    df["dh"] = df.dh + 0.01 * df.rain_int      # make rain *deepen* water: unphysical
    c = Physics().fit(df).coef["shallow"]
    assert c["rain_int"] <= 1e-12               # the constraint wins


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(1)
    n = 4000
    df = pd.DataFrame({"aquifer": "deep", "scale_past": 1.0, "dh": rng.normal(0, 2, n)})
    yhat = np.zeros(n)
    cal, test = df.iloc[:2000], df.iloc[2000:]
    c = Conformal((0.8, 0.9)).fit(cal, yhat[:2000])
    rep = c.coverage_report(test, yhat[2000:])
    for cov, emp in zip(rep.nominal, rep.empirical):
        assert abs(cov - emp) < 0.03
