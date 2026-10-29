"""Area aggregation, scenario masks, footprint and freezing.  Run:  python -m pytest -q

The tests that need the shipped data and deployed model are skipped if those are absent.
"""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from gwtwin import areas as A
from gwtwin.config import ARTIFACTS, PROCESSED, ROOT

HAVE_DATA = (PROCESSED / "wells.csv").exists() and (ARTIFACTS / "deploy_model.pkl").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="processed data / deployed model not present")


def _toy():
    return pd.DataFrame({
        "well_id": list("abcdefg"),
        "aquifer": ["deep", "deep", "deep", "shallow", "shallow", "deep", "deep"],
        "district": ["K", "K", "C", "K", "C", None, "C"],
        "taluk": ["T1", "T1", "T2", "T1", "T2", None, "T3"],
        "in_study": [True, True, True, True, True, False, True],
        "lat": 13.0, "lon": 78.0})


def test_membership_excludes_buffer_wells():
    m = A.membership(_toy())
    assert m.in_totals.tolist() == [True, True, True, True, True, False, True]
    assert pd.isna(m.loc[5, "taluk"])


def test_dug_wells_never_summarised_by_taluk():
    m = A.membership(_toy())
    rows = A.aggregate(m, {"v": np.arange(7.0)})
    assert not [r for r in rows if r["level"] == "taluk" and r["aquifer"] == "shallow"]
    assert all(r["indicative"] for r in rows if r["aquifer"] == "shallow")
    reg = next(r for r in rows if r["level"] == "region" and r["aquifer"] == "deep")
    assert reg["n_wells"] == 4                   # a, b, c, g — not the buffer well f
    assert reg["v"]["median"] == pytest.approx(np.median([0, 1, 2, 6]))


def test_low_data_flag():
    m = A.membership(_toy())
    idx = {a["name"]: a for a in A.area_index(m)}
    assert idx["T1"]["low_data"]                 # 2 bore wells < min_deep_wells


def test_lateral_kernel_shape():
    x = np.array([-50_000.0, -1000, 0, 1000, 50_000])
    g = A.lateral_kernel(x, 365, 2000.0)
    assert g[0] == pytest.approx(1, abs=1e-6) and g[-1] == pytest.approx(0, abs=1e-6)
    assert g[2] == pytest.approx(0.5)
    assert np.all(np.diff(g) <= 0)


@needs_data
def test_footprint_zones_and_weights():
    w = pd.read_csv(PROCESSED / "wells.csv")
    f = A.footprint(w, A.normalise_area({"level": "taluk", "name": "Malur"}))
    inside = f.zone == "inside"
    assert inside.sum() == (w.taluk == "Malur").sum()
    assert (f.loc[~inside, ["w_rain", "w_pump"]] == 0).all().all()
    assert (f.loc[f.zone == "outside", "w_tank"] == 0).all()
    assert f.loc[f.zone == "buffer", "edge_km"].between(0, 3).all()
    assert f.w_tank.between(0, 1).all()


@needs_data
def test_area_scenario_has_no_effect_beyond_buffer_and_matches_inside():
    from gwtwin.service import TwinService

    svc = TwinService()
    r_all = svc.scenario(-20, 20, 50, horizon=2, agro=False)
    r_tal = svc.scenario(-20, 20, 50, horizon=2, agro=False, area={"level": "taluk", "name": "Kolar"})
    a = pd.DataFrame(r_all["wells"]).set_index("well_id")
    t = pd.DataFrame(r_tal["wells"]).set_index("well_id")
    out = t.zone == "outside"
    assert out.sum() > 300
    assert np.allclose(t.loc[out, "delta_m"], 0.0)
    buf = t.zone == "buffer"
    assert np.allclose(t.loc[buf, "rain_m"], 0) and np.allclose(t.loc[buf, "pump_m"], 0)
    ins = (t.zone == "inside") & (t.tank_weight == 1.0)
    assert np.allclose(t.loc[ins, "delta_m"], a.loc[ins[ins].index, "delta_m"], atol=1e-6)
    # pumping range brackets the central value
    assert (t.delta_lo_m <= t.delta_m + 1e-9).all() and (t.delta_hi_m >= t.delta_m - 1e-9).all()


@needs_data
def test_area_none_reproduces_frozen_forecast():
    from gwtwin.service import TwinService

    frozen = ROOT / "forecasts" / "frozen" / "2026-10-30" / "wells.json"
    if not frozen.exists():
        pytest.skip("no frozen forecast")
    svc = TwinService()
    fc = svc.forecast_cached(4)
    now = svc.nowcast()["depths"][-1]
    got = pd.Series(fc["median"][-1] - now, index=svc.twin.wid)
    w = pd.DataFrame(json.loads(frozen.read_text())).set_index("well_id")
    assert np.nanmax(np.abs(got.reindex(w.index) - w.forecast_change_m)) < 1e-3


def test_freeze_refuses_overwrite_and_detects_tampering(tmp_path, monkeypatch):
    from gwtwin import freeze as F

    src = tmp_path / "web"
    src.mkdir()
    (src / "meta.json").write_text(json.dumps({"now_round": "2026-11-05", "forcing_observed_until": "2026-10-26"}))
    (src / "wells.json").write_text(json.dumps([{"well_id": "x", "forecast": [{"date": "2027-01-05"}]}]))
    monkeypatch.setattr(F, "FROZEN", tmp_path / "frozen")
    out = F.freeze("t1", src=src)
    assert F.verify(out)
    with pytest.raises(SystemExit):
        F.freeze("t1", src=src)
    (out / "wells.json").write_text("[]")
    assert not F.verify(out)
    shutil.rmtree(tmp_path / "frozen")
