"""Freeze a forecast before the readings it predicts exist, then score it later.

    python -m gwtwin freeze            copy artifacts/web/{wells,meta,wells_detail}.json into
                                       forecasts/frozen/<today>/ with SHA-256 hashes (MANIFEST.json)
    python -m gwtwin score [--frozen forecasts/frozen/2026-10-30]
                                       compare a frozen forecast with readings that arrived later

Why: the Nov 2026 round and the 2027 rounds have not been measured yet. A forecast that is
dated and hashed before they appear is an out-of-sample test that nobody can call leaky.
Commit the frozen folder (or e-mail the MANIFEST hash to someone) so the date is provable.

What is scored
  * nowcast_depth   -> the Nov 2026 round (the frozen file's "now_round")
  * forecast[h]     -> the four rounds Jan 2027 ... Nov 2027
Per well: error of the predicted depth and of the predicted CHANGE from the last real reading,
against no-change (= last real reading). Per taluk / district / region (study wells only):
median change, the same comparison on the area medians, and coverage of the 80 % well bands.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACTS, PROCESSED, ROOT

FROZEN = ROOT / "forecasts" / "frozen"
FILES = ("wells.json", "meta.json", "wells_detail.json")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze(tag: str | None = None, src: Path | None = None) -> Path:
    src = src or (ARTIFACTS / "web")
    tag = tag or date.today().isoformat()
    out = FROZEN / tag
    if out.exists():
        raise SystemExit(f"{out} already exists; frozen forecasts are never overwritten. Use another tag.")
    out.mkdir(parents=True)
    meta = json.loads((src / "meta.json").read_text())
    man = {"frozen_on": tag, "source_dir": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
           "now_round": meta.get("now_round"), "assimilation_start": meta.get("assimilation_start"),
           "forcing_observed_until": meta.get("forcing_observed_until"),
           "trained_through": meta.get("trained_through"), "files": {}}
    for name in FILES:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
            man["files"][name] = {"sha256": sha256(out / name), "bytes": (out / name).stat().st_size}
    model = ARTIFACTS / "deploy_model.pkl"
    if model.exists():          # the model itself is not copied (large); its hash pins which one made the forecast
        man["deploy_model_sha256"] = sha256(model)
    wells = json.loads((out / "wells.json").read_text())
    targets = sorted({f["date"] for w in wells for f in w["forecast"]})
    man["targets"] = {"nowcast": meta.get("now_round"), "forecast": targets}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(f"frozen forecast -> {out}")
    print(f"   wells.json sha256 {man['files']['wells.json']['sha256']}")
    return out


def verify(folder: Path) -> bool:
    man = json.loads((folder / "MANIFEST.json").read_text())
    ok = True
    for name, info in man["files"].items():
        h = sha256(folder / name)
        if h != info["sha256"]:
            print(f"   HASH MISMATCH {name}: {h} != {info['sha256']}")
            ok = False
    return ok


def _readings_after(cutoff: str) -> pd.DataFrame:
    """Round values newer than the frozen forecast's training data (from data/processed/rounds.csv)."""
    r = pd.read_csv(PROCESSED / "rounds.csv", parse_dates=["date"])
    return r[r.date > pd.Timestamp(cutoff)]


def score(folder: Path | None = None) -> dict:
    folder = folder or max(FROZEN.iterdir())
    if not verify(folder):
        raise SystemExit("frozen files were modified after freezing; refusing to score")
    man = json.loads((folder / "MANIFEST.json").read_text())
    wells = pd.DataFrame(json.loads((folder / "wells.json").read_text()))
    targets = {man["targets"]["nowcast"]: "nowcast"} | {d: f"h{i + 1}" for i, d in enumerate(man["targets"]["forecast"])}
    new = _readings_after(man["forcing_observed_until"])
    new["round"] = [d.strftime("%Y-%m-%d") for d in new.date]
    new = new[new["round"].isin(targets)]
    if new.empty:
        msg = (f"no readings yet for {', '.join(targets)}. Ingest new data (README: 'Updating the twin'), "
               "run `python -m gwtwin build`, then score again. Do NOT re-run deploy first if you want "
               "the frozen file, not a refreshed one, to be scored (scoring reads only the frozen folder).")
        print(msg)
        return {"status": "waiting", "message": msg}
    rows = []
    for w in wells.itertuples():
        pred = {man["targets"]["nowcast"]: (w.nowcast_depth, None, None)}
        for f in w.forecast:
            pred[f["date"]] = (f["median"], f["lo80"], f["hi80"])
        for o in new[new.well_id == w.well_id].itertuples():
            p, lo, hi = pred.get(o.round, (None, None, None))
            if p is None or w.last_obs_depth is None or not np.isfinite(w.last_obs_depth):
                continue
            rows.append({"well_id": w.well_id, "aquifer": w.aquifer, "district": w.district, "taluk": w.taluk,
                         "round": o.round, "target": targets[o.round], "obs": o.depth_m, "pred": p,
                         "last_obs": w.last_obs_depth, "lo80": lo, "hi80": hi})
    d = pd.DataFrame(rows)
    if d.empty:
        return {"status": "waiting", "message": "new readings exist but none match frozen wells"}
    d["dh_obs"], d["dh_pred"] = d.obs - d.last_obs, d.pred - d.last_obs
    d["in80"] = np.where(d.lo80.notna(), (d.obs >= d.lo80) & (d.obs <= d.hi80), np.nan)

    def skill(g):
        mae_m, mae_p = np.mean(np.abs(g.dh_obs - g.dh_pred)), np.mean(np.abs(g.dh_obs))
        return {"n": int(len(g)), "MAE_model_m": float(mae_m), "MAE_no_change_m": float(mae_p),
                "skill_vs_no_change": float(1 - mae_m / mae_p) if mae_p > 0 else None,
                "coverage80": float(np.nanmean(g.in80)) if g.in80.notna().any() else None}

    out = {"status": "scored", "frozen": man["frozen_on"], "wells": {}, "taluks": [], "districts": []}
    for (aq, t), g in d.groupby(["aquifer", "target"]):
        out["wells"][f"{aq}/{t}"] = skill(g)
    st = d[d.taluk.notna()]
    for (aq, t, tal), g in st[st.aquifer == "deep"].groupby(["aquifer", "target", "taluk"]):
        out["taluks"].append({"aquifer": aq, "target": t, "taluk": tal, "n_wells": int(len(g)),
                              "median_dh_obs": float(g.dh_obs.median()), "median_dh_pred": float(g.dh_pred.median())})
    for (aq, t, dist), g in st.groupby(["aquifer", "target", "district"]):
        out["districts"].append({"aquifer": aq, "target": t, "district": dist, "n_wells": int(len(g)),
                                 "median_dh_obs": float(g.dh_obs.median()), "median_dh_pred": float(g.dh_pred.median())})
    # headline: skill of the taluk medians against no change, per target round (bore wells)
    tl = pd.DataFrame(out["taluks"])
    out["taluk_skill"] = {}
    for t, g in (tl.groupby("target") if len(tl) else []):
        mae, mae0 = np.mean(np.abs(g.median_dh_obs - g.median_dh_pred)), np.mean(np.abs(g.median_dh_obs))
        out["taluk_skill"][t] = {"n_taluks": int(len(g)), "skill_vs_no_change": float(1 - mae / mae0) if mae0 > 0 else None,
                                 "corr": float(np.corrcoef(g.median_dh_obs, g.median_dh_pred)[0, 1]) if len(g) > 2 else None}
    (folder / "SCORE.json").write_text(json.dumps(out, indent=1))
    d.to_csv(folder / "score_rows.csv", index=False)
    print(json.dumps({"wells": out["wells"], "taluk_skill": out["taluk_skill"]}, indent=1))
    return out
