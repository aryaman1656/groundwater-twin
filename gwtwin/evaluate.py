"""Train, compare and stress-test all models. Writes reports/ and artifacts/.

Experiments (each maps to a hypothesis in the project spec):
  1. Model comparison on held-out years (test = 2023+)              -> H1, baseline-first rule
  2. Feature ablation of the hybrid model                          -> H2 (which information matters)
  3. Sparse-monitoring test: hide 10/20/30% of training readings    -> H3 (does the graph help more?)
  4. Conformal coverage on test years, per aquifer & horizon       -> H5
  5. Multi-step recursive forecasts (1–4 rounds ahead)             -> forecast horizons
Scenario sanity (H4) and K.C. Valley validation live in scenarios.py / kcvalley.py.
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd

from .config import ARTIFACTS, PROCESSED, REPORTS, cfg
from .data import panel as panel_mod
from .models.conformal import Conformal
from .models.core import (FEATURES, FORCING, GRAPH, STATIC, Hybrid, Persistence, Physics, SeasonalMean, XGB,
                          metrics, prep)

LABEL_COLS = ["depth_m", "dh", "rain_int", "et0_int"]


def labelled(p: pd.DataFrame) -> pd.DataFrame:
    return prep(p.dropna(subset=LABEL_COLS))


def compare_models(lab: pd.DataFrame, use_gnn: bool = True, panel=None, edges=None):
    tr, ca, te = (lab[lab.split == s] for s in ("train", "calib", "test"))
    models = [Persistence(), SeasonalMean(), XGB(), Physics(), Hybrid()]
    preds, rows, fitted = {}, [], {}
    for m in models:
        t0 = time.time()
        m.fit(tr)
        fitted[m.name] = m
        for nm, d in (("calib", ca), ("test", te)):
            yhat = m.predict(d)
            preds[(m.name, nm)] = yhat
            for aq in ("shallow", "deep"):
                idx = np.asarray(d.aquifer == aq)
                if idx.sum():
                    rows.append({"model": m.name, "split": nm, "aquifer": aq, **metrics(d.dh[idx], yhat[idx])})
        print(f"   {m.name:<34s} {time.time() - t0:5.1f}s")
    gnn = None
    if use_gnn:
        try:
            from .models.gnn import STGNN, build_tensors

            t0 = time.time()
            T = build_tensors(panel, edges)
            gnn = STGNN().fit(T)
            fitted[gnn.name] = gnn
            for nm, d in (("calib", ca), ("test", te)):
                yhat = gnn.predict_rows(T, d)
                preds[(gnn.name, nm)] = yhat
                for aq in ("shallow", "deep"):
                    idx = np.asarray(d.aquifer == aq)
                    if idx.sum():
                        rows.append({"model": gnn.name, "split": nm, "aquifer": aq, **metrics(d.dh[idx], yhat[idx])})
            print(f"   {gnn.name:<34s} {time.time() - t0:5.1f}s")
        except Exception as e:  # noqa: BLE001
            print("   ST-GNN skipped:", e)
    res = pd.DataFrame(rows)
    pers = res[res.model == "Persistence"].set_index(["split", "aquifer"]).MAE
    res["skill_vs_persistence"] = [1 - r.MAE / pers.get((r.split, r.aquifer), np.nan) for r in res.itertuples()]
    return res, preds, fitted


def ablation(lab: pd.DataFrame) -> pd.DataFrame:
    tr, te = lab[lab.split == "train"], lab[lab.split == "test"]
    variants = {
        "full hybrid": FEATURES,
        "− graph (neighbour) features": [f for f in FEATURES if f not in GRAPH],
        "− rainfall/ET/tank forcing": [f for f in FEATURES if f not in FORCING],
        "− tank water only": [f for f in FEATURES if not f.startswith("water_")],
        "− static (terrain/soil/land cover)": [f for f in FEATURES if f not in STATIC],
        "− past groundwater state": [f for f in FEATURES if f not in
                                     ("depth_m", "anom", "past_mean", "dh_prev", "dh_season_past", "frac_depth")],
    }
    rows = []
    for name, feats in variants.items():
        m = Hybrid(features=feats).fit(tr)
        yhat = m.predict(te)
        for aq in ("shallow", "deep"):
            idx = np.asarray(te.aquifer == aq)
            rows.append({"variant": name, "aquifer": aq, **metrics(te.dh[idx], yhat[idx])})
    rows_phys = Physics().fit(tr)
    yhat = rows_phys.predict(te)
    for aq in ("shallow", "deep"):
        idx = np.asarray(te.aquifer == aq)
        rows.append({"variant": "physics only (no ML)", "aquifer": aq, **metrics(te.dh[idx], yhat[idx])})
    return pd.DataFrame(rows)


def sparse_experiment(levels=(0.0, 0.1, 0.2, 0.3), seeds=(0, 1)) -> pd.DataFrame:
    """Hide a fraction of TRAINING-period readings, rebuild all features, retrain, test on 2022+."""
    rounds = pd.read_csv(PROCESSED / "rounds.csv")
    train_end = pd.Timestamp(cfg()["time"]["train_end"])
    rows = []
    for frac in levels:
        for seed in seeds if frac > 0 else (0,):
            rng = np.random.default_rng(seed)
            r = rounds.copy()
            in_train = pd.to_datetime(r.date) <= train_end
            drop = in_train & (rng.random(len(r)) < frac)
            p = panel_mod.build(rounds_override=r[~drop], save=False, verbose=False)
            lab = labelled(p)
            # always test on the SAME, fully observed test set
            full = labelled(pd.read_parquet(PROCESSED / "panel.parquet"))
            te = full[full.split == "test"]
            tr = lab[lab.split == "train"]
            for name, m in (("Hybrid (with graph)", Hybrid()),
                            ("Hybrid (no graph)", Hybrid(features=[f for f in FEATURES if f not in GRAPH])),
                            ("XGBoost (no physics)", XGB())):
                m.fit(tr)
                yhat = m.predict(te)
                for aq in ("shallow", "deep"):
                    idx = np.asarray(te.aquifer == aq)
                    rows.append({"hidden_frac": frac, "seed": seed, "model": name, "aquifer": aq,
                                 **metrics(te.dh[idx], yhat[idx])})
            print(f"   sparse {frac:.0%} seed {seed} done")
    return pd.DataFrame(rows)


def recursive_errors(twin, origins: list[int], H: int, lab_split: str) -> pd.DataFrame:
    """Run the twin from each origin round with observed forcing; collect h-step errors vs observations."""
    obs = twin.obs
    rows = []
    for k0 in origins:
        out = twin.rollout(k0, H)
        d0 = out["depths"][0]
        observed0 = obs[k0].to_numpy(float) if k0 in obs.columns else np.full(twin.n, np.nan)
        for h in range(1, H + 1):
            k = k0 + h
            if k not in obs.columns:
                continue
            y = obs[k].to_numpy(float)
            ok = np.isfinite(y) & np.isfinite(observed0)
            for i in np.where(ok)[0]:
                rows.append({"well_id": twin.wid[i], "aquifer": twin.aquifer[i], "k0": k0, "h": h,
                             "depth0": d0[i], "pred": out["depths"][h][i], "obs": y[i],
                             "scale_past": twin.wells.scale_past.iat[i], "split": lab_split})
    df = pd.DataFrame(rows)
    df["dh"] = df.obs - df.depth0
    df["dh_hat"] = df.pred - df.depth0
    return df


def run(use_gnn: bool = True, sparse: bool = True):
    from .twin import Twin
    from .data.wells import round_index

    t_all = time.time()
    p = pd.read_parquet(PROCESSED / "panel.parquet")
    edges = pd.read_csv(PROCESSED / "edges.csv")
    lab = labelled(p)
    print("1) model comparison")
    res, preds, fitted = compare_models(lab, use_gnn=use_gnn, panel=p, edges=edges)
    res.to_csv(REPORTS / "model_comparison.csv", index=False)
    print(res[res.split == "test"].pivot_table(index="model", columns="aquifer", values="MAE").round(3).to_string())

    hybrid = fitted["Hybrid (physics + monotone ML)"]
    ca, te = lab[lab.split == "calib"], lab[lab.split == "test"]
    print("2) conformal intervals (1-step)")
    cov_rows = []
    for name in [n for n in fitted if n not in ("Persistence",)]:
        conf = Conformal(cfg()["uncertainty"]["coverages"]).fit(ca, preds[(name, "calib")])
        rep = conf.coverage_report(te, preds[(name, "test")])
        rep.insert(0, "model", name)
        cov_rows.append(rep)
    cov = pd.concat(cov_rows)
    cov.to_csv(REPORTS / "coverage_1step.csv", index=False)
    print(cov[cov.model.str.startswith("Hybrid")].round(3).to_string(index=False))

    print("3) ablation")
    abl = ablation(lab)
    abl.to_csv(REPORTS / "ablation.csv", index=False)
    print(abl.pivot_table(index="variant", columns="aquifer", values="MAE").round(3).to_string())

    print("4) recursive multi-step forecasts + horizon-wise conformal")
    twin = Twin(p, edges, hybrid)
    t_end, c_end = pd.Timestamp(cfg()["time"]["train_end"]), pd.Timestamp(cfg()["time"]["calib_end"])
    cal_y = pd.Timestamp(cfg()["time"]["calib_end"]).year
    calib_origins = [k for k in range(round_index(cal_y - 1, 8), round_index(cal_y, 11) + 1)]
    test_origins = [round_index(y, m) for y in range(cal_y + 1, cal_y + 4) for m in (1, 5, 8, 11)]
    H = 4
    rc = recursive_errors(twin, calib_origins, H, "calib")
    tdates = pd.Series([twin_k_date(k + h) for k, h in zip(rc.k0, rc.h)])
    rc = rc[((tdates > t_end) & (tdates <= c_end)).to_numpy()]
    rt = recursive_errors(twin, test_origins, H, "test")
    conf_by_h, hrows = {}, []
    for h in range(1, H + 1):
        c_h = rc[rc.h == h]
        t_h = rt[rt.h == h]
        if len(c_h) < 30 or len(t_h) == 0:
            continue
        conf = Conformal(cfg()["uncertainty"]["coverages"]).fit(c_h, c_h.dh_hat.to_numpy())
        conf_by_h[h] = conf
        rep = conf.coverage_report(t_h, t_h.dh_hat.to_numpy())
        rep.insert(0, "horizon_rounds", h)
        for aq in ("shallow", "deep"):
            tt = t_h[t_h.aquifer == aq]
            if len(tt):
                mm = metrics(tt.obs, tt.pred)
                pm = metrics(tt.obs, tt.depth0)
                rep.loc[rep.aquifer == aq, "MAE_twin"] = mm["MAE"]
                rep.loc[rep.aquifer == aq, "MAE_persistence"] = pm["MAE"]
        hrows.append(rep)
    hz = pd.concat(hrows) if hrows else pd.DataFrame()
    hz.to_csv(REPORTS / "horizon_coverage.csv", index=False)
    print(hz.round(3).to_string(index=False))

    sp = None
    if sparse:
        print("5) sparse-monitoring experiment")
        sp = sparse_experiment()
        sp.to_csv(REPORTS / "sparse_experiment.csv", index=False)
        print(sp.groupby(["hidden_frac", "model", "aquifer"]).MAE.mean().unstack().round(3).to_string())

    # ---- save the evaluation-configuration model and summary
    with open(ARTIFACTS / "eval_models.pkl", "wb") as f:
        pickle.dump({"hybrid": hybrid, "conf_by_h": conf_by_h}, f)
    phys = hybrid.phys.summary()
    phys.to_csv(REPORTS / "physics_coefficients.csv", index=False)
    summary = {
        "test_metrics": res[res.split == "test"].to_dict("records"),
        "coverage_1step": cov.to_dict("records"),
        "horizon": hz.to_dict("records") if len(hz) else [],
        "ablation": abl.to_dict("records"),
        "sparse": sp.to_dict("records") if sp is not None else [],
        "physics": phys.to_dict("records"),
        "n_train": int((lab.split == "train").sum()), "n_calib": int((lab.split == "calib").sum()),
        "n_test": int((lab.split == "test").sum()),
        "runtime_s": round(time.time() - t_all, 1),
    }
    (REPORTS / "evaluation.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"evaluation done in {summary['runtime_s']} s")
    return summary


def twin_k_date(k):
    from .data.wells import index_to_round, round_date

    return round_date(*index_to_round(k))
