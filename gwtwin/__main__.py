"""Command line entry point.

    python -m gwtwin download [--only groundwater imd ...]   fetch raw data (skips what exists)
    python -m gwtwin build                                    clean wells, climate, static, panel
    python -m gwtwin evaluate [--no-gnn] [--no-sparse]        train + compare + stress-test models
    python -m gwtwin deploy                                   train deployed model, K.C. Valley test, export
    python -m gwtwin serve [--port 8000]                      start the API + dashboard
    python -m gwtwin all                                      build -> evaluate -> deploy (data must exist)
    python -m gwtwin freeze [--tag 2026-10-30]                dated, hashed copy of the current forecast
    python -m gwtwin score [--frozen DIR]                     score a frozen forecast against newer readings
    python -m gwtwin spatial                                  evidence figures: residual correlation, K.C. Valley by distance
    python -m gwtwin areas                                    area-level skill and rolling conformal bands (reports/area_eval.json)
"""
from __future__ import annotations

import argparse
import sys
import time


def build():
    from .data import climate, panel, static, wells

    t = time.time()
    print("== wells"); wells.build()
    from .config import INTERIM, RAW

    if (RAW / "imd" / "rain").exists():
        print("== climate"); climate.build()
    elif (INTERIM / "climate_daily.parquet").exists():
        print("== climate: raw IMD files not present, reusing data/interim/climate_daily.parquet")
    else:
        raise SystemExit("No climate data. Run `python -m gwtwin download --only imd` first.")
    print("== static features"); static.build_static(); static.build_water_yearly()
    print("== panel"); panel.build()
    print(f"build done in {time.time() - t:.0f}s")


def evaluate(gnn=True, sparse=True):
    from . import evaluate as ev

    ev.run(use_gnn=gnn, sparse=sparse)


def deploy():
    from . import area_eval, kcvalley, service, spatial
    from .export import export_all

    service.train_deploy()
    import pickle

    import pandas as pd

    from .config import ARTIFACTS, PROCESSED
    from .twin import Twin

    ev = pickle.load(open(ARTIFACTS / "eval_models.pkl", "rb"))
    twin = Twin(pd.read_parquet(PROCESSED / "panel.parquet"), pd.read_csv(PROCESSED / "edges.csv"), ev["hybrid"])
    print("== K.C. Valley natural experiment"); kcvalley.run(twin)
    print("== area-level skill and rolling bands"); area_eval.run()
    print("== spatial evidence"); spatial.run()
    print("== export"); export_all()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gwtwin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download"); d.add_argument("--only", nargs="*")
    sub.add_parser("build")
    e = sub.add_parser("evaluate"); e.add_argument("--no-gnn", action="store_true"); e.add_argument("--no-sparse", action="store_true")
    sub.add_parser("deploy")
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8000); s.add_argument("--host", default="127.0.0.1")
    fz = sub.add_parser("freeze"); fz.add_argument("--tag")
    sc = sub.add_parser("score"); sc.add_argument("--frozen")
    sub.add_parser("spatial")
    sub.add_parser("areas")
    a = sub.add_parser("all"); a.add_argument("--no-gnn", action="store_true"); a.add_argument("--no-sparse", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "download":
        from .data.download import run
        run(args.only)
    elif args.cmd == "build":
        build()
    elif args.cmd == "evaluate":
        evaluate(not args.no_gnn, not args.no_sparse)
    elif args.cmd == "deploy":
        deploy()
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("api.main:app", host=args.host, port=args.port, reload=False)
    elif args.cmd == "freeze":
        from .freeze import freeze
        freeze(args.tag)
    elif args.cmd == "score":
        from pathlib import Path

        from .freeze import score
        score(Path(args.frozen) if args.frozen else None)
    elif args.cmd == "areas":
        from .area_eval import run as areas_run
        areas_run()
    elif args.cmd == "spatial":
        from .spatial import run as spatial_run
        spatial_run()
    elif args.cmd == "all":
        build(); evaluate(not args.no_gnn, not args.no_sparse); deploy()


if __name__ == "__main__":
    sys.exit(main())
