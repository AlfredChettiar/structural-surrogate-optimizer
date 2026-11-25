"""FRAMED benchmark pipeline. python -m src.framed.run --stage {fetch,stats,baselines,ours,optimize,report,all} [--fast]"""
from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import psutil

from src.framed import config as fc


def _save(name: str, obj: dict) -> None:
    fc.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (fc.MODELS_DIR / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float))


def load_artifact(name: str):
    p = fc.MODELS_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def env_info() -> dict:
    import sklearn, torch, xgboost
    from src.framed import automl
    vm, du = psutil.virtual_memory(), psutil.disk_usage(str(fc.MODELS_DIR.parent))
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")),
               platform.processor()) if platform.system() == "Linux" else platform.processor()
    return {"cpu": cpu, "cores": psutil.cpu_count(),
            "ram_gb": round(vm.total / 1e9, 1), "disk_free_gb": round(du.free / 1e9, 1),
            "python": platform.python_version(), "os": platform.platform(),
            "sklearn": sklearn.__version__, "torch": torch.__version__, "xgboost": xgboost.__version__,
            "autogluon": automl.available(), "n_jobs": fc.N_JOBS}


def stage_fetch(args):
    from src.framed.data import fetch_framed
    print({k: str(v) for k, v in fetch_framed().items()})
    _save("env", env_info())


def stage_stats(args):
    from src.framed.data import load_framed
    from src.framed.paper_stats import stats_comparison
    df = stats_comparison(*load_framed())
    _save("stats", {"rows": df.to_dict(orient="records")})
    print(df.round(3).to_string())


def _cv_all(models: dict, X, Y, folds, tag: str, keys=None) -> dict:
    out, oofs = {}, {}
    for k in keys or models:
        t0 = time.perf_counter()
        n_jobs = fc.N_JOBS  # AutoGluon workers use 1 CPU each
        f = folds[:5] if k.startswith("autogluon") else folds  # AutoGluon bags internally; 1x5 bounds run time
        r = cross_validate(models[k], X, Y, f, n_jobs=n_jobs)
        oofs[k] = r.pop("oof_pred")
        r["wall_seconds"] = time.perf_counter() - t0
        out[k] = r
        print(f"{tag} {k}: R2 {r['r2']['mean']:.3f}±{r['r2']['std']:.3f}  MAE {r['mae']['mean']:.3f}  "
              f"MSE {r['mse']['mean']:.3f}  ({r['wall_seconds']:.0f}s, {r['n_folds']} folds)", flush=True)
    path = fc.MODELS_DIR / f"{tag}_oof.npz"
    if path.exists():  # keep OOF predictions of models not re-run
        with np.load(path) as old:
            oofs = {**{k: old[k] for k in old.files}, **oofs}
    np.savez(path, **oofs)
    return out


from src.framed.baselines import cross_validate  # noqa: E402


def _folds(n, fast):
    from src.framed.data import fold_indices
    return fold_indices(n, 5, 1 if fast else 10)


def stage_baselines(args):
    from src.framed.baselines import paper_models, ranking_preserved, reproduction_table
    from src.framed.data import load_framed
    X, Y = load_framed()
    fc.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = paper_models()
    keys = args.models.split(",") if args.models else list(models)
    prev = (load_artifact("reproduction") or {}).get("models", {})
    res = {**prev, **_cv_all(models, X, Y.values, _folds(len(X), args.fast), "reproduction", keys)}
    table = reproduction_table(res)
    _save("reproduction", {"models": res, "table": table.reset_index().to_dict(orient="records"),
                           "ranking_preserved": ranking_preserved(table), "fast": args.fast,
                           "protocol": "per-fold StandardScaler on X and Y; uniform mean over 10 standardised "
                                       "targets; 10x5 KFold (AutoGluon 1x5)"})
    print(table.round(3).to_string())


def paired_tests(ours: dict, paper: dict) -> dict:
    """Wilcoxon signed-rank on per-fold R2 (same folds), plus one-sample against the published Table-4 value."""
    from scipy.stats import wilcoxon
    from src.framed import paper_reference as ref
    out = {}
    for o, ro in ours.items():
        a = np.array(ro["fold_scores"]["r2"])
        row = {"vs_published": {}}
        for p, val in ref.TABLE4.items():
            row["vs_published"][p] = {"delta": float(a.mean() - val["r2"]),
                                      "p_value": float(wilcoxon(a - val["r2"], alternative="greater").pvalue)}
        for p, rp in paper.items():
            b = np.array(rp["fold_scores"]["r2"])
            n = min(len(a), len(b))
            row[p] = {"n_folds": n, "delta": float((a[:n] - b[:n]).mean()), "wins": int((a[:n] > b[:n]).sum()),
                      "p_value": float(wilcoxon(a[:n], b[:n], alternative="greater").pvalue)}
        out[o] = row
    return out


def stage_ours(args):
    from src import config as v1
    from src.framed.data import load_framed
    from src.framed.surrogates import interval_coverage, latency_report, our_models
    X, Y = load_framed()
    models = our_models()
    keys = args.models.split(",") if args.models else list(models)
    folds = _folds(len(X), args.fast)
    prev = (load_artifact("ours") or {}).get("models", {})
    res = {**prev, **_cv_all(models, X, Y.values, folds, "ours", keys)}
    cov = (load_artifact("ours") or {}).get("coverage", {})
    for k in keys:
        if getattr(models[k](), "supports_uncertainty", False):
            cov[k] = interval_coverage(models[k], X, Y.values, folds[:5])
    fitted = {k: models[k]().fit(X, Y.values) for k in keys if not k.startswith("autogluon")}
    lat = {**(load_artifact("ours") or {}).get("latency", {}).get("models", {}),
           **latency_report(fitted, X, v1.NOMINAL_INDUSTRIAL_FE_SECONDS)["models"]}
    latency = {**latency_report({}, X, v1.NOMINAL_INDUSTRIAL_FE_SECONDS), "models": lat}
    paper = (load_artifact("reproduction") or {}).get("models", {})
    best = max(res, key=lambda k: res[k]["r2"]["mean"])
    _save("ours", {"models": res, "coverage": cov, "latency": latency, "paired_tests": paired_tests(res, paper),
                   "best": best, "fast": args.fast})
    print("best:", best, round(res[best]["r2"]["mean"], 3))


def stage_optimize(args):
    import pandas as pd
    from src import config as v1
    from src.framed.data import load_framed
    from src.framed.optimize import run_study
    from src.framed.surrogates import our_models
    X, Y = load_framed()
    ours = load_artifact("ours") or {"models": {}}
    models = our_models()
    uq = [k for k in models if getattr(models[k](), "supports_uncertainty", False)]
    pick = max(uq, key=lambda k: ours["models"].get(k, {}).get("r2", {}).get("mean", -1))
    sur = models[pick]().fit(X, Y.values)
    gens = 15 if args.fast else 60
    studies, fronts = {}, []
    for mat in ("Titanium", "Steel", "Aluminum"):
        r = run_study(sur, X, Y, mat, generations=gens)
        studies[mat] = {"algorithms": r["algorithms"], "table": r["table"].to_dict(orient="records")}
        fronts.append(r["front"])
        print(mat, r["table"][["design", "mass_kg", "dropout_y_mm", "sf1_lo", "sf3_lo", "feasible",
                                "nearest_distance"]].round(3).to_string(), sep="\n")
    pd.concat(fronts).to_csv(v1.DATA_DIR / "framed_pareto.csv", index=False)
    _save("optimize", {"surrogate": pick, "studies": studies, "fast": args.fast,
                       "reference_index": 3853, "verification": "surrogate 95 % CI + nearest simulated design; "
                                                                "no FE re-check"})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--models", default="", help="comma list; results merge into the existing artefact")
    args = ap.parse_args(argv)
    stages = {"fetch": stage_fetch, "stats": stage_stats, "baselines": stage_baselines, "ours": stage_ours, "optimize": stage_optimize}
    for name in (stages if args.stage == "all" else [args.stage]):
        print(f"=== {name}", flush=True)
        stages[name](args)


if __name__ == "__main__":
    main()
