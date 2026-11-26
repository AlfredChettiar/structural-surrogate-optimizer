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
            "autogluon": _ag_version(), "n_jobs": fc.N_JOBS}


def _ag_version() -> str:
    import subprocess
    from src.framed import automl
    if not automl.available():
        return "not installed"
    r = subprocess.run([str(automl.AG_PYTHON), "-c", "import autogluon.tabular as a; print(a.__version__)"],
                       capture_output=True, text=True)
    return f"{r.stdout.strip()} (separate venv .venv-ag)"


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


def corrected_ttest(a: np.ndarray, b: np.ndarray, test_train_ratio: float = 0.25) -> dict:
    """Nadeau-Bengio corrected resampled t-test (one-sided, a > b) for repeated k-fold CV scores."""
    from scipy.stats import t as student_t
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = np.var(d, ddof=1)
    tstat = d.mean() / np.sqrt((1 / J + test_train_ratio) * var) if var > 0 else (
        np.inf * np.sign(d.mean()) if d.mean() != 0 else 0.0)
    return {"n_folds": J, "delta": float(d.mean()), "wins": int((d > 0).sum()), "t": float(tstat),
            "p_value": float(student_t.sf(tstat, J - 1))}


def paired_tests(ours: dict, paper: dict) -> dict:
    """Corrected resampled t-test on per-fold R2 against each reproduced paper model (identical folds), plus the
    plain difference to the published Table-4 value (descriptive only: different, inferred protocol and data)."""
    from src.framed import paper_reference as ref
    out = {}
    for o, ro in ours.items():
        a = np.array(ro["fold_scores"]["r2"])
        row = {"vs_published_delta": {p: float(a.mean() - v["r2"]) for p, v in ref.TABLE4.items()}}
        for p, rp in paper.items():
            b = np.array(rp["fold_scores"]["r2"])
            n = min(len(a), len(b))
            row[p] = corrected_ttest(a[:n], b[:n])
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
    for k, r in res.items():
        r.setdefault("fast", args.fast if k in keys else None)
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
    from src.framed.optimize import run_study, typical_nn_distance
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
        for robust in (False, True):
            r = run_study(sur, X, Y, mat, generations=gens, robust=robust)
            key = f"{mat} · {'robust (95 % lower bound)' if robust else 'nominal (mean)'}"
            studies[key] = {"material": mat, "robust": robust, "algorithms": r["algorithms"],
                            "table": r["table"].to_dict(orient="records")}
            fronts.append(r["front"])
            print(key, r["table"][["design", "mass_kg", "dropout_y_mm", "sf1_lo", "sf3_lo", "feasible",
                                    "nearest_distance"]].round(3).to_string(), sep="\n")
    pd.concat(fronts).to_csv(v1.DATA_DIR / "framed_pareto.csv", index=False)
    _save("optimize", {"surrogate": pick, "studies": studies, "fast": args.fast,
                       "reference_index": 3853, "typical_nn_distance": typical_nn_distance(X),
                       "reference_fe": {k: float(Y.loc[3853, k]) for k in ("Model Mass", "Sim 1 Safety Factor",
                                                                          "Sim 3 Safety Factor", "Sim 1 Dropout Y Disp.")}, "verification": "surrogate 95 % CI + nearest simulated design; "
                                                                "no FE re-check"})


def stage_report(args):
    from src.framed.data import load_framed
    from src.framed.report import build_report
    print(build_report(data=load_framed()))


def stage_figures(args):
    """Static PNGs of the report figures for docs/FRAMED_PROJECT.md (needs kaleido + a Chrome/Chromium)."""
    import pandas as pd
    from src import config as v1
    from src.framed import figures as F
    from src.framed.data import load_framed
    from src.framed.paper_stats import stats_comparison
    out = v1.ROOT / "docs" / "framed" / "img"
    out.mkdir(parents=True, exist_ok=True)
    X, Y = load_framed()
    repro, ours, opt = load_artifact("reproduction"), load_artifact("ours"), load_artifact("optimize")
    figs = {"01_materials": F.fig_material_breakdown(X), "02_claims": F.fig_check_status(stats_comparison(X, Y),
            "Dataset claims: deviation from the paper (grey = context only)"),
            "03_performance_pairs": F.fig_performance_pairs(Y), "04_correlation": F.fig_objective_correlation(Y),
            "05_physical_validation": F.fig_physical_validation()}
    if repro:
        t = pd.DataFrame(repro["table"]).set_index("model")
        figs["06_table4"] = F.fig_table4_comparison(t)
        with np.load(fc.MODELS_DIR / "reproduction_oof.npz") as z:
            figs["07_parity_paper_automl"] = F.fig_parity_grid(Y, z["autogluon"] if "autogluon" in z.files else
                                                                z["xgboost"], "Paper's best model (reproduced), out-of-fold")
    if ours:
        with np.load(fc.MODELS_DIR / "ours_oof.npz") as z:
            figs["08_parity_ours"] = F.fig_parity_grid(Y, z[ours["best"]], f"{F.MODEL_LABELS[ours['best']]}, out-of-fold")
            figs["09_parity_deep_ensemble"] = F.fig_parity_grid(Y, z["deep_ensemble_phys"],
                                                                 "Deep ensemble + physics, out-of-fold")
    if opt:
        fr = pd.read_csv(v1.DATA_DIR / "framed_pareto.csv")
        for i, (key, st) in enumerate(opt["studies"].items()):
            sel = fr[(fr["material"] == st["material"]) & (fr["mode"] == ("robust" if st["robust"] else "nominal"))]
            if len(sel):
                figs[f"1{i}_pareto_{st['material'].lower()}_{'robust' if st['robust'] else 'nominal'}"] = \
                    F.fig_framed_pareto(sel, pd.DataFrame(st["table"]), key)
    for name, fig in figs.items():
        w = 1200 if name.startswith(("03", "07", "08", "09", "04")) else 1000
        fig.write_image(out / f"{name}.png", width=w, height=fig.layout.height or 450, scale=1.5)
        print(out / f"{name}.png")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--models", default="", help="comma list; results merge into the existing artefact")
    args = ap.parse_args(argv)
    stages = {"fetch": stage_fetch, "stats": stage_stats, "baselines": stage_baselines, "ours": stage_ours, "optimize": stage_optimize, "report": stage_report, "figures": stage_figures}
    order = ["fetch", "stats", "baselines", "ours", "optimize", "report"]  # "figures" is opt-in (needs kaleido)
    for name in (order if args.stage == "all" else [args.stage]):
        print(f"=== {name}", flush=True)
        stages[name](args)


if __name__ == "__main__":
    main()
