"""Benchmark every optimiser on the selected surrogate, verify fronts with the FE solver, select the best.

Each stochastic optimiser is repeated over ``config.OPT_SEEDS`` seeds; indicators are reported as
mean +/- std and the scorecard uses the means. Fronts, convergence curves and the headline design
come from the first seed.

Run:  .\\.venv\\Scripts\\python.exe -m src.optimizer
Outputs: models/optimizer_benchmark.json, data/pareto_fronts.csv, data/lhs_candidates.csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src import config
from src.guardrails import safety_factor
from src.optimizer import (ALGORITHM_LABELS, ALGORITHMS, ChassisProblem, knee_point, non_dominated_mask,
                           pareto_front, run_algorithm, screen_candidates)
from src.physics import frame_fe_batch
from src.surrogate_zoo import load_surrogate

BASELINE_GEOMETRY = {"tube_od_mm": 32.0, "wall_thickness_mm": 2.5, "gusset_thickness_mm": 4.0,
                     "bend_radius_mm": 60.0}
MAX_FE_VERIFY = 60
SCORE_WEIGHTS = {
    "hypervolume_fe_verified": 0.30,   # what survives contact with the physics
    "hypervolume": 0.20,               # surrogate-space front quality
    "igd": 0.15,                       # closeness/coverage vs. the combined reference front
    "fe_feasible_rate": 0.20,          # fraction of claimed-optimal designs that are truly feasible
    "evals_to_99pct_hv": 0.10,         # sample efficiency
    "wall_seconds": 0.05,
}
LOWER_IS_BETTER = {"igd", "evals_to_99pct_hv", "wall_seconds"}
SEED_METRICS = ["hypervolume", "hypervolume_fe_verified", "igd", "spacing", "evals_to_99pct_hv",
                "fe_feasible_rate", "fe_mass_mape_pct", "fe_stress_mape_pct", "wall_seconds",
                "n_evaluations", "n_feasible", "n_pareto"]


# ----------------------------------------------------------------------------- indicators
def hypervolume_2d(Fn: np.ndarray, ref=(1.1, 1.1)) -> float:
    """Exact 2-D hypervolume of normalised minimisation points w.r.t. ``ref``."""
    if len(Fn) == 0:
        return 0.0
    P = Fn[np.all(Fn < np.asarray(ref), axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[non_dominated_mask(P)]
    P = P[np.argsort(P[:, 0])]
    hv, prev_f2 = 0.0, ref[1]
    for f1, f2 in P:
        hv += (ref[0] - f1) * (prev_f2 - f2)
        prev_f2 = f2
    return float(hv)


def igd(front: np.ndarray, reference: np.ndarray) -> float:
    if len(front) == 0 or len(reference) == 0:
        return float("inf")
    d = np.linalg.norm(reference[:, None, :] - front[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def spacing(front: np.ndarray) -> float:
    """Schott's spacing: std of nearest-neighbour distances (0 = perfectly uniform)."""
    if len(front) < 3:
        return float("nan")
    d = np.linalg.norm(front[:, None, :] - front[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.std(d.min(axis=1)))


def convergence_curve(F: np.ndarray, cv: np.ndarray, normalise, n_points: int = 30):
    n = len(F)
    checkpoints = np.unique(np.clip(np.geomspace(10, max(n, 11), n_points).astype(int), 1, n))
    hv = []
    for c in checkpoints:
        idx = pareto_front(F[:c], cv[:c])
        hv.append(hypervolume_2d(normalise(F[:c][idx])) if len(idx) else 0.0)
    return checkpoints.tolist(), hv


# ----------------------------------------------------------------------------- one seed
def _algo_kwargs(seed: int, fast: bool) -> dict:
    return {
        "lhs_screening": {"seed": seed},
        "nsga2": {"seed": seed, **({"generations": 25} if fast else {})},
        "epsilon_constraint": {"seed": seed, **({"n_levels": 6, "maxiter": 15} if fast else {})},
        "weighted_sum_slsqp": {"seed": seed, **({"n_weights": 6, "n_starts": 2} if fast else {})},
        "parego": {"seed": seed, **({"budget": 80} if fast else {})},
    }


def _run_seed(surrogate, scenario: dict, seed: int, fast: bool):
    problem = ChassisProblem(surrogate, scenario)
    problem.objective_bounds()
    runs = {}
    kw = _algo_kwargs(seed, fast)
    for name in ALGORITHMS:
        secs = run_algorithm(problem, name, **kw[name])
        G, F, cv = problem.archive_for(name)
        idx = pareto_front(F, cv)
        runs[name] = {"G": G, "F": F, "cv": cv, "front_G": G[idx], "front_F": F[idx], "wall_seconds": secs}
    return problem, runs


def _fe_verify(problem: ChassisProblem, surrogate, Gp: np.ndarray, Fp: np.ndarray):
    """Re-solve up to MAX_FE_VERIFY evenly spread front designs with the FE model."""
    if len(Gp) > MAX_FE_VERIFY:
        order = np.argsort(Fp[:, 0])
        sel = order[np.unique(np.linspace(0, len(Gp) - 1, MAX_FE_VERIFY).astype(int))]
    else:
        sel = np.arange(len(Gp))
    if not len(sel):
        return sel, np.empty((0, 5)), np.zeros(0, dtype=bool), np.empty((0, 5))
    X = problem.full_inputs(Gp[sel])
    Yfe = frame_fe_batch(X)
    ok = np.all(problem.violations(Yfe, Gp[sel]) <= 0, axis=1)
    return sel, Yfe, ok, surrogate.predict(X)


# ----------------------------------------------------------------------------- benchmark
def run_benchmark(surrogate_name: str | None = None, scenario: dict | None = None, seed: int = config.SEED,
                  fast: bool = False, n_seeds: int = config.OPT_SEEDS, surrogate=None) -> dict:
    scenario = scenario or dict(config.DEFAULT_SCENARIO)
    surrogate = surrogate if surrogate is not None else load_surrogate(surrogate_name)
    seeds = [seed + k for k in range(max(1, n_seeds))]
    seed_runs = []
    for s in seeds:
        problem, runs = _run_seed(surrogate, scenario, s, fast)
        seed_runs.append((problem, runs))
        print(f"  seed {s}: " + ", ".join(f"{a} {r['wall_seconds']:.1f}s/{len(r['front_F'])} pts"
                                          for a, r in runs.items()))

    # common normalisation across every seed and algorithm
    all_front = np.vstack([r["front_F"] for _, runs in seed_runs for r in runs.values() if len(r["front_F"])])
    ref_front = all_front[non_dominated_mask(all_front)]
    ideal, nadir = ref_front.min(axis=0), all_front.max(axis=0)
    scale = np.where(nadir - ideal > 0, nadir - ideal, 1.0)

    def norm(F):
        return (F - ideal) / scale

    per_seed = {a: [] for a in ALGORITHMS}
    first_extra = {}
    for k, (problem, runs) in enumerate(seed_runs):
        for a in ALGORITHMS:
            r = runs[a]
            ckpt, hv_curve = convergence_curve(r["F"], r["cv"], norm)
            hv_final = hypervolume_2d(norm(r["front_F"]))
            reach = next((c for c, h in zip(ckpt, hv_curve) if h >= 0.99 * hv_final), ckpt[-1]) \
                if hv_final > 0 else ckpt[-1]
            sel, Yfe, ok, Ysur = _fe_verify(problem, surrogate, r["front_G"], r["front_F"])
            err = np.abs(Ysur[:, :2] - Yfe[:, :2]) / np.clip(Yfe[:, :2], 1e-9, None) * 100 if len(sel) else None
            per_seed[a].append({
                "seed": seeds[k],
                "hypervolume": hv_final,
                "hypervolume_fe_verified": hypervolume_2d(norm(Yfe[ok][:, :2])) if ok.any() else 0.0,
                "igd": igd(norm(r["front_F"]), norm(ref_front)),
                "spacing": spacing(norm(r["front_F"])),
                "evals_to_99pct_hv": int(reach),
                "fe_feasible_rate": float(ok.mean()) if len(sel) else 0.0,
                "fe_mass_mape_pct": float(err[:, 0].mean()) if err is not None else float("nan"),
                "fe_stress_mape_pct": float(err[:, 1].mean()) if err is not None else float("nan"),
                "wall_seconds": r["wall_seconds"],
                "n_evaluations": int(len(r["F"])),
                "n_feasible": int((r["cv"] <= 0).sum()),
                "n_pareto": int(len(r["front_F"])),
            })
            if k == 0:
                first_extra[a] = {"convergence": {"evaluations": ckpt, "hypervolume": hv_curve},
                                  "verify": (sel, Yfe, ok)}

    results = {}
    for a in ALGORITHMS:
        rows = per_seed[a]
        res = {"label": ALGORITHM_LABELS[a], "convergence": first_extra[a]["convergence"],
               "per_seed": rows, "std": {}}
        for m in SEED_METRICS:
            vals = np.array([r[m] for r in rows], dtype=float)
            finite = vals[np.isfinite(vals)]
            res[m] = float(finite.mean()) if len(finite) else float("nan")
            res["std"][m] = float(finite.std()) if len(finite) > 1 else 0.0
        results[a] = res

    # scorecard on seed means
    scores = {n: 0.0 for n in ALGORITHMS}
    for metric, w in SCORE_WEIGHTS.items():
        vals = np.array([results[n][metric] for n in ALGORITHMS], dtype=float)
        if not np.isfinite(vals).all():
            fill = np.nanmax(vals[np.isfinite(vals)]) if np.isfinite(vals).any() else 0.0
            vals = np.where(np.isfinite(vals), vals, fill)
        if metric in ("evals_to_99pct_hv", "wall_seconds"):
            vals = np.log10(np.clip(vals, 1e-3, None))
        span = vals.max() - vals.min()
        normed = np.ones_like(vals) if span == 0 else (vals - vals.min()) / span
        if metric in LOWER_IS_BETTER:
            normed = 1.0 - normed
        for n, v in zip(ALGORITHMS, normed):
            scores[n] += w * float(v)
            results[n].setdefault("score_breakdown", {})[metric] = round(w * float(v), 4)
    for n in ALGORITHMS:
        results[n]["score"] = scores[n]
    best = max(scores, key=scores.get)

    # Pareto rows (first seed) with FE verification columns
    problem0, runs0 = seed_runs[0]
    sy, dlf = scenario["yield_strength_mpa"], scenario["dynamic_load_factor"]
    pareto_rows = []
    for a in ALGORITHMS:
        Gp = runs0[a]["front_G"]
        if not len(Gp):
            continue
        Yp = surrogate.predict(problem0.full_inputs(Gp))
        sel, Yfe, ok = first_extra[a]["verify"]
        verified = dict(zip(sel.tolist(), range(len(sel))))
        for i in range(len(Gp)):
            row = {"algorithm": a, **dict(zip(config.GEOMETRY_FEATURES, Gp[i])),
                   "yield_strength_mpa": sy, "dynamic_load_factor": dlf,
                   **{t: Yp[i, j] for j, t in enumerate(config.TARGETS)},
                   "safety_factor": float(safety_factor(sy, Yp[i, 1])),
                   "d_over_t": float(Gp[i, 0] / Gp[i, 1]), "bend_ratio": float(Gp[i, 3] / Gp[i, 0]),
                   "fe_verified": i in verified}
            if i in verified:
                k = verified[i]
                row.update({f"fe_{t}": Yfe[k, j] for j, t in enumerate(config.TARGETS)})
                row["fe_safety_factor"] = float(safety_factor(sy, Yfe[k, 1]))
                row["fe_feasible"] = bool(ok[k])
            pareto_rows.append(row)

    headline = _headline(pd.DataFrame(pareto_rows), best, problem0, sy, dlf)
    return {
        "surrogate": surrogate_name or getattr(surrogate, "name", "surrogate"),
        "scenario": scenario,
        "constraints": problem0.constraints,
        "seeds": seeds,
        "normalisation": {"ideal": ideal.tolist(), "nadir": nadir.tolist(), "reference_point": [1.1, 1.1]},
        "score_weights": SCORE_WEIGHTS,
        "algorithms": results,
        "selected_optimizer": best,
        "headline": headline,
        "_pareto_rows": pareto_rows,
    }


def _headline(pdf: pd.DataFrame, best: str, problem: ChassisProblem, sy: float, dlf: float) -> dict:
    """Baseline vs. (a) lightest FE-feasible design at <= baseline stress and (b) the knee-point design."""
    base_g = np.array([list(BASELINE_GEOMETRY.values())])
    base_fe = frame_fe_batch(problem.full_inputs(base_g))[0]
    base_ok = bool(np.all(problem.violations(base_fe[None, :], base_g) <= 0))
    h = {"baseline_geometry": BASELINE_GEOMETRY, "baseline_fe": dict(zip(config.TARGETS, base_fe.tolist())),
         "baseline_feasible": base_ok}
    if pdf.empty or "fe_feasible" not in pdf:
        return h
    cand = pdf[(pdf.algorithm == best) & (pdf["fe_feasible"].fillna(False).astype(bool))]
    if cand.empty:
        return h
    same_stress = cand[cand["fe_peak_von_mises_mpa"] <= base_fe[1]]
    pick = (same_stress if len(same_stress) else cand).sort_values("fe_mass_kg").iloc[0]
    h.update({
        "optimised_geometry": {f: float(pick[f]) for f in config.GEOMETRY_FEATURES},
        "optimised_fe": {t: float(pick[f"fe_{t}"]) for t in config.TARGETS},
        "mass_saving_pct": float(100 * (base_fe[0] - pick["fe_mass_kg"]) / base_fe[0]),
        "stress_change_pct": float(100 * (pick["fe_peak_von_mises_mpa"] - base_fe[1]) / base_fe[1]),
        "comparison": "lightest FE-feasible design from the selected optimiser with stress <= baseline"
        if len(same_stress) else "lightest FE-feasible design from the selected optimiser",
    })
    F = cand[["fe_mass_kg", "fe_peak_von_mises_mpa"]].to_numpy()
    nd = non_dominated_mask(F)
    front = cand[nd]
    knee = front.iloc[knee_point(F[nd])]
    h["recommended"] = {
        "rule": "knee point of the FE-verified front (best-balanced mass/stress compromise)",
        "geometry": {f: float(knee[f]) for f in config.GEOMETRY_FEATURES},
        "fe": {t: float(knee[f"fe_{t}"]) for t in config.TARGETS},
        "safety_factor": float(sy / knee["fe_peak_von_mises_mpa"]),
        "mass_saving_pct": float(100 * (base_fe[0] - knee["fe_mass_kg"]) / base_fe[0]),
        "stress_change_pct": float(100 * (knee["fe_peak_von_mises_mpa"] - base_fe[1]) / base_fe[1]),
    }
    return h


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surrogate", default=None, help="defaults to models/model_selection.json")
    ap.add_argument("--fast", action="store_true", help="smaller budgets, 1 seed")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--seeds", type=int, default=None, help=f"number of repeats (default {config.OPT_SEEDS})")
    args = ap.parse_args(argv)
    name = args.surrogate or json.loads(config.MODEL_SELECTION_JSON.read_text())["selected_model"]
    n_seeds = args.seeds or (1 if args.fast else config.OPT_SEEDS)
    print(f"Optimiser benchmark on surrogate '{name}' | scenario {config.DEFAULT_SCENARIO} | {n_seeds} seed(s)")
    out = run_benchmark(name, seed=args.seed, fast=args.fast, n_seeds=n_seeds)
    rows = out.pop("_pareto_rows")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.PARETO_CSV, index=False)
    screen_candidates(load_surrogate(name), seed=args.seed).to_csv(config.CANDIDATES_CSV, index=False)
    config.OPTIMIZATION_JSON.write_text(json.dumps(out, indent=2, default=float))
    for a, r in out["algorithms"].items():
        print(f"  {a:20s} score {r['score']:.3f} | HV {r['hypervolume']:.3f} +/- {r['std']['hypervolume']:.3f} "
              f"| FE-feasible {r['fe_feasible_rate']:.0%}")
    print("Selected optimiser:", out["selected_optimizer"])
    h = out["headline"]
    if "mass_saving_pct" in h:
        print(f"Lightest at <= baseline stress: {h['mass_saving_pct']:.1f}% lighter (stress {h['stress_change_pct']:+.1f}%)")
    if "recommended" in h:
        r = h["recommended"]
        print(f"Recommended (knee): {r['mass_saving_pct']:.1f}% lighter, stress {r['stress_change_pct']:+.1f}%, "
              f"SF {r['safety_factor']:.2f}, geometry {r['geometry']}")


if __name__ == "__main__":
    main()
