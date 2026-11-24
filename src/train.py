"""Train, benchmark and select the surrogate model.

Run:  .\\.venv\\Scripts\\python.exe -m src.train            (full study, ~5-8 min on a laptop CPU)
      .\\.venv\\Scripts\\python.exe -m src.train --fast     (3-fold CV, shorter NN training)

Outputs (models/):
  <model>.joblib / <model>.pt     trained artifacts
  metrics.json                    R2, MAE, Max % error, latency, speedup, CV per model
  model_selection.json            selected model + decision trace
  evaluation_extras.json          learning curves, calibration, extrapolation, Sobol, PDP, NN history
  evaluation_test_predictions.npz test-set predictions for parity / residual plots
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from src import config
from src.evaluation import (calibration_curve, extrapolation_mask, partial_dependence,
                            permutation_importance, regression_metrics, sobol_indices)
from src.physics import frame_fe_batch, frame_fe_solve
from src.surrogate_zoo import (ClosedFormReference, GaussianProcess, Surrogate, available_models,
                               make_model, timed_predict, to_transformed)

R2_GATE = 0.98            # every target must reach this CV R2 to be eligible


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(config.DATASET_CSV)
    return df[config.FEATURES].to_numpy(float), df[config.TARGETS].to_numpy(float)


def _factory(name: str, gp_kernels, epochs: int | None, seed: int):
    if name == GaussianProcess.name and gp_kernels is not None:
        return GaussianProcess(seed=seed, fixed_kernels=gp_kernels)
    if name in ("residual_mlp", "physics_guided_mlp"):
        return make_model(name, seed=seed, epochs=epochs)
    return make_model(name, seed=seed)


def measure_latency(model: Surrogate, X: np.ndarray) -> tuple[float, float]:
    """(single-design ms, per-design ms within a 1000-design batch)."""
    single = np.median([timed_predict(model, X[i:i + 1], repeats=3) for i in range(20)])
    batch = timed_predict(model, np.resize(X, (1000, X.shape[1])), repeats=3) / 1000.0
    return float(single), float(batch)


def fe_latency_ms(X: np.ndarray, n: int = 30) -> float:
    times = []
    for row in X[:n]:
        t0 = time.perf_counter()
        frame_fe_solve(*row)
        times.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(times))


def cross_validate(name, X, Y, folds, gp_kernels, epochs, seed) -> dict:
    kf = KFold(folds, shuffle=True, random_state=seed)
    scores = {t: [] for t in config.TARGETS}
    for tr, va in kf.split(X):
        m = _factory(name, gp_kernels, epochs, seed).fit(X[tr], Y[tr])
        res = regression_metrics(Y[va], m.predict(X[va]))["per_target"]
        for t in config.TARGETS:
            scores[t].append(res[t]["r2"])
    out = {t: {"mean": float(np.mean(v)), "std": float(np.std(v))} for t, v in scores.items()}
    out["min_mean_r2"] = float(min(v["mean"] for k, v in out.items() if k in config.TARGETS))
    return out


SELECTION_WEIGHTS = {
    # criterion: (weight, higher_is_better, log_scale)
    "stress_max_err_pct": (0.25, False, True),    # worst-case stress error vs FE (safety-critical)
    "stress_mape_pct": (0.20, False, False),      # typical stress error vs FE
    "mean_mape_pct": (0.15, False, False),        # typical error across all 5 outputs
    "mean_max_err_pct": (0.10, False, True),      # worst-case error averaged across outputs
    "extrapolation_stress_mape_pct": (0.15, False, True),  # behaviour outside the training region
    "cv_min_r2": (0.05, True, False),             # stability on unseen folds
    "uncertainty": (0.05, True, False),           # provides a confidence interval
    "batch_latency_ms": (0.05, False, True),      # what optimisers pay per design
}


def selection_criteria(name: str, m: dict, extrap: dict | None) -> dict:
    fe = m["test_vs_clean_fe"]["per_target"]
    ex = extrap["models"].get(name, {}) if extrap else {}
    return {
        "stress_max_err_pct": fe["peak_von_mises_mpa"]["max_rel_err_pct"],
        "stress_mape_pct": fe["peak_von_mises_mpa"]["mape_pct"],
        "mean_mape_pct": float(np.mean([fe[t]["mape_pct"] for t in config.TARGETS])),
        "mean_max_err_pct": float(np.mean([fe[t]["max_rel_err_pct"] for t in config.TARGETS])),
        "extrapolation_stress_mape_pct": ex.get("peak_von_mises_mpa", {}).get("mape_pct", float("nan")),
        "cv_min_r2": m["cv"]["min_mean_r2"],
        "uncertainty": 1.0 if m["supports_uncertainty"] else 0.0,
        "batch_latency_ms": m["latency_ms_per_design_batch"],
    }


def select_model(metrics: dict, extrap: dict | None = None) -> dict:
    """Transparent two-stage rule (v1):
    1. gate: every target CV R2 >= R2_GATE (fallback: all trainable models if none pass);
    2. weighted composite score over SELECTION_WEIGHTS; each criterion min-max normalised across the
       eligible models (log scale where marked), 1 = best. Highest total wins.
    """
    trace = []
    trainable = {k: v for k, v in metrics.items() if v["trainable"]}
    eligible = {k: v for k, v in trainable.items() if v["cv"]["min_mean_r2"] >= R2_GATE}
    for k, v in trainable.items():
        trace.append(f"{k}: CV min R2 = {v['cv']['min_mean_r2']:.4f} -> "
                     f"{'eligible' if k in eligible else 'rejected (below gate)'}")
    if not eligible:
        eligible = trainable
        trace.append("No model passed the R2 gate; falling back to all trainable models.")

    names = list(eligible)
    crit = {n: selection_criteria(n, eligible[n], extrap) for n in names}
    scores = {n: 0.0 for n in names}
    breakdown = {n: {} for n in names}
    for c, (w, higher, logscale) in SELECTION_WEIGHTS.items():
        vals = np.array([crit[n][c] for n in names], dtype=float)
        if not np.isfinite(vals).all():
            fill = (np.nanmin if higher else np.nanmax)(vals) if np.isfinite(vals).any() else 0.0
            vals = np.where(np.isfinite(vals), vals, fill)
        if logscale:
            vals = np.log10(np.clip(vals, 1e-9, None))
        span = vals.max() - vals.min()
        normed = np.ones_like(vals) if span == 0 else (vals - vals.min()) / span
        if not higher:
            normed = 1.0 - normed
        for n, v in zip(names, normed):
            scores[n] += w * float(v)
            breakdown[n][c] = round(w * float(v), 4)
    ranking = sorted(names, key=lambda n: -scores[n])
    chosen = ranking[0]
    for n in ranking:
        trace.append(f"{n}: composite score {scores[n]:.3f}")
    trace.append(f"Highest composite score -> {chosen}")
    return {
        "selected_model": chosen,
        "rule": {"r2_gate": R2_GATE, "method": "weighted composite score (min-max normalised, 1 = best)",
                 "weights": {k: v[0] for k, v in SELECTION_WEIGHTS.items()}},
        "eligible_ranking": ranking,
        "scores": scores,
        "score_breakdown": breakdown,
        "criteria": crit,
        "trace": trace,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="3-fold CV and shorter NN training")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args(argv)
    folds, epochs = (3, 200) if args.fast else (5, None)
    seed = args.seed
    rng = np.random.default_rng(seed)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, Y = load_dataset()
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=config.TEST_FRACTION, random_state=seed)
    print(f"Train {len(X_tr)} / test {len(X_te)} | models: {available_models()}")
    Y_te_clean = frame_fe_batch(X_te)
    fe_ms = fe_latency_ms(X_te)
    noise_floor = {t: float(100 * np.max(np.abs(Y_te[:, j] - Y_te_clean[:, j])
                                         / np.maximum(np.abs(Y_te_clean[:, j]), 1.0 if j == 4 else 1e-9)))
                   for j, t in enumerate(config.TARGETS)}

    metrics: dict = {}
    preds: dict = {"X_test": X_te, "Y_test": Y_te, "Y_test_clean_fe": Y_te_clean}
    models: dict[str, Surrogate] = {}
    names = [ClosedFormReference.name] + available_models()
    for name in names:
        model = ClosedFormReference() if name == ClosedFormReference.name else _factory(name, None, epochs, seed)
        model.fit(X_tr, Y_tr)
        models[name] = model
        P = model.predict(X_te)
        preds[f"pred__{name}"] = P
        single, batch = measure_latency(model, X_te)
        metrics[name] = {
            "label": model.label, "family": model.family, "trainable": model.trainable,
            "supports_uncertainty": model.supports_uncertainty, "fit_seconds": float(model.fit_seconds),
            "test": regression_metrics(Y_te, P),
            "test_vs_clean_fe": regression_metrics(Y_te_clean, P),
            "latency_ms_single": single, "latency_ms_per_design_batch": batch,
            "speedup_vs_fe_single": fe_ms / single, "speedup_vs_fe_batch": fe_ms / batch,
            "projected_speedup_vs_industrial_fe": 1e3 * config.NOMINAL_INDUSTRIAL_FE_SECONDS / single,
        }
        if model.trainable:
            model.save(config.MODELS_DIR)
        m = metrics[name]
        print(f"  {name:24s} fit {m['fit_seconds']:7.1f}s | min R2 {m['test']['min_r2']:.4f} | "
              f"stress max err (vs FE) {m['test_vs_clean_fe']['per_target']['peak_von_mises_mpa']['max_rel_err_pct']:6.2f}% | "
              f"{single:.2f} ms | {m['speedup_vs_fe_single']:.1f}x vs FE")

    gp_kernels = models[GaussianProcess.name].fitted_kernels() if GaussianProcess.name in models else None

    # ---- cross-validation
    print(f"Cross-validation ({folds}-fold)...")
    for name in names:
        if not metrics[name]["trainable"]:
            metrics[name]["cv"] = {"min_mean_r2": metrics[name]["test"]["min_r2"]}
            continue
        metrics[name]["cv"] = cross_validate(name, X_tr, Y_tr, folds, gp_kernels, epochs, seed)

    # ---- learning curves
    print("Learning curves...")
    sizes = [s for s in (100, 200, 400, 800) if s < len(X_tr)] + [len(X_tr)]
    learning = {"sizes": sizes, "models": {}}
    for name in names:
        if not metrics[name]["trainable"]:
            continue
        rows = []
        for s in sizes:
            idx = rng.choice(len(X_tr), s, replace=False) if s < len(X_tr) else np.arange(len(X_tr))
            m = models[name] if s == len(X_tr) else _factory(name, gp_kernels, epochs, seed).fit(X_tr[idx], Y_tr[idx])
            r = regression_metrics(Y_te_clean, m.predict(X_te))
            rows.append({"mean_r2": r["mean_r2"], "min_r2": r["min_r2"],
                         "stress_mape_pct": r["per_target"]["peak_von_mises_mpa"]["mape_pct"]})
        learning["models"][name] = rows

    # ---- extrapolation: train without the large/thick-tube corner, test inside it
    print("Extrapolation study...")
    mask_tr = extrapolation_mask(X_tr)
    mask_all = extrapolation_mask(X)
    X_corner, Y_corner_clean = X[mask_all], frame_fe_batch(X[mask_all])
    extrap = {"n_train_excluded": int(mask_tr.sum()), "n_corner_test": int(mask_all.sum()),
              "region": "tube_od_mm > 34 and wall_thickness_mm > 3.0", "models": {}}
    for name in names:
        m = models[name] if not metrics[name]["trainable"] else \
            _factory(name, gp_kernels, epochs, seed).fit(X_tr[~mask_tr], Y_tr[~mask_tr])
        r = regression_metrics(Y_corner_clean, m.predict(X_corner))
        extrap["models"][name] = {t: {"mape_pct": r["per_target"][t]["mape_pct"],
                                      "max_rel_err_pct": r["per_target"][t]["max_rel_err_pct"]}
                                  for t in config.TARGETS}

    selection = select_model(metrics, extrap)
    best = selection["selected_model"]
    print("Selected surrogate:", best)
    for line in selection["trace"]:
        print("   ", line)

    # ---- uncertainty calibration (GP)
    calibration = {}
    if GaussianProcess.name in models:
        mu, sd = models[GaussianProcess.name].predict_transformed_std(X_te)
        calibration[GaussianProcess.name] = calibration_curve(to_transformed(Y_te), mu, sd)
        lo, _, hi = models[GaussianProcess.name].predict_interval(X_te)
        preds["gp_lower95"], preds["gp_upper95"] = lo, hi

    # ---- sensitivity on the selected model
    print("Sensitivity analysis on", best, "...")
    bm = models[best]
    sensitivity = {
        "model": best,
        "sobol": sobol_indices(bm.predict, n_base=2048, seed=seed),
        "permutation": permutation_importance(bm.predict, X_te, Y_te, seed=seed),
        "partial_dependence": partial_dependence(bm.predict, X_tr[rng.choice(len(X_tr), 200, replace=False)]),
    }

    histories = {n: models[n].history for n in names if hasattr(models[n], "history") and models[n].history}

    out_metrics = {
        "split": {"n_train": int(len(X_tr)), "n_test": int(len(X_te)), "seed": seed, "cv_folds": folds},
        "fe_baseline_ms_per_design": fe_ms,
        "nominal_industrial_fe_seconds": config.NOMINAL_INDUSTRIAL_FE_SECONDS,
        "label_noise_floor_max_rel_err_pct": noise_floor,
        "selected_model": best,
        "models": metrics,
    }
    config.METRICS_JSON.write_text(json.dumps(out_metrics, indent=2))
    config.MODEL_SELECTION_JSON.write_text(json.dumps(selection, indent=2))
    config.EVALUATION_JSON.write_text(json.dumps({
        "learning_curves": learning, "extrapolation": extrap, "calibration": calibration,
        "sensitivity": sensitivity, "training_history": histories,
    }, indent=2))
    np.savez_compressed(config.EVALUATION_NPZ, **preds)
    print(f"Saved metrics -> {config.METRICS_JSON}")


if __name__ == "__main__":
    main()
