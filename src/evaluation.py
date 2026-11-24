"""Model evaluation: accuracy metrics, uncertainty calibration, extrapolation and sensitivity."""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import norm, qmc
from sklearn.metrics import mean_absolute_error, r2_score

from src import config


def max_relative_error_pct(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> float:
    """Max relative error in %. For log10 fatigue life the denominator is max(|y|, 1 decade)
    so that near-zero lives do not explode the metric."""
    denom = np.abs(y_true)
    if target == "log10_fatigue_life_cycles":
        denom = np.maximum(denom, 1.0)
    return float(100.0 * np.max(np.abs(y_pred - y_true) / np.clip(denom, 1e-9, None)))


def regression_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    per_target = {}
    for j, t in enumerate(config.TARGETS):
        yt, yp = Y_true[:, j], Y_pred[:, j]
        per_target[t] = {
            "r2": float(r2_score(yt, yp)),
            "mae": float(mean_absolute_error(yt, yp)),
            "max_rel_err_pct": max_relative_error_pct(yt, yp, t),
            "mape_pct": float(100 * np.mean(np.abs(yp - yt) / np.maximum(np.abs(yt), 1.0 if j == 4 else 1e-9))),
        }
    r2s = [v["r2"] for v in per_target.values()]
    return {"per_target": per_target, "mean_r2": float(np.mean(r2s)), "min_r2": float(np.min(r2s))}


# ----------------------------------------------------------------------------- uncertainty
def calibration_curve(y_t: np.ndarray, mu_t: np.ndarray, sd_t: np.ndarray,
                      levels=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)) -> dict:
    """Empirical coverage of central prediction intervals, per target, in transformed space."""
    out = {"levels": list(levels), "coverage": {}, "mean_interval_width": {}}
    for j, t in enumerate(config.TARGETS):
        z_abs = np.abs(y_t[:, j] - mu_t[:, j]) / np.clip(sd_t[:, j], 1e-12, None)
        cov, width = [], []
        for lv in levels:
            z = norm.ppf(0.5 + lv / 2.0)
            cov.append(float(np.mean(z_abs <= z)))
            width.append(float(np.mean(2 * z * sd_t[:, j])))
        out["coverage"][t] = cov
        out["mean_interval_width"][t] = width
    return out


# ----------------------------------------------------------------------------- extrapolation
def extrapolation_mask(X: np.ndarray) -> np.ndarray:
    """Hold-out corner of the design space: large, thick tubes (unseen during training)."""
    od = X[:, config.FEATURES.index("tube_od_mm")]
    wt = X[:, config.FEATURES.index("wall_thickness_mm")]
    return (od > 34.0) & (wt > 3.0)


# ----------------------------------------------------------------------------- sensitivity
def _bounds():
    lo = np.array([config.BOUNDS[f][0] for f in config.FEATURES])
    hi = np.array([config.BOUNDS[f][1] for f in config.FEATURES])
    return lo, hi


def sobol_indices(predict: Callable[[np.ndarray], np.ndarray], n_base: int = 4096,
                  seed: int = config.SEED) -> dict:
    """First-order (Saltelli 2010) and total-order (Jansen) Sobol indices over the full
    design space, per target, using a Sobol' low-discrepancy base sample."""
    d = len(config.FEATURES)
    lo, hi = _bounds()
    base = qmc.Sobol(2 * d, scramble=True, seed=seed).random(n_base)
    A = lo + base[:, :d] * (hi - lo)
    B = lo + base[:, d:] * (hi - lo)
    fA, fB = predict(A), predict(B)
    var = np.var(np.vstack([fA, fB]), axis=0)
    S1 = np.zeros((d, fA.shape[1]))
    ST = np.zeros_like(S1)
    for i in range(d):
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        fABi = predict(ABi)
        S1[i] = np.mean(fB * (fABi - fA), axis=0) / np.clip(var, 1e-12, None)
        ST[i] = 0.5 * np.mean((fA - fABi) ** 2, axis=0) / np.clip(var, 1e-12, None)
    S1, ST = np.clip(S1, 0.0, 1.0), np.clip(ST, 0.0, 1.0)
    return {
        "features": config.FEATURES,
        "S1": {t: S1[:, j].round(5).tolist() for j, t in enumerate(config.TARGETS)},
        "ST": {t: ST[:, j].round(5).tolist() for j, t in enumerate(config.TARGETS)},
        "n_evaluations": int(n_base * (d + 2)),
    }


def permutation_importance(predict, X: np.ndarray, Y: np.ndarray, n_repeats: int = 5,
                           seed: int = config.SEED) -> dict:
    """Increase in MAE (normalised by the target std) when one feature column is shuffled."""
    rng = np.random.default_rng(seed)
    base_mae = np.mean(np.abs(predict(X) - Y), axis=0)
    scale = Y.std(axis=0) + 1e-12
    imp = np.zeros((len(config.FEATURES), Y.shape[1]))
    for i in range(len(config.FEATURES)):
        acc = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, i] = rng.permutation(Xp[:, i])
            acc.append(np.mean(np.abs(predict(Xp) - Y), axis=0) - base_mae)
        imp[i] = np.mean(acc, axis=0) / scale
    return {"features": config.FEATURES,
            "importance": {t: imp[:, j].round(5).tolist() for j, t in enumerate(config.TARGETS)}}


def partial_dependence(predict, X_ref: np.ndarray, n_grid: int = 25) -> dict:
    """1-D partial dependence of every target on every feature (averaged over X_ref)."""
    lo, hi = _bounds()
    out = {}
    for i, f in enumerate(config.FEATURES):
        grid = np.linspace(lo[i], hi[i], n_grid)
        curves = []
        for g in grid:
            Xg = X_ref.copy()
            Xg[:, i] = g
            curves.append(predict(Xg).mean(axis=0))
        curves = np.array(curves)
        out[f] = {"grid": grid.round(4).tolist(),
                  **{t: curves[:, j].round(5).tolist() for j, t in enumerate(config.TARGETS)}}
    return out
