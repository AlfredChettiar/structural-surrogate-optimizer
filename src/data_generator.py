"""Generate the 1,500-sample Latin Hypercube + Central Composite Design chassis dataset.

Run:  .\\.venv\\Scripts\\python.exe -m src.data_generator
Outputs: data/chassis_doe_dataset.csv, models/dataset_summary.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src import config
from src.guardrails import apply_guardrails, safety_factor
from src.physics import fatigue_log10_life, frame_fe_batch

# realistic measurement / solver scatter (lognormal sigma, i.e. ~ relative std)
NOISE_SIGMA = {
    "mass_kg": 0.005,             # weighing / CAD-mass tolerance
    "peak_von_mises_mpa": 0.03,   # strain-gauge & mesh-sensitivity scatter
    "max_deflection_mm": 0.03,
    "first_mode_hz": 0.015,       # modal test scatter
}
FATIGUE_EXTRA_SIGMA = 0.05        # additional scatter in log10(life)


def _bounds_arrays():
    lo = np.array([config.BOUNDS[f][0] for f in config.FEATURES])
    hi = np.array([config.BOUNDS[f][1] for f in config.FEATURES])
    return lo, hi


def ccd_face_centred(n_factors: int) -> np.ndarray:
    """Face-centred CCD in unit space [0, 1]: 2^k corners, 2k axial points, 1 centre point."""
    corners = np.array(list(itertools.product([0.0, 1.0], repeat=n_factors)))
    axial = []
    for j in range(n_factors):
        for v in (0.0, 1.0):
            p = np.full(n_factors, 0.5)
            p[j] = v
            axial.append(p)
    return np.vstack([corners, np.array(axial), np.full((1, n_factors), 0.5)])


def build_design(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return physical-unit design matrix and DOE-source labels."""
    d = len(config.FEATURES)
    ccd = ccd_face_centred(d)
    n_lhs = n_samples - len(ccd)
    if n_lhs <= 0:
        raise ValueError(f"n_samples must exceed the {len(ccd)} CCD points")
    lhs = qmc.LatinHypercube(d=d, optimization="random-cd", seed=seed).random(n_lhs)
    unit = np.vstack([lhs, ccd])
    lo, hi = _bounds_arrays()
    labels = np.array(["LHS"] * n_lhs + ["CCD"] * len(ccd))
    return lo + unit * (hi - lo), labels


def add_noise(Y: np.ndarray, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    Yn = Y.copy()
    for j, col in enumerate(config.TARGETS[:4]):
        Yn[:, j] = Y[:, j] * np.exp(rng.normal(0.0, NOISE_SIGMA[col], len(Y)))
    # fatigue life is re-derived from the *measured* stress, then gets its own scatter
    Yn[:, 4] = fatigue_log10_life(Yn[:, 1], X[:, 4], X[:, 5]) + rng.normal(0.0, FATIGUE_EXTRA_SIGMA, len(Y))
    return apply_guardrails(Yn)


def generate(n_samples: int = config.N_SAMPLES, seed: int = config.SEED) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    X, labels = build_design(n_samples, seed)
    t0 = time.perf_counter()
    Y_clean = frame_fe_batch(X)
    fe_seconds = time.perf_counter() - t0
    Y = add_noise(Y_clean, X, rng)

    df = pd.DataFrame(X, columns=config.FEATURES)
    for j, col in enumerate(config.TARGETS):
        df[col] = Y[:, j]
    df["doe_source"] = labels

    lo, hi = _bounds_arrays()
    unit = (X - lo) / (hi - lo)
    sf = safety_factor(df["yield_strength_mpa"], df["peak_von_mises_mpa"])
    c = config.CONSTRAINTS
    d_over_t = df["tube_od_mm"] / df["wall_thickness_mm"]
    bend_ratio = df["bend_radius_mm"] / df["tube_od_mm"]
    manufacturable = (d_over_t <= c["max_d_over_t"]) & (bend_ratio >= c["min_bend_ratio"])
    feasible = manufacturable & (
        (sf >= c["min_safety_factor"])
        & (df["first_mode_hz"] >= c["min_first_mode_hz"])
        & (df["max_deflection_mm"] <= c["max_deflection_mm"])
        & (df["log10_fatigue_life_cycles"] >= c["min_log10_fatigue_life"])
    )
    summary = {
        "n_samples": int(len(df)),
        "n_lhs": int((labels == "LHS").sum()),
        "n_ccd": int((labels == "CCD").sum()),
        "seed": seed,
        "features": config.FEATURES,
        "targets": config.TARGETS,
        "bounds": config.BOUNDS,
        "noise_sigma": {**NOISE_SIGMA, "log10_fatigue_life_cycles(additive)": FATIGUE_EXTRA_SIGMA},
        "centered_l2_discrepancy_lhs": float(qmc.discrepancy(unit[labels == "LHS"], method="CD")),
        "fe_solver_seconds_total": round(fe_seconds, 3),
        "fe_solver_ms_per_design": round(1e3 * fe_seconds / len(df), 3),
        "constraint_pass_rate": {
            "safety_factor": float((sf >= c["min_safety_factor"]).mean()),
            "first_mode": float((df["first_mode_hz"] >= c["min_first_mode_hz"]).mean()),
            "deflection": float((df["max_deflection_mm"] <= c["max_deflection_mm"]).mean()),
            "fatigue": float((df["log10_fatigue_life_cycles"] >= c["min_log10_fatigue_life"]).mean()),
            "manufacturability": float(manufacturable.mean()),
            "all": float(feasible.mean()),
        },
        "stats": df[config.FEATURES + config.TARGETS].describe().T.round(5).to_dict(orient="index"),
        "correlation": df[config.FEATURES + config.TARGETS].corr().round(4).to_dict(),
    }
    return df, summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-samples", type=int, default=config.N_SAMPLES)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args(argv)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df, summary = generate(args.n_samples, args.seed)
    df.to_csv(config.DATASET_CSV, index=False)
    config.DATASET_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Saved {len(df)} samples ({summary['n_lhs']} LHS + {summary['n_ccd']} CCD) -> {config.DATASET_CSV}")
    print(f"FE solver: {summary['fe_solver_ms_per_design']} ms/design | "
          f"feasible fraction: {summary['constraint_pass_rate']['all']:.1%}")


if __name__ == "__main__":
    main()
