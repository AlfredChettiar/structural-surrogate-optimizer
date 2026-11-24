"""Physical boundary guardrails applied to every prediction leaving a model, optimizer or UI."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BOUNDS, FEATURES, GUARDRAILS, TARGETS


def clip_inputs(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clip design inputs to the trained design space.

    Returns the clipped array and a boolean mask of rows that were outside the bounds
    (i.e. extrapolation requests the caller should flag).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    lo = np.array([BOUNDS[f][0] for f in FEATURES])
    hi = np.array([BOUNDS[f][1] for f in FEATURES])
    outside = np.any((X < lo - 1e-9) | (X > hi + 1e-9), axis=1)
    return np.clip(X, lo, hi), outside


def apply_guardrails(Y):
    """Clip target predictions (columns in ``config.TARGETS`` order) to physical bounds.

    Accepts an (n, 5) array or a DataFrame containing any subset of the target columns.
    Non-finite values are replaced by the conservative bound (low for capacity-like
    quantities, high for demand-like quantities).
    """
    if isinstance(Y, pd.DataFrame):
        out = Y.copy()
        for col in TARGETS + ["safety_factor"]:
            if col in out:
                out[col] = _clip_col(out[col].to_numpy(dtype=float), col)
        return out
    arr = np.array(Y, dtype=float, copy=True)
    squeeze = arr.ndim == 1
    arr = np.atleast_2d(arr)
    for j, col in enumerate(TARGETS[: arr.shape[1]]):
        arr[:, j] = _clip_col(arr[:, j], col)
    return arr[0] if squeeze else arr


# demand-like targets: unknown -> assume worst (high); capacity-like: unknown -> assume worst (low)
_CONSERVATIVE_HIGH = {"mass_kg", "peak_von_mises_mpa", "max_deflection_mm"}


def _clip_col(values: np.ndarray, col: str) -> np.ndarray:
    lo, hi = GUARDRAILS[col]
    fill = hi if col in _CONSERVATIVE_HIGH else lo
    values = np.where(np.isfinite(values), values, fill)
    return np.clip(values, lo, hi)


def safety_factor(yield_mpa, peak_vm) -> np.ndarray:
    """Yield safety factor, bounded to the guardrail range."""
    lo, hi = GUARDRAILS["safety_factor"]
    sf = np.asarray(yield_mpa, dtype=float) / np.clip(np.asarray(peak_vm, dtype=float), 1e-6, None)
    return np.clip(sf, lo, hi)


def guard_interval(lower, mean, upper, col: str):
    """Order and clip a (lower, mean, upper) confidence band for one target."""
    lower, mean, upper = (np.asarray(v, dtype=float) for v in (lower, mean, upper))
    lo_b = np.minimum(lower, mean)
    hi_b = np.maximum(upper, mean)
    return _clip_col(lo_b, col), _clip_col(mean, col), _clip_col(hi_b, col)
