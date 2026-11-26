"""Download (pinned, checksummed) and load the FRAMED dataset."""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.framed import config as fc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(path: Path, sha: str) -> None:
    got = _sha256(path)
    if got != sha:
        raise ValueError(f"sha256 mismatch for {path.name}: expected {sha}, got {got}")


def fetch_framed(dest: Path = fc.FRAMED_DIR, force: bool = False) -> dict[str, Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, (url, sha) in fc.FILES.items():
        path = dest / fc.FILENAMES[name]
        if force or not path.exists():
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            _verify(tmp, sha)  # verify before replacing a good file
            tmp.replace(path)
        _verify(path, sha)
        out[name] = path
    return out


def load_framed(path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(X, Y): X = first 39 columns, Y = 10 outputs with abs() as in the authors' Analysis.ipynb."""
    # FRAMED_CSV env var lets you train on your own file with the same columns (docs/FRAMED_PROJECT.md, section 8)
    df = pd.read_csv(path or os.environ.get("FRAMED_CSV") or fc.FRAMED_DIR / fc.FILENAMES["data"], index_col=0)
    missing = [c for c in fc.Y_COLS if c not in df.columns]
    if df.shape[1] < fc.N_X or missing:
        raise ValueError(f"expected 39 input columns then the 10 outputs; missing outputs: {missing}")
    X = df.iloc[:, : fc.N_X].astype(float)
    Y = df[fc.Y_COLS].astype(float).copy()
    Y[fc.ABS_COLS] = Y[fc.ABS_COLS].abs()
    return X, Y


def fold_indices(n: int, n_folds: int = 5, n_repeats: int = 10, seed: int = fc.SEED):
    folds = []
    for r in range(n_repeats):
        folds += list(KFold(n_folds, shuffle=True, random_state=seed + r).split(np.arange(n)))
    return folds
