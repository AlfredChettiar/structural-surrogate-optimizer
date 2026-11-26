"""Load pipeline artifacts (JSON / NPZ / CSV) for the app and report. Missing files -> None."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_all() -> dict:
    preds = None
    if config.EVALUATION_NPZ.exists():
        with np.load(config.EVALUATION_NPZ) as npz:
            preds = {k: npz[k] for k in npz.files}
    return {
        "dataset": pd.read_csv(config.DATASET_CSV) if config.DATASET_CSV.exists() else None,
        "summary": _json(config.DATASET_SUMMARY_JSON),
        "metrics": _json(config.METRICS_JSON),
        "selection": _json(config.MODEL_SELECTION_JSON),
        "extras": _json(config.EVALUATION_JSON),
        "preds": preds,
        "optimization": _json(config.OPTIMIZATION_JSON),
        "pareto": pd.read_csv(config.PARETO_CSV) if config.PARETO_CSV.exists() else None,
        "candidates": pd.read_csv(config.CANDIDATES_CSV) if config.CANDIDATES_CSV.exists() else None,
    }


def missing(art: dict) -> list[str]:
    return [k for k, v in art.items() if v is None]
