"""Optional AutoGluon model. AutoGluon lives in its own venv (.venv-ag) because it pins older library versions;
we call it through a subprocess with .npz files."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from src import config as v1

AG_PYTHON = v1.ROOT / ".venv-ag" / "bin" / "python"
RUNNER = Path(__file__).with_name("automl_runner.py")
TIME_LIMIT_S = 90  # per target per fold


def available() -> bool:
    if not AG_PYTHON.exists():
        return False
    return subprocess.run([str(AG_PYTHON), "-c", "import autogluon.tabular"], capture_output=True).returncode == 0


class AutoGluonModel:
    """Fit happens at predict time so a fold is one subprocess call (train + predict)."""

    def __init__(self, seed: int, time_limit: int = TIME_LIMIT_S, columns=None):
        self.seed, self.time_limit = seed, time_limit

    def fit(self, X, Y):
        self.X, self.Y = np.asarray(X, float), np.asarray(Y, float)
        return self

    def predict(self, X):
        with tempfile.TemporaryDirectory() as d:
            inp, out = Path(d) / "in.npz", Path(d) / "out.npy"
            np.savez(inp, X=self.X, Y=self.Y, Xp=np.asarray(X, float))
            subprocess.run([str(AG_PYTHON), str(RUNNER), str(inp), str(out), str(self.time_limit), str(self.seed),
                            str(Path(d) / "ag")], check=True, capture_output=True)
            return np.load(out)
