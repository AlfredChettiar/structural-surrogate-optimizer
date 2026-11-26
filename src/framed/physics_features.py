"""Beam-theory features for FRAMED frames (SI units: m, kg, Pa)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.framed import config as fc
from src.framed import paper_reference as ref

LENGTH_COLS = {"ST": "ST Length", "HT": "HT Length", "DT": "DT Length", "CS": "CS Length", "BB": "BB Length"}
PAIRED = {"CS": 2, "SS": 2}  # stays come in left/right pairs


def section_properties(od, t):
    """Area, second moment and polar moment of a round tube; t >= od/2 is a solid rod."""
    od = np.asarray(od, float)
    di = np.clip(od - 2 * np.minimum(np.asarray(t, float), od / 2), 0.0, None)
    A = np.pi / 4 * (od**2 - di**2)
    I = np.pi / 64 * (od**4 - di**4)
    return A, I, 2 * I


def material_props(X: pd.DataFrame) -> pd.DataFrame:
    oh = X[fc.MATERIAL_COLS].to_numpy()
    if not (np.isin(oh, (0, 1)).all() and (oh.sum(1) == 1).all()):
        raise ValueError("material columns must be exactly one-hot")
    props = pd.DataFrame([ref.MATERIALS[m] for m in fc.MATERIALS])
    out = pd.DataFrame(oh @ props[["E", "G", "rho", "sigma_y"]].to_numpy(), index=X.index,
                       columns=["E", "G", "rho", "sigma_y"])
    return out


def analytic_mass_prior(X: pd.DataFrame) -> np.ndarray:
    # shortcut: top tube and seat stays have no explicit length in FRAMED, so they are left out; add them if a
    # geometry reconstruction is ever built
    rho = material_props(X)["rho"].to_numpy()
    m = np.zeros(len(X))
    for tube, lcol in LENGTH_COLS.items():
        A, _, _ = section_properties(X[f"{tube} OD"], X[f"{tube} Thickness"])
        m += rho * A * X[lcol].to_numpy() * PAIRED.get(tube, 1)
    return m


def add_physics_features(X: pd.DataFrame) -> pd.DataFrame:
    mp = material_props(X)
    feats = {}
    for tube in fc.TUBES:
        A, I, J = section_properties(X[f"{tube} OD"], X[f"{tube} Thickness"])
        feats[f"A_{tube}"] = A
        feats[f"log_EI_{tube}"] = np.log(mp["E"].to_numpy() * I)
        feats[f"log_GJ_{tube}"] = np.log(mp["G"].to_numpy() * J)
        if tube in LENGTH_COLS:
            feats[f"slender_{tube}"] = X[LENGTH_COLS[tube]].to_numpy() / X[f"{tube} OD"].to_numpy()
    feats["mass_prior"] = analytic_mass_prior(X)
    feats["sigma_y"] = mp["sigma_y"].to_numpy()
    feats["rho"] = mp["rho"].to_numpy()
    return pd.concat([X, pd.DataFrame(feats, index=X.index)], axis=1)
