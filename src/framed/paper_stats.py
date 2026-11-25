"""Re-derive the paper's dataset statistics (Sec. 5) from the released data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.framed import paper_reference as ref

PAIR_OBJECTIVES = ["Sim 1 Dropout Y Disp.", "Sim 2 Bottom Bracket Z Disp.", "Sim 3 Bottom Bracket X Rot.",
                   "Sim 1 Safety Factor", "Model Mass"]
PAIR_SENSE = np.array([-1, -1, -1, 1, -1])  # -1 minimise, +1 maximise (Analysis.ipynb)


def feasibility_counts(Y: pd.DataFrame) -> dict[str, int]:
    f1, f3 = Y["Sim 1 Safety Factor"] < 1, Y["Sim 3 Safety Factor"] < 1
    return {"both_fail": int((f1 & f3).sum()), "sf1_only": int((f1 & ~f3).sum()),
            "sf3_only": int((~f1 & f3).sum()), "valid": int((~f1 & ~f3).sum())}


def non_dominated_mask(F: np.ndarray, sense: np.ndarray, chunk: int = 512) -> np.ndarray:
    G = -np.asarray(F, float) * sense  # convert to all-minimise
    dominated = np.zeros(len(G), bool)
    for s in range(0, len(G), chunk):
        g = G[s:s + chunk, None, :]
        dominated[s:s + chunk] = ((G[None] <= g).all(2) & (G[None] < g).any(2)).any(1)
    return ~dominated


def stats_comparison(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    rows = []
    counts = feasibility_counts(Y)
    for k, v in ref.NOTEBOOK_FEASIBILITY.items():
        rows.append((f"Frames: {k.replace('_', ' ')}", "authors' notebook", v, counts[k], ref.COUNT_TOLERANCE))
    nd = int(non_dominated_mask(Y[PAIR_OBJECTIVES].values, PAIR_SENSE).sum())
    rows.append(("Non-dominated designs (5 objectives)", "authors' notebook", ref.NOTEBOOK_N_NON_DOMINATED, nd,
                 ref.COUNT_TOLERANCE))
    rows.append(("Frames failing SF >= 1 in any case", "paper Sec. 5.1", ref.PAPER_N_FAIL_ANY,
                 len(Y) - counts["valid"], None))  # paper counted 4118 frames, release has 4047: context only
    rows.append(("Geometrically valid frames simulated", "paper Sec. 3.2", ref.PAPER_N_VALID_GEOMETRY, len(Y), None))
    for idx, cs in ref.CASE_STUDIES.items():
        rows.append((f"Frame {idx} mass (kg)", "paper Sec. 5.2", cs["mass"], float(Y.loc[idx, "Model Mass"]), 0.005))
        rows.append((f"Frame {idx} SF1", "paper Sec. 5.2", cs["sf1"], float(Y.loc[idx, "Sim 1 Safety Factor"]), 0.005))
    df = pd.DataFrame(rows, columns=["quantity", "source", "paper", "ours", "tol"])
    df["abs_dev"] = df["ours"] - df["paper"]
    df["pct_dev"] = 100 * df["abs_dev"] / df["paper"]
    df["pass"] = [None if t is None or np.isnan(t) else bool(abs(d) <= t) for d, t in zip(df["abs_dev"], df["tol"])]
    return df
