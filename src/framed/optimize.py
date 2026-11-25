"""Uncertainty-aware lightweighting of a FRAMED reference frame (the paper's Future Work, Sec. 8).

Rider-fit geometry is frozen to the reference frame (paper Sec. 5.2.3 warns that shrinking a frame trivially
'optimises' it). Variables: OD and wall thickness of the top, down and seat tubes and the stays. Verification is by
surrogate uncertainty and distance to the nearest simulated FRAMED design only: no FE re-check (user scope decision).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.framed import config as fc
from src.optimizer import ChassisProblem, knee_point, nsga2, parego, pareto_front

REF_INDEX = 3853  # paper Sec. 5.2 "Frame 1": titanium road frame, 2.84 kg
OPT_TUBES = ["TT", "DT", "ST", "SS", "CS"]
VARS = [f"{t} OD" for t in OPT_TUBES] + [f"{t} Thickness" for t in OPT_TUBES]
MAX_D_OVER_T = 50.0
J = {c: fc.Y_COLS.index(c) for c in fc.Y_COLS}


def snap_material(X: pd.DataFrame) -> pd.DataFrame:
    """Force the material columns to a valid one-hot (argmax), e.g. after an optimiser or slider edit."""
    X = X.copy()
    oh = np.zeros((len(X), 3))
    oh[np.arange(len(X)), X[fc.MATERIAL_COLS].to_numpy().argmax(1)] = 1
    X[fc.MATERIAL_COLS] = oh
    return X


@dataclass
class FramedProblem(ChassisProblem):
    """Duck-types src.optimizer: xl, xu, n_var, evaluate(G) -> {F, cv, ...}, normalise(F), archive."""

    X_all: pd.DataFrame = None
    material: str = "Titanium"
    max_bb_z: float = np.inf  # transverse BB deflection limit (m)

    def __post_init__(self):
        self.base = snap_material(self.X_all.loc[[REF_INDEX]].assign(
            **{c: float(c == f"Material={self.material}") for c in fc.MATERIAL_COLS}))
        self.xl = self.X_all[VARS].quantile(0.01).to_numpy()
        self.xu = self.X_all[VARS].quantile(0.99).to_numpy()
        self.n_var = len(VARS)
        self.archive, self.tag, self.n_evals = [], "untagged", 0
        self._f_lo = self._f_hi = None

    def full_inputs(self, G: np.ndarray) -> pd.DataFrame:
        X = pd.concat([self.base] * len(G), ignore_index=True)
        X[VARS] = np.atleast_2d(G)
        return snap_material(X)

    def predict(self, G):
        return self.surrogate.predict_interval(self.full_inputs(G))

    def evaluate(self, G: np.ndarray) -> dict:
        G = np.clip(np.atleast_2d(np.asarray(G, float)), self.xl, self.xu)
        lo, Y, hi = self.predict(G)
        od, t = G[:, :5], G[:, 5:]
        V = np.column_stack([
            1.0 - lo[:, J["Sim 1 Safety Factor"]],  # SF >= 1 on the 95 % lower bound
            1.0 - lo[:, J["Sim 3 Safety Factor"]],
            (Y[:, J["Sim 2 Bottom Bracket Z Disp."]] - self.max_bb_z) / self.max_bb_z,
            ((od / t - MAX_D_OVER_T) / MAX_D_OVER_T).max(1),  # thin-wall manufacturability
            ((2 * t - od) / od).max(1),  # wall cannot exceed the radius
        ])
        cv = np.clip(V, 0, None).sum(1)
        F = Y[:, [J[fc.MASS_COL], J["Sim 1 Dropout Y Disp."]]]
        if self.record:
            self.archive.append((self.tag, G.copy(), F.copy(), cv.copy()))
            self.n_evals += len(G)
        return {"Y": Y, "lo": lo, "hi": hi, "F": F, "V": V, "cv": cv, "feasible": cv <= 0}


def nearest_simulated(Xq: pd.DataFrame, X_all: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = X_all.mean(), X_all.std().replace(0, 1)
    A, B = ((Xq - mu) / sd).to_numpy(), ((X_all - mu) / sd).to_numpy()
    d = np.sqrt(((A[:, None, :] - B[None]) ** 2).sum(-1))
    i = d.argmin(1)
    return X_all.index.to_numpy()[i], d[np.arange(len(Xq)), i]


def _rows(problem: FramedProblem, G: np.ndarray, labels: list[str], Y_all: pd.DataFrame) -> pd.DataFrame:
    ev = problem.evaluate(G)
    X = problem.full_inputs(G)
    nn, dist = nearest_simulated(X, problem.X_all)
    df = pd.DataFrame(G * 1000, columns=[f"{v} (mm)" for v in VARS])
    df.insert(0, "design", labels)
    df["mass_kg"] = ev["Y"][:, J[fc.MASS_COL]]
    df["mass_lo"], df["mass_hi"] = ev["lo"][:, J[fc.MASS_COL]], ev["hi"][:, J[fc.MASS_COL]]
    df["dropout_y_mm"] = 1000 * ev["Y"][:, J["Sim 1 Dropout Y Disp."]]
    df["bb_z_mm"] = 1000 * ev["Y"][:, J["Sim 2 Bottom Bracket Z Disp."]]
    df["sf1"], df["sf1_lo"] = ev["Y"][:, J["Sim 1 Safety Factor"]], ev["lo"][:, J["Sim 1 Safety Factor"]]
    df["sf3"], df["sf3_lo"] = ev["Y"][:, J["Sim 3 Safety Factor"]], ev["lo"][:, J["Sim 3 Safety Factor"]]
    df["feasible"] = ev["feasible"]
    df["nearest_framed_index"], df["nearest_distance"] = nn, dist
    df["nearest_mass_kg"] = Y_all.loc[nn, fc.MASS_COL].to_numpy()
    return df


def run_study(surrogate, X_all: pd.DataFrame, Y_all: pd.DataFrame, material: str, seed: int = fc.SEED,
              pop_size: int = 100, generations: int = 60) -> dict:
    ref_bb_z = float(Y_all.loc[REF_INDEX, "Sim 2 Bottom Bracket Z Disp."])
    p = FramedProblem(surrogate=surrogate, X_all=X_all, material=material, max_bb_z=ref_bb_z)
    g0 = X_all.loc[REF_INDEX, VARS].to_numpy()[None]
    for name, algo, kw in (("nsga2", nsga2, dict(pop_size=pop_size, generations=generations)),
                           ("parego", parego, dict(budget=150))):
        p.tag = name
        algo(p, seed=seed, **kw)
    out = {"material": material, "algorithms": {}}
    fronts = []
    for name in ("nsga2", "parego"):
        Gs, Fs, cvs = p.archive_for(name)
        front = pareto_front(Fs, cvs)
        out["algorithms"][name] = {"evaluations": int(len(Gs)), "n_feasible": int((cvs <= 0).sum()),
                                   "front_size": int(len(front))}
        fronts.append(pd.DataFrame(Gs[front], columns=VARS).assign(algorithm=name))
    Gs, Fs, cvs = p.archive_for("nsga2")
    front = pareto_front(Fs, cvs)
    front = front[np.argsort(Fs[front, 0])]
    picks, labels = [g0[0]], ["reference (frame 3853 geometry)"]
    if len(front):
        picks += [Gs[front[knee_point(Fs[front])]], Gs[front[0]]]
        labels += ["knee", "lightest feasible"]
    table = _rows(p, np.vstack(picks), labels, Y_all)
    base = table.iloc[0]
    table["mass_change_pct"] = 100 * (table["mass_kg"] / base["mass_kg"] - 1)
    table["dropout_y_change_pct"] = 100 * (table["dropout_y_mm"] / base["dropout_y_mm"] - 1)
    front_df = _rows(p, Gs[front], ["pareto"] * len(front), Y_all) if len(front) else pd.DataFrame()
    out["table"] = table
    out["front"] = front_df.assign(material=material)
    return out
