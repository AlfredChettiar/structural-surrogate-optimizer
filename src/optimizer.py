"""Multi-objective weight optimisation of the sub-frame on a trained surrogate.

Objectives (minimise): mass_kg, peak_von_mises_mpa
Constraints: SF = yield / peak_vm >= 1.5, f1 >= 45 Hz, deflection <= 2.5 mm, fatigue life >= 1e6 cycles
Design variables: tube OD, wall thickness, gusset thickness, bend radius
Scenario (fixed per run): material yield strength, dynamic load factor

Algorithms:
  lhs_screening        5,000-candidate Latin Hypercube screening + non-dominated filter (spec baseline)
  nsga2                NSGA-II with constraint domination, SBX crossover, polynomial mutation
  epsilon_constraint   epsilon-constraint scalarisation solved by SciPy differential evolution
  weighted_sum_slsqp   weighted-sum scalarisation solved by gradient-based SLSQP (multi-start)
  parego               ParEGO: GP-based Bayesian optimisation with augmented Tchebycheff scalarisation

Run the full benchmark:  .\\.venv\\Scripts\\python.exe -m src.optimizer
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import NonlinearConstraint, differential_evolution, minimize
from scipy.stats import norm, qmc

from src import config
from src.guardrails import safety_factor

ALGORITHMS = ["lhs_screening", "nsga2", "epsilon_constraint", "weighted_sum_slsqp", "parego"]
ALGORITHM_LABELS = {
    "lhs_screening": "LHS screening (5,000)",
    "nsga2": "NSGA-II",
    "epsilon_constraint": "epsilon-constraint + DE",
    "weighted_sum_slsqp": "Weighted-sum SLSQP",
    "parego": "ParEGO (Bayesian)",
}
OBJECTIVES = ["mass_kg", "peak_von_mises_mpa"]


# ----------------------------------------------------------------------------- problem
@dataclass
class ChassisProblem:
    surrogate: object
    scenario: dict = field(default_factory=lambda: dict(config.DEFAULT_SCENARIO))
    constraints: dict = field(default_factory=lambda: dict(config.CONSTRAINTS))
    record: bool = True

    def __post_init__(self):
        self.xl = np.array([config.BOUNDS[f][0] for f in config.GEOMETRY_FEATURES])
        self.xu = np.array([config.BOUNDS[f][1] for f in config.GEOMETRY_FEATURES])
        self.n_var = len(self.xl)
        self.archive: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        self.tag = "untagged"
        self.n_evals = 0
        self._f_lo = self._f_hi = None

    # -- helpers
    def full_inputs(self, G: np.ndarray) -> np.ndarray:
        G = np.atleast_2d(G)
        sc = np.tile([self.scenario["yield_strength_mpa"], self.scenario["dynamic_load_factor"]], (len(G), 1))
        return np.hstack([G, sc])

    def violations(self, Y: np.ndarray, G: np.ndarray | None = None) -> np.ndarray:
        """Normalised constraint values; > 0 means violated.

        Columns: safety factor, first mode, deflection, fatigue life (from the predictions Y) and,
        when the geometry G is given, the two exact manufacturability checks (D/t, bend ratio).
        """
        c = self.constraints
        Y = np.atleast_2d(Y)
        sf = safety_factor(self.scenario["yield_strength_mpa"], Y[:, 1])
        cols = [
            (c["min_safety_factor"] - sf) / c["min_safety_factor"],
            (c["min_first_mode_hz"] - Y[:, 3]) / c["min_first_mode_hz"],
            (Y[:, 2] - c["max_deflection_mm"]) / c["max_deflection_mm"],
            (c["min_log10_fatigue_life"] - Y[:, 4]) / c["min_log10_fatigue_life"],
        ]
        if G is not None:
            cols += list(manufacturing_violations(np.atleast_2d(G), c).T)
        return np.column_stack(cols)

    def evaluate(self, G: np.ndarray) -> dict:
        G = np.clip(np.atleast_2d(np.asarray(G, dtype=float)), self.xl, self.xu)
        Y = self.surrogate.predict(self.full_inputs(G))  # guard-railed inside the surrogate
        V = self.violations(Y, G)
        cv = np.clip(V, 0.0, None).sum(axis=1)
        F = Y[:, [0, 1]]
        if self.record:
            self.archive.append((self.tag, G.copy(), F.copy(), cv.copy()))
            self.n_evals += len(G)
        return {"Y": Y, "F": F, "V": V, "cv": cv, "feasible": cv <= 0.0}

    def objective_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Robust normalisation bounds from an unrecorded pilot sample."""
        if self._f_lo is None:
            rec, self.record = self.record, False
            pilot = self.evaluate(lhs(self.xl, self.xu, 512, seed=7))
            self.record = rec
            self._f_lo = np.percentile(pilot["F"], 1, axis=0)
            self._f_hi = np.percentile(pilot["F"], 99, axis=0)
        return self._f_lo, self._f_hi

    def normalise(self, F):
        lo, hi = self.objective_bounds()
        return (F - lo) / (hi - lo)

    def archive_for(self, tag: str):
        rows = [a for a in self.archive if a[0] == tag]
        if not rows:
            return np.empty((0, self.n_var)), np.empty((0, 2)), np.empty(0)
        return (np.vstack([r[1] for r in rows]), np.vstack([r[2] for r in rows]),
                np.concatenate([r[3] for r in rows]))


# ----------------------------------------------------------------------------- utilities
def manufacturing_violations(G: np.ndarray, constraints: dict | None = None) -> np.ndarray:
    """Exact geometric manufacturability checks for rotary-draw tube bending (> 0 = violated).

    G columns follow ``config.GEOMETRY_FEATURES`` (OD, wall, gusset, bend radius)."""
    c = constraints or config.CONSTRAINTS
    od, wall, radius = G[:, 0], G[:, 1], G[:, 3]
    d_over_t = od / wall
    bend_ratio = radius / od
    return np.column_stack([(d_over_t - c["max_d_over_t"]) / c["max_d_over_t"],
                            (c["min_bend_ratio"] - bend_ratio) / c["min_bend_ratio"]])


def knee_point(F: np.ndarray) -> int:
    """Index of the knee of a 2-objective front: the point farthest from the straight line joining
    the two extreme designs (in normalised objective space) - the best-balanced compromise."""
    if len(F) <= 2:
        return int(np.argmin(F[:, 0])) if len(F) else -1
    lo, hi = F.min(axis=0), F.max(axis=0)
    Fn = (F - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
    a, b = Fn[np.argmin(Fn[:, 0])], Fn[np.argmin(Fn[:, 1])]
    d = b - a
    norm_d = np.hypot(*d) or 1.0
    dist = np.abs(d[0] * (Fn[:, 1] - a[1]) - d[1] * (Fn[:, 0] - a[0])) / norm_d
    below = (Fn[:, 0] - a[0]) * d[1] - (Fn[:, 1] - a[1]) * d[0] >= 0  # on the ideal side of the chord
    dist = np.where(below, dist, -dist)
    return int(np.argmax(dist))


def lhs(xl, xu, n, seed):
    return xl + qmc.LatinHypercube(d=len(xl), seed=seed).random(n) * (xu - xl)


def non_dominated_mask(F: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (minimisation)."""
    n = len(F)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort(F.T[::-1])
    keep = np.zeros(n, dtype=bool)
    best_f2 = np.inf
    for i in order:  # 2-objective sweep: sorted by f1 then f2
        if F[i, 1] < best_f2:
            keep[i] = True
            best_f2 = F[i, 1]
    if F.shape[1] > 2:  # generic fallback
        keep = np.array([not np.any(np.all(F <= F[i], 1) & np.any(F < F[i], 1)) for i in range(n)])
    return keep


def pareto_front(F: np.ndarray, cv: np.ndarray) -> np.ndarray:
    """Indices of the feasible non-dominated set."""
    feas = np.where(cv <= 0.0)[0]
    if len(feas) == 0:
        return feas
    return feas[non_dominated_mask(F[feas])]


def _constrained_ranks(F: np.ndarray, cv: np.ndarray) -> np.ndarray:
    """Deb's constraint-domination non-dominated sorting -> front rank per individual."""
    n = len(F)
    dom = np.zeros((n, n), dtype=bool)
    feas = cv <= 0
    for i in range(n):
        better_obj = np.all(F[i] <= F, 1) & np.any(F[i] < F, 1)
        dom[i] = np.where(feas[i] & feas, better_obj,
                          np.where(feas[i] & ~feas, True, np.where(~feas[i] & ~feas, cv[i] < cv, False)))
    dom_count = dom.sum(axis=0)
    ranks = np.full(n, -1)
    current = np.where(dom_count == 0)[0]
    r = 0
    while len(current):
        ranks[current] = r
        dom_count = dom_count - dom[current].sum(axis=0)
        dom_count[ranks >= 0] = -1
        current = np.where(dom_count == 0)[0]
        r += 1
    return ranks


def _crowding(F: np.ndarray) -> np.ndarray:
    n = len(F)
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(F.shape[1]):
        o = np.argsort(F[:, j])
        span = F[o[-1], j] - F[o[0], j] or 1.0
        d[o[0]] = d[o[-1]] = np.inf
        d[o[1:-1]] += (F[o[2:], j] - F[o[:-2], j]) / span
    return d


# ----------------------------------------------------------------------------- algorithms
def lhs_screening(problem: ChassisProblem, n: int = config.N_OPT_CANDIDATES, seed: int = config.SEED):
    problem.evaluate(lhs(problem.xl, problem.xu, n, seed))


def nsga2(problem: ChassisProblem, pop_size: int = 100, generations: int = 50, seed: int = config.SEED,
          eta_c: float = 15.0, eta_m: float = 20.0, p_c: float = 0.9):
    rng = np.random.default_rng(seed)
    xl, xu, nv = problem.xl, problem.xu, problem.n_var
    P = lhs(xl, xu, pop_size, seed)
    ev = problem.evaluate(P)
    F, cv = problem.normalise(ev["F"]), ev["cv"]
    for _ in range(generations - 1):
        ranks = _constrained_ranks(F, cv)
        crowd = np.zeros(len(P))
        for r in np.unique(ranks):
            m = ranks == r
            crowd[m] = _crowding(F[m])
        # binary tournament
        a, b = rng.integers(0, len(P), (2, pop_size))
        win = np.where((ranks[a] < ranks[b]) | ((ranks[a] == ranks[b]) & (crowd[a] > crowd[b])), a, b)
        parents = P[win]
        # SBX crossover
        children = parents.copy()
        for i in range(0, pop_size - 1, 2):
            if rng.random() < p_c:
                u = rng.random(nv)
                beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
                p1, p2 = parents[i], parents[i + 1]
                children[i] = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                children[i + 1] = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        # polynomial mutation
        mut = rng.random(children.shape) < 1.0 / nv
        u = rng.random(children.shape)
        delta = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
        children = np.clip(children + mut * delta * (xu - xl), xl, xu)
        evc = problem.evaluate(children)
        P = np.vstack([P, children])
        F = np.vstack([F, problem.normalise(evc["F"])])
        cv = np.concatenate([cv, evc["cv"]])
        # environmental selection
        ranks = _constrained_ranks(F, cv)
        keep = []
        for r in np.unique(ranks):
            idx = np.where(ranks == r)[0]
            if len(keep) + len(idx) <= pop_size:
                keep.extend(idx)
            else:
                cd = _crowding(F[idx])
                keep.extend(idx[np.argsort(-cd)][: pop_size - len(keep)])
                break
        keep = np.array(keep)
        P, F, cv = P[keep], F[keep], cv[keep]


def epsilon_constraint(problem: ChassisProblem, n_levels: int = 12, popsize: int = 10, maxiter: int = 25,
                       seed: int = config.SEED):
    """min mass  s.t. stress <= eps_k (+ engineering constraints), via penalised DE."""
    lo, hi = problem.objective_bounds()
    eps_levels = np.linspace(lo[1], hi[1] * 0.6, n_levels)
    span = (problem.xu - problem.xl)

    def make_obj(eps):
        def obj(Gu):  # vectorised: Gu shape (n_var, S) in unit space
            G = problem.xl + Gu.T * span
            ev = problem.evaluate(G)
            fn = problem.normalise(ev["F"])
            pen = ev["cv"] + np.clip(ev["F"][:, 1] - eps, 0, None) / eps
            return fn[:, 0] + 1e3 * pen
        return obj

    for k, eps in enumerate(eps_levels):
        differential_evolution(make_obj(eps), bounds=[(0.0, 1.0)] * problem.n_var, popsize=popsize,
                               maxiter=maxiter, tol=1e-6, polish=False, seed=seed + k, vectorized=True,
                               updating="deferred")


def weighted_sum_slsqp(problem: ChassisProblem, n_weights: int = 11, n_starts: int = 3, seed: int = config.SEED):
    span = problem.xu - problem.xl
    starts = qmc.LatinHypercube(d=problem.n_var, seed=seed).random(n_weights * n_starts)
    cache: dict = {}

    def ev(u):
        key = tuple(np.round(u, 12))
        if key not in cache:
            cache.clear() if len(cache) > 64 else None
            cache[key] = problem.evaluate(problem.xl + np.clip(u, 0, 1) * span)
        return cache[key]

    cons = NonlinearConstraint(lambda u: -ev(u)["V"][0], 0.0, np.inf)
    for k, w in enumerate(np.linspace(0.0, 1.0, n_weights)):
        for s in range(n_starts):
            u0 = starts[k * n_starts + s]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                minimize(lambda u: float(w * problem.normalise(ev(u)["F"])[0, 0]
                                         + (1 - w) * problem.normalise(ev(u)["F"])[0, 1]),
                         u0, method="SLSQP", bounds=[(0.0, 1.0)] * problem.n_var, constraints=[cons],
                         options={"maxiter": 60, "ftol": 1e-7})


def parego(problem: ChassisProblem, budget: int = 150, n_init: int = 40, seed: int = config.SEED,
           n_candidates: int = 3000, rho: float = 0.05):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    rng = np.random.default_rng(seed)
    span = problem.xu - problem.xl
    U = qmc.LatinHypercube(d=problem.n_var, seed=seed).random(n_init)
    ev = problem.evaluate(problem.xl + U * span)
    Fn, cv = problem.normalise(ev["F"]), ev["cv"]
    weights = np.linspace(0, 1, 11)
    kernel = ConstantKernel(1.0) * Matern(np.ones(problem.n_var) * 0.3, (1e-2, 1e2), nu=2.5) + WhiteKernel(1e-4)
    for _ in range(budget - n_init):
        w1 = rng.choice(weights)
        lam = np.array([w1, 1 - w1])
        s = np.max(lam * Fn, axis=1) + rho * (lam * Fn).sum(axis=1) + 10.0 * cv
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=0, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(U, s)
        best = s.min()
        best_u = U[np.argmin(s)]
        cand = np.vstack([rng.random((n_candidates, problem.n_var)),
                          np.clip(best_u + rng.normal(0, 0.05, (300, problem.n_var)), 0, 1)])
        mu, sd = gp.predict(cand, return_std=True)
        sd = np.clip(sd, 1e-9, None)
        z = (best - mu) / sd
        ei = (best - mu) * norm.cdf(z) + sd * norm.pdf(z)
        u_next = cand[np.argmax(ei)][None, :]
        evn = problem.evaluate(problem.xl + u_next * span)
        U = np.vstack([U, u_next])
        Fn = np.vstack([Fn, problem.normalise(evn["F"])])
        cv = np.concatenate([cv, evn["cv"]])


ALGORITHM_FUNCS = {
    "lhs_screening": lhs_screening,
    "nsga2": nsga2,
    "epsilon_constraint": epsilon_constraint,
    "weighted_sum_slsqp": weighted_sum_slsqp,
    "parego": parego,
}


def run_algorithm(problem: ChassisProblem, name: str, **kwargs) -> float:
    """Run one algorithm, tagging its evaluations in the problem archive. Returns wall seconds."""
    problem.tag = name
    t0 = time.perf_counter()
    ALGORITHM_FUNCS[name](problem, **kwargs)
    return time.perf_counter() - t0


def screen_candidates(surrogate, scenario=None, constraints=None, n: int = config.N_OPT_CANDIDATES,
                      seed: int = config.SEED) -> pd.DataFrame:
    """Fast LHS screening returning a tidy DataFrame (used by the app for live re-runs)."""
    prob = ChassisProblem(surrogate, scenario or dict(config.DEFAULT_SCENARIO),
                          constraints or dict(config.CONSTRAINTS), record=False)
    G = lhs(prob.xl, prob.xu, n, seed)
    ev = prob.evaluate(G)
    df = pd.DataFrame(prob.full_inputs(G), columns=config.FEATURES)
    for j, t in enumerate(config.TARGETS):
        df[t] = ev["Y"][:, j]
    df["safety_factor"] = safety_factor(prob.scenario["yield_strength_mpa"], ev["Y"][:, 1])
    df["d_over_t"] = G[:, 0] / G[:, 1]
    df["bend_ratio"] = G[:, 3] / G[:, 0]
    df["feasible"] = ev["feasible"]
    df["pareto"] = False
    df.loc[pareto_front(ev["F"], ev["cv"]), "pareto"] = True
    return df


if __name__ == "__main__":
    from src.optimizer_benchmark import main

    main()
