"""This project's surrogates on FRAMED: physics features, log targets, guardrails and uncertainty."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from src.framed import config as fc
from src.framed.baselines import TorchMLP, XGBEarlyStop
from src.framed.physics_features import add_physics_features, analytic_mass_prior

Z95 = 1.959963984540054
MASS_J = fc.Y_COLS.index(fc.MASS_COL)


# ----------------------------------------------------------------------------- transforms + guardrails
def target_eps(Y: np.ndarray) -> np.ndarray:
    return 0.01 * np.median(np.abs(Y), axis=0) + 1e-12


def transform_targets(Y: np.ndarray, eps: np.ndarray) -> np.ndarray:
    return np.log(np.clip(Y, 0, None) + eps)


def inverse_transform_targets(T: np.ndarray, eps: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(T, -50, 50)) - eps


def guard_framed(Y: np.ndarray) -> np.ndarray:
    """Physical guardrails: finite, mass > 0, displacement magnitudes >= 0, 0 <= SF <= SF_CAP."""
    Y = np.atleast_2d(np.asarray(Y, float)).copy()
    big = np.nanmax(np.where(np.isfinite(Y), Y, np.nan), axis=0)
    Y = np.where(np.isposinf(Y), np.nan_to_num(big, nan=0.0)[None], Y)  # +inf -> largest finite value in column
    Y = np.clip(np.nan_to_num(Y, nan=0.0, neginf=0.0), 0.0, None)
    sf = [fc.Y_COLS.index(c) for c in fc.SF_COLS]
    Y[:, sf] = np.clip(Y[:, sf], 0.0, fc.SF_CAP)
    Y[:, MASS_J] = np.maximum(Y[:, MASS_J], 1e-6)
    return Y


def clip_framed_inputs(X: pd.DataFrame, lo: pd.Series, hi: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    flag = ((X < lo) | (X > hi)).any(axis=1).to_numpy()
    return X.clip(lo, hi, axis=1), flag


# ----------------------------------------------------------------------------- model wrapper
class PhysicsSurrogate:
    """physics features -> standardise -> estimator on log targets -> inverse -> guardrails.

    mass_residual=True learns log(mass) - log(mass_prior) (physics-guided residual on the analytic prior)."""

    supports_uncertainty = False

    def __init__(self, est, mass_residual: bool = False, features: bool = True):
        self.est, self.mass_residual, self.features = est, mass_residual, features

    def _feats(self, X):
        return add_physics_features(X) if self.features else X

    def _prior(self, X):
        return np.log(analytic_mass_prior(X) + self.eps[MASS_J]) if self.mass_residual else 0.0

    def fit(self, X: pd.DataFrame, Y: np.ndarray):
        Y = np.asarray(Y, float)
        self.lo, self.hi = X.min(), X.max()
        self.eps = target_eps(Y)
        T = transform_targets(Y, self.eps)
        T[:, MASS_J] -= self._prior(X)
        self.xs, self.ts = StandardScaler(), StandardScaler()
        self.est.fit(self.xs.fit_transform(self._feats(X).to_numpy()), self.ts.fit_transform(T))
        return self

    def _t(self, Tz, X):
        T = self.ts.inverse_transform(np.atleast_2d(Tz))
        T[:, MASS_J] += self._prior(X)
        return T

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X, _ = clip_framed_inputs(X, self.lo, self.hi)
        Z = self.xs.transform(self._feats(X).to_numpy())
        return guard_framed(inverse_transform_targets(self._t(self.est.predict(Z), X), self.eps))

    def predict_interval(self, X: pd.DataFrame, level: float = 0.95):
        X, _ = clip_framed_inputs(X, self.lo, self.hi)
        Z = self.xs.transform(self._feats(X).to_numpy())
        mu, sd = self.est.predict_dist(Z)  # standardised log space
        z = Z95 if level == 0.95 else float(__import__("scipy").stats.norm.ppf(0.5 + level / 2))
        f = lambda A: guard_framed(inverse_transform_targets(self._t(A, X), self.eps))
        return f(mu - z * sd), f(mu), f(mu + z * sd)


class DeepEnsemble:
    def __init__(self, seed: int, n: int = 5, **mlp):
        self.members = [TorchMLP(seed + i, **mlp) for i in range(n)]

    def fit(self, X, Y):
        for m in self.members:
            m.fit(X, Y)
        # aleatoric part: each member's residual variance on its own held-out early-stopping split
        self.noise_var = np.mean([np.var(Y[m.val_mask] - m.predict(X[m.val_mask]), axis=0) for m in self.members], 0)
        return self

    def predict_dist(self, X):
        P = np.stack([m.predict(X) for m in self.members])
        return P.mean(0), np.sqrt(P.var(0) + self.noise_var)

    def predict(self, X):
        return self.predict_dist(X)[0]


class PerTargetGP:
    """One GP per target. Kernel hyperparameters are fitted on a random subsample (bounds O(n^3) cost),
    then the GP is conditioned on the full training set with those hyperparameters fixed."""

    def __init__(self, seed: int, n_features: int = 12, n_opt: int = 400):
        self.seed, self.k, self.n_opt = seed, n_features, n_opt

    def fit(self, X, Y):
        rng = np.random.default_rng(self.seed)
        sub = rng.choice(len(X), min(self.n_opt, len(X)), replace=False)
        self.cols, self.gps = [], []
        for j in range(Y.shape[1]):
            corr = np.abs([np.corrcoef(X[:, i], Y[:, j])[0, 1] if X[:, i].std() > 0 else 0 for i in range(X.shape[1])])
            cols = np.argsort(np.nan_to_num(corr))[::-1][: self.k]
            kern = ConstantKernel(1.0) * Matern(np.ones(len(cols)), (1e-2, 1e3), nu=2.5) + WhiteKernel(1e-2, (1e-6, 1))
            g = GaussianProcessRegressor(kern, normalize_y=True, random_state=self.seed).fit(X[sub][:, cols], Y[sub, j])
            full = GaussianProcessRegressor(g.kernel_, optimizer=None, normalize_y=True).fit(X[:, cols], Y[:, j])
            self.cols.append(cols)
            self.gps.append(full)
        return self

    def predict_dist(self, X):
        out = [g.predict(X[:, c], return_std=True) for g, c in zip(self.gps, self.cols)]
        return np.column_stack([o[0] for o in out]), np.column_stack([o[1] for o in out])

    def predict(self, X):
        return self.predict_dist(X)[0]


class UQ(PhysicsSurrogate):
    supports_uncertainty = True


def our_models(seed: int = fc.SEED) -> dict:
    mlp = dict(hidden=(128, 128, 128), act=torch.nn.SiLU, dropout=0.05, lr=1e-3, batch_size=128, patience=40,
               weight_decay=1e-5)
    models = {
        "xgboost_phys": lambda: PhysicsSurrogate(XGBEarlyStop(seed)),
        "deep_ensemble_phys": lambda: UQ(DeepEnsemble(seed, **mlp)),
        "pg_mlp": lambda: PhysicsSurrogate(TorchMLP(seed, **mlp), mass_residual=True),
        "gp_phys": lambda: UQ(PerTargetGP(seed)),
    }
    from src.framed import automl
    if automl.available():
        models["autogluon_phys"] = lambda: PhysicsSurrogate(automl.AutoGluonModel(seed))
    return models


# ----------------------------------------------------------------------------- UQ + latency
def interval_coverage(factory, X: pd.DataFrame, Y: np.ndarray, folds) -> dict:
    """Observed coverage of the 95 % interval per target (physical units), pooled over the given folds."""
    hits, n, width = np.zeros(Y.shape[1]), 0, []
    for tr, te in folds:
        m = factory().fit(X.iloc[tr], Y[tr])
        lo, _, hi = m.predict_interval(X.iloc[te])
        hits += ((Y[te] >= lo) & (Y[te] <= hi)).sum(0)
        n += len(te)
        width.append(np.median((hi - lo) / np.maximum(np.abs(Y[te]), 1e-12), axis=0))
    return {c: {"coverage": float(hits[j] / n), "median_rel_width": float(np.mean([w[j] for w in width]))}
            for j, c in enumerate(fc.Y_COLS)}


def _time(fn, repeats=5) -> float:
    fn()
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


def latency_report(fitted: dict, X: pd.DataFrame, nominal_fe_s: float) -> dict:
    one, batch = X.iloc[:1], X.iloc[: min(1000, len(X))]
    analytic_ms = 1e3 * _time(lambda: analytic_mass_prior(one))
    out = {"analytic_baseline_ms": analytic_ms, "nominal_fe_seconds": nominal_fe_s, "models": {}}
    for k, m in fitted.items():
        s = 1e3 * _time(lambda: m.predict(one))
        b = 1e3 * _time(lambda: m.predict(batch), repeats=3) / len(batch)
        out["models"][k] = {"single_ms": s, "batch_ms_per_design": b, "speedup_vs_analytic_single": analytic_ms / s,
                            "speedup_vs_nominal_fe_batch": nominal_fe_s * 1e3 / b}
    return out
