"""Uniform wrappers around every candidate surrogate.

Every surrogate:
  * takes inputs in physical units (columns = ``config.FEATURES``),
  * learns standardised, transformed targets (log for positive quantities),
  * returns guard-railed predictions in physical units from ``predict``.
Models with ``supports_uncertainty`` also implement ``predict_interval``.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from src import config
from src.guardrails import apply_guardrails, clip_inputs
from src.physics import closed_form

_LOG_MASK = np.array([t in config.LOG_TARGETS for t in config.TARGETS])
Z95 = 1.959963984540054


def to_transformed(Y: np.ndarray) -> np.ndarray:
    Y = np.atleast_2d(np.asarray(Y, dtype=float)).copy()
    Y[:, _LOG_MASK] = np.log(np.clip(Y[:, _LOG_MASK], 1e-9, None))
    return Y


def from_transformed(T: np.ndarray) -> np.ndarray:
    T = np.atleast_2d(np.asarray(T, dtype=float)).copy()
    T[:, _LOG_MASK] = np.exp(np.clip(T[:, _LOG_MASK], -30, 30))
    return T


class Surrogate:
    """Base class: scaling, target transform, guardrails and persistence."""

    name = "base"
    label = "Base"
    family = "base"
    supports_uncertainty = False
    trainable = True

    def __init__(self, seed: int = config.SEED):
        self.seed = seed
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.fit_seconds = float("nan")

    # ---- hooks for subclasses (operate on standardised arrays)
    def _fit(self, Xs: np.ndarray, Ts: np.ndarray, X_raw: np.ndarray) -> None:
        raise NotImplementedError

    def _predict(self, Xs: np.ndarray, X_raw: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # ---- public API
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "Surrogate":
        t0 = time.perf_counter()
        X = np.asarray(X, dtype=float)
        T = self._targets_for_fit(X, Y)
        Xs = self.x_scaler.fit_transform(X)
        Ts = self.y_scaler.fit_transform(T)
        self._fit(Xs, Ts, X)
        self.fit_seconds = time.perf_counter() - t0
        return self

    def _targets_for_fit(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return to_transformed(Y)

    def _untransform(self, Ts: np.ndarray, X_raw: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse_transform(Ts)

    def predict_transformed(self, X: np.ndarray) -> np.ndarray:
        Xc, _ = clip_inputs(X)
        return self._untransform(self._predict(self.x_scaler.transform(Xc), Xc), Xc)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return apply_guardrails(from_transformed(self.predict_transformed(X)))

    def save(self, directory: Path) -> Path:
        path = Path(directory) / f"{self.name}.joblib"
        joblib.dump(self, path)
        return path


# ----------------------------------------------------------------------------- reference
class ClosedFormReference(Surrogate):
    """Uncalibrated hand calculation (nominal stress). Reference only - not trained."""

    name, label, family, trainable = "closed_form", "Closed-form hand calc", "physics", False

    def fit(self, X, Y):
        self.fit_seconds = 0.0
        return self

    def predict_transformed(self, X):
        Xc, _ = clip_inputs(X)
        return to_transformed(closed_form(Xc))


# ----------------------------------------------------------------------------- sklearn models
class RidgePolynomial(Surrogate):
    name, label, family = "ridge_poly", "Ridge polynomial (deg 3)", "linear"

    def _fit(self, Xs, Ts, X_raw):
        self.model = make_pipeline(PolynomialFeatures(3, include_bias=False), StandardScaler(),
                                   RidgeCV(alphas=np.logspace(-5, 2, 15)))
        self.model.fit(Xs, Ts)

    def _predict(self, Xs, X_raw):
        return self.model.predict(Xs)


class GaussianProcess(Surrogate):
    name, label, family, supports_uncertainty = "gaussian_process", "Gaussian process (Matern 5/2 ARD)", "kernel", True

    def __init__(self, seed: int = config.SEED, n_restarts: int = 0, fixed_kernels=None):
        """``fixed_kernels``: reuse already-optimised kernels (no hyper-parameter search).
        Used in CV / learning-curve / extrapolation studies to bound compute cost."""
        super().__init__(seed)
        self.n_restarts = n_restarts
        self.fixed_kernels = fixed_kernels

    def _fit(self, Xs, Ts, X_raw):
        d = Xs.shape[1]
        self.models = []
        for j in range(Ts.shape[1]):
            if self.fixed_kernels is not None:
                gp = GaussianProcessRegressor(self.fixed_kernels[j], optimizer=None, random_state=self.seed)
            else:
                kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(np.ones(d), (1e-2, 1e5), nu=2.5) \
                    + WhiteKernel(1e-3, (1e-8, 1e-1))
                gp = GaussianProcessRegressor(kernel, normalize_y=False, n_restarts_optimizer=self.n_restarts,
                                              random_state=self.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(Xs, Ts[:, j])
            self.models.append(gp)

    def fitted_kernels(self):
        return [m.kernel_ for m in self.models]

    def _predict(self, Xs, X_raw):
        return np.column_stack([m.predict(Xs) for m in self.models])

    def predict_transformed_std(self, X):
        """Mean and std in transformed (log / log10) space."""
        Xc, _ = clip_inputs(X)
        Xs = self.x_scaler.transform(Xc)
        mus, sds = zip(*(m.predict(Xs, return_std=True) for m in self.models))
        mu = self.y_scaler.inverse_transform(np.column_stack(mus))
        sd = np.column_stack(sds) * self.y_scaler.scale_
        return mu, sd

    def predict_interval(self, X, level: float = 0.95):
        from scipy.stats import norm

        z = norm.ppf(0.5 + level / 2.0) if level != 0.95 else Z95
        mu, sd = self.predict_transformed_std(X)
        lo = apply_guardrails(from_transformed(mu - z * sd))
        mean = apply_guardrails(from_transformed(mu))
        hi = apply_guardrails(from_transformed(mu + z * sd))
        return lo, mean, hi


class RandomForest(Surrogate):
    name, label, family = "random_forest", "Random forest", "tree"

    def _fit(self, Xs, Ts, X_raw):
        self.model = RandomForestRegressor(n_estimators=200, min_samples_leaf=1, max_features=0.8,
                                           n_jobs=-1, random_state=self.seed)
        self.model.fit(Xs, Ts)
        self.model.set_params(n_jobs=1)  # single-design queries are faster without a thread pool

    def _predict(self, Xs, X_raw):
        return self.model.predict(Xs)


class GradientBoosting(Surrogate):
    name, label, family = "hist_gradient_boosting", "Hist gradient boosting", "tree"

    def _fit(self, Xs, Ts, X_raw):
        base = HistGradientBoostingRegressor(max_iter=800, learning_rate=0.05, max_leaf_nodes=31,
                                             l2_regularization=1e-3, early_stopping=True,
                                             validation_fraction=0.1, n_iter_no_change=40,
                                             random_state=self.seed)
        self.model = MultiOutputRegressor(base)
        self.model.fit(Xs, Ts)

    def _predict(self, Xs, X_raw):
        return self.model.predict(Xs)


class SupportVector(Surrogate):
    name, label, family = "svr", "Support vector regression (RBF)", "kernel"

    def _fit(self, Xs, Ts, X_raw):
        self.model = MultiOutputRegressor(SVR(C=30.0, epsilon=0.01, gamma="scale"))
        self.model.fit(Xs, Ts)

    def _predict(self, Xs, X_raw):
        return self.model.predict(Xs)


# ----------------------------------------------------------------------------- torch models
class _TorchSurrogate(Surrogate):
    family = "neural"
    width, n_blocks, epochs = 96, 3, 800

    def __init__(self, seed: int = config.SEED, epochs: int | None = None):
        super().__init__(seed)
        if epochs is not None:
            self.epochs = epochs
        self.history: dict = {}

    def _net_inputs(self, Xs, X_raw):
        return Xs

    def _fit(self, Xs, Ts, X_raw):
        from src.model import ResidualMLP, train_network

        Z = self._net_inputs(Xs, X_raw)
        self.net = ResidualMLP(Z.shape[1], Ts.shape[1], self.width, self.n_blocks)
        self.history = train_network(self.net, Z, Ts, epochs=self.epochs, seed=self.seed)

    def _predict(self, Xs, X_raw):
        from src.model import predict_network

        return predict_network(self.net, self._net_inputs(Xs, X_raw))

    def save(self, directory: Path) -> Path:
        import torch

        path = Path(directory) / f"{self.name}.pt"
        state = self.net.state_dict()
        self.net = None  # pickle the wrapper (scalers etc.) without the module
        try:
            joblib.dump(self, Path(directory) / f"{self.name}.wrapper.joblib")
        finally:
            from src.model import ResidualMLP

            n_in = state["inp.0.weight"].shape[1]
            n_out = state["head.weight"].shape[0]
            self.net = ResidualMLP(n_in, n_out, self.width, self.n_blocks)
            self.net.load_state_dict(state)
            self.net.eval()
        torch.save(state, path)
        return path

    @classmethod
    def load_from(cls, directory: Path, name: str):
        import torch
        from src.model import ResidualMLP

        wrapper = joblib.load(Path(directory) / f"{name}.wrapper.joblib")
        state = torch.load(Path(directory) / f"{name}.pt", map_location="cpu", weights_only=True)
        n_in = state["inp.0.weight"].shape[1]
        n_out = state["head.weight"].shape[0]
        wrapper.net = ResidualMLP(n_in, n_out, wrapper.width, wrapper.n_blocks)
        wrapper.net.load_state_dict(state)
        wrapper.net.eval()
        return wrapper


class ResidualMLPSurrogate(_TorchSurrogate):
    name, label = "residual_mlp", "Residual MLP (PyTorch)"


class PhysicsGuidedMLP(_TorchSurrogate):
    """Multi-fidelity: the network sees the hand-calc outputs and learns the correction
    ``T(FE) - T(hand calc)`` - i.e. the stress-concentration / bend-flexibility physics the
    hand calc misses. Predictions are hand calc + learned correction."""

    name, label = "physics_guided_mlp", "Physics-guided residual MLP"

    def _cf(self, X_raw):
        return to_transformed(closed_form(X_raw))

    def _targets_for_fit(self, X, Y):
        return to_transformed(Y) - self._cf(X)

    def _net_inputs(self, Xs, X_raw):
        cf = self._cf(X_raw)
        if not hasattr(self, "cf_scaler"):
            self.cf_scaler = StandardScaler().fit(cf)
        return np.hstack([Xs, self.cf_scaler.transform(cf)])

    def fit(self, X, Y):
        if hasattr(self, "cf_scaler"):
            del self.cf_scaler
        return super().fit(X, Y)

    def _untransform(self, Ts, X_raw):
        return self.y_scaler.inverse_transform(Ts) + self._cf(X_raw)


# ----------------------------------------------------------------------------- registry
REGISTRY: dict[str, type[Surrogate]] = {
    cls.name: cls
    for cls in (RidgePolynomial, GaussianProcess, ResidualMLPSurrogate, PhysicsGuidedMLP,
                RandomForest, GradientBoosting, SupportVector)
}
TORCH_MODELS = {ResidualMLPSurrogate.name, PhysicsGuidedMLP.name}


def available_models() -> list[str]:
    try:
        import torch  # noqa: F401

        return list(REGISTRY)
    except ImportError:
        return [n for n in REGISTRY if n not in TORCH_MODELS]


def make_model(name: str, **kwargs) -> Surrogate:
    return REGISTRY[name](**kwargs)


def load_surrogate(name: str, directory: Path = config.MODELS_DIR) -> Surrogate:
    if name == ClosedFormReference.name:
        return ClosedFormReference()
    if name in TORCH_MODELS:
        return REGISTRY[name].load_from(directory, name)
    return joblib.load(Path(directory) / f"{name}.joblib")


def timed_predict(model: Surrogate, X: np.ndarray, repeats: int = 5) -> float:
    """Median wall-time per call in milliseconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(X)
        times.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(times))
