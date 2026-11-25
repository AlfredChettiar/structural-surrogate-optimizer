"""The paper's Table-4 models with the authors' best hyperparameters, and the shared CV protocol.

Protocol (inferred, the authors' AutoML code is not public): per-fold StandardScaler on X and Y; R2/MSE/MAE are the
uniform mean over the 10 standardised targets; 10 repeats x 5-fold KFold.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.framed import config as fc
from src.framed import paper_reference as ref

HP = ref.HYPERPARAMS


# ----------------------------------------------------------------------------- models
class Scaled:
    """Standardise X and Y around any multi-output estimator (what the paper's models see)."""

    def __init__(self, est):
        self.est, self.xs, self.ys = est, StandardScaler(), StandardScaler()

    def fit(self, X, Y):
        self.est.fit(self.xs.fit_transform(np.asarray(X, float)), self.ys.fit_transform(Y))
        return self

    def predict(self, X):
        return self.ys.inverse_transform(np.atleast_2d(self.est.predict(self.xs.transform(np.asarray(X, float)))))


class XGBEarlyStop:
    """XGBoost with the paper's early stopping on a 10 % inner validation split."""

    def __init__(self, seed: int, **kw):
        self.seed, self.kw = seed, kw

    def fit(self, X, Y):
        rng = np.random.default_rng(self.seed)
        val = rng.random(len(X)) < 0.1
        p = dict(HP["xgboost"])
        self.m = XGBRegressor(n_estimators=2000, tree_method="hist", multi_strategy="one_output_per_tree",
                              random_state=self.seed, n_jobs=1, **p, **self.kw)
        self.m.fit(X[~val], Y[~val], eval_set=[(X[val], Y[val])], verbose=False)
        return self

    def predict(self, X):
        return self.m.predict(X)


class TorchMLP:
    """MLP with dropout and early stopping (paper Table 9 by default)."""

    def __init__(self, seed: int, hidden=tuple(HP["neural_net"]["hidden"]), dropout=HP["neural_net"]["dropout"],
                 lr=HP["neural_net"]["lr"], batch_size=HP["neural_net"]["batch_size"],
                 patience=HP["neural_net"]["patience"], act=torch.nn.ReLU, max_epochs: int = 3000,
                 weight_decay: float = 0.0):
        self.seed, self.hidden, self.dropout, self.lr = seed, hidden, dropout, lr
        self.batch_size, self.patience, self.act, self.max_epochs, self.wd = batch_size, patience, act, max_epochs, weight_decay

    def _net(self, n_in, n_out):
        layers, d = [], n_in
        for h in self.hidden:
            layers += [torch.nn.Linear(d, h), self.act(), torch.nn.Dropout(self.dropout)]
            d = h
        return torch.nn.Sequential(*layers, torch.nn.Linear(d, n_out))

    def fit(self, X, Y):
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
        g = torch.Generator().manual_seed(self.seed)
        val = torch.rand(len(X), generator=g) < 0.1
        Xt, Yt, Xv, Yv = X[~val], Y[~val], X[val], Y[val]
        self.net = self._net(X.shape[1], Y.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_state, wait = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.net.train()
            for idx in torch.randperm(len(Xt), generator=g).split(self.batch_size):
                opt.zero_grad()
                torch.nn.functional.mse_loss(self.net(Xt[idx]), Yt[idx]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = torch.nn.functional.mse_loss(self.net(Xv), Yv).item()
            if v < best - 1e-6:
                best, wait = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            elif (wait := wait + 1) >= self.patience:
                break
        if best_state is None:
            raise RuntimeError("validation loss never improved (NaN or empty validation split)")
        self.net.load_state_dict(best_state)
        self.val_mask = val.numpy()
        self.epochs = _ + 1
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.tensor(X, dtype=torch.float32)).numpy()


def paper_models(seed: int = fc.SEED) -> dict[str, Callable[[], object]]:
    """Factories taking (X DataFrame, Y ndarray) in physical units."""
    models = {
        "decision_tree": lambda: Scaled(DecisionTreeRegressor(random_state=seed, **HP["decision_tree"])),
        "knn": lambda: Scaled(KNeighborsRegressor(**HP["knn"])),
        "xgboost": lambda: Scaled(XGBEarlyStop(seed)),
        "neural_net": lambda: Scaled(TorchMLP(seed)),
    }
    from src.framed import automl
    if automl.available():
        models["autogluon"] = lambda: automl.AutoGluonModel(seed)
    return models


# ----------------------------------------------------------------------------- metrics + CV
def max_rel_err_pct(yt: np.ndarray, yp: np.ndarray, floor: float) -> float:
    return float(100 * np.max(np.abs(yp - yt) / np.maximum(np.abs(yt), floor)))


def rel_floors(Y_train: np.ndarray) -> np.ndarray:
    """Denominator floor for max relative error: 5 % of the training median for displacements, ~0 otherwise."""
    floors = np.full(Y_train.shape[1], 1e-12)
    for j, c in enumerate(fc.Y_COLS):
        if c in fc.DISP_COLS:
            floors[j] = 0.05 * np.median(np.abs(Y_train[:, j]))
    return floors


def fold_metrics(Y_tr: np.ndarray, Y_te: np.ndarray, P: np.ndarray) -> dict:
    s = StandardScaler().fit(Y_tr)
    zt, zp = s.transform(Y_te), s.transform(P)
    fl = rel_floors(Y_tr)
    per = {c: {"r2": float(r2_score(Y_te[:, j], P[:, j])), "mae": float(mean_absolute_error(Y_te[:, j], P[:, j])),
               "max_rel_err_pct": max_rel_err_pct(Y_te[:, j], P[:, j], fl[j]),
               "r2_std": float(r2_score(zt[:, j], zp[:, j]))} for j, c in enumerate(fc.Y_COLS)}
    return {"r2": float(r2_score(zt, zp)), "mse": float(mean_squared_error(zt, zp)),
            "mae": float(mean_absolute_error(zt, zp)), "per_target": per}


def _one_fold(factory, X, Y, tr, te):
    t0 = time.perf_counter()
    m = factory().fit(X.iloc[tr], Y[tr])
    fit_s = time.perf_counter() - t0
    P = np.asarray(m.predict(X.iloc[te]), float)
    return fold_metrics(Y[tr], Y[te], P), P, fit_s


def cross_validate(factory, X: pd.DataFrame, Y: np.ndarray, folds, n_jobs: int = fc.N_JOBS) -> dict:
    Y = np.asarray(Y, float)
    res = Parallel(n_jobs=n_jobs)(delayed(_one_fold)(factory, X, Y, tr, te) for tr, te in folds)
    oof = np.full_like(Y, np.nan)
    for (tr, te), (_, P, _) in zip(folds[:5], res[:5]):
        oof[te] = P
    ms = [r[0] for r in res]
    agg = lambda k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms])))
    per = {c: {k: float(np.mean([m["per_target"][c][k] for m in ms])) for k in ms[0]["per_target"][c]}
           for c in fc.Y_COLS}
    # max error is a worst case: take the max over folds, not the mean
    for c in fc.Y_COLS:
        per[c]["max_rel_err_pct"] = float(np.max([m["per_target"][c]["max_rel_err_pct"] for m in ms]))
    out = {k: {"mean": agg(k)[0], "std": agg(k)[1]} for k in ("r2", "mse", "mae")}
    out.update(per_target=per, fold_scores={k: [m[k] for m in ms] for k in ("r2", "mae")},
               fit_seconds=float(np.mean([r[2] for r in res])), n_folds=len(folds), oof_pred=oof)
    return out


def reproduction_table(results: dict) -> pd.DataFrame:
    rows = []
    for m, paper in ref.TABLE4.items():
        r = results.get(m)
        ours = (r or {}).get("r2", {}).get("mean", np.nan)
        ours_mae = (r or {}).get("mae", {}).get("mean", np.nan)
        ok = abs(ours - paper["r2"]) <= ref.R2_TOLERANCE and abs(ours_mae - paper["mae"]) <= ref.MAE_TOLERANCE
        rows.append({"model": m, "paper_r2": paper["r2"], "ours_r2": ours,
                     "ours_r2_std": (r or {}).get("r2", {}).get("std", np.nan), "delta_r2": ours - paper["r2"],
                     "paper_mae": paper["mae"], "ours_mae": (r or {}).get("mae", {}).get("mean", np.nan),
                     "ours_mae_std": (r or {}).get("mae", {}).get("std", np.nan),
                     "paper_mse": paper["mse"], "ours_mse": (r or {}).get("mse", {}).get("mean", np.nan),
                     "n_folds": (r or {}).get("n_folds", 0),
                     "status": "not run" if r is None else ("pass" if ok else "outside tolerance")})
    return pd.DataFrame(rows).set_index("model")


def ranking_preserved(table: pd.DataFrame) -> dict:
    """Paper order AutoML >= XGB > NN > DT > KNN, checked on the models that ran (strict except AutoML-XGB)."""
    ran = [m for m in ref.TABLE4_RANKING if not np.isnan(table.loc[m, "ours_r2"])]
    vals = [table.loc[m, "ours_r2"] for m in ran]
    ok = all((a >= b) if (m == "autogluon") else (a > b) for m, a, b in zip(ran, vals, vals[1:]))
    return {"compared": ran, "complete": len(ran) == len(ref.TABLE4_RANKING),
            "preserved": bool(ok) if len(ran) >= 2 else None}
