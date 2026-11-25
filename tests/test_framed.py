"""Tests for the FRAMED benchmark package. Data-dependent tests skip when the dataset is not downloaded."""
import numpy as np
import pandas as pd
import pytest

from src.framed import config as fc
from src.framed import paper_reference as ref
from src.framed.data import _verify, fold_indices, load_framed

DATA = fc.FRAMED_DIR / fc.FILENAMES["data"]
needs_data = pytest.mark.skipif(not DATA.exists(), reason="FRAMED data not downloaded")


@pytest.fixture(scope="module")
def framed():
    return load_framed()


# ----------------------------------------------------------------------------- Task 1: data
def test_sha_mismatch_raises(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("x")
    with pytest.raises(ValueError, match="sha256"):
        _verify(p, "0" * 64)


@needs_data
def test_load_shapes(framed):
    X, Y = framed
    assert X.shape == (4047, 39) and Y.shape == (4047, 10)
    assert not X.isna().any().any() and not Y.isna().any().any()
    assert (Y[fc.ABS_COLS] >= 0).all().all()
    assert np.allclose(X[fc.MATERIAL_COLS].sum(axis=1), 1.0)


def test_fold_indices():
    folds = fold_indices(23, n_folds=5, n_repeats=10, seed=1)
    assert len(folds) == 50
    for r in range(10):
        tests = np.concatenate([te for _, te in folds[5 * r: 5 * r + 5]])
        assert sorted(tests.tolist()) == list(range(23))
    a, b = fold_indices(23, seed=1), fold_indices(23, seed=1)
    assert all((x[1] == y[1]).all() for x, y in zip(a, b))


@needs_data
def test_case_study_frames_match_paper(framed):
    X, Y = framed
    for idx, cs in ref.CASE_STUDIES.items():
        assert X.loc[idx, f"Material={cs['material']}"] == 1
        assert Y.loc[idx, fc.MASS_COL] == pytest.approx(cs["mass"], abs=0.005)
        assert Y.loc[idx, "Sim 1 Safety Factor"] == pytest.approx(cs["sf1"], abs=0.005)
        assert Y.loc[idx, "Sim 3 Safety Factor"] == pytest.approx(cs["sf3"], abs=0.005)


# ----------------------------------------------------------------------------- Task 2: dataset claims
from src.framed import figures as F  # noqa: E402
from src.framed.paper_stats import PAIR_OBJECTIVES, PAIR_SENSE, feasibility_counts, non_dominated_mask, stats_comparison  # noqa: E402


def test_feasibility_counts_logic():
    Y = pd.DataFrame({"Sim 1 Safety Factor": [0.5, 0.5, 2, 2], "Sim 3 Safety Factor": [0.5, 2, 0.5, 2]})
    assert feasibility_counts(Y) == {"both_fail": 1, "sf1_only": 1, "sf3_only": 1, "valid": 1}


def test_non_dominated_logic():
    F_ = np.array([[1, 1], [2, 2], [0, 3], [1, 1]])  # minimise both; duplicates are not dominated
    assert non_dominated_mask(F_, np.array([-1, -1])).tolist() == [True, False, True, True]


@needs_data
def test_feasibility_matches_notebook(framed):
    got = feasibility_counts(framed[1])
    for k, v in ref.NOTEBOOK_FEASIBILITY.items():
        assert abs(got[k] - v) <= ref.COUNT_TOLERANCE


@needs_data
def test_non_dominated_matches_paper(framed):
    n = non_dominated_mask(framed[1][PAIR_OBJECTIVES].values, PAIR_SENSE).sum()
    assert abs(n - ref.NOTEBOOK_N_NON_DOMINATED) <= ref.COUNT_TOLERANCE


@needs_data
def test_stats_comparison_all_checks_pass(framed):
    df = stats_comparison(*framed)
    assert all(p is None or p for p in df["pass"])


@needs_data
def test_figures_build(framed):
    X, Y = framed
    for fig in (F.fig_material_breakdown(X), F.fig_performance_pairs(Y), F.fig_objective_correlation(Y),
                F.fig_physical_validation(), F.fig_check_status(stats_comparison(X, Y), "t")):
        assert len(fig.data) > 0
    assert len(F.fig_parity_grid(Y, Y.values, "t").data) == 20


# ----------------------------------------------------------------------------- Task 3: physics features
from src.framed.physics_features import add_physics_features, analytic_mass_prior, material_props, section_properties  # noqa: E402


def test_hollow_tube_area():
    A, I, J = section_properties(0.0286, 0.0009)
    assert A * 1e6 == pytest.approx(np.pi / 4 * (28.6**2 - 26.8**2), rel=1e-6)
    assert A * 1e6 == pytest.approx(78.32, rel=1e-4)
    assert I * 1e12 == pytest.approx(np.pi / 64 * (28.6**4 - 26.8**4), rel=1e-6)
    assert J == pytest.approx(2 * I)


def test_solid_rod_when_thick():
    A, I, _ = section_properties(0.00235, 0.010)
    assert A == pytest.approx(np.pi / 4 * 0.00235**2) and np.isfinite(I) and I > 0


def test_material_props_rejects_bad_onehot():
    X = pd.DataFrame([[0.5, 0.5, 0.0]], columns=fc.MATERIAL_COLS)
    with pytest.raises(ValueError, match="one-hot"):
        material_props(X)


@needs_data
def test_mass_prior_tracks_mass(framed):
    X, Y = framed
    r = np.corrcoef(analytic_mass_prior(X), Y[fc.MASS_COL])[0, 1]
    assert r > 0.8
    feats = add_physics_features(X)
    assert np.isfinite(feats.to_numpy()).all()


# ----------------------------------------------------------------------------- Task 4: Table 4 reproduction
from src.framed import automl  # noqa: E402
from src.framed.baselines import cross_validate, fold_metrics, paper_models, ranking_preserved, reproduction_table  # noqa: E402


class _Oracle:
    def __init__(self, Y):
        self.Y = Y

    def fit(self, X, Y):
        return self

    def predict(self, X):
        return self.Y[X.index.to_numpy()]


def test_cv_metric_on_perfect_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((40, 3)))
    Y = rng.random((40, 10)) + 0.1
    r = cross_validate(lambda: _Oracle(Y), X, Y, fold_indices(40, 5, 1), n_jobs=1)
    assert r["r2"]["mean"] == pytest.approx(1.0) and r["mae"]["mean"] == pytest.approx(0.0, abs=1e-12)


def test_fold_metrics_standardised():
    rng = np.random.default_rng(1)
    Ytr, Yte = rng.random((50, 10)), rng.random((20, 10))
    m = fold_metrics(Ytr, Yte, np.repeat(Ytr.mean(0, keepdims=True), 20, 0))
    assert m["r2"] < 0.1 and set(m["per_target"]) == set(fc.Y_COLS)


def test_paper_hyperparameters():
    M = paper_models(0)
    knn, dt = M["knn"]().est, M["decision_tree"]().est
    assert knn.n_neighbors == 70 and knn.weights == "distance" and knn.metric == "euclidean"
    assert dt.max_depth == 7 and dt.min_samples_leaf == 11 and dt.criterion == "absolute_error"
    assert M["neural_net"]().est.hidden == (92,)


def test_autogluon_missing_is_skipped(monkeypatch):
    monkeypatch.setattr(automl, "available", lambda: False)
    assert "autogluon" not in paper_models(0)
    table = reproduction_table({"knn": {"r2": {"mean": 0.3, "std": 0}, "mae": {"mean": 0.4, "std": 0},
                                        "mse": {"mean": 1, "std": 0}}})
    assert table.loc["autogluon", "status"] == "not run" and table.loc["knn", "status"] == "pass"
    assert ranking_preserved(table) is None


@needs_data
@pytest.mark.slow
def test_reproduction_ranking_xgb_over_knn(framed):
    X, Y = framed
    M, f = paper_models(0), fold_indices(len(X), 5, 1)
    r = {k: cross_validate(M[k], X, Y.values, f) for k in ("knn", "decision_tree", "xgboost")}
    assert r["xgboost"]["r2"]["mean"] > r["decision_tree"]["r2"]["mean"] > r["knn"]["r2"]["mean"]


# ----------------------------------------------------------------------------- Task 5: our surrogates
from src.framed.surrogates import (  # noqa: E402
    DeepEnsemble, UQ, clip_framed_inputs, guard_framed, interval_coverage, inverse_transform_targets,
    our_models, target_eps, transform_targets)


def _fixture_xy(n=200, seed=0):
    X, Y = load_framed()
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    return X.iloc[idx], Y.values[idx]


def test_transform_roundtrip():
    Y = np.abs(np.random.default_rng(0).standard_cauchy((50, 10)))
    Y[0] = 0.0
    eps = target_eps(Y)
    T = transform_targets(Y, eps)
    assert np.isfinite(T).all()
    assert np.allclose(inverse_transform_targets(T, eps), Y, rtol=1e-9, atol=1e-12)


def test_guard_framed():
    Y = np.full((1, 10), 0.01)
    Y[0, fc.Y_COLS.index(fc.MASS_COL)] = -1
    Y[0, 0] = np.nan
    Y[0, fc.Y_COLS.index("Sim 1 Safety Factor")] = 50
    G = guard_framed(Y)
    assert np.isfinite(G).all() and (G >= 0).all()
    assert G[0, fc.Y_COLS.index(fc.MASS_COL)] > 0 and G[0, fc.Y_COLS.index("Sim 1 Safety Factor")] == fc.SF_CAP


def test_clip_inputs_flags_extrapolation():
    X = pd.DataFrame({"a": [0.0, 5.0, -1.0]})
    Xc, flag = clip_framed_inputs(X, pd.Series({"a": 0.0}), pd.Series({"a": 1.0}))
    assert flag.tolist() == [False, True, True] and Xc["a"].tolist() == [0.0, 1.0, 0.0]


@needs_data
def test_interval_contains_mean():
    X, Y = _fixture_xy()
    m = UQ(DeepEnsemble(0, n=2, hidden=(16,), max_epochs=30, patience=5, lr=1e-3, batch_size=64, dropout=0.0)).fit(X, Y)
    lo, mu, hi = m.predict_interval(X)
    assert (lo <= mu + 1e-12).all() and (mu <= hi + 1e-12).all()


@needs_data
@pytest.mark.slow
def test_physics_features_beat_raw_xgboost(framed):
    X, Y = framed
    f = fold_indices(len(X), 5, 1)
    raw = cross_validate(paper_models(0)["xgboost"], X, Y.values, f)
    phys = cross_validate(our_models(0)["xgboost_phys"], X, Y.values, f)
    assert phys["r2"]["mean"] > raw["r2"]["mean"]


# ----------------------------------------------------------------------------- Task 6: optimisation
from src.framed.optimize import FramedProblem, VARS, snap_material  # noqa: E402
from src.optimizer import nsga2  # noqa: E402


class _StubSurrogate:
    """mean SF = 1.2, lower bound 0.9: infeasible under the CI-lower-bound constraint."""

    def predict_interval(self, X):
        n = len(X)
        Y = np.full((n, 10), 1e-3)
        Y[:, fc.Y_COLS.index(fc.MASS_COL)] = X[VARS].to_numpy().sum(1)
        Y[:, [fc.Y_COLS.index(c) for c in fc.SF_COLS]] = 1.2
        lo = Y.copy()
        lo[:, [fc.Y_COLS.index(c) for c in fc.SF_COLS]] = 0.9
        return lo, Y, Y * 1.1


def test_onehot_snapped():
    X = pd.DataFrame([[0.2, 0.7, 0.1]], columns=fc.MATERIAL_COLS)
    assert snap_material(X).iloc[0].tolist() == [0, 1, 0]


@needs_data
def test_problem_duck_types_optimizer(framed):
    p = FramedProblem(surrogate=_StubSurrogate(), X_all=framed[0], max_bb_z=1.0)
    p.tag = "nsga2"
    nsga2(p, pop_size=8, generations=2, seed=0)
    G, F, cv = p.archive_for("nsga2")
    assert G.shape[1] == len(VARS) and len(F) == 16 and F.shape[1] == 2


@needs_data
def test_ci_lower_bound_constraint(framed):
    p = FramedProblem(surrogate=_StubSurrogate(), X_all=framed[0], max_bb_z=1.0)
    ev = p.evaluate(framed[0].loc[[3853], VARS].to_numpy())
    assert ev["V"][0, 0] > 0 and ev["cv"][0] > 0 and not ev["feasible"][0]
