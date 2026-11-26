"""Unit and smoke tests for the structural surrogate & Pareto optimiser pipeline."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src import config
from src.data_generator import build_design, ccd_face_centred, generate
from src.guardrails import apply_guardrails, clip_inputs, guard_interval, safety_factor
from src.optimizer import (ALGORITHMS, ChassisProblem, non_dominated_mask, nsga2, pareto_front, run_algorithm)
from src.optimizer_benchmark import hypervolume_2d, igd
from src.physics import (bend_factors, closed_form, fatigue_log10_life, frame_fe_batch, frame_fe_solve, joint_scf)
from src.surrogate_zoo import GaussianProcess, RidgePolynomial, load_surrogate
from src.train import select_model

ROOT = Path(__file__).resolve().parents[1]
CENTRE = np.array([np.mean(config.BOUNDS[f]) for f in config.FEATURES])


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def small_data():
    df, summary = generate(n_samples=160, seed=1)
    return df, summary


@pytest.fixture(scope="module")
def ridge(small_data):
    df, _ = small_data
    return RidgePolynomial().fit(df[config.FEATURES].to_numpy(), df[config.TARGETS].to_numpy())


# ----------------------------------------------------------------------------- dataset
def test_ccd_has_expected_points_in_unit_cube():
    ccd = ccd_face_centred(6)
    assert ccd.shape == (2**6 + 2 * 6 + 1, 6)
    assert ccd.min() >= 0.0 and ccd.max() <= 1.0


def test_design_respects_bounds():
    X, labels = build_design(200, seed=0)
    lo = np.array([config.BOUNDS[f][0] for f in config.FEATURES])
    hi = np.array([config.BOUNDS[f][1] for f in config.FEATURES])
    assert X.shape == (200, 6)
    assert np.all(X >= lo - 1e-9) and np.all(X <= hi + 1e-9)
    assert (labels == "CCD").sum() == 77


def test_generated_dataset_is_physical(small_data):
    df, summary = small_data
    assert len(df) == 160 and set(config.FEATURES + config.TARGETS) <= set(df.columns)
    assert not df[config.TARGETS].isna().any().any()
    assert (df[["mass_kg", "peak_von_mises_mpa", "max_deflection_mm", "first_mode_hz"]] > 0).all().all()
    assert df["log10_fatigue_life_cycles"].between(0, config.FATIGUE_LOG10_CAP).all()
    assert 0.0 <= summary["constraint_pass_rate"]["all"] <= 1.0


# ----------------------------------------------------------------------------- physics
def test_thicker_wall_is_heavier_stiffer_and_less_stressed():
    thin, thick = CENTRE.copy(), CENTRE.copy()
    thin[1], thick[1] = 1.8, 3.2
    a, b = frame_fe_solve(*thin), frame_fe_solve(*thick)
    assert b["mass_kg"] > a["mass_kg"]
    assert b["peak_von_mises_mpa"] < a["peak_von_mises_mpa"]
    assert b["max_deflection_mm"] < a["max_deflection_mm"]


def test_load_factor_scales_stress_not_frequency():
    lo_, hi_ = CENTRE.copy(), CENTRE.copy()
    lo_[5], hi_[5] = 2.0, 4.0
    a, b = frame_fe_solve(*lo_), frame_fe_solve(*hi_)
    assert b["peak_von_mises_mpa"] == pytest.approx(2.0 * a["peak_von_mises_mpa"], rel=1e-6)
    assert b["first_mode_hz"] == pytest.approx(a["first_mode_hz"], rel=1e-9)


def test_bend_and_joint_factors():
    k, i = bend_factors(np.array([22.0, 38.0]), np.array([3.5, 1.6]), np.array([90.0, 40.0]))
    assert np.all(k >= 1.0) and np.all(i >= 1.0)
    assert i[1] > i[0]  # large thin tube in a tight bend intensifies stress
    assert joint_scf(2.0, 5.0) < joint_scf(2.0, 2.0)  # thicker gusset relieves the joint


def test_fatigue_life_monotonic_and_capped():
    s = np.array([50.0, 150.0, 300.0])
    life = fatigue_log10_life(s, 355.0, 3.0)
    assert np.all(np.diff(life) < 0)
    assert life.max() <= config.FATIGUE_LOG10_CAP and life.min() >= 0.0


def test_closed_form_mass_matches_fe_and_misses_stress_concentration():
    fe = frame_fe_batch(CENTRE[None, :])[0]
    cf = closed_form(CENTRE)[0]
    assert cf[0] == pytest.approx(fe[0], rel=0.03)
    assert cf[1] < fe[1]  # nominal hand-calc stress has no SCF


# ----------------------------------------------------------------------------- guardrails
def test_guardrails_clip_and_sanitise():
    raw = np.array([[-1.0, -50.0, -0.1, np.nan, 12.0]])
    g = apply_guardrails(raw)[0]
    assert g[0] >= config.GUARDRAILS["mass_kg"][0]
    assert g[1] >= config.GUARDRAILS["peak_von_mises_mpa"][0]
    assert g[2] >= 0.0
    assert g[3] == config.GUARDRAILS["first_mode_hz"][0]  # unknown capacity -> conservative low
    assert g[4] == config.FATIGUE_LOG10_CAP


def test_safety_factor_bounds_and_interval_order():
    assert safety_factor(355.0, 1e-12) == config.GUARDRAILS["safety_factor"][1]
    assert safety_factor(355.0, 1e9) >= 0.0
    lo, mu, hi = guard_interval(np.array([5.0]), np.array([4.0]), np.array([3.0]), "mass_kg")
    assert lo <= mu <= hi


def test_clip_inputs_flags_extrapolation():
    X = np.vstack([CENTRE, CENTRE * 10])
    Xc, outside = clip_inputs(X)
    assert outside.tolist() == [False, True]
    assert Xc[1, 0] == config.BOUNDS["tube_od_mm"][1]


# ----------------------------------------------------------------------------- surrogates
def test_ridge_predicts_positive_and_accurate_on_train(ridge, small_data):
    df, _ = small_data
    P = ridge.predict(df[config.FEATURES].to_numpy())
    assert P.shape == (len(df), 5) and np.all(P[:, :4] > 0)
    rel = np.abs(P[:, 0] - df["mass_kg"]) / df["mass_kg"]
    assert rel.mean() < 0.02


def test_gp_interval_contains_mean(small_data):
    df, _ = small_data
    gp = GaussianProcess().fit(df[config.FEATURES].to_numpy()[:100], df[config.TARGETS].to_numpy()[:100])
    lo, mu, hi = gp.predict_interval(df[config.FEATURES].to_numpy()[100:120])
    assert np.all(lo <= mu + 1e-9) and np.all(mu <= hi + 1e-9)
    assert np.all(lo[:, :4] > 0)


def test_sklearn_save_load_roundtrip(ridge, tmp_path):
    ridge.save(tmp_path)
    again = load_surrogate(ridge.name, tmp_path)
    np.testing.assert_allclose(again.predict(CENTRE), ridge.predict(CENTRE))


def test_torch_models_train_and_roundtrip(small_data, tmp_path):
    pytest.importorskip("torch")
    from src.model import ResidualMLP
    from src.surrogate_zoo import PhysicsGuidedMLP, ResidualMLPSurrogate

    import torch

    net = ResidualMLP(6, 5)
    assert net(torch.zeros(3, 6)).shape == (3, 5)
    df, _ = small_data
    X, Y = df[config.FEATURES].to_numpy(), df[config.TARGETS].to_numpy()
    for cls in (ResidualMLPSurrogate, PhysicsGuidedMLP):
        m = cls(epochs=15).fit(X, Y)
        p = m.predict(X[:5])
        assert p.shape == (5, 5) and np.all(np.isfinite(p)) and np.all(p[:, :4] > 0)
        m.save(tmp_path)
        np.testing.assert_allclose(load_surrogate(m.name, tmp_path).predict(X[:5]), p, rtol=1e-5)


# ----------------------------------------------------------------------------- optimisation
def test_non_dominated_mask_and_pareto_front():
    F = np.array([[1, 5], [2, 3], [3, 4], [4, 1], [2, 3.5]], dtype=float)
    assert non_dominated_mask(F).tolist() == [True, True, False, True, False]
    cv = np.array([0, 0.1, 0, 0, 0])
    assert sorted(pareto_front(F, cv).tolist()) == [0, 3, 4]


def test_hypervolume_and_igd_known_values():
    assert hypervolume_2d(np.array([[0.0, 1.0], [1.0, 0.0]]), ref=(2.0, 2.0)) == pytest.approx(3.0)
    ref = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert igd(ref, ref) == pytest.approx(0.0)


def test_nsga2_front_is_feasible_and_non_dominated(ridge):
    prob = ChassisProblem(ridge)
    prob.tag = "nsga2"
    nsga2(prob, pop_size=24, generations=6, seed=0)
    G, F, cv = prob.archive_for("nsga2")
    idx = pareto_front(F, cv)
    assert len(idx) > 0 and np.all(cv[idx] <= 0)
    assert non_dominated_mask(F[idx]).all()
    ev = prob.evaluate(G[idx])  # constraints hold when re-evaluated
    assert np.all(ev["feasible"])


def test_every_optimizer_runs_with_small_budget(ridge):
    prob = ChassisProblem(ridge)
    small = {"lhs_screening": {"n": 200}, "nsga2": {"pop_size": 16, "generations": 3},
             "epsilon_constraint": {"n_levels": 2, "popsize": 5, "maxiter": 3},
             "weighted_sum_slsqp": {"n_weights": 2, "n_starts": 1}, "parego": {"budget": 14, "n_init": 10}}
    for name in ALGORITHMS:
        secs = run_algorithm(prob, name, **small[name])
        _, F, _ = prob.archive_for(name)
        assert secs >= 0 and len(F) > 0
        assert np.all(F > 0)  # guard-railed physical objectives


def _entry(stress_max, stress_mape, r2=0.99, uq=False, lat=0.01, other=1.0):
    pt = {t: {"max_rel_err_pct": other, "mape_pct": other / 4} for t in config.TARGETS}
    pt["peak_von_mises_mpa"] = {"max_rel_err_pct": stress_max, "mape_pct": stress_mape}
    return {"trainable": True, "supports_uncertainty": uq, "latency_ms_per_design_batch": lat,
            "cv": {"min_mean_r2": r2}, "test_vs_clean_fe": {"per_target": pt}}


def test_model_selection_gate_and_composite_score():
    metrics = {"accurate": _entry(10.0, 1.0, uq=True), "worst_case_only": _entry(9.0, 3.0, other=5.0),
               "gated_out": _entry(1.0, 0.1, r2=0.90)}
    extrap = {"models": {"accurate": {"peak_von_mises_mpa": {"mape_pct": 2.0}},
                         "worst_case_only": {"peak_von_mises_mpa": {"mape_pct": 5.0}}}}
    sel = select_model(metrics, extrap)
    assert "gated_out" not in sel["eligible_ranking"]          # fails the R2 gate despite great errors
    assert sel["selected_model"] == "accurate"                 # better on most criteria, not just one
    assert abs(sum(sel["score_breakdown"]["accurate"].values()) - sel["scores"]["accurate"]) < 1e-3


def test_manufacturing_constraints_and_knee_point():
    from src.optimizer import knee_point, manufacturing_violations

    V = manufacturing_violations(np.array([[38.0, 1.6, 5.0, 90.0], [30.0, 2.0, 4.0, 40.0]]))
    assert V[0, 0] > 0 and V[0, 1] <= 0     # D/t = 23.75 > 20, bend ratio 2.4 OK
    assert V[1, 0] <= 0 and V[1, 1] > 0     # D/t = 15 OK, bend ratio 1.33 < 1.5
    F = np.array([[0.0, 1.0], [0.1, 0.3], [0.5, 0.1], [1.0, 0.0]])
    assert knee_point(F) == 1


def test_multi_seed_benchmark_runs_and_selects(ridge):
    from src.optimizer_benchmark import run_benchmark

    out = run_benchmark(surrogate=ridge, fast=True, n_seeds=2)
    assert out["selected_optimizer"] in out["algorithms"]
    for r in out["algorithms"].values():
        assert len(r["per_seed"]) == 2 and 0.0 <= r["fe_feasible_rate"] <= 1.0
        assert "hypervolume" in r["std"]
    rows = out["_pareto_rows"]
    assert rows and all(r["d_over_t"] <= config.CONSTRAINTS["max_d_over_t"] + 1e-9 for r in rows)


# ----------------------------------------------------------------------------- prompt-refiner hook
def test_prompt_hook_logs_and_routes(tmp_path):
    hook = ROOT / ".claude" / "hooks" / "refine_prompt.py"
    if not hook.exists():  # not yet installed via claude_setup/install.ps1
        hook = ROOT / "claude_setup" / "hooks" / "refine_prompt.py"
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(tmp_path)}
    long = subprocess.run([sys.executable, str(hook)], input=json.dumps({"prompt": "Add a residual plot to tab 3 please"}),
                          capture_output=True, text=True, env=env, timeout=30)
    short = subprocess.run([sys.executable, str(hook)], input=json.dumps({"prompt": "yes"}),
                           capture_output=True, text=True, env=env, timeout=30)
    assert long.returncode == 0 and short.returncode == 0
    assert "prompt-refiner" in json.loads(long.stdout)["hookSpecificOutput"]["additionalContext"]
    assert short.stdout.strip() == ""
    lines = (tmp_path / "prompts" / "prompt_history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["refine"] for line in lines] == [True, False]


# ----------------------------------------------------------------------------- artifact smoke tests
# Committed JSON results alone are not enough: the figures and app also need the git-ignored dataset and model files
_trained = config.DATASET_CSV.exists() and any(config.MODELS_DIR.glob("*.joblib"))
needs_artifacts = pytest.mark.skipif(not (config.METRICS_JSON.exists() and _trained),
                                     reason="run src.data_generator and src.train first")


@needs_artifacts
def test_metrics_artifact_reports_required_kpis():
    m = json.loads(config.METRICS_JSON.read_text())
    assert m["selected_model"] in m["models"]
    for name, v in m["models"].items():
        for t in config.TARGETS:
            pt = v["test"]["per_target"][t]
            assert {"r2", "mae", "max_rel_err_pct"} <= set(pt)
        assert v["latency_ms_single"] > 0 and v["speedup_vs_fe_batch"] > 0
    assert m["models"][m["selected_model"]]["test"]["min_r2"] > 0.95


@needs_artifacts
def test_report_figures_build():
    pytest.importorskip("plotly")
    from src.artifacts import load_all
    from src.report import build_figures

    figs = build_figures(load_all())
    assert len(figs) >= 14


@pytest.mark.skipif(not (config.METRICS_JSON.exists() and config.OPTIMIZATION_JSON.exists() and _trained),
                    reason="run src.train and src.optimizer first")
def test_streamlit_app_runs_without_exceptions():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
