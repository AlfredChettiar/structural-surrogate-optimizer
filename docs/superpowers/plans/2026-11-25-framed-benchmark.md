# FRAMED Benchmark Implementation Plan (reproduce → improve → report)

> **For agentic workers:** superpowers:executing-plans, native mode. Reviewer = subagent via
> superpowers:requesting-code-review at gates 🔍A (after Task 4) and 🔍C (after Task 7). Report uses `impeccable:impeccable`.
> **Context budget:** run ONE task per fresh session ("execute Task N only"). Never open raw CSV/PDF/notebooks; log long runs to `logs\` and report summaries only.

**Goal:** On the user's Windows PC, validate the published FRAMED paper's claims from its data, reproduce its surrogate
benchmark (Table 4), benchmark this project's physics-guided surrogates against it, run an uncertainty-aware optimisation,
and publish an impeccable report comparing every number with the paper.

**Execution (user decision, 25 Nov):** everything runs in the cloud session (4 vCPU, 16 GB), `.venv/bin/python` on Linux.

**Scope cut (user decision):** no OptiStruct/HyperMesh, no frame-geometry reconstruction, no FE re-verification, no
generated-frame OOD study. Optimised designs are checked only by surrogate CI and nearest simulated FRAMED design.

**Architecture:** new package `src/framed/`; V1 pipeline untouched. Reuses `src/optimizer.py` (duck-typed problem:
`xl, xu, n_var, evaluate(G)->{"F","cv",...}, normalise(F)`) and Plotly style from `src/plots.py`.

**Tech Stack:** existing (NumPy, Pandas, SciPy, scikit-learn, PyTorch, Plotly, Streamlit, Pytest) + `xgboost`, `PyMuPDF`,
`psutil`; optional `autogluon.tabular`.

## Source

- Paper [1]: Regenwetter, Weaver, Ahmed, *FRAMED: An AutoML Approach for Structural Performance Prediction of Bicycle Frames*, Computer-Aided Design (Elsevier) 2023, arXiv:2201.10459, MIT DeCoDE Lab. PDF: `github.com/Lyleregenwetter/Lyleregenwetter.github.io/papers/2022_regenwetter_framed.pdf`.
- Data: `github.com/Lyleregenwetter/FRAMED`, MIT, commit `b881634bb3b79240f08fc4285483becf6ec7d7c5`, file `all_structural_data_oh.csv` (4047 × 50), sha256 `8506eb3ec69f85fd35c551c630e5eba68c5ce5b2c78b3a42a33ab681f2697b5e`.
- 39 inputs (18 geometry, 9 ODs, 7 thicknesses, 2 bridge flags, 3 material one-hot); 10 outputs (7 displacements/rotation, 2 SF, mass); SolidWorks FE, 3 load cases (paper Sec. 3.3).

## Paper reference values → `src/framed/paper_reference.py` (single source for all comparisons)

- Table 2 (steel/Al/Ti): E 205/69/105 GPa; ρ 7850/2700/4429 kg/m³; σy 460/275/827 MPa; ν 0.285/0.330/0.310; G 80/26/41 GPa.
- Table 3 physical validation (actual→simulated; front, rear defl. mm; mass kg): DeRosa 0.40→0.297, 0.15→0.116, 1.966→1.69; Casati 0.44→0.3028, 0.15→0.124, 1.966→1.80; Holland 0.38→0.26, 0.13→0.107, 1.962→1.77.
- Table 4 (DT/KNN/XGB/NN/AutoML): R² 0.440/0.306/0.595/0.545/0.605; MSE 1.156/1.270/0.978/1.031/0.995; MAE 0.297/0.412/0.208/0.263/0.182.
- Tables 6–9 hyperparameters — DT: depth 7, best, absolute_error, split 5, leaf 11, max_features 32. KNN: k 70, distance, euclidean. XGB: depth 6, λ 72.41212, η 0.275542, γ 0.059818, min_child_weight 4.819113, early stop 8. NN: 1×92 ReLU, dropout 0.578647, lr 7.2e-5, batch 135, patience 44, no batch-norm.
- Counts (repo `Analysis.ipynb`): both SF<1 1966, SF1-only 1272, SF3-only 182, valid 626; non-dominated (5-objective) 76. Paper text: 3419/4118 fail.
- Sec. 5.2 case studies: #3853 Ti 2.84 kg SF 1.29/1.073; #2108 steel 22.83 kg SF1 0.0926; #857 Ti 1.60 kg SF 2.416/2.986 (verified against pinned data).
- Figures to compare: Fig. 4 materials, Fig. 6 pair plot, Fig. 8/9 parity plots (page numbers filled in Task 2).

## Design decisions

1. Preprocessing per `Analysis.ipynb`: X = first 39 cols; `abs()` on 7 displacement cols + mass.
2. Table-4 protocol (inferred): per-fold StandardScaler on X and Y; uniform mean R²/MSE/MAE over 10 standardised targets; 10×5-fold `KFold(shuffle=True, seed 42)`. R² and MAE both gate (±0.05 each); MSE reported only (paper's MSE inconsistent with its R²).
3. Reproduction passes: each model within ±0.05 R² of Table 4 and ranking AutoML ≥ XGB > NN > DT > KNN. Misses are reported, never tuned away.
4. "My model": beam-theory features (A, I, J, EI, GJ per tube, `mass_prior` = Σ ρ·A·L over tubes with explicit length ST/HT/DT/CS/BB, σy), log target transforms, GP (95 % CI), deep ensemble (CI), physics-guided MLP, plus XGB/AutoGluon on the same features. Same 50 folds as reproduction; paired Nadeau–Bengio corrected resampled t-test (Gate A: Wilcoxon on overlapping folds overstates significance); the difference to published Table 4 values is descriptive only.
5. Guardrails: mass > 0, |disp| ≥ 0, SF ∈ [0, 20], finite; inputs clipped to training range and flagged.
6. Speedup baseline: analytic `mass_prior` computation (latency); nominal FE 180 s (`config.NOMINAL_INDUSTRIAL_FE_SECONDS`) labelled nominal.
7. Paper figure pages rendered locally only (`reports/framed/paper_pages/`, git-ignored); public report cites "Fig. N in [1]" unless arXiv licence permits reuse.

## Global Constraints

- `.\.venv\Scripts\python.exe` everywhere; `-m pytest` green before every commit; data/model-dependent tests `skipif`.
- Never read raw files in `data/`/`models/`; report R², MAE, max rel. error % (displacement denominator floor = 5 % of training median), speedup.
- `n_jobs = os.cpu_count() - 1`. Artefacts in `data/external/framed/`, `models/framed/` (git-ignored).
- AutoGluon optional (`try` import → rows "not run"). Credit FRAMED (MIT) + paper. Never commit `prompts/` or data. Delete nothing without user OK.

## Review Focus

1. Thickness ≥ OD/2 (stays 2.35 mm OD, walls ≤ 10 mm) → solid rod, never negative inner diameter (Task 3).
2. Heavy-tailed outputs (|dropout Y| to 1.386 m) → log transform finite at 0; metrics on standardised and physical scales (Task 5).
3. Non-one-hot material rows from optimiser/app → reject or snap (Tasks 3, 6).
4. AutoGluon absent/slow → pipeline finishes, rows "not run" (Task 4).
5. Scaler leakage across folds → fit inside each fold only (Task 4).

## File Structure

`src/framed/{__init__,config,paper_reference,data,paper_stats,baselines,physics_features,surrogates,optimize,figures,report,run}.py`,
`tests/test_framed.py`, `app.py` (tab 5). CLI: `python -m src.framed.run --stage {fetch,stats,baselines,ours,optimize,report,all} [--fast]`.

---

### Task 0: Local setup (no code)
- [ ] `git status`; V1 tests need `models/` → run `powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Fast` (log to `logs\`, summary only).
- [ ] Cleanup: list caches/stale worktrees/untracked files with sizes; delete only what the user approves. Never `.venv`, `.git`, source, docs.
- [ ] `pip install -r requirements.txt xgboost PyMuPDF psutil`; try `autogluon.tabular`, record result.
- [ ] Record CPU/cores/RAM/disk/Python in `models/framed/env.json`; `-m pytest` green.
- [ ] Commit `.gitignore` (`data/external/`, `reports/framed/paper_pages/`) + `requirements.txt`: `chore: local setup for FRAMED benchmark`.

### Task 1: Data, paper, reference values
**Produces:** `fetch_framed(dest=config.FRAMED_DIR, force=False) -> dict[str, Path]` (raw.githubusercontent URL at pinned commit + PDF; sha256 check, `ValueError` on mismatch); `load_framed() -> (X, Y)`; `fold_indices(n, n_folds=5, n_repeats=10, seed=42) -> list[(train, test)]`.
- [ ] Tests: `test_sha_mismatch_raises`; `test_load_shapes` (X (4047,39), Y (4047,10), no NaN, ABS cols ≥ 0, one-hot sum 1); `test_fold_indices` (50 splits, partitions, deterministic); `test_case_study_frames_match_paper` (±0.005).
- [ ] red → implement (stdlib `urllib`) → `--stage fetch` → green → commit `feat(framed): pinned dataset and paper reference values`.

### Task 2: Validate paper dataset claims + figures
**Produces:** `feasibility_counts(Y) -> dict`; `non_dominated_mask(F, sense) -> ndarray`; `render_paper_pages(pdf, pages, out, dpi=150)`; figures `fig_material_breakdown`, `fig_performance_pairs` (Fig. 6 splom, valid/invalid, case studies marked), `fig_objective_correlation`, `fig_physical_validation` (Table 3); `stats_comparison() -> DataFrame[quantity, paper, ours, abs_dev, pct_dev]`.
- [ ] Tests: `test_feasibility_counts_logic` (4-row fixture); `test_feasibility_matches_notebook` (±2); `test_non_dominated_matches_paper` (76 ± 2); `test_figures_build`.
- [ ] red → implement → fill `PAPER_FIGURE_PAGES` → green → commit `feat(framed): validate dataset claims`.

### Task 3: Physics features
**Produces:** `section_properties(od, t) -> (A, I, J)` (t clipped to od/2); `material_props(X) -> DataFrame[E, G, rho, sigma_y]` (`ValueError` if not one-hot); `analytic_mass_prior(X)`; `add_physics_features(X)`.
- [ ] Tests: `test_hollow_tube_area` (OD 28.6, t 0.9 mm → A 78.32 mm², I = π/64(28.6⁴−26.8⁴)); `test_solid_rod_when_thick` (OD 2.35, t 10 mm); `test_material_props_rejects_bad_onehot`; `test_mass_prior_tracks_mass` (skipif no data; report Pearson r, gate r > 0.8).
- [ ] red → implement → green → commit `feat(framed): beam-theory features`.

### Task 4: Reproduce Table 4 (paper models)
**Produces:** `paper_models(seed) -> dict[str, factory]` (`decision_tree, knn, xgboost, neural_net, autogluon?`); `cross_validate(factory, X, Y, folds) -> dict` (r2/mse/mae mean±std standardised, `per_target` physical R²/MAE/max rel err, `fold_scores`, `oof_pred`); `reproduction_table(results)`; `fig_parity_grid`, `fig_table4_comparison`.
- [ ] Tests: `test_cv_metric_on_perfect_model`; `test_paper_hyperparameters`; `test_autogluon_missing_is_skipped`; `test_reproduction_knn_dt` (`slow`, 1×5 folds, ±0.05). Register `slow` in `pytest.ini`.
- [ ] red → implement → `--stage baselines` (10×5 folds; AutoGluon 1×5 in its own `.venv-ag` via subprocess, `medium_quality`, 90 s/target) → `models/framed/reproduction.json` → green → commit `feat(framed): reproduce Table 4`.
- [ ] 🔍 **Gate A** reviewer subagent on Tasks 1–4: protocol vs decision 2, hyperparameters vs Tables 6–9, fold leakage, honest pass/fail. Fix, re-test.

### Task 5: My surrogates vs the paper
**Produces:** `transform_targets / inverse_transform_targets` (log(y+eps), eps = 1 % training median); `guard_framed(Y)`; `clip_framed_inputs(X, lo, hi) -> (X, flag)`; `our_models(seed)` → `xgboost_phys`, `autogluon_phys?`, `gp_phys` (ARD Matern 2.5 + White, top-12 |corr| features, kernel fitted on a 400-row subsample: the 20-feature version took >10 min per fold), `deep_ensemble_phys` (5× MLP 3×128 SiLU), `pg_mlp` (mass as log(mass/mass_prior)); UQ: `predict_interval(X, 0.95) -> (lo, mean, hi)`; `latency_report(models, X)`.
- [ ] Tests: `test_transform_roundtrip`; `test_guard_framed`; `test_interval_contains_mean`; `test_improved_beats_raw_xgboost` (`slow`).
- [ ] red → implement → `--stage ours` → `models/framed/ours.json` (same 50 folds, per-target metrics, CI coverage, Wilcoxon vs AutoML or vs published 0.605, latency) → green → commit `feat(framed): physics-informed surrogates with UQ`.

### Task 6: Uncertainty-aware lightweighting (paper's Future Work, Sec. 8)
**Produces:** `FramedProblem` (duck-types `src/optimizer.py`; reference frame #3853; 10 vars = OD, t of TT/DT/ST/SS/CS; bounds = FRAMED 1st–99th pct; D/t ≤ 50; material fixed per run); objectives min mass, min |Sim 1 dropout Y|; constraints SF1, SF3 ≥ 1.0 on CI lower bound, |Sim 2 BB Z| ≤ reference; `run_study(material, seed) -> DataFrame` (Pareto, knee, lightest, CI, nearest FRAMED design + distance).
- [ ] Tests: `test_problem_duck_types_optimizer` (stub, `nsga2(pop_size=8, generations=2)`); `test_ci_lower_bound_constraint`; `test_onehot_snapped`.
- [ ] red → implement → `--stage optimize` (steel/Al/Ti) → `models/framed/optimize.json`, `data/framed_pareto.csv` → green → commit `feat(framed): uncertainty-aware lightweighting`.

### Task 7: Report (impeccable, Read mode), app tab, README
- [ ] Test `test_report_builds_from_fixtures` (HTML + MD contain "Regenwetter", every Table 4 value, pass/fail per check).
- [ ] `build_report(artifacts_dir, out_dir)`: `reports/framed/index.html` (Plotly embedded, pinned CDN) + `reports/framed_benchmark.md`. Sections: summary (3 headline numbers) · source & credibility · claim validation (paper/ours/deviation) · Table 4 reproduction + figure pairs · my models vs paper (paired stats, CI calibration, speedup) · optimisation (surrogate-only, flagged) · limitations (bicycle not motorcycle; inferred protocol; 4047 vs 4118 rows; MSE inconsistency; no FE re-check) · hardware/run times · references & licences.
- [ ] impeccable: `impeccable context` → `init` (PRODUCT.md: engineering portfolio report, recruiters + CAE engineers, Read mode) → new-work → `craft-floor.md` before edits → keep `src/plots.py` palette → `impeccable shot` desktop + phone → fix → `critique` + `audit`.
- [ ] App tab 5 (reads JSON/CSV only); README section between `<!-- FRAMED:START/END -->`; `run_pipeline.ps1 -Framed`.
- [ ] Full `-m pytest` green → commit `docs(framed): benchmark report, app tab, README`.
- [ ] 🔍 **Gate C** reviewer subagent, whole branch: every report number traces to JSON or `paper_reference`; no claim beyond evidence; licences; no data/prompts committed. Fix, re-test, push.
