# FRAMED Benchmark (Reproduce, then Improve) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second, peer-reviewed data track to the project: load the published FRAMED dataset, reproduce the
paper's statistics, figures and surrogate results (Table 4), then run this project's own physics-guided surrogates,
uncertainty, out-of-distribution test and Pareto optimiser on the same design space and show the gains.

**Architecture:** A new self-contained package `src/framed/` (data, paper reproduction, physics features,
surrogates, optimisation, figures, report). The V1 synthetic motorcycle pipeline stays untouched. Only the Pareto
algorithms in `src/optimizer.py` and the Plotly style in `src/plots.py` are reused. `src/optimizer.py` is duck-typed:
any object with `xl, xu, n_var, evaluate(G) -> {"F","cv",...}, normalise(F)` works with `nsga2`, `parego` and the
other algorithms.

**Tech Stack:** NumPy, Pandas, SciPy, scikit-learn, PyTorch, Plotly, Streamlit, Pytest (existing). New:
`xgboost` (required, to reproduce the paper's XGBoost baseline) and `autogluon.tabular` (optional, for the paper's
AutoML model; every step degrades gracefully when it is missing).

**Spec:** This document, sections "Source" and "Design decisions" below. Source of truth for every reproduced
number is the paper (text copy used for this plan: Regenwetter et al., CAD preprint 29 Oct 2022).

---

## Source

| Item | Value |
|---|---|
| Paper | L. Regenwetter, C. Weaver, F. Ahmed, *FRAMED: An AutoML Approach for Structural Performance Prediction of Bicycle Frames*, Computer-Aided Design (Elsevier), 2023. arXiv:2201.10459. MIT DeCoDE Lab. |
| Data + code | `https://github.com/Lyleregenwetter/FRAMED`, MIT licence, pinned commit `b881634bb3b79240f08fc4285483becf6ec7d7c5` |
| FE solver | SolidWorks Simulation, solid mesh, blended curvature-based, 0.03 mm minimum cell; mesh-convergence study in paper Sec. 3.5 |
| Physical validation in paper | Rinard (1996) frame-deflection tests, 3 frames, Table 3 (17–32 % deflection error, 8–14 % mass error) |
| Design space | 39 model inputs (37 parameters, material one-hot to 3): 18 geometry, 9 tube ODs, 7 wall thicknesses (Sobol, log-scaled 0.5–10 mm), 2 bridge flags, material {steel 4130, Al 6061-T6, Ti-6Al-4V} |
| Load cases | (1) in-plane: 2000 N up at dropouts, 2000 N down at bottom bracket (BB), head tube fixed; (2) transverse: 500 N lateral at BB; (3) eccentric: 2000 N + 140 N·m at BB |
| Outputs (10) | Sim 1 dropout X/Y disp., Sim 1 BB X/Y disp., Sim 2 BB Z disp., Sim 3 BB Y disp., Sim 3 BB X rot., Sim 1 SF, Sim 3 SF, model mass |
| Files used | `all_structural_data_oh.csv` (4047 × 50) sha256 `8506eb3ec69f85fd35c551c630e5eba68c5ce5b2c78b3a42a33ab681f2697b5e`; `all_structural_data_oh_aug.csv` (14852 × 50; rows `Gen 0…Gen 10804` are 10,805 AI-generated frames simulated with the same SolidWorks pipeline) sha256 `3373f943b9968b40923e09a912207c53aeda70e4c2c0f1941f8205c13aab0eb7` |

### Paper numbers to reproduce

- Counts on repo data (`Analysis.ipynb` output): SF1<1 & SF3<1 = 1966; SF1<1 only = 1272; SF3<1 only = 182; both ≥1 (valid) = 626. Non-dominated designs in the 5-objective subset = 76. Paper text: 3419 of 4118 fail at least one case.
- Table 4 (regression, standardised targets, 10 × 5-fold CV):

| Metric | Decision Tree | KNN | XGBoost | Neural Net | AutoML |
|---|---|---|---|---|---|
| R² | 0.440 | 0.306 | 0.595 | 0.545 | 0.605 |
| MSE | 1.156 | 1.270 | 0.978 | 1.031 | 0.995 |
| MAE | 0.297 | 0.412 | 0.208 | 0.263 | 0.182 |

- Best hyperparameters (appendix Tables 6–9):
  - DT: max_depth 7, splitter best, criterion absolute_error, min_samples_split 5, min_samples_leaf 11, max_features 32
  - KNN: n_neighbors 70, weights distance, metric euclidean
  - XGBoost: max_depth 6, reg_lambda 72.41212, learning_rate 0.275542, gamma 0.059818, min_child_weight 4.819113, early-stopping patience 8
  - NN: 1 hidden layer of 92, ReLU, no batch-norm, dropout 0.578647, lr 7.2e-5, batch 135, patience 44
- Figures: Fig. 6 (5-output pair plot coloured valid/invalid), objective correlation heat-map, Fig. 4 material breakdown, Fig. 8/9 parity plots (train and validation, all 10 outputs), Table 3 physical validation.

## Design decisions

1. **Why FRAMED.** It is the only public, peer-reviewed tube-parametric frame dataset found: a two-wheeler,
   tubular, FE-labelled with physical validation, and published with code and baselines. No public motorcycle
   sub-frame dataset exists (searched: NAFEMS, Zenodo/Figshare/HF, SAE, arXiv). NAFEMS publishes solver
   verification cases, not design-space datasets. CarHoods10k (Honda Research Institute / OSU) is the fallback
   if a second dataset is wanted later; it is not tubular.
2. **Preprocessing follows the authors' `Analysis.ipynb`:** X = first 39 columns of `all_structural_data_oh.csv`;
   Y = the 10 output columns; `abs()` on the seven displacement/rotation columns and mass. The repo has 4047
   rows against the paper's 4118, so reproduction tolerances allow for that.
3. **Table 4 protocol (inferred, the paper's AutoML code is not in the repo):** StandardScaler on X and on Y,
   fitted per training fold. R² = uniform mean over the 10 standardised targets; MAE and MSE averaged the same
   way; 10 repeats × 5-fold `KFold(shuffle=True)`. The paper's MSE (0.995 at R² 0.605) is inconsistent with unit-variance
   targets, so **R² and MAE gate the reproduction and MSE is only reported**. The write-up must say so.
4. **Reproduction acceptance:** each baseline's R² within ±0.05 of Table 4, and the ranking
   AutoML ≥ XGBoost > NN > DT > KNN preserved. Anything outside this is reported as a discrepancy, never tuned away.
5. **"My model" = this project's surrogate methodology on FRAMED's data:** physics-informed features from beam
   theory (tube A, I, J, E·I, analytic mass prior), log target transforms, GP with 95 % CI, deep ensemble with
   CI, physics-guided MLP (residual on the analytic prior), plus XGBoost and AutoGluon on the same physics
   features so the gain from features is separated from the gain from the model.
6. **Same folds for everything:** every model in Tasks 3 and 5 runs on the identical `fold_indices(seed=42)`
   splits, so all comparisons with Table 4 are paired.
7. **Analytical baseline for speedup (CLAUDE.md):** `analytic_mass_prior` plus beam-theory tube stiffness is the
   in-repo analytical baseline for latency. Speedup against SolidWorks is projected from
   `config.NOMINAL_INDUSTRIAL_FE_SECONDS` (180 s) and labelled as nominal, since the paper gives no run time.
8. **Optimisation implements the paper's Future Work (Sec. 8):** constrained optimisation with the surrogate.
   Per the paper's warning (Sec. 5.2.3), rider-fit geometry is frozen to a reference frame and only tube ODs,
   thicknesses and material vary. **No SolidWorks re-check is possible here.** Optimum designs are instead
   checked with (a) the CI lower bound on SF and (b) distance to the nearest simulated FRAMED design, and both
   are reported as such.
9. **Data handling (CLAUDE.md):** downloaded files go to `data/external/framed/` (git-ignored by the existing
   `data/*.csv` rule plus a new `data/external/` line). They are only ever inspected through the loaders or short
   Python commands, never opened raw.
10. **Guardrails on all FRAMED predictions:** mass > 0, displacement magnitudes ≥ 0, SF clipped to [0, 20],
    finite outputs; inputs outside the training [min, max] are clipped and flagged.

## Global Constraints

- Run Python as `.\.venv\Scripts\python.exe` (Windows). In a Linux container with no `.venv`, `python -m` is the
  equivalent, after `pip install -r requirements.txt`.
- Never read raw files in `data/` or `models/`; use loaders or short Python commands.
- Report R², MAE, max relative error (%) and speedup against the analytical baseline for every surrogate.
- Max relative error on displacement-type targets uses a denominator floor of 5 % of that target's training
  median, to stop near-zero values blowing it up. Record the floor in the metrics JSON.
- `.\.venv\Scripts\python.exe -m pytest` passes before each task is committed.
- Tests must not need the network: logic tests use an in-memory fixture; tests against the real data are marked
  `@pytest.mark.skipif(not FRAMED_CSV.exists(), ...)`.
- `autogluon` is optional: `import` inside a `try`, and a missing package skips that model with a logged note.
- Credit FRAMED (MIT, copyright 2022 Lyleregenwetter) in the README and report; no FRAMED files are committed.
- Do not commit or push `prompts/` content.

## Review Focus

1. **Thickness ≥ OD/2** (FRAMED samples thicknesses up to 10 mm on stays as thin as 2.35 mm OD): section
   properties must treat these tubes as solid rods, not return negative inner diameters. Tested in Task 4.
2. **Heavy-tailed outputs** (|Sim 1 dropout Y| up to 1.386 m against a 0.05 m std): log transforms must stay
   finite for zero values, and metrics must be reported on both standardised and physical scales. Tested in Task 5.
3. **Generated-frame set includes designs outside the original ranges:** the OOD evaluation must report the
   extrapolation fraction and must not silently clip it away. Tested in Task 6.
4. **Material one-hot rows that are not exactly one-hot** (from the optimiser or app sliders) must be rejected
   or snapped, not passed to the model. Tested in Task 7.
5. **AutoGluon absent or slow:** the pipeline must finish and the report must mark AutoML rows as "not run"
   instead of crashing or leaving blanks. Tested in Task 3.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/framed/__init__.py` | empty |
| `src/framed/config.py` | paths, pinned URL/commit/sha256, column lists, Table 2 material properties, Table 4 numbers, paper hyperparameters |
| `src/framed/data.py` | download + checksum, load original/generated, fold indices |
| `src/framed/paper_stats.py` | feasibility counts, non-dominated set, correlations |
| `src/framed/baselines.py` | paper models with the paper's hyperparameters, CV protocol, metrics |
| `src/framed/physics_features.py` | tube section properties, material mapping, analytic mass prior |
| `src/framed/surrogates.py` | improved models, target transforms, guardrails, UQ |
| `src/framed/optimize.py` | `FramedProblem` for the shared optimisers, reference-frame study |
| `src/framed/figures.py` | Plotly figures (reuses `src.plots._layout` and `SERIES`) |
| `src/framed/run.py` | CLI: `python -m src.framed.run [--fast] [--skip-autogluon]`, writes `models/framed/*.json`, `reports/framed/*` |
| `tests/test_framed.py` | all FRAMED tests |
| `app.py` | new tab 5, "FRAMED benchmark" |
| `README.md`, `run_pipeline.ps1`, `requirements.txt`, `.gitignore` | docs, pipeline step, `xgboost`, `data/external/` |

---

### Task 1: Data acquisition and loading

**Files:**
- Create: `src/framed/__init__.py`, `src/framed/config.py`, `src/framed/data.py`, `tests/test_framed.py`
- Modify: `.gitignore` (add `data/external/`), `requirements.txt` (add `xgboost`, pinned to the version installed)

**Interfaces:**
- Produces:
  - `config.FRAMED_DIR = DATA_DIR / "external" / "framed"`, `config.FILES: dict[str, tuple[str, str]]` (name → (repo path, sha256))
  - `config.X_COLS: list[str]` (39), `config.Y_COLS: list[str]` (10), `config.ABS_COLS: list[str]` (8, as in `Analysis.ipynb`)
  - `fetch_framed(dest: Path = config.FRAMED_DIR, force: bool = False) -> dict[str, Path]`: downloads `https://raw.githubusercontent.com/Lyleregenwetter/FRAMED/<commit>/<file>` and falls back to `git clone --depth 1` when raw access is blocked. Raises `ValueError` on a sha256 mismatch.
  - `load_framed(kind: Literal["original", "generated"] = "original") -> tuple[pd.DataFrame, pd.DataFrame]`, returning (X, Y)
  - `fold_indices(n: int, n_folds: int = 5, n_repeats: int = 10, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]`

- [ ] **Step 1: Write failing tests**
  - `test_sha_mismatch_raises`: write a temporary file with the wrong content, call `_verify(path, "00…")`, expect `ValueError`.
  - `test_load_shapes` (skipif no data): original X `(4047, 39)`, Y `(4047, 10)`; generated X `(10805, 39)`; no NaN; `(Y[ABS_COLS] >= 0).all().all()`; the material one-hot row sum is 1.
  - `test_fold_indices`: 50 splits for (5, 10); the test sets of each repeat partition `range(n)`; deterministic for a given seed.
- [ ] **Step 2:** Run `.\.venv\Scripts\python.exe -m pytest tests/test_framed.py -v`. Expected: FAIL (module not found).
- [ ] **Step 3:** Implement `config.py` and `data.py` with the signatures above. Use `urllib.request` from the standard library; no new dependency.
- [ ] **Step 4:** Run `.\.venv\Scripts\python.exe -c "from src.framed.data import fetch_framed; fetch_framed()"`, then pytest. Expected: PASS, data tests not skipped.
- [ ] **Step 5:** Commit: `feat(framed): pinned, checksummed FRAMED dataset loader`.

### Task 2: Reproduce the paper's dataset statistics and figures

**Files:**
- Create: `src/framed/paper_stats.py`, `src/framed/figures.py`
- Test: `tests/test_framed.py`

**Interfaces:**
- Consumes: `load_framed`, `config.Y_COLS`
- Produces:
  - `feasibility_counts(Y: pd.DataFrame) -> dict[str, int]` with keys `both_fail, sf1_only, sf3_only, valid`
  - `non_dominated_mask(Y5: np.ndarray, sense: np.ndarray) -> np.ndarray` (objective sense per column: −1 = minimise, +1 = maximise); 5 objectives: |Sim 1 dropout Y|, |Sim 2 BB Z|, |Sim 3 BB X rot|, Sim 1 SF (maximise), mass
  - `figures.fig_performance_pairs(Y) -> go.Figure` (Fig. 6, Plotly `splom`), `fig_objective_correlation(Y)`, `fig_material_breakdown(X)`, `fig_physical_validation()` (Table 3 constants in `config.PAPER_TABLE3`, captioned "values from Regenwetter et al., Table 3"), `fig_parity_grid(Y_true, Y_pred_train, Y_pred_val, title)` (10 subplots, Fig. 8/9 style)

- [ ] **Step 1: Write failing tests**
  - `test_feasibility_counts_logic`: a 4-row fixture with one row per class gives `{both_fail:1, sf1_only:1, sf3_only:1, valid:1}`.
  - `test_feasibility_matches_paper` (skipif no data): each count within ±2 of 1966 / 1272 / 182 / 626.
  - `test_non_dominated_matches_paper` (skipif no data): `mask.sum()` within 76 ± 2.
  - `test_case_study_frames_match_paper` (skipif no data, Sec. 5.2): index 3853 has mass 2.84, SF1 1.29, SF3 1.073 (paper: 2.84 kg, SF 1.3/1.1); 2108 has mass 22.83 and SF1 < 0.1; 857 has mass 1.60, SF1 2.416, SF3 2.986 (paper: 1.6 kg, 2.42/2.99). All three were verified against the pinned data while writing this plan.
  - `test_figures_build`: each figure function returns a `go.Figure`; `fig_parity_grid` has 20 scatter traces.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement. Use a vectorised dominance check on 4047 points; an O(n²) NumPy broadcast in chunks of 512 is fine.
- [ ] **Step 4:** Run pytest. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(framed): reproduce FRAMED dataset statistics and figures`.

### Task 3: Reproduce Table 4 (paper baselines)

**Files:**
- Create: `src/framed/baselines.py`
- Test: `tests/test_framed.py`

**Interfaces:**
- Consumes: `load_framed`, `fold_indices`
- Produces:
  - `paper_models(seed: int) -> dict[str, Callable[[], Regressor]]` with keys `decision_tree, knn, xgboost, neural_net, autogluon`. Hyperparameters are copied from "Paper numbers to reproduce". `autogluon` is present only if it imports.
  - `Regressor` protocol: `fit(X: np.ndarray, Y: np.ndarray) -> self`, `predict(X) -> np.ndarray` (multi-output; DT and KNN are native; XGBoost uses `multi_strategy="one_output_per_tree"` with a 10 % inner validation split for early stopping; the NN is a small torch module; AutoGluon is one `TabularPredictor` per target, `presets="medium_quality"`, `time_limit` per target from CLI)
  - `cross_validate(factory, X, Y, folds) -> dict` returning `{"r2", "mse", "mae"}` (mean ± std over folds, standardised scale), plus `per_target` with R², MAE and max relative error (%) on the physical scale, and `oof_pred: np.ndarray` (out-of-fold predictions from repeat 0, for parity plots)
  - `reproduction_table(results: dict) -> pd.DataFrame` with columns paper R², ours R², ΔR², paper MAE, ours MAE, within_tolerance

- [ ] **Step 1: Write failing tests**
  - `test_cv_metric_on_perfect_model`: a model that returns the standardised truth gives R² = 1 and MAE = 0.
  - `test_paper_hyperparameters`: `paper_models(0)["knn"]().get_params()` has `n_neighbors == 70` and `weights == "distance"`; the DT has `max_depth == 7` and `min_samples_leaf == 11`.
  - `test_autogluon_missing_is_skipped`: monkeypatch the import to fail; `"autogluon" not in paper_models(0)` and `run.py` writes `"status": "not run"` for it.
  - `test_reproduction_within_tolerance` (skipif no data, marked `slow`): KNN and DT on 1 × 5 folds are within ±0.05 R² of 0.306 and 0.440.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement. Register the `slow` marker in `pytest.ini`.
- [ ] **Step 4:** Run `.\.venv\Scripts\python.exe -m src.framed.run --stage baselines` (10 × 5 folds; AutoGluon 1 × 5 folds with `--ag-time 60` s per target). Expected: `models/framed/reproduction.json`, and the table prints with the ranking check. Record any discrepancy in the JSON's `notes`.
- [ ] **Step 5:** Run pytest, then commit: `feat(framed): reproduce Table 4 baselines with the paper's hyperparameters`.

### Task 4: Physics-informed features and analytical baseline

**Files:**
- Create: `src/framed/physics_features.py`
- Test: `tests/test_framed.py`

**Interfaces:**
- Consumes: `config.MATERIALS` (Table 2: E 205/69/105 GPa, ν 0.285/0.330/0.310, ρ 7850/2700/4429 kg/m³, σy 460/275/827 MPa for steel/Al/Ti)
- Produces:
  - `section_properties(od: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning A, I, J in SI units; `t` is clipped to `od/2` (a solid rod)
  - `material_props(X: pd.DataFrame) -> pd.DataFrame` with columns `E, rho, sigma_y`; raises `ValueError` when the one-hot row sum is not 1
  - `analytic_mass_prior(X: pd.DataFrame) -> np.ndarray` = Σ ρ·A·L over the tubes with an explicit length (`ST, HT, DT, CS, BB`) plus top-tube and seat-stay lengths from the frame-geometry relations (derive from `Stack`, `ST Angle`, `HT Angle`, `ST Length`, `CS Length`, `BB Drop`; document the formula in one comment line)
  - `add_physics_features(X: pd.DataFrame) -> pd.DataFrame`: X plus, per tube, `A_*`, `EI_*`, `GJ_*`, slenderness `L/OD` where L is known, plus `mass_prior`, `sigma_y`

- [ ] **Step 1: Write failing tests**
  - `test_hollow_tube_area`: OD 28.6 mm, t 0.9 mm gives A = π/4·(28.6² − 26.8²) mm² = 78.32 mm² (rel 1e-4), and I = π/64·(28.6⁴ − 26.8⁴) mm⁴.
  - `test_solid_rod_when_thick`: OD 2.35 mm, t 10 mm gives A = π/4·2.35² mm², all finite and positive.
  - `test_material_props_rejects_bad_onehot`: row `[0.5, 0.5, 0]` raises `ValueError`.
  - `test_mass_prior_tracks_mass` (skipif no data): Pearson r(mass_prior, Model Mass) > 0.9, and the median ratio is within 0.7–1.3.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run pytest. Expected: PASS. If the mass-prior correlation gate fails, report the observed r rather than weaken the test without asking.
- [ ] **Step 5:** Commit: `feat(framed): beam-theory features and analytic mass prior`.

### Task 5: Improved surrogates with uncertainty (the "show the improvement" step)

**Files:**
- Create: `src/framed/surrogates.py`
- Test: `tests/test_framed.py`

**Interfaces:**
- Consumes: `add_physics_features`, `analytic_mass_prior`, `cross_validate`, `fold_indices`, `paper_models`
- Produces:
  - `transform_targets(Y: np.ndarray) -> np.ndarray` and `inverse_transform_targets(T) -> np.ndarray`: `log(y + eps_j)` per column, where `eps_j` = 1 % of the column's training median; mass and SF are logged too
  - `guard_framed(Y: np.ndarray) -> np.ndarray`, applying decision 10, and `clip_framed_inputs(X, lo, hi) -> tuple[np.ndarray, np.ndarray]` returning (clipped X, extrapolation flag)
  - `our_models(seed: int) -> dict[str, Callable[[], Regressor]]` with keys `xgboost_phys`, `autogluon_phys` (optional), `gp_phys` (sklearn GP, ARD Matern 2.5 + White, fitted on the 20 highest-|corr| features to bound cost), `deep_ensemble_phys` (5 MLPs, 3×128, SiLU), `pg_mlp` (predicts log(mass/mass_prior) for mass, physics features for the rest). Every model wraps transform → fit → inverse → `guard_framed`.
  - UQ models expose `predict_interval(X, level=0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]` (lo, mean, hi)
  - `latency_report(models, X) -> dict` with ms per design (single, batch 1000) for each model and for the analytical baseline, speedup vs analytical, and projected speedup vs 180 s

- [ ] **Step 1: Write failing tests**
  - `test_transform_roundtrip`: random positive Y including zeros round-trips to 1e-9.
  - `test_guard_framed`: an input containing −1 mass, NaN and SF 50 gives mass > 0, finite values and SF ≤ 20.
  - `test_interval_contains_mean`: for `deep_ensemble_phys` trained on a 200-row fixture, `lo <= mean <= hi` everywhere.
  - `test_improved_beats_raw_xgboost` (skipif no data, `slow`): on 1 × 5 identical folds, `xgboost_phys` mean R² ≥ `xgboost` mean R².
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `.\.venv\Scripts\python.exe -m src.framed.run --stage ours`. Expected output in `models/framed/ours.json`: the same metrics as `reproduction.json`, plus 95 % CI coverage per target for the GP and the ensemble, plus `latency_report`. The headline comparison table is paper AutoML vs best of ours, paired on folds, with a Wilcoxon signed-rank p-value over the 50 fold scores (`scipy.stats.wilcoxon`). Whatever the result, report it as measured.
- [ ] **Step 5:** Run pytest, then commit: `feat(framed): physics-informed surrogates with UQ on FRAMED`.

### Task 6: Out-of-distribution test on the 10,805 generated frames (beyond the paper)

**Files:**
- Modify: `src/framed/surrogates.py` (add a function), `src/framed/run.py`
- Test: `tests/test_framed.py`

**Interfaces:**
- Consumes: `load_framed("generated")`, the models from Tasks 3 and 5 refit on all 4047 originals
- Produces: `evaluate_ood(models: dict, X_tr, Y_tr, X_gen, Y_gen) -> dict` returning, per model: R², MAE and max relative error per target; `extrapolation_fraction` (rows with any input outside the training [min, max]); the same metrics split into in-range and extrapolated rows; and CI coverage for the UQ models

- [ ] **Step 1: Write failing test** `test_ood_reports_extrapolation`: a fixture where 3 of 10 generated rows lie outside the range gives `extrapolation_fraction == 0.3`, with metrics present for both subsets.
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3:** Implement. Do not clip away the extrapolated rows; evaluate them as they are, flagged.
- [ ] **Step 4:** Run `python -m src.framed.run --stage ood` to write `models/framed/ood.json`, then pytest. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(framed): OOD generalisation test on generated frames`.

### Task 7: Constrained lightweighting optimisation (the paper's Future Work)

**Files:**
- Create: `src/framed/optimize.py`
- Test: `tests/test_framed.py`

**Interfaces:**
- Consumes: `src.optimizer.nsga2`, `src.optimizer.parego`, `src.optimizer.knee_point`, the best UQ surrogate from Task 5
- Produces:
  - `class FramedProblem` with attributes `xl, xu, n_var, archive, tag, n_evals` and methods `evaluate(G) -> {"Y","F","V","cv","feasible"}` and `normalise(F)`, matching what `src/optimizer.py` expects. The reference frame is `X.loc[3853]` (paper "Example frame 1": a titanium road frame, 2.84 kg, SF3 1.073, so it is already marginal on SF3; assert its mass within 0.01 kg). Variables: OD and thickness of TT, DT, ST, SS and CS (10 variables); HT and BB stay fixed as standard interfaces. Bounds are the 1st–99th percentile of FRAMED. Material is fixed per run (steel, Al, Ti: 3 runs).
  - Objectives: min mass, min |Sim 1 dropout Y disp.| (in-plane stiffness). Constraints: SF1 ≥ 1.0 and SF3 ≥ 1.0 on the **95 % CI lower bound** (FRAMED validity definition, made uncertainty-aware); |Sim 2 BB Z| ≤ the reference frame's value (no loss of transverse stiffness); D/t ≤ 50 (thin-wall manufacturability, kept in `config.MAX_D_OVER_T`).
  - `nearest_simulated(G_full: np.ndarray, X_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]` returning (index, standardised distance) to the closest FRAMED design, as the stand-in verification
  - `run_study(material: str, seed: int) -> pd.DataFrame` with columns for the Pareto designs, predictions, CI and nearest-design distance

- [ ] **Step 1: Write failing tests**
  - `test_problem_duck_types_optimizer`: `nsga2(FramedProblem(stub_surrogate), pop_size=8, generations=2)` runs and returns arrays.
  - `test_reference_frame_mass` (skipif no data): `Y.loc[3853, "Model Mass"]` is within 2.84 ± 0.01.
  - `test_ci_lower_bound_constraint`: a stub surrogate with mean SF 1.2 and lower bound 0.9 gives `cv > 0`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `python -m src.framed.run --stage optimize` to write `data/framed_pareto.csv` and `models/framed/optimize.json` (baseline vs knee vs lightest feasible per material: mass saving %, stiffness change %, SF lower bound, nearest-design distance). Then run pytest.
- [ ] **Step 5:** Commit: `feat(framed): uncertainty-aware lightweighting of a FRAMED reference frame`.

### Task 8: Report, app tab, README, pipeline

**Files:**
- Modify: `src/framed/run.py` (`--stage report`, `--stage all`), `src/framed/figures.py`, `app.py`, `README.md`, `run_pipeline.ps1`
- Test: `tests/test_framed.py`

**Interfaces:**
- Consumes: the JSON outputs of Tasks 3, 5, 6 and 7, and the figures from Task 2
- Produces:
  - Figures: `fig_table4_comparison(repro, ours)` (grouped bars: paper vs reproduced vs ours, R² and MAE), `fig_ood(ood)`, `fig_framed_pareto(df)`
  - `reports/framed/index.html` (gallery) and `reports/framed_benchmark.md` (sections: Source & citation, Reproduction with a pass/fail table, Discrepancies, Improvements with paired statistics, OOD, Optimisation, Limitations)
  - App tab "FRAMED benchmark" with sub-views: Reproduction, Our models, OOD, Optimiser. It reads only JSON/CSV through a small loader in `src/framed/run.py`, and when the artefacts are missing it shows the command to generate them.
  - README section "Benchmark on published data (FRAMED)", with the citation, the MIT attribution, the headline table between new `<!-- FRAMED:START -->`/`<!-- FRAMED:END -->` markers, and honest limitations: bicycle, not motorcycle; SolidWorks solid FE that cannot be re-run here; 4047 rows against the paper's 4118; inferred CV protocol; MSE inconsistency.
  - `run_pipeline.ps1` gets a `-Framed` switch that runs `python -m src.framed.run --stage all`.

- [ ] **Step 1: Write failing test** `test_report_builds_from_fixtures`: given fixture JSONs in `tmp_path`, `build_report(tmp_path)` writes an `index.html` and an `.md` that contains "Regenwetter".
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3:** Implement, and use the `run` skill to launch the Streamlit app and screenshot tab 5.
- [ ] **Step 4:** Run `.\.venv\Scripts\python.exe -m pytest` (full suite, including the 26 V1 tests). Expected: all pass.
- [ ] **Step 5:** Commit: `docs(framed): benchmark report, app tab and README section`.

---

## Expected outcomes and risks

| Risk | Mitigation |
|---|---|
| AutoGluon may not install on Python 3.13, or may be slow on a laptop | Optional; `--skip-autogluon`; report the paper's AutoML number as the reference and compare against it |
| Reproduced R² misses ±0.05 because of the inferred protocol or the 4047-row data | Report as a discrepancy, with two protocol variants (uniform vs variance-weighted R²); never tune to match |
| The physics features may not beat AutoML on SF (paper: stress is mesh-sensitive, Sec. 3.5) | Report per target; the expected gains are in mass and stiffness, where beam theory applies |
| No FE re-check of optimised frames | CI lower-bound constraints + nearest-design distance; offer a SolidWorks re-run with the authors' `PreSW.py`/`PostSW.py` templates if a licence is available |
| Run time: 5 models × 50 folds + GP | `--fast` (1 × 5 folds); the GP uses the top 20 features |
