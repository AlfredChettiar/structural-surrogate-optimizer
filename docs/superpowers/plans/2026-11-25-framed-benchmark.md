# FRAMED Benchmark (Reproduce, Cross-check with OptiStruct, Improve, Report) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (native mode, chosen by the user).
> Reviews are done by a **subagent** using superpowers:requesting-code-review at the three review gates marked
> 🔍 below. The report task uses the `impeccable:impeccable` skill. Steps use checkbox (`- [ ]`) syntax.

**Goal:** On the user's Windows machine, reproduce the published FRAMED bicycle-frame study (data, statistics, figures,
surrogate results), cross-check its SolidWorks FE with an independent Altair OptiStruct beam model, run this project's
physics-guided surrogates and uncertainty-aware Pareto optimiser on the same design space, FE-verify the optimised
frames in OptiStruct, and publish a portfolio-grade report that compares every number and plot with the paper.

**Architecture:** A self-contained package `src/framed/` next to the untouched V1 pipeline. It holds data,
paper reference values, reproduction, an OptiStruct deck writer and runner, physics features, surrogates,
optimisation, figures and a report builder. From V1 it reuses only `src/optimizer.py` (duck-typed problem
interface) and the Plotly style in `src/plots.py`. All ML runs locally on the user's CPU, RAM and SSD. OptiStruct
runs locally in batch mode from Python.

**Tech Stack:** NumPy, Pandas, SciPy, scikit-learn, PyTorch, Plotly, Streamlit, Pytest (existing). Added: `xgboost`,
`PyMuPDF` (render the paper's figure pages for the side-by-side comparison), `psutil` (record hardware in the
report), and optionally `autogluon.tabular` (paper's AutoML). External tools: Altair OptiStruct 2026 (batch solver),
HyperMesh 2026 (visual check of decks and results only).

**Spec:** This document ("Source", "Paper reference values", "Design decisions").

---

## Where and how this runs

- **Machine:** the user's Windows PC, `C:\Users\alfre\Projects\structural-surrogate-optimizer`. This cloud session
  cannot reach that folder: Remote Control lists it but does not serve it to cloud sessions. Execution therefore
  happens in a **local Claude Code session** (desktop app or CLI) opened in that folder. This plan is committed on branch
  `claude/optimistic-meitner-vvd4ne`, so the local session starts by pulling it (Task 0).
- **Python:** `.\.venv\Scripts\python.exe` for everything (CLAUDE.md).
- **Execution mode:** native, meaning one session implements every task. A fresh reviewer subagent runs at 🔍 Gate A
  (after Task 4), 🔍 Gate B (after Task 6) and 🔍 Gate C (after Task 10). Its findings are fixed before moving on.

## Source

| Item | Value |
|---|---|
| Paper [1] | L. Regenwetter, C. Weaver, F. Ahmed, *FRAMED: An AutoML Approach for Structural Performance Prediction of Bicycle Frames*, Computer-Aided Design (Elsevier), 2023. arXiv:2201.10459. MIT DeCoDE Lab. |
| Paper PDF | author's site repo `github.com/Lyleregenwetter/Lyleregenwetter.github.io`, `papers/2022_regenwetter_framed.pdf` (preprint dated 29 Oct 2022) |
| Data + code | `github.com/Lyleregenwetter/FRAMED`, MIT licence, pinned commit `b881634bb3b79240f08fc4285483becf6ec7d7c5` |
| Files | `all_structural_data_oh.csv` (4047 × 50) sha256 `8506eb3ec69f85fd35c551c630e5eba68c5ce5b2c78b3a42a33ab681f2697b5e`; `all_structural_data_oh_aug.csv` (14852 × 50; rows `Gen 0…Gen 10804` = 10,805 AI-generated frames simulated with the same pipeline) sha256 `3373f943b9968b40923e09a912207c53aeda70e4c2c0f1941f8205c13aab0eb7` |
| Paper FE | SolidWorks Simulation, solid mesh, 0.03 mm minimum cell, mesh-convergence study (Sec. 3.5); physical validation against Rinard (1996) (Table 3) |
| Design space | 39 model inputs (37 parameters, material one-hot to 3): 18 geometry (m, deg), 9 tube ODs (m), 7 wall thicknesses (m, Sobol log-scaled 0.5–10 mm), 2 bridge flags, material {AISI 4130, 6061-T6, Ti-6Al-4V} |
| Load cases | (1) in-plane: 2000 N up at dropouts, 2000 N down at BB, head tube fixed; (2) transverse: 500 N lateral at BB, head tube fixed, dropouts laterally restrained; (3) eccentric: 2000 N down + 140 N·m at BB, head tube fixed |
| Outputs (10) | Sim 1 dropout X/Y disp., Sim 1 BB X/Y disp., Sim 2 BB Z disp., Sim 3 BB Y disp., Sim 3 BB X rot., Sim 1 SF, Sim 3 SF, model mass |

## Paper reference values

All go into `src/framed/paper_reference.py` as constants, each with its paper location. **Every comparison in the
report reads from this one file.**

- Table 2 materials (steel / Al / Ti): E 205 / 69 / 105 GPa; ν 0.285 / 0.330 / 0.310; G 80 / 26 / 41 GPa; ρ 7850 / 2700 / 4429 kg/m³; UTS 731 / 310 / 1050 MPa; σy 460 / 275 / 827 MPa.
- Table 3 physical validation (actual / simulated, front defl. mm, rear defl. mm, mass kg): DeRosa SLX 0.40/0.297, 0.15/0.116, 1.966/1.69; Casati Gold Line 0.44/0.3028, 0.15/0.124, 1.966/1.80; Holland SL/SP 0.38/0.26, 0.13/0.107, 1.962/1.77.
- Table 4 regression (DT / KNN / XGBoost / NN / AutoML): R² 0.440 / 0.306 / 0.595 / 0.545 / 0.605; MSE 1.156 / 1.270 / 0.978 / 1.031 / 0.995; MAE 0.297 / 0.412 / 0.208 / 0.263 / 0.182.
- Table 5 classification (DT / KNN / XGB / NN / SVM / AutoML) F1 0.750 / 0.736 / 0.722 / 0.736 / 0.620 / 0.915; ROC AUC 0.817 / 0.844 / 0.936 / 0.892 / 0.924 / 0.998.
- Tables 6–9 best hyperparameters:
  - DT: max_depth 7, splitter best, criterion absolute_error, min_samples_split 5, min_samples_leaf 11, max_features 32
  - KNN: n_neighbors 70, weights distance, metric euclidean
  - XGBoost: max_depth 6, reg_lambda 72.41212, learning_rate 0.275542, gamma 0.059818, min_child_weight 4.819113, early-stopping 8
  - NN: 1 hidden layer of 92, ReLU, no batch-norm, dropout 0.578647, lr 7.2e-5, batch 135, patience 44
- Sec. 5 counts: 3419 of 4118 fail at least one SF ≥ 1 check. On the repo data (`Analysis.ipynb` output): both fail 1966, SF1 only 1272, SF3 only 182, valid 626; non-dominated in the 5-objective subset 76.
- Sec. 5.2 case studies (verified against the pinned data while writing this plan): index 3853 (Ti) 2.84 kg, SF1 1.29, SF3 1.073; index 2108 (steel) 22.83 kg, SF1 0.0926; index 857 (Ti) 1.60 kg, SF1 2.416, SF3 2.986.
- Paper figures to compare against: Fig. 4 material breakdown, Fig. 5 mesh convergence, Fig. 6 performance pair plot, Fig. 7 case-study frames, Fig. 8 and Fig. 9 AutoML parity plots (train and validation, 10 outputs). Their page numbers go in `PAPER_FIGURE_PAGES`, filled in during Task 2 by rendering the PDF.

## Design decisions

1. **FRAMED** is the only peer-reviewed, public, tube-parametric two-wheeler frame dataset found that comes with FE
   labels, physical validation, code and published baselines. NAFEMS publishes solver verification cases, not
   design-space datasets.
2. **Preprocessing follows the authors' `Analysis.ipynb`:** X = first 39 columns; Y = 10 outputs; `abs()` on the
   seven displacement/rotation columns and mass.
3. **Table 4 protocol (inferred; the AutoML code is not in the repo):** StandardScaler on X and Y fitted per training
   fold; R², MSE and MAE are uniform means over the 10 standardised targets; 10 repeats × 5-fold `KFold(shuffle=True)`.
   The paper's MSE does not fit its R² for unit-variance targets, so **R² and MAE decide whether the reproduction
   passes; MSE is reported only.**
4. **Reproduction passes** when each baseline is within ±0.05 R² of Table 4 and the ranking
   AutoML ≥ XGBoost > NN > DT > KNN holds. Misses are reported as discrepancies, never tuned away.
5. **Figure comparison:** our Plotly figure goes side by side with the paper's figure page rendered from the PDF
   (PyMuPDF at 150 dpi). Rendered paper pages stay **local only** (`reports/framed/paper_pages/`, git-ignored)
   unless the arXiv licence permits reuse; the published report then cites "Fig. N in [1]" with our version plus
   the numeric comparison. Where the paper prints numbers (tables, case studies, counts), the comparison is a
   numeric table with absolute and % deviation.
6. **OptiStruct cross-check.** We have OptiStruct, not SolidWorks, so we build an **independent 3-D beam model**
   (CBEAM, PBEAML TUBE/ROD, MAT1 from Table 2, 3 subcases). It reproduces the paper's load cases, and stays a
   beam model by intent (ponytail: a solid model would need the authors' CAD template, which is not public).
   - **Geometry check first:** frame-node coordinates are rebuilt from the 18 BikeCAD geometry parameters.
     Because mass depends only on geometry, OptiStruct mass vs FRAMED mass is the proof that the geometry is
     right. Gate: median |error| ≤ 5 %.
   - **Solver cross-check:** OptiStruct is run on a stratified sample of FRAMED designs, and displacements and SF are
     compared with SolidWorks. The expected and reported outcome: displacements correlate strongly; SF differs
     systematically, because the paper says its SF captures junction stress singularities (Sec. 3.5) and a beam
     model has no SCF.
   - **Verifying optimised designs:** each optimised frame and its reference frame are both solved in
     OptiStruct, and the claim is made on the **ratio** (mass saving at equal or better stiffness and SF). A
     systematic beam-vs-solid bias cancels in a ratio.
7. **"My model":** this project's method on FRAMED's data. That means beam-theory features (A, I, J, EI, GJ, an
   analytic mass prior), log target transforms, a GP with 95 % CI, a deep ensemble with CI and a physics-guided
   MLP, plus XGBoost and AutoGluon on the same features, which separates the gain from features from the gain
   from the model. All of them use the **same folds** as the reproduction, compared with a paired Wilcoxon test.
8. **Optional multi-fidelity step (Task 7), run only if Task 5 shows Pearson r ≥ 0.9 on displacements:** OptiStruct
   beam results for all 4047 designs become a low-fidelity input feature or prior, the way the V1 hand calc is used.
9. **Optimisation implements the paper's Future Work (Sec. 8)** on reference frame 3853 (Ti road frame, 2.84 kg).
   Rider-fit geometry is frozen (the paper warns that shrinking the frame trivially "optimises" it, Sec. 5.2.3).
   The variables are the OD and wall thickness of TT, DT, ST, SS and CS; the objectives are min mass and min in-plane
   dropout deflection; the constraints are SF1, SF3 ≥ 1.0 on the 95 % CI lower bound and transverse BB deflection no
   worse than the reference.
10. **Guardrails on every FRAMED prediction:** mass > 0, displacement magnitudes ≥ 0, SF in [0, 20], finite outputs.
    Inputs outside the training range are clipped and flagged.
11. **Speedup (CLAUDE.md):** measured against two baselines, (a) the analytic mass/stiffness calculation and
    (b) **the measured OptiStruct wall time per design**, which replaces the old nominal 180 s.
12. **Report:** a single static HTML page plus a Markdown twin, built with `impeccable:impeccable` in **Read mode**
    (the visitor needs to understand an engineering study). It follows impeccable's flow: `impeccable context` →
    `init` (PRODUCT.md) → new-work for the report surface → `craft-floor.md` before any edit → verify with
    `impeccable shot` at desktop and phone widths.

## Global Constraints

- Run all Python with `.\.venv\Scripts\python.exe`. Never read raw files in `data/` or `models/`; use loaders or short Python commands.
- Report R², MAE, max relative error (%) and speedup for every surrogate. Max relative error on displacement targets uses a denominator floor of 5 % of the training median; record the floor.
- Run `.\.venv\Scripts\python.exe -m pytest` before every commit. Tests needing data or OptiStruct are `skipif`-marked, so the suite still passes on a machine without them.
- `n_jobs` defaults to `os.cpu_count() - 1`. Heavy artefacts go to `models/framed/`, `data/external/framed/` and `runs/optistruct/`, all git-ignored.
- OptiStruct path comes from env var `OPTISTRUCT_EXE`, defaulting to `C:\Program Files\Altair\2026\hwsolvers\scripts\optistruct.bat`. Verify the path in Task 0; never hard-code it anywhere else.
- AutoGluon is optional (`try` import). Missing means its rows read "not run".
- Credit FRAMED (MIT, copyright 2022 Lyleregenwetter) and the paper in the README and report; commit no FRAMED data files.
- Never commit or push `prompts/`. Delete nothing in Task 0 without the user's explicit OK on the listed items.

## Review Focus

1. **Thickness ≥ OD/2** (stays as thin as 2.35 mm OD with walls up to 10 mm): section properties and the PBEAML deck must switch to a solid ROD, not write a negative inner diameter. Tested in Task 3 and Task 5.
2. **Heavy-tailed outputs** (|Sim 1 dropout Y| up to 1.386 m against a 0.05 m std): log transforms must stay finite at 0, and metrics are reported on both the standardised and the physical scale. Tested in Task 6.
3. **OptiStruct failures** (singular stiffness, licence errors, a missing `.pch`): the runner must record a status per design and continue, and comparisons exclude failed runs with the count reported. Tested in Task 5.
4. **Extrapolated generated frames:** the OOD evaluation reports the extrapolation fraction and does not clip it away. Tested in Task 8.
5. **Non-one-hot material rows** from the optimiser or sliders are rejected or snapped. Tested in Tasks 3 and 9.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/framed/config.py` | paths, pinned URLs, sha256s, column lists, `OPTISTRUCT_EXE` |
| `src/framed/paper_reference.py` | every paper number and figure page, with its source location |
| `src/framed/data.py` | fetch and checksum data and PDF, loaders, fold indices |
| `src/framed/paper_stats.py` | feasibility counts, non-dominated set, correlations |
| `src/framed/baselines.py` | paper models, CV protocol, reproduction table |
| `src/framed/physics_features.py` | section properties, materials, analytic mass prior |
| `src/framed/geometry.py` | 18 geometry parameters → 3-D node coordinates and tube list |
| `src/framed/optistruct.py` | `.fem` deck writer, batch runner, `.pch` parser |
| `src/framed/surrogates.py` | improved models, transforms, guardrails, UQ, OOD evaluation, latency |
| `src/framed/optimize.py` | `FramedProblem`, study runner, OptiStruct verification |
| `src/framed/figures.py` | Plotly figures; paper-page renderer |
| `src/framed/report.py` | builds `reports/framed/index.html` and `reports/framed_benchmark.md` |
| `src/framed/run.py` | CLI `--stage {fetch,stats,baselines,optistruct,ours,multifidelity,ood,optimize,report,all} [--fast]` |
| `tests/test_framed.py` | all FRAMED tests |
| `app.py` | tab 5, "FRAMED benchmark" (reads artefacts only) |

---

### Task 0: Local setup and folder cleanup (local session, no code)

- [ ] **Step 1: Sync.** Run `git status`, `git stash list` and `git worktree list`, then `git fetch origin`, `git switch claude/optimistic-meitner-vvd4ne` and `git pull`. If the worktree `.claude\worktrees\admiring-ishizaka-412344` still holds uncommitted `prompts/PROMPT_HISTORY.md` edits, **ask the user** whether to keep or discard them before touching that worktree.
- [ ] **Step 2: Inventory.** Run `Get-ChildItem -Force -Recurse -Directory | ...` (top 2 levels, with sizes) and list the cleanup candidates in a table: `__pycache__`, `.pytest_cache`, `logs\`, stale worktrees, stray files not tracked by git and not ignored (`git status --porcelain --ignored`), and old V1 outputs that `run_pipeline.ps1` can regenerate. **Never list** `.venv`, `.git`, `src`, `tests`, `app.py`, docs, or `prompts\` unless the user names it.
- [ ] **Step 3: Ask the user** to approve the table, then delete only the approved items. Report freed space.
- [ ] **Step 4: Environment.** Run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt xgboost PyMuPDF psutil`, then try `pip install autogluon.tabular` and record success or failure. If AutoGluon needs a different Python, create `.venv-ag` and record that in `models/framed/env.json`.
- [ ] **Step 5: Baseline check.** Run `.\.venv\Scripts\python.exe -m pytest`. Expected: the 26 V1 tests pass. Record CPU model, cores, RAM, free SSD space, Python version and OptiStruct version (`& $env:OPTISTRUCT_EXE -v`, or locate it under `C:\Program Files\Altair\2026`) in `models/framed/env.json`.
- [ ] **Step 6:** Commit `.gitignore` (add `data/external/`, `runs/`, `reports/framed/paper_pages/`) and `requirements.txt`, with the message `chore: local setup for FRAMED benchmark`.

### Task 1: Data and paper acquisition

**Files:** Create `src/framed/__init__.py`, `config.py`, `paper_reference.py`, `data.py`, `tests/test_framed.py`

**Interfaces (produces):**
- `fetch_framed(dest: Path = config.FRAMED_DIR, force: bool = False) -> dict[str, Path]`: downloads the data files from `https://raw.githubusercontent.com/Lyleregenwetter/FRAMED/<commit>/<file>` and the paper PDF, verifies the sha256, and raises `ValueError` on a mismatch. Record the PDF's sha256 on first download.
- `load_framed(kind: Literal["original", "generated"] = "original") -> tuple[pd.DataFrame, pd.DataFrame]`, returning (X, Y)
- `fold_indices(n: int, n_folds: int = 5, n_repeats: int = 10, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]`

- [ ] **Step 1: Tests.**
  - `test_sha_mismatch_raises`
  - `test_load_shapes`: original X `(4047, 39)` and Y `(4047, 10)`; generated X `(10805, 39)`; no NaN; ABS columns ≥ 0; one-hot row sum 1
  - `test_fold_indices`: 50 splits, each repeat partitions the rows, deterministic
  - `test_case_study_frames_match_paper`: the three Sec. 5.2 frames match the values in `paper_reference` (mass ±0.005 kg, SF ±0.005)
- [ ] **Step 2:** Run the tests. Expected: FAIL. **Step 3:** Implement with the standard library's `urllib`. **Step 4:** Run `python -m src.framed.run --stage fetch`, then pytest. Expected: PASS. **Step 5:** Commit `feat(framed): pinned dataset, paper and reference values`.

### Task 2: Reproduce the dataset statistics and figures, then compare with the paper

**Files:** Create `src/framed/paper_stats.py`, `src/framed/figures.py`

**Interfaces (produces):**
- `feasibility_counts(Y) -> dict[str, int]` (`both_fail, sf1_only, sf3_only, valid`)
- `non_dominated_mask(F: np.ndarray, sense: np.ndarray) -> np.ndarray`, with sense −1 = minimise and +1 = maximise
- `render_paper_pages(pdf: Path, pages: list[int], out: Path, dpi: int = 150) -> list[Path]`
- Figures (each returns `go.Figure`): `fig_material_breakdown`, `fig_performance_pairs` (Fig. 6 as a splom, valid vs invalid, with the 3 case-study frames marked), `fig_objective_correlation`, `fig_physical_validation` (Table 3)
- `stats_comparison() -> pd.DataFrame` with columns quantity, paper, ours, abs dev, % dev

- [ ] **Step 1: Tests.**
  - `test_feasibility_counts_logic`: a 4-row fixture
  - `test_feasibility_matches_notebook`: each count within ±2 of 1966 / 1272 / 182 / 626
  - `test_non_dominated_matches_paper`: 76 ± 2
  - `test_figures_build`
- [ ] **Step 2–4:** red → implement → green. Render the PDF pages once, find each figure's page, and fill in `PAPER_FIGURE_PAGES`.
- [ ] **Step 5:** Commit `feat(framed): dataset statistics and paper-figure comparison`.

### Task 3: Physics features and analytical baseline

**Files:** Create `src/framed/physics_features.py`

**Interfaces (produces):**
- `section_properties(od, t) -> (A, I, J)`: SI units, with `t` clipped to `od/2`
- `material_props(X) -> DataFrame[E, G, rho, sigma_y]`: raises `ValueError` for a non-one-hot row
- `analytic_mass_prior(X) -> np.ndarray` = Σ ρ·A·L over tube lengths from `geometry.tube_lengths` (Task 5). Until Task 5 lands, it uses only the five explicit lengths (ST, HT, DT, CS, BB), noted as `shortcut:` in a comment.
- `add_physics_features(X) -> DataFrame`: X plus per-tube `A_*`, `EI_*`, `GJ_*`, plus `mass_prior` and `sigma_y`

- [ ] **Step 1: Tests.**
  - `test_hollow_tube_area`: OD 28.6 mm, t 0.9 mm gives A = 78.32 mm² (rel 1e-4) and I = π/64·(28.6⁴ − 26.8⁴) mm⁴
  - `test_solid_rod_when_thick`: OD 2.35 mm, t 10 mm gives a solid rod, finite and > 0
  - `test_material_props_rejects_bad_onehot`
- [ ] **Step 2–4:** red → implement → green. **Step 5:** Commit `feat(framed): beam-theory features`.

### Task 4: Reproduce Table 4 locally (the paper's models)

**Files:** Create `src/framed/baselines.py`; modify `pytest.ini` (`slow` marker)

**Interfaces (produces):**
- `paper_models(seed) -> dict[str, Callable[[], Regressor]]` with keys `decision_tree, knn, xgboost, neural_net, autogluon?`, using the hyperparameters from `paper_reference`
- `Regressor`: `fit(X, Y) -> self`, `predict(X) -> ndarray` (multi-output)
- `cross_validate(factory, X, Y, folds) -> dict`: `r2, mse, mae` (mean ± std, standardised scale), `per_target` (R², MAE, max relative error % on the physical scale), `fold_scores` (list of 50), `oof_pred` (repeat 0)
- `reproduction_table(results) -> DataFrame`: columns paper R², ours R², ΔR², paper MAE, ours MAE, pass
- `fig_parity_grid(Y, oof, title)`: 10 panels, Fig. 9 layout; `fig_table4_comparison(repro, ours=None)`

- [ ] **Step 1: Tests.**
  - `test_cv_metric_on_perfect_model`: R² 1, MAE 0
  - `test_paper_hyperparameters`: KNN k = 70 and distance weights; DT depth 7 and leaf 11
  - `test_autogluon_missing_is_skipped`
  - `test_reproduction_knn_dt` (`slow`): 1 × 5 folds within ±0.05 of 0.306 and 0.440
- [ ] **Step 2–3:** red → implement.
- [ ] **Step 4: Full local run.** `python -m src.framed.run --stage baselines` runs 10 × 5 folds for DT, KNN, XGB and NN, and AutoGluon on 1 × 5 folds with `presets="medium_quality"` and `time_limit=120` s per target. It writes `models/framed/reproduction.json` (with wall time and hardware) and the parity figure for the best model, next to rendered paper Fig. 9.
- [ ] **Step 5:** Run pytest, then commit `feat(framed): reproduce Table 4 locally`.
- [ ] 🔍 **Gate A, reviewer subagent.** Use superpowers:requesting-code-review on Tasks 1–4. Brief: check the protocol against decision 3, that the hyperparameters match Tables 6–9, data leakage (scalers per fold), and that the reproduction table's pass/fail is honest. Fix the findings, then rerun pytest.

### Task 5: OptiStruct beam model and solver cross-check

**Files:** Create `src/framed/geometry.py`, `src/framed/optistruct.py`

**Interfaces (produces):**
- `frame_nodes(x: pd.Series) -> dict[str, np.ndarray]`: 3-D coordinates (m) of BB centre, BB ends, ST top, HT top and bottom, TT and DT junctions, and the left and right dropouts and stay junctions. BikeCAD conventions: origin at the BB, x rearward-negative, y up, z lateral. Uses `Stack`, `HT Angle`, `HT Length`, `HT LX`, `HT UX`, `ST Angle`, `ST Length`, `ST UX`, `DT Length`, `CS Length`, `BB Drop`, `BB Length`, `CS F`, `SS E`, `SS Z`, `Dropout Offset`, `SSB/CSB Offset`. One comment line per derived point.
- `tube_lengths(x) -> dict[str, float]`, also used by `analytic_mass_prior`
- `write_deck(x: pd.Series, path: Path, elems_per_tube: int = 10) -> Path`. Writes a `.fem` with:
  - GRID and CBEAM entries; PBEAML TUBE (`DIM1 = OD/2`, `DIM2 = OD/2 − t`) or ROD when `t ≥ OD/2`; MAT1 from Table 2
  - three SUBCASEs built from SPC/FORCE/MOMENT per the load cases in "Source"
  - `DISPLACEMENT(PUNCH) = ALL`, `STRESS(PUNCH) = ALL`, and a mass output (`PARAM,GRDPNT,0`, so the `.out` has total mass)
  - bridges included as CBEAMs when `SSB_Include` / `CSB_Include` are set
- `run_optistruct(decks: list[Path], n_parallel: int, ncpu_per_job: int = 1, timeout_s: int = 300) -> pd.DataFrame`, with columns deck, status, wall_s. Uses `subprocess`, `-nt ncpu_per_job` and a thread pool.
- `parse_results(deck: Path) -> dict`: the 10 FRAMED outputs in FRAMED's sign and units, from `.pch` and `.out`. SF = σy / max beam von Mises per subcase.

- [ ] **Step 1: Tests (no OptiStruct needed).**
  - `test_geometry_reference_frame`: for frame 3853, the ST length between nodes equals `ST Length`, the DT length equals `DT Length` (±1e-6 m), and the chain-stay length equals `CS Length`
  - `test_deck_has_three_subcases`: on 3853, the deck contains 3 `SUBCASE` entries, a PBEAML for every tube, and no negative dimensions
  - `test_rod_when_thick`
  - `test_parse_pch_fixture`: a small hand-written `.pch` fixture gives known numbers
  - `test_runner_records_failure`: a fake executable that exits 1 gives status `"failed"`, and the run continues
- [ ] **Step 2–3:** red → implement.
- [ ] **Step 4: Smoke run.** Write the deck for 3853, run OptiStruct, then open the deck and `.h3d` in **HyperMesh** and save two screenshots (mesh, and subcase 1 deformed) to `reports/framed/img/`. Check that the BCs and load directions look right.
- [ ] **Step 5: Geometry gate.** Compute beam-model mass in Python (Σ ρ·A·L from `tube_lengths`) for all 4047 designs and compare it with FRAMED mass. Gate: median |error| ≤ 5 %. Confirm on 5 smoke-run designs that the OptiStruct `.out` grid-point-weight mass equals the Python mass within 0.5 %. On failure, fix `frame_nodes` and do not continue.
- [ ] **Step 6: Cross-check.** Run the full static solve on a 300-design sample stratified by material × valid/invalid. Write `models/framed/optistruct_xcheck.json` with Pearson r, R², median ratio OptiStruct/SolidWorks and Bland–Altman bias per output, the failure count and the mean wall time per design. Figures: `fig_solver_crosscheck` (10-panel parity, OptiStruct vs SolidWorks).
- [ ] **Step 7:** Run pytest, then commit `feat(framed): independent OptiStruct beam model and cross-check`.

### Task 6: Improved surrogates with UQ

**Files:** Create `src/framed/surrogates.py`

**Interfaces (produces):**
- `transform_targets(Y, eps) -> T` and `inverse_transform_targets(T, eps) -> Y`, with `log(y + eps_j)` and `eps_j` = 1 % of the training median
- `guard_framed(Y) -> Y` and `clip_framed_inputs(X, lo, hi) -> (X, flag)`
- `our_models(seed) -> dict` with keys:
  - `xgboost_phys`; `autogluon_phys` (optional)
  - `gp_phys`: ARD Matern 2.5 + White on the 20 highest-|corr| features
  - `deep_ensemble_phys`: 5 × MLP 3×128 SiLU
  - `pg_mlp`: predicts log(mass/mass_prior) for mass
- UQ models expose `predict_interval(X, level=0.95) -> (lo, mean, hi)`
- `latency_report(models, X, optistruct_wall_s) -> dict`: ms per design (single and batch of 1000), speedup vs analytic, speedup vs measured OptiStruct

- [ ] **Step 1: Tests.**
  - `test_transform_roundtrip`
  - `test_guard_framed`: mass −1, NaN and SF 50 become a valid output
  - `test_interval_contains_mean`
  - `test_improved_beats_raw_xgboost` (`slow`, 1 × 5 folds)
- [ ] **Step 2–3:** red → implement.
- [ ] **Step 4: Full local run.** `--stage ours` writes `models/framed/ours.json`: Table-4 metrics on the same 50 folds, per-target physical-scale metrics, 95 % CI coverage, the paired Wilcoxon test against the paper's AutoML (or its published 0.605 when AutoGluon did not run), and the latency report.
- [ ] **Step 5:** Run pytest, then commit `feat(framed): physics-informed surrogates with UQ`.
- [ ] 🔍 **Gate B, reviewer subagent.** Use superpowers:requesting-code-review on Tasks 5–6. Brief: whether the geometry reconstruction is plausible, BC and load fidelity against the paper's Sec. 3.3, `.pch` parsing units and signs, fair comparison on the same folds, and whether any metric is computed on transformed instead of physical values without saying so.

### Task 7 (conditional): Multi-fidelity with OptiStruct as the low-fidelity source

Run this only if Task 5 gives r ≥ 0.9 on at least 5 of the 7 displacement outputs. Otherwise record "skipped: r = …" and go to Task 8.

- [ ] **Step 1:** Solve all 4047 originals in OptiStruct in batch (time estimated from the Task 5 wall time × 4047 / n_parallel; tell the user the estimate before starting). Cache the results in `models/framed/optistruct_all.parquet`.
- [ ] **Step 2:** Add an `mf_*` variant of the best Task 6 model that takes the OptiStruct outputs as features (and, for `pg_mlp`, learns the SolidWorks/OptiStruct log-ratio). Test: `test_mf_feature_alignment` checks that row indices match FRAMED.
- [ ] **Step 3:** `--stage multifidelity` runs the same 50 folds and adds the results to `ours.json`. Then run pytest and commit `feat(framed): OptiStruct multi-fidelity surrogate`.

### Task 8: Out-of-distribution test on the 10,805 generated frames

- [ ] **Step 1: Test** `test_ood_reports_extrapolation`: 3 of 10 rows out of range give a fraction of 0.3 and metrics for both subsets.
- [ ] **Step 2–3:** red → implement `evaluate_ood(models, X_tr, Y_tr, X_gen, Y_gen) -> dict` in `surrogates.py`. Refit on all 4047 originals, evaluate the paper's models and ours, and report CI coverage for the UQ models.
- [ ] **Step 4:** `--stage ood` writes `models/framed/ood.json`. Run pytest, then commit `feat(framed): OOD test on generated frames`.

### Task 9: Uncertainty-aware lightweighting with OptiStruct verification

**Files:** Create `src/framed/optimize.py`

**Interfaces (produces):**
- `class FramedProblem`: attributes `xl, xu, n_var, archive, tag, n_evals`; methods `evaluate(G) -> {"Y","F","V","cv","feasible"}` and `normalise(F)`, matching `src/optimizer.py`. Reference frame `X.loc[3853]`; 10 variables (OD and t of TT, DT, ST, SS, CS); bounds at the FRAMED 1st–99th percentile; D/t ≤ 50; material fixed per run.
- `run_study(material: str, seed: int, algorithms=("nsga2","parego")) -> pd.DataFrame`: the Pareto set, knee point (`src.optimizer.knee_point`), the lightest feasible design and the CI bounds
- `verify_with_optistruct(designs: pd.DataFrame) -> pd.DataFrame`: writes decks for the reference, knee and lightest designs (plus the full Pareto set if cheap), runs them, and reports OptiStruct mass, displacements and SF with **ratios to the OptiStruct reference** (decision 6)

- [ ] **Step 1: Tests.**
  - `test_problem_duck_types_optimizer`: a stub surrogate works with `nsga2(pop_size=8, generations=2)`
  - `test_ci_lower_bound_constraint`: mean SF 1.2 with lower bound 0.9 gives `cv > 0`
  - `test_onehot_snapped`
- [ ] **Step 2–3:** red → implement.
- [ ] **Step 4:** `--stage optimize` writes `data/framed_pareto.csv` and `models/framed/optimize.json` for steel, Al and Ti. Each OptiStruct-verified headline (mass saving %, stiffness change %, SF ratio) is reported only if OptiStruct agrees on the direction of every constraint; otherwise it is flagged.
- [ ] **Step 5:** Run pytest, then commit `feat(framed): lightweighting with OptiStruct verification`.

### Task 10: Report (impeccable, Read mode), app tab, README

**Files:** Create `src/framed/report.py`, `PRODUCT.md` (via impeccable init); modify `app.py`, `README.md`, `run_pipeline.ps1`

- [ ] **Step 1: Builder test.** `test_report_builds_from_fixtures`: fixture JSONs give an `index.html` and an `.md` that contain "Regenwetter", every `paper_reference` Table 4 value and a pass/fail cell per check.
- [ ] **Step 2:** Implement `build_report(artifacts_dir, out_dir) -> Path`. It writes a static HTML page with embedded Plotly figures (`include_plotlyjs="cdn"`, pinned version) and a Markdown twin. Report outline:
  1. Summary: three headline numbers (reproduction pass rate, best surrogate vs paper AutoML, OptiStruct-verified mass saving)
  2. Source and why it is credible
  3. Reproduction: statistics and Table 4, each with paper / ours / deviation, and the figure pairs side by side
  4. OptiStruct cross-check, with HyperMesh screenshots
  5. Improvements, with paired statistics and CI calibration
  6. OOD generalisation
  7. Optimisation and FE verification
  8. Discrepancies and limitations: bicycle not motorcycle; beam vs solid; 4047 vs 4118 rows; inferred protocol; MSE inconsistency; any failed gate
  9. Hardware and run times
  10. References and licences
- [ ] **Step 3: Design with impeccable.** Invoke `impeccable:impeccable`. Run `impeccable context`, then `init`: PRODUCT.md for "an engineering portfolio report: reproduce → cross-check → improve", with recruiters and CAE engineers as the audience and Read mode. Then follow new-work for the report surface and read `craft-floor.md` before editing the template. Keep the project's existing Plotly palette (`src/plots.py`) as the chart system, so figures and page read as one.
- [ ] **Step 4: Verify.** Run `impeccable shot reports/framed/index.html` at desktop and phone widths, fix what one look shows, and shoot again. Then run `impeccable critique` and `audit` on the page, and fix the material findings.
- [ ] **Step 5: App and docs.**
  - App tab 5 has sub-views Reproduction, OptiStruct, Our models, OOD and Optimiser. It reads JSON and CSV only, and shows the generating command when artefacts are missing.
  - README section "Benchmark on published data (FRAMED)" sits between `<!-- FRAMED:START/END -->` markers, with the citation and MIT attribution.
  - `run_pipeline.ps1 -Framed` runs `python -m src.framed.run --stage all`.
- [ ] **Step 6:** Run the full `.\.venv\Scripts\python.exe -m pytest` (V1 and FRAMED). Expected: all pass. Commit `docs(framed): benchmark report, app tab, README`.
- [ ] 🔍 **Gate C, reviewer subagent.** A whole-branch review with superpowers:requesting-code-review. Brief: whether every number in the report traces to a JSON artefact or `paper_reference`, no claim is stronger than its evidence, the licences and attribution are correct, and no data or prompts are committed. Fix the findings, rerun pytest, and push.

---

## Risks

| Risk | Mitigation |
|---|---|
| The folder cannot be reached from the cloud | Run in a local Claude Code session (Task 0); this plan is on the branch |
| Geometry reconstruction misses the 5 % mass gate | Stop at Task 5 Step 5, report the per-tube error breakdown, and ask the user before relaxing the gate |
| OptiStruct SF far from SolidWorks SF | Expected (singular junction stresses vs beam); report it, and verify optima by ratio |
| AutoGluon will not install on Windows or this Python | `.venv-ag`, or compare with the published 0.605 |
| Long runs (50-fold CV, OptiStruct batch) | `--fast`; time estimate shown before any run longer than 30 min |
| Paper figure reuse in a public report | Local-only side-by-side unless the arXiv licence permits it; otherwise cite the figure number |
