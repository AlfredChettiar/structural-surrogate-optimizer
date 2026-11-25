# Physics-Guided Structural Surrogate & Pareto Weight Optimizer

Lightweighting a **tubular two-wheeler rear sub-frame** with machine-learning surrogates that are
benchmarked against a finite-element ground truth, then driven by a benchmarked set of
multi-objective optimisers, with every candidate design re-checked against the physics.

> Built as a portfolio project around a CAE-durability workflow: DOE -> FE -> surrogate -> optimisation
> -> FE verification, with physical guardrails at every step.

## What it does

| Stage | What happens | Module |
|---|---|---|
| Physics | Planar **Euler-Bernoulli frame FE** model of the rail + strut: static solve, generalised eigen-solve for f1, ASME B31.3-style **pipe-bend flexibility & SIF**, gusseted weld-toe **SCF**, **Goodman + Basquin** fatigue life. A deliberately simple **closed-form hand calc** (nominal stress, no SCF) is the low-fidelity prior. | `src/physics.py` |
| Data | **1,500 designs**: 1,423 Latin Hypercube + 77 face-centred CCD points over 6 inputs, with realistic measurement scatter. | `src/data_generator.py` |
| Surrogates | **7 models**: Ridge polynomial, Gaussian process (95 % CI), Residual MLP, **Physics-guided MLP** (learns the FE-minus-hand-calc correction), random forest, gradient boosting, SVR. | `src/surrogate_zoo.py`, `src/model.py` |
| Benchmark | R², MAE, max % error (vs noisy labels **and** noise-free FE), CV, latency, speedup, learning curves, GP calibration, extrapolation, Sobol / permutation / PDP sensitivity. A transparent rule selects the best model. | `src/train.py`, `src/evaluation.py` |
| Optimisation | **5 optimisers**: 5,000-candidate LHS screening, NSGA-II, epsilon-constraint + differential evolution, weighted-sum SLSQP, ParEGO (Bayesian). Objectives: min mass, min peak stress. Constraints: SF >= 1.5, f1 >= 45 Hz, deflection <= 2.5 mm, fatigue life >= 1e6 cycles. | `src/optimizer.py` |
| Selection | Hypervolume, IGD, spacing, evaluations-to-99 % HV, wall time, and **FE re-verification** of every claimed Pareto design -> weighted scorecard -> best optimiser + headline mass saving. | `src/optimizer_benchmark.py` |
| Guardrails | Inputs clipped to the trained design space (extrapolation flagged); outputs non-negative, finite, bounded SF and fatigue life. Used by every model, optimiser and UI. | `src/guardrails.py` |
| Outputs | Streamlit app (4 tabs) and a gallery of interactive Plotly figures for the website. | `app.py`, `src/plots.py`, `src/report.py` |

## Architecture

```
config.py ──> physics.py ──> data_generator.py ──> data/chassis_doe_dataset.csv
                  │                                        │
                  │ (closed-form prior)                    v
                  └──────────> surrogate_zoo.py <── model.py (PyTorch)
                                     │
                     train.py + evaluation.py ──> models/*.joblib|*.pt, metrics.json,
                                     │             model_selection.json, evaluation_extras.json
                                     v
                 optimizer.py + optimizer_benchmark.py ──> models/optimizer_benchmark.json,
                                     │                     data/pareto_fronts.csv (FE-verified)
                                     v
                 plots.py ──> report.py (reports/)   and   app.py (Streamlit)
                 guardrails.py is applied on every prediction path
```

## Run it (Windows, project venv)

Install dependencies first: `pip install -r requirements.txt`.

```powershell
# whole pipeline, one log per step in .\logs\ (about 10-15 min on a laptop CPU)
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1        # add -Fast for a quick smoke run

# or step by step
.\.venv\Scripts\python.exe -m src.data_generator
.\.venv\Scripts\python.exe -m src.train          # --fast: 3-fold CV, shorter NN training
.\.venv\Scripts\python.exe -m src.optimizer      # benchmark of all 5 optimisers
.\.venv\Scripts\python.exe -m src.report         # reports/figures/*.html + reports/index.html
.\.venv\Scripts\python.exe -m pytest

# dashboard
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Static PNG export of figures needs `kaleido` (`.\.venv\Scripts\pip install kaleido`); without it the
report writes interactive HTML only, which can be embedded on a website with an `<iframe>`.

## Dashboard

1. **Real-time surrogate explorer**: geometry/material/load sliders; mass, stress, SF, deflection, f1, fatigue
   life with GP 95 % CI; pass/fail per constraint; live FE solve of the same design for side-by-side
   verification and measured speedup.
2. **Pareto weight optimisation**: all optimisers' fronts over the 5,000-candidate cloud (colour = first
   mode), live constraint filters, custom material/load re-screening, CSV download of optimal dimensions,
   baseline-vs-optimised FE comparison.
3. **Model validation & sensitivity**: parity, residual histograms, Q-Q, residual-vs-prediction for any
   model; Sobol, permutation importance and partial dependence for the selected model.
4. **Benchmark & model selection**: model and optimiser scorecards, accuracy-vs-latency map, radar chart,
   learning curves, calibration, extrapolation, speedup, convergence, FE re-verification, parallel coordinates.

## Results

The decision write-ups with full tables are generated into `reports/model_selection.md` and
`reports/optimizer_selection.md`; the figure gallery is `reports/index.html`.

<!-- RESULTS:START -->
Numbers from the **V1** run (seed 42, laptop CPU, no GPU). Test set: 300 held-out designs;
errors vs **noise-free FE**. Full tables: `reports/model_selection.md`, `reports/optimizer_selection.md`.

**Surrogates** (selected by a weighted composite score after an R² gate)

| Model | CV min R² | Worst stress err | Stress MAPE | Worst mass err | Extrapolation stress MAPE | 95% CI | Score |
|---|---|---|---|---|---|---|---|
| **Gaussian process (selected)** | 0.992 | 12.4 % | **1.4 %** | **0.4 %** | **1.9 %** | yes | **0.95** |
| Ridge polynomial (deg 3) | 0.987 | 18.5 % | 2.0 % | 1.3 % | 2.5 % | no | 0.39 |
| Residual MLP | 0.988 | 14.4 % | 2.3 % | 6.9 % | 5.1 % | no | 0.33 |
| Physics-guided MLP | 0.985 | 20.3 % | 2.2 % | 0.5 % | 3.8 % | no | 0.18 |
| Hist gradient boosting | 0.975 (rejected) | 19.0 % | 4.2 % | 3.5 % | 9.7 % | no | - |
| SVR | 0.956 (rejected) | 43.3 % | 3.9 % | 18.0 % | 11.4 % | no | - |
| Random forest | 0.935 (rejected) | 31.0 % | 7.3 % | 10.2 % | 13.8 % | no | - |
| Closed-form hand calc (reference) | -0.03 | 62.0 % | 46.3 % | 0.1 % | 51.8 % | - | - |

- Label-noise floor for worst-case stress error: 8.7 %.
- GP 95 % intervals are well calibrated: observed coverage 94.7-97.3 % across all five outputs.
- Peak stress is driven by tube OD (total Sobol index 0.62) and the dynamic load factor (0.35).
- Speedup vs the ~4-7 ms beam FE solve: 6.5x per single design for the GP, up to ~4,300x in batches for ridge;
  ~2.6 x 10^5 projected vs a 180 s production FE run.

**Optimisers** (surrogate: GP; yield 355 MPa, DLF 3.0; mean of 3 seeds)

| Optimiser | Score | Hypervolume (± std) | FE-verified HV | IGD | Evals to 99 % HV |
|---|---|---|---|---|---|
| **NSGA-II (selected)** | **0.91** | 1.010 ± 0.001 | 1.001 | 0.002 | 2,459 |
| epsilon-constraint + DE | 0.78 | 0.993 ± 0.002 | 0.988 | 0.012 | 3,385 |
| Weighted-sum SLSQP | 0.76 | 0.986 ± 0.007 | 0.983 | 0.017 | 726 |
| ParEGO (Bayesian) | 0.30 | 0.929 ± 0.011 | 0.926 | 0.059 | 113 |
| LHS screening (5,000) | 0.28 | 0.928 ± 0.004 | 0.927 | 0.050 | 4,097 |

All re-checked Pareto designs of every optimiser were FE-feasible. ParEGO plateaus in ~110 evaluations, which makes
it the choice when each evaluation is a real FE run.

**Headline (FE-verified, all constraints incl. manufacturability):**

| Design | Geometry (OD / wall / gusset / bend R, mm) | Mass | Peak stress | SF | f1 | Fatigue life |
|---|---|---|---|---|---|---|
| Baseline | 32.0 / 2.50 / 4.0 / 60 | 3.74 kg | 234 MPa | 1.52 | 47.4 Hz | 1.2e6 |
| **Lightest at ≤ baseline stress** | 35.8 / 1.79 / 4.8 / 84 | **3.20 kg (-14.6 %)** | 189 MPa (-19.3 %) | 1.88 | 45.2 Hz | 1.2e7 |
| **Recommended (knee point)** | 38.0 / 2.00 / 5.0 / 89 | 3.70 kg (-1.1 %) | 155 MPa (-33.7 %) | 2.29 | 51.4 Hz | 9.0e7 |

The lightest design sits on the D/t = 20 and 45 Hz limits; the knee design spends almost the baseline mass on a
much larger margin.
<!-- RESULTS:END -->

<!-- FRAMED:START -->
## Benchmark on published data (FRAMED)

To check the method against a reputed external source, the pipeline was re-run on **FRAMED** [Regenwetter, Weaver & Ahmed,
*Computer-Aided Design* (Elsevier) 2023, MIT DeCoDE Lab; data MIT-licensed]: 4,047 SolidWorks-simulated bicycle frames
(39 inputs, 10 outputs: deflections, safety factors, mass). Code: `src/framed/`; run `python -m src.framed.run --stage all`.
Report: `reports/framed/index.html` (also `reports/framed_benchmark.md`).

- **Paper claims:** 11 of 11 checkable dataset claims reproduce from the released data (frame counts within ±2, case-study frames exact).
- **Table 4 reproduction:** the paper's five models, rebuilt with the authors' hyperparameters on 10 × 5-fold CV (AutoGluon 1 × 5), keep the published ranking but score higher than published (data release has 71 fewer frames than the paper; protocol inferred). Reproduced AutoML MAE equals the paper's 0.182.
- **This project's additions** (beam-theory tube features, log targets, guardrails, calibrated UQ) on identical folds:

| Model | Origin | R² (mean ± std) | MAE (std.) | Paper Table 4 R² | Folds |
|---|---|---|---|---|---|
| AutoGluon + physics | this project | 0.747 ± 0.052 | 0.157 |  | 5 |
| Deep ensemble + physics | this project | 0.734 ± 0.048 | 0.189 |  | 50 |
| Physics-guided MLP | this project | 0.717 ± 0.048 | 0.203 |  | 50 |
| XGBoost + physics | this project | 0.705 ± 0.055 | 0.198 |  | 50 |
| Gaussian process + physics | this project | 0.651 ± 0.084 | 0.236 |  | 50 |
| AutoML (AutoGluon) | paper, reproduced | 0.737 ± 0.044 | 0.182 | 0.605 | 5 |
| XGBoost | paper, reproduced | 0.691 ± 0.049 | 0.227 | 0.595 | 50 |
| Neural network | paper, reproduced | 0.592 ± 0.058 | 0.309 | 0.545 | 50 |
| Decision tree | paper, reproduced | 0.514 ± 0.046 | 0.338 | 0.440 | 50 |
| K-nearest neighbours | paper, reproduced | 0.378 ± 0.030 | 0.447 | 0.306 | 50 |

  Physics features lift AutoGluon by +0.010 R² (not significant over 5 folds, p = 0.15) and cut its MAE by -13 %; gains over the other four paper models are significant (corrected resampled t-test, p < 0.01). Deep-ensemble 95 % intervals cover 94.6–98.4 % of held-out frames (per output, 5 folds).
- **Optimisation** of the paper's titanium frame 3853 (surrogate only, not FE re-verified): -23.8 % mass and -30.5 % in-plane deflection at mean SF ≥ 1; requiring SF ≥ 1 on the 95 % lower bound instead needs +20.4 % mass.
- **Limits:** bicycle, not motorcycle; SF labels are mesh-sensitive (paper Sec. 3.5); no FE re-check of optimised frames.
<!-- FRAMED:END -->

## What changed in V1

| Change | Why |
|---|---|
| Composite model-selection score (worst + typical stress error, all-output error, extrapolation, CV R², confidence interval, speed) | v0 ranked on worst-case stress error alone, which ignored average and mass accuracy |
| Longer neural-network training (up to 800 epochs, patience 120, at least 150 epochs) | v0's early stopping cut the physics-guided MLP off at 57 epochs. Result: physics-guided MLP worst stress error 27.5 % -> 20.3 %; the residual MLP did not improve (9.3 % -> 14.4 %), so the limit is the data and noise, not training time |
| Manufacturability constraints: tube D/t <= 20, bend radius >= 1.5 x OD | v0's optimum (D/t 23 at a 38 mm OD) was not realistically bendable |
| Optimiser benchmark repeated over 3 seeds, reported as mean +/- std | NSGA-II, DE and ParEGO are random; one run can be lucky |
| Recommended design = knee point of the FE-verified front, starred on the Pareto plot | Gives one defensible, best-balanced design instead of only the lightest |
| Surrogate scorecard figure and decision write-up | Shows why the selected model wins |

## Engineering notes & honest limitations

- **Speedup baseline.** The in-repo beam FE solver takes only a few milliseconds per design, so single-design
  speedups are modest; batched surrogate inference (how optimisers use it) is orders of magnitude faster.
  The report also projects speedup against a nominal 180 s production FE run.
- **Noise floor.** Labels carry 3 % stress scatter, so worst-case errors near ~9 % sit at the data's
  resolution; accuracy is therefore also reported against noise-free FE.
- **GP studies.** Cross-validation, learning-curve and extrapolation studies reuse the GP kernel fitted on the
  full training set to bound compute; this slightly favours the GP in those studies.
- **Scenario vs. geometry.** The optimiser varies geometry only; yield strength and load factor are fixed per
  scenario (material grade, load case) and can be changed in the app.
- **Manufacturability limits are rules of thumb.** D/t <= 20 and bend radius >= 1.5 x OD are typical for
  rotary-draw bending without a mandrel; adjust them in `src/config.py` or the app sidebar to a supplier's capability.
- **Run-to-run variation.** A full rerun on Linux (Python 3.13, versions in `requirements.txt`) passed all 26 tests and
  reproduced the Gaussian-process selection, NSGA-II selection, GP accuracy (1.4 % stress MAPE), GP calibration, Sobol
  indices and the 14.6 % mass saving. Neural-network rows and the exact headline geometry vary slightly: the same run gave
  -17.2 % peak stress for the lightest design instead of -19.3 %. The numbers above are from the original V1 run.
- **Not modelled yet.** Weld-detail S-N classes, PSD-based random-vibration damage, 3-D frame torsion, uncertainty for
  the neural networks (e.g. deep ensembles).

## Prompt-refiner (development tooling)

`claude_setup/` contains a Claude Code subagent (`prompt-refiner`) and a `UserPromptSubmit` hook that log
every prompt given in this project and restructure it into a sectioned brief without losing meaning
(`prompts/PROMPT_HISTORY.md`, local and git-ignored). Install into `.claude/` with
`powershell -ExecutionPolicy Bypass -File .\claude_setup\install.ps1`.

## License

MIT, see [LICENSE](LICENSE).
