# Project 1 Spec: Physics-Guided Structural Surrogate & Pareto Weight Optimizer

## 1. Synthetic Physics Dataset (`src/data_generator.py`)
- Generate 1,500 design samples (Latin Hypercube + Central Composite Design) for a parametric tubular two-wheeler chassis sub-frame assembly.
- Inputs (6 features): `tube_od_mm` (22–38), `wall_thickness_mm` (1.6–3.5), `gusset_thickness_mm` (2.0–5.0), `bend_radius_mm` (40–90), `yield_strength_mpa` (250–480), `dynamic_load_factor` (1.5–4.0).
- Outputs (4 targets via nonlinear structural mechanics + realistic measurement noise): `mass_kg`, `peak_von_mises_mpa`, `max_deflection_mm`, `first_mode_hz`.
- Save dataset to `data/chassis_doe_dataset.csv` and dataset summary stats to `models/dataset_summary.json`.

## 2. Surrogate Models & Uncertainty Quantification (`src/train.py`)
- Train and compare 3 models on an 80/20 split:
  1. Ridge Polynomial Regression (Baseline)
  2. Gaussian Process Regressor (provides predictive mean + 95% confidence interval)
  3. PyTorch Residual MLP (`src/model.py`)
- Save trained artifacts to `models/` and benchmark metrics (R2, MAE, Max % Error, Inference Latency in ms) to `models/metrics.json`.

## 3. Multi-Objective Pareto Optimizer (`src/optimizer.py`)
- Evaluate 5,000+ candidate geometries using the trained surrogate under engineering constraints:
  - `Safety Factor (yield_strength / peak_von_mises) >= 1.5`
  - `First Natural Frequency >= 45.0 Hz`
  - `Max Deflection <= 2.5 mm`
- Extract the non-dominated Pareto frontier minimizing `mass_kg` and `peak_von_mises_mpa`.

## 4. Interactive Streamlit Dashboard (`app.py`)
- **Tab 1 — Real-Time Surrogate Explorer:** Geometry & load sliders with instant stress, mass, safety factor, 95% confidence interval, and inference speedup KPIs.
- **Tab 2 — Pareto Weight Optimization:** Interactive Plotly Pareto frontier (Mass vs. Stress colored by Natural Frequency) with constraint filters and downloadable optimal dimensions table.
- **Tab 3 — Model Validation & Sensitivity:** Parity plots (Actual vs. Predicted), residual error distributions, and feature sensitivity bar chart.
