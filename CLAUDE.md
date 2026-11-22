# Project Rules & Engineering Standards

## Environment & Tech Stack
- Virtual environment: `.venv` (activate via `.\.venv\Scripts\Activate.ps1`; run Python via `.\.venv\Scripts\python.exe`)
- Libraries: NumPy, Pandas, SciPy, Scikit-Learn, PyTorch, Streamlit, Plotly, Pytest

## Architecture & Guardrails
- Keep physics/synthetic dataset generation, surrogate training, Pareto optimization, and Streamlit UI modular under `src/`.
- Enforce physical boundary guardrails on all predictions (non-negative mass/stress, safety factor bounds).
- Never read raw files inside `data/` or `models/` directly; inspect shapes/metrics via short Python commands.
- Report R2, MAE, Max Relative Error (%), and inference speedup vs. analytical baseline.

## Verification
- Run `.\.venv\Scripts\python.exe -m pytest` before finishing any module to confirm zero runtime errors.
