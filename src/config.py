"""Central configuration: paths, design-space bounds, targets, constraints and seeds."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATASET_CSV = DATA_DIR / "chassis_doe_dataset.csv"
DATASET_SUMMARY_JSON = MODELS_DIR / "dataset_summary.json"
METRICS_JSON = MODELS_DIR / "metrics.json"
MODEL_SELECTION_JSON = MODELS_DIR / "model_selection.json"
EVALUATION_NPZ = MODELS_DIR / "evaluation_test_predictions.npz"
EVALUATION_JSON = MODELS_DIR / "evaluation_extras.json"
OPTIMIZATION_JSON = MODELS_DIR / "optimizer_benchmark.json"
PARETO_CSV = DATA_DIR / "pareto_fronts.csv"
CANDIDATES_CSV = DATA_DIR / "lhs_candidates.csv"

SEED = 42

# ----------------------------------------------------------------------------- design space
FEATURES: list[str] = [
    "tube_od_mm",
    "wall_thickness_mm",
    "gusset_thickness_mm",
    "bend_radius_mm",
    "yield_strength_mpa",
    "dynamic_load_factor",
]
BOUNDS: dict[str, tuple[float, float]] = {
    "tube_od_mm": (22.0, 38.0),
    "wall_thickness_mm": (1.6, 3.5),
    "gusset_thickness_mm": (2.0, 5.0),
    "bend_radius_mm": (40.0, 90.0),
    "yield_strength_mpa": (250.0, 480.0),
    "dynamic_load_factor": (1.5, 4.0),
}
FEATURE_LABELS: dict[str, str] = {
    "tube_od_mm": "Tube OD [mm]",
    "wall_thickness_mm": "Wall thickness [mm]",
    "gusset_thickness_mm": "Gusset thickness [mm]",
    "bend_radius_mm": "Bend radius [mm]",
    "yield_strength_mpa": "Yield strength [MPa]",
    "dynamic_load_factor": "Dynamic load factor [-]",
}

# Geometry variables the optimizer is allowed to change. Material yield strength and the
# dynamic load factor are *scenario* parameters (material grade + load case), not geometry.
GEOMETRY_FEATURES: list[str] = FEATURES[:4]
DEFAULT_SCENARIO: dict[str, float] = {"yield_strength_mpa": 355.0, "dynamic_load_factor": 3.0}

# ----------------------------------------------------------------------------- targets
TARGETS: list[str] = [
    "mass_kg",
    "peak_von_mises_mpa",
    "max_deflection_mm",
    "first_mode_hz",
    "log10_fatigue_life_cycles",
]
TARGET_LABELS: dict[str, str] = {
    "mass_kg": "Mass [kg]",
    "peak_von_mises_mpa": "Peak von Mises [MPa]",
    "max_deflection_mm": "Max deflection [mm]",
    "first_mode_hz": "First mode [Hz]",
    "log10_fatigue_life_cycles": "log10 Fatigue life [cycles]",
}
# Positive targets are modelled in log space so back-transformed predictions stay > 0.
LOG_TARGETS: list[str] = ["mass_kg", "peak_von_mises_mpa", "max_deflection_mm", "first_mode_hz"]
FATIGUE_LOG10_CAP = 9.0  # run-out: lives beyond 1e9 cycles are reported as 1e9

# ----------------------------------------------------------------------------- constraints
CONSTRAINTS: dict[str, float] = {
    "min_safety_factor": 1.5,
    "min_first_mode_hz": 45.0,
    "max_deflection_mm": 2.5,
    "min_log10_fatigue_life": 6.0,  # >= 1e6 cycles at the design load spectrum
    # manufacturability (exact geometric checks, no surrogate needed). Rule-of-thumb limits for
    # rotary-draw bending of steel tube without a mandrel; adjust to the supplier's capability.
    "max_d_over_t": 20.0,           # tube OD / wall thickness ("wall factor")
    "min_bend_ratio": 1.5,          # bend centre-line radius / tube OD
}

# ----------------------------------------------------------------------------- guardrail bounds
GUARDRAILS: dict[str, tuple[float, float]] = {
    "mass_kg": (0.05, 50.0),
    "peak_von_mises_mpa": (0.1, 5000.0),
    "max_deflection_mm": (0.0, 200.0),
    "first_mode_hz": (0.1, 2000.0),
    "log10_fatigue_life_cycles": (0.0, FATIGUE_LOG10_CAP),
    "safety_factor": (0.0, 20.0),
}

# ----------------------------------------------------------------------------- DOE
N_SAMPLES = 1500
TEST_FRACTION = 0.2
N_OPT_CANDIDATES = 5000
OPT_SEEDS = 3                 # optimiser benchmark repeats (stochastic algorithms)

# Representative wall-clock of one production FE run (linear static + modal, OptiStruct),
# used only to *project* industrial speedup. The measured speedup is vs. the in-repo beam FE.
NOMINAL_INDUSTRIAL_FE_SECONDS = 180.0
