"""Every number taken from the FRAMED paper [1], with its location. All comparisons read from here.

[1] L. Regenwetter, C. Weaver, F. Ahmed, "FRAMED: An AutoML Approach for Structural Performance
    Prediction of Bicycle Frames", Computer-Aided Design (Elsevier), 2023. arXiv:2201.10459.
"""
CITATION = ("L. Regenwetter, C. Weaver, F. Ahmed, \"FRAMED: An AutoML Approach for Structural Performance "
            "Prediction of Bicycle Frames\", Computer-Aided Design (Elsevier), 2023. arXiv:2201.10459.")

# Table 2: material properties (SI)
MATERIALS = {
    "Steel":    {"E": 205e9, "nu": 0.285, "G": 80e9, "rho": 7850.0, "uts": 731e6, "sigma_y": 460e6},
    "Aluminum": {"E": 69e9,  "nu": 0.330, "G": 26e9, "rho": 2700.0, "uts": 310e6, "sigma_y": 275e6},
    "Titanium": {"E": 105e9, "nu": 0.310, "G": 41e9, "rho": 4429.0, "uts": 1050e6, "sigma_y": 827e6},
}

# Table 3: physical validation vs Rinard (1996). (front defl. mm, rear defl. mm, mass kg)
TABLE3 = {
    "DeRosa SLX":       {"actual": (0.40, 0.15, 1.966), "simulated": (0.297, 0.116, 1.69)},
    "Casati Gold Line": {"actual": (0.44, 0.15, 1.966), "simulated": (0.3028, 0.124, 1.80)},
    "Holland SL/SP":    {"actual": (0.38, 0.13, 1.962), "simulated": (0.26, 0.107, 1.77)},
}
TABLE3_REPORTED_ERR_PCT = {"DeRosa SLX": (26, 23, 14), "Casati Gold Line": (31, 17, 8),
                           "Holland SL/SP": (32, 18, 10)}

# Table 4: regression, standardised targets
TABLE4 = {
    "decision_tree": {"r2": 0.440, "mse": 1.156, "mae": 0.297},
    "knn":           {"r2": 0.306, "mse": 1.270, "mae": 0.412},
    "xgboost":       {"r2": 0.595, "mse": 0.978, "mae": 0.208},
    "neural_net":    {"r2": 0.545, "mse": 1.031, "mae": 0.263},
    "autogluon":     {"r2": 0.605, "mse": 0.995, "mae": 0.182},
}
TABLE4_RANKING = ["autogluon", "xgboost", "neural_net", "decision_tree", "knn"]
R2_TOLERANCE = 0.05

# Tables 6-9: best hyperparameters found by the authors' Bayesian optimisation
HYPERPARAMS = {
    "decision_tree": {"max_depth": 7, "splitter": "best", "criterion": "absolute_error",
                      "min_samples_split": 5, "min_samples_leaf": 11, "max_features": 32},
    "knn": {"n_neighbors": 70, "weights": "distance", "metric": "euclidean"},
    "xgboost": {"max_depth": 6, "reg_lambda": 72.41212, "learning_rate": 0.275542, "gamma": 0.059818,
                "min_child_weight": 4.819113, "early_stopping_rounds": 8},
    "neural_net": {"hidden": [92], "activation": "relu", "batch_norm": False, "dropout": 0.578647,
                   "lr": 7.20e-5, "batch_size": 135, "patience": 44},
}

# Sec. 5.1 (paper text) and the authors' Analysis.ipynb outputs on the released data
PAPER_N_VALID_GEOMETRY = 4118
PAPER_N_FAIL_ANY = 3419
NOTEBOOK_FEASIBILITY = {"both_fail": 1966, "sf1_only": 1272, "sf3_only": 182, "valid": 626}
NOTEBOOK_N_NON_DOMINATED = 76
COUNT_TOLERANCE = 2

# Sec. 5.2 case studies: index -> (material, mass kg, SF1, SF3); paper text rounds these
CASE_STUDIES = {
    3853: {"label": "Frame 1 (random valid road bike)", "material": "Titanium", "mass": 2.84, "sf1": 1.29, "sf3": 1.073},
    2108: {"label": "Frame 2 (heaviest, artistic)", "material": "Steel", "mass": 22.83, "sf1": 0.09256, "sf3": 0.2098},
    857:  {"label": "Frame 3 (lightest valid)", "material": "Titanium", "mass": 1.60, "sf1": 2.416, "sf3": 2.986},
}
CASE_STUDIES_PAPER_TEXT = {3853: "2.84 kg, SF 1.3 / 1.1", 2108: "22.83 kg, SF < 0.1", 857: "1.6 kg, SF 2.42 / 2.99"}

# Sec. 6.3 / Table 5 (classification, reported for context only; not reproduced: validity labels are not in the
# regression file)
TABLE5_F1 = {"decision_tree": 0.750, "knn": 0.736, "xgboost": 0.722, "neural_net": 0.736, "svm": 0.620,
             "autogluon": 0.915}


PAPER_FIGURE_PAGES = {"fig4_materials": 13, "fig5_convergence": 14, "fig6_pairs": 17, "fig8_parity": 20,
                      "fig9_parity_all": 26, "table3": 15, "table4": 21}

# Fig. 9 axis maxima as printed (units: cm for displacements, deg for rotation, kg, N/A for SF)
FIG9_UNITS = {"Sim 1 Dropout X Disp.": "cm", "Sim 1 Dropout Y Disp.": "cm", "Sim 1 Bottom Bracket X Disp.": "cm",
              "Sim 1 Bottom Bracket Y Disp.": "cm", "Sim 2 Bottom Bracket Z Disp.": "cm",
              "Sim 3 Bottom Bracket Y Disp.": "cm", "Sim 3 Bottom Bracket X Rot.": "deg",
              "Sim 1 Safety Factor": "N/A", "Sim 3 Safety Factor": "N/A", "Model Mass": "kg"}
FIG9_AXIS_MAX = {"Sim 1 Dropout X Disp.": 9.5, "Sim 1 Dropout Y Disp.": 15.0, "Sim 1 Bottom Bracket X Disp.": 11.0,
                 "Sim 1 Bottom Bracket Y Disp.": 10.0, "Sim 2 Bottom Bracket Z Disp.": 0.5,
                 "Sim 3 Bottom Bracket Y Disp.": 2.8, "Sim 3 Bottom Bracket X Rot.": 0.75,
                 "Sim 1 Safety Factor": 11.0, "Sim 3 Safety Factor": 4.2, "Model Mass": 11.0}
