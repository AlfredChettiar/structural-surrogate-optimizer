"""FRAMED benchmark configuration: pinned sources, paths and column lists."""
from __future__ import annotations

import os

from src import config as v1

FRAMED_DIR = v1.DATA_DIR / "external" / "framed"
MODELS_DIR = v1.MODELS_DIR / "framed"
REPORT_DIR = v1.REPORTS_DIR / "framed"
PAPER_PAGES_DIR = REPORT_DIR / "paper_pages"  # local only (git-ignored): paper figures are not ours to republish

COMMIT = "b881634bb3b79240f08fc4285483becf6ec7d7c5"
DATA_URL = f"https://raw.githubusercontent.com/Lyleregenwetter/FRAMED/{COMMIT}/"
PDF_URL = ("https://raw.githubusercontent.com/Lyleregenwetter/Lyleregenwetter.github.io/"
           "master/papers/2022_regenwetter_framed.pdf")

# name -> (url, sha256)
FILES = {
    "data": (DATA_URL + "all_structural_data_oh.csv",
             "8506eb3ec69f85fd35c551c630e5eba68c5ce5b2c78b3a42a33ab681f2697b5e"),
    "paper": (PDF_URL, "ec70c86b8e4300983b3e08fadcb2d521a36577c84da93e69bb680059d19908cc"),
}
FILENAMES = {"data": "all_structural_data_oh.csv", "paper": "regenwetter_2023_framed.pdf"}

MATERIALS = ["Steel", "Aluminum", "Titanium"]
MATERIAL_COLS = [f"Material={m}" for m in MATERIALS]
DISP_COLS = ["Sim 1 Dropout X Disp.", "Sim 1 Dropout Y Disp.", "Sim 1 Bottom Bracket X Disp.",
             "Sim 1 Bottom Bracket Y Disp.", "Sim 2 Bottom Bracket Z Disp.", "Sim 3 Bottom Bracket Y Disp.",
             "Sim 3 Bottom Bracket X Rot."]
SF_COLS = ["Sim 1 Safety Factor", "Sim 3 Safety Factor"]
MASS_COL = "Model Mass"
Y_COLS = DISP_COLS + SF_COLS + [MASS_COL]
ABS_COLS = DISP_COLS + [MASS_COL]
# Output units in the released CSV: metres, radians, kg (SolidWorks SI; the paper's Fig. 9 shows the same values in
# cm and deg, e.g. dropout Y 99th pct 0.18 m vs a 15 cm axis; Analysis.ipynb labels dropout Y "(m)")
UNITS = {**{c: "m" for c in DISP_COLS}, "Sim 3 Bottom Bracket X Rot.": "rad", **{c: "-" for c in SF_COLS},
         MASS_COL: "kg"}  # as in the authors' Analysis.ipynb
N_X = 39

TUBES = ["BB", "TT", "HT", "DT", "CS", "SS", "ST"]
SEED = 42
N_JOBS = max(1, (os.cpu_count() or 2) - 1)
SF_CAP = 20.0
