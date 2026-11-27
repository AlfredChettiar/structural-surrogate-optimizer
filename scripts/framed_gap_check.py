"""Why do the reproduced Table-4 scores differ from the paper? Tests two candidate explanations.

1. Data version: the paper used 4,118 frames, the release has 4,047. The dataset is rebuilt from the raw SolidWorks batch
   files with the authors' own preprocessing (scripts/framed_authors, Prepare Datasets.ipynb cell 1), keeping the frames
   that the final validity check drops.
2. Unshuffled folds: KFold(shuffle=False) on frames ordered by ID.

Usage (needs a clone of https://github.com/Lyleregenwetter/FRAMED):
    git clone --depth 1 https://github.com/Lyleregenwetter/FRAMED ../FRAMED
    python scripts/framed_gap_check.py ../FRAMED
Writes models/framed/gap_check.json.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "framed_authors")]
import getInvalid  # noqa: E402
import PostSW  # noqa: E402

from src.framed import config as fc  # noqa: E402
from src.framed import paper_reference as ref  # noqa: E402
from src.framed.baselines import cross_validate, paper_models  # noqa: E402
from src.framed.data import fold_indices, load_framed  # noqa: E402

ORDER = ["Sim 1 Dropout X Disp.", "Sim 1 Dropout Y Disp.", "Sim 1 Bottom Bracket X Disp.", "Sim 1 Bottom Bracket Y Disp.",
         "Sim 2 Bottom Bracket Z Disp.", "Sim 3 Bottom Bracket Y Disp.", "Sim 3 Bottom Bracket X Rot.",
         "Sim 1 Safety Factor", "Sim 3 Safety Factor", "Model Mass"]


def rebuild(repo: Path) -> tuple[pd.DataFrame, list]:
    """Authors' pipeline; returns (dataset before the final validity drop, indices that drop removes)."""
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        iv = 200
        batch = lambda pat, n, idx, tag: PostSW.extract_data_batch(
            [pat % (i * iv, i * iv + 199) for i in range(n)], idx, [f"{tag}_{x + 1}" for x in range(n)], iv)
        info = "Original frame simulation results/Batch Info/"
        data = batch("Original frame simulation results/Batch_1/results_new_%d_%d.csv", 22,
                     pd.read_csv("Processed_structural_data_backup.csv", index_col=0).index, "1")
        for b in (batch("Original frame simulation results/Batch_2/results_resim_%d_%d.csv", 9,
                        np.load(info + "batch2_backup.npy"), "2"),
                  batch("Original frame simulation results/Batch_3/results_mat_%d_%d.csv", 6,
                        np.load(info + "batch3.npy"), "3")):  # batches 2 and 3 correct some of batch 1
            inter = list(set(b.index) & set(data.index))
            data = pd.concat([data.drop(inter), b.loc[inter]])
        data = data.sort_index()
        # calcQOA, vectorised: its chained .iloc assignment is a no-op under pandas >= 3 copy-on-write
        params, tag, d = data.iloc[:, :-13], data.iloc[:, -1], data.iloc[:, -13:-1].astype(float)
        d.columns = ORDER[:5] + ["R", "L", "Sim 1 Safety Factor", "s1", "Sim 3 Safety Factor", "s3", "Model Mass"]
        d["Sim 3 Bottom Bracket Y Disp."] = (d["L"] + d["R"]) / 2
        d["Sim 3 Bottom Bracket X Rot."] = (d["L"] - d["R"]) / .14
        data = PostSW.renameMats(pd.concat([params, d[ORDER], tag], axis=1))
        valid = sorted(set(data.index) - set(getInvalid.getInvalid(pd.read_csv("biked_predrop.csv", index_col=0))[0]))
        data = PostSW.setBridges(data.loc[valid])
        return PostSW.oneHot(data), getInvalid.getInvalid(data)[0]
    finally:
        os.chdir(cwd)


def main(repo: Path) -> dict:
    pre, dropped = rebuild(repo)
    released = pd.read_csv(fc.FRAMED_DIR / fc.FILENAMES["data"], index_col=0)
    rebuilt = pre.drop(dropped)
    num = [c for c in released.columns if c != "batch"]
    max_diff = float(np.nanmax(np.abs(released.loc[rebuilt.index, num].astype(float).values
                                      - rebuilt[num].astype(float).values)))
    assert set(rebuilt.index) == set(released.index) and max_diff < 1e-9, "rebuild does not match the release"

    with tempfile.TemporaryDirectory() as d:
        pre.to_csv(Path(d) / "pre.csv")
        Xp, Yp = load_framed(Path(d) / "pre.csv")
    Xr, Yr = load_framed()
    drop_rows = Yp.loc[dropped]
    out = {"pre_drop_rows": len(pre), "dropped_rows": len(dropped), "released_rows": len(released),
           "rebuild_max_abs_diff": max_diff,
           "dropped_wall_thicker_than_od": int(sum((pre.loc[dropped, f"{t} Thickness"] > pre.loc[dropped, f"{t} OD"])
                                                   .astype(int) for t in fc.TUBES).gt(0).sum()),
           "dropped_outputs_max": drop_rows.max().to_dict(), "released_outputs_max": Yr.max().to_dict(), "variants": {}}
    unshuffled = lambda n: list(KFold(5, shuffle=False).split(np.arange(n)))
    variants = {"released, shuffled": (Xr, Yr, fold_indices(len(Xr), 5, 2)),
                "pre-drop, shuffled": (Xp, Yp, fold_indices(len(Xp), 5, 2)),
                "released, not shuffled": (Xr, Yr, unshuffled(len(Xr))),
                "pre-drop, not shuffled": (Xp, Yp, unshuffled(len(Xp)))}
    models = paper_models()
    for name, (X, Y, folds) in variants.items():
        out["variants"][name] = {}
        for k in ("decision_tree", "knn", "xgboost"):
            r = cross_validate(models[k], X, Y.values, folds)
            out["variants"][name][k] = {m: r[m]["mean"] for m in ("r2", "mae", "mse")}
        print(name, {k: round(v["r2"], 3) for k, v in out["variants"][name].items()}, flush=True)
    out["paper"] = {k: ref.TABLE4[k] for k in ("decision_tree", "knn", "xgboost")}
    (fc.MODELS_DIR / "gap_check.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]))
