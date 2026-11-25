"""Runs inside .venv-ag: fit one AutoGluon TabularPredictor per target, predict, save. Usage: in out time seed dir"""
import sys

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

inp, out, tl, seed, d = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
z = np.load(inp)
X, Y, Xp = z["X"], z["Y"], z["Xp"]
cols = [f"x{i}" for i in range(X.shape[1])]
P = np.zeros((len(Xp), Y.shape[1]))
for j in range(Y.shape[1]):
    tr = pd.DataFrame(X, columns=cols).assign(y=Y[:, j])
    pr = TabularPredictor(label="y", problem_type="regression", path=f"{d}/t{j}", verbosity=0).fit(
        tr, presets="medium_quality", time_limit=tl, ag_args_fit={"num_cpus": 1})
    P[:, j] = pr.predict(pd.DataFrame(Xp, columns=cols)).to_numpy()
np.save(out, P)
