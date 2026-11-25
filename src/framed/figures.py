"""Plotly figures for the FRAMED benchmark, styled like the V1 figures (src.plots)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.framed import config as fc
from src.framed import paper_reference as ref
from src.framed.paper_stats import PAIR_OBJECTIVES
from src.plots import NEUTRAL, SERIES, STATUS, _layout

SHORT = {"Sim 1 Dropout X Disp.": "S1 dropout X", "Sim 1 Dropout Y Disp.": "S1 dropout Y",
         "Sim 1 Bottom Bracket X Disp.": "S1 BB X", "Sim 1 Bottom Bracket Y Disp.": "S1 BB Y",
         "Sim 2 Bottom Bracket Z Disp.": "S2 BB Z", "Sim 3 Bottom Bracket Y Disp.": "S3 BB Y",
         "Sim 3 Bottom Bracket X Rot.": "S3 BB rot", "Sim 1 Safety Factor": "S1 SF",
         "Sim 3 Safety Factor": "S3 SF", "Model Mass": "Mass"}
MODEL_LABELS = {"decision_tree": "Decision tree", "knn": "K-nearest neighbours", "xgboost": "XGBoost",
                "neural_net": "Neural network", "autogluon": "AutoML (AutoGluon)",
                "xgboost_phys": "XGBoost + physics", "autogluon_phys": "AutoGluon + physics",
                "gp_phys": "Gaussian process + physics", "deep_ensemble_phys": "Deep ensemble + physics",
                "pg_mlp": "Physics-guided MLP"}


def to_paper_units(col: str, v: np.ndarray) -> np.ndarray:
    """Fig. 9 units: displacements cm, rotation deg."""
    u = ref.FIG9_UNITS[col]
    return v * 100 if u == "cm" else np.degrees(v) if u == "deg" else v


def validity(Y: pd.DataFrame) -> np.ndarray:
    return np.where((Y["Sim 1 Safety Factor"] >= 1) & (Y["Sim 3 Safety Factor"] >= 1), "Valid", "Invalid")


def fig_material_breakdown(X: pd.DataFrame) -> go.Figure:
    counts = X[fc.MATERIAL_COLS].sum().values
    fig = go.Figure(go.Bar(x=fc.MATERIALS, y=counts, marker_color=SERIES[:3], text=counts.astype(int),
                           textposition="outside"))
    return _layout(fig, "Frames by material in the released data", height=340, yaxis_title="Frames")


def fig_performance_pairs(Y: pd.DataFrame) -> go.Figure:
    """Paper Fig. 6: pair plot of five outputs, valid vs invalid, case studies highlighted."""
    v = validity(Y)
    dims = lambda m: [dict(label=SHORT[c], values=to_paper_units(c, Y.loc[m, c].values)) for c in PAIR_OBJECTIVES]
    fig = go.Figure()
    for name, color in (("Invalid", SERIES[1]), ("Valid", SERIES[0])):
        fig.add_trace(go.Splom(dimensions=dims(v == name), name=name, marker=dict(size=3, color=color, opacity=0.5),
                               diagonal_visible=False, showupperhalf=False))
    cs = Y.index.isin(list(ref.CASE_STUDIES))
    fig.add_trace(go.Splom(dimensions=dims(cs), name="Case-study frames", text=[str(i) for i in Y.index[cs]],
                           marker=dict(size=11, symbol="x", color=SERIES[3]), diagonal_visible=False,
                           showupperhalf=False))
    return _layout(fig, "Performance space (cf. paper Fig. 6)", height=760)


def fig_objective_correlation(Y: pd.DataFrame) -> go.Figure:
    c = Y.corr(method="spearman")
    labels = [SHORT[k] for k in c.columns]
    fig = go.Figure(go.Heatmap(z=c.values, x=labels, y=labels, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
                               text=np.round(c.values, 2), texttemplate="%{text}", hovertemplate="%{x} / %{y}: %{z:.2f}"))
    return _layout(fig, "Spearman correlation between the ten outputs", height=560)


def fig_physical_validation() -> go.Figure:
    names = list(ref.TABLE3)
    qty = ["Front deflection (mm)", "Rear deflection (mm)", "Mass (kg)"]
    fig = make_subplots(1, 3, subplot_titles=qty)
    for j in range(3):
        for k, (key, color) in enumerate((("actual", NEUTRAL), ("simulated", SERIES[0]))):
            fig.add_trace(go.Bar(x=names, y=[ref.TABLE3[n][key][j] for n in names], name=key.title(),
                                 marker_color=color, showlegend=j == 0), 1, j + 1)
    return _layout(fig, "Paper Table 3: SolidWorks vs physical test (Rinard 1996)", height=360, barmode="group")


def fig_parity_grid(Y: pd.DataFrame, pred: np.ndarray, title: str, train_pred: np.ndarray | None = None) -> go.Figure:
    """Paper Fig. 9 layout: 10 panels, paper units and axis limits."""
    cols = fc.Y_COLS
    fig = make_subplots(3, 4, subplot_titles=[f"{SHORT[c]} ({ref.FIG9_UNITS[c]})" for c in cols],
                        horizontal_spacing=0.06, vertical_spacing=0.09)
    for j, c in enumerate(cols):
        r, k = divmod(j, 4)
        lim = ref.FIG9_AXIS_MAX[c]
        yt = to_paper_units(c, Y[c].values)
        sets = [("Validation (out-of-fold)", pred, SERIES[4])]
        if train_pred is not None:
            sets.insert(0, ("Train", train_pred, SERIES[3]))
        for name, P, color in sets:
            fig.add_trace(go.Scattergl(x=yt, y=to_paper_units(c, P[:, j]), mode="markers", name=name,
                                       marker=dict(size=3, color=color, opacity=0.5), showlegend=j == 0),
                          r + 1, k + 1)
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color=NEUTRAL, width=1),
                                 showlegend=False, hoverinfo="skip"), r + 1, k + 1)
        fig.update_xaxes(range=[0, lim], row=r + 1, col=k + 1)
        fig.update_yaxes(range=[0, lim], row=r + 1, col=k + 1)
    return _layout(fig, title, height=820)


def fig_table4_comparison(rows: pd.DataFrame) -> go.Figure:
    """rows: index = model key, columns paper_r2, ours_r2, paper_mae, ours_mae (NaN where not available)."""
    labels = [MODEL_LABELS.get(m, m) for m in rows.index]
    fig = make_subplots(1, 2, subplot_titles=["R² (higher is better)", "MAE, standardised (lower is better)"])
    for k, metric in enumerate(("r2", "mae")):
        fig.add_trace(go.Bar(x=labels, y=rows[f"paper_{metric}"], name="Paper (Table 4)", marker_color=NEUTRAL,
                             showlegend=k == 0), 1, k + 1)
        fig.add_trace(go.Bar(x=labels, y=rows[f"ours_{metric}"], name="This study", marker_color=SERIES[0],
                             error_y=dict(type="data", array=rows.get(f"ours_{metric}_std")), showlegend=k == 0),
                      1, k + 1)
    return _layout(fig, "Table 4: paper vs this study", height=420, barmode="group")


def fig_check_status(df: pd.DataFrame, title: str) -> go.Figure:
    """Horizontal % deviation bars, coloured by pass/fail, for a stats/reproduction comparison table."""
    colors = [NEUTRAL if p is None else STATUS["good"] if p else STATUS["critical"] for p in df["pass"]]
    fig = go.Figure(go.Bar(y=df["quantity"], x=df["pct_dev"], orientation="h", marker_color=colors,
                           text=df["pct_dev"].round(2).astype(str) + " %", textposition="outside"))
    return _layout(fig, title, height=40 * len(df) + 120, xaxis_title="Deviation from paper (%)")
