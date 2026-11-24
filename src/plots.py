"""Plotly figure factory shared by the Streamlit app and the static portfolio report.

Colour rules (validated reference palette): every model / optimiser owns a fixed categorical slot
that never changes with filters; magnitude uses one blue ramp; feasibility uses status colours
together with text labels / symbols so colour never carries meaning alone. One y-axis per chart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from src import config

# ----------------------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#8a8984"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}

MODEL_ORDER = ["gaussian_process", "physics_guided_mlp", "residual_mlp", "ridge_poly",
               "hist_gradient_boosting", "random_forest", "svr"]
MODEL_COLORS = {m: SERIES[i] for i, m in enumerate(MODEL_ORDER)}
MODEL_COLORS["closed_form"] = NEUTRAL
ALGO_ORDER = ["nsga2", "parego", "epsilon_constraint", "weighted_sum_slsqp", "lhs_screening"]
ALGO_COLORS = {a: SERIES[i] for i, a in enumerate(ALGO_ORDER)}
ALGO_SYMBOLS = {"nsga2": "circle", "parego": "diamond", "epsilon_constraint": "square",
                "weighted_sum_slsqp": "triangle-up", "lhs_screening": "x"}

SHORT = {"mass_kg": "Mass", "peak_von_mises_mpa": "Stress", "max_deflection_mm": "Deflection",
         "first_mode_hz": "First mode", "log10_fatigue_life_cycles": "Fatigue life"}


def _layout(fig: go.Figure, title: str | None = None, height: int = 420, **kw) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=16, color=TEXT)) if title else None,
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=12, color=TEXT_2),
        margin=dict(l=60, r=24, t=64 if title else 24, b=56),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=TEXT_2)),
        hoverlabel=dict(bgcolor="white", font=dict(color=TEXT)),
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, ticks="outside", tickcolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, ticks="outside", tickcolor=AXIS)
    return fig


def model_label(name: str, metrics: dict | None = None) -> str:
    if metrics and name in metrics.get("models", {}):
        return metrics["models"][name]["label"]
    return name.replace("_", " ").title()


# ============================================================================ dataset
def fig_doe_coverage(df: pd.DataFrame, max_lhs: int = 600) -> go.Figure:
    lhs_df = df[df.doe_source == "LHS"].sample(min(max_lhs, (df.doe_source == "LHS").sum()), random_state=0)
    ccd_df = df[df.doe_source == "CCD"]
    dims = [dict(label=config.FEATURE_LABELS[f].split(" [")[0], values=None) for f in config.FEATURES]
    fig = go.Figure()
    for sub, name, color, size, sym in ((lhs_df, f"LHS (showing {len(lhs_df)})", SERIES[0], 3, "circle"),
                                        (ccd_df, f"CCD ({len(ccd_df)})", SERIES[1], 6, "diamond")):
        fig.add_trace(go.Splom(
            dimensions=[dict(d, values=sub[f]) for d, f in zip(dims, config.FEATURES)],
            name=name, marker=dict(color=color, size=size, symbol=sym, line=dict(width=0.5, color=SURFACE)),
            diagonal_visible=False, showupperhalf=False))
    return _layout(fig, "Design-of-experiments coverage (LHS + face-centred CCD)", height=720)


# ============================================================================ accuracy
def fig_parity(Y_true: np.ndarray, Y_pred: np.ndarray, model: str, band: float = 0.10) -> go.Figure:
    fig = make_subplots(rows=1, cols=5, subplot_titles=[config.TARGET_LABELS[t] for t in config.TARGETS],
                        horizontal_spacing=0.06)
    color = MODEL_COLORS.get(model, SERIES[0])
    for j, t in enumerate(config.TARGETS):
        yt, yp = Y_true[:, j], Y_pred[:, j]
        lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        line = np.array([lo, hi])
        fig.add_trace(go.Scatter(x=np.r_[line, line[::-1]], y=np.r_[line * (1 + band), (line * (1 - band))[::-1]],
                                 fill="toself", fillcolor="rgba(42,120,214,0.08)", line=dict(width=0),
                                 hoverinfo="skip", showlegend=j == 0, name=f"±{band:.0%} band"), 1, j + 1)
        fig.add_trace(go.Scatter(x=line, y=line, mode="lines", line=dict(color=AXIS, width=1.5, dash="dash"),
                                 hoverinfo="skip", showlegend=False), 1, j + 1)
        fig.add_trace(go.Scatter(x=yt, y=yp, mode="markers", name="Test design", showlegend=j == 0,
                                 marker=dict(color=color, size=6, opacity=0.75, line=dict(width=1, color=SURFACE)),
                                 hovertemplate="actual %{x:.3g}<br>predicted %{y:.3g}<extra></extra>"), 1, j + 1)
        fig.update_xaxes(title_text="Actual", row=1, col=j + 1)
    fig.update_yaxes(title_text="Predicted", row=1, col=1)
    return _layout(fig, f"Parity: {model.replace('_', ' ')} on held-out test set", height=380)


def relative_residuals(Y_true, Y_pred) -> np.ndarray:
    R = (Y_pred - Y_true) / np.clip(np.abs(Y_true), 1e-9, None) * 100
    R[:, 4] = (Y_pred[:, 4] - Y_true[:, 4]) * 100 / np.maximum(np.abs(Y_true[:, 4]), 1.0)
    return R


def fig_residual_hist(Y_true, Y_pred, model: str) -> go.Figure:
    R = relative_residuals(Y_true, Y_pred)
    fig = make_subplots(rows=1, cols=5, subplot_titles=[SHORT[t] for t in config.TARGETS], horizontal_spacing=0.05)
    for j in range(5):
        fig.add_trace(go.Histogram(x=R[:, j], nbinsx=30, marker=dict(color=MODEL_COLORS.get(model, SERIES[0]),
                                   line=dict(color=SURFACE, width=1)), showlegend=False,
                                   hovertemplate="%{x:.2f} %%<br>%{y} designs<extra></extra>"), 1, j + 1)
        fig.add_vline(x=0, line=dict(color=AXIS, dash="dash"), row=1, col=j + 1)
        fig.update_xaxes(title_text="Error [%]", row=1, col=j + 1)
    fig.update_yaxes(title_text="Designs", row=1, col=1)
    return _layout(fig, "Relative residual distribution (test set)", height=320)


def fig_qq(Y_true, Y_pred, model: str) -> go.Figure:
    R = relative_residuals(Y_true, Y_pred)
    fig = make_subplots(rows=1, cols=5, subplot_titles=[SHORT[t] for t in config.TARGETS], horizontal_spacing=0.05)
    for j in range(5):
        z = (R[:, j] - R[:, j].mean()) / (R[:, j].std() + 1e-12)
        (osm, osr), _ = stats.probplot(z, dist="norm")
        fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", showlegend=False,
                                 marker=dict(size=5, color=MODEL_COLORS.get(model, SERIES[0])),
                                 hovertemplate="theoretical %{x:.2f}<br>observed %{y:.2f}<extra></extra>"), 1, j + 1)
        lim = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=AXIS, dash="dash"),
                                 hoverinfo="skip", showlegend=False), 1, j + 1)
        fig.update_xaxes(title_text="Normal quantile", row=1, col=j + 1)
    fig.update_yaxes(title_text="Standardised residual", row=1, col=1)
    return _layout(fig, "Q-Q plot of residuals (normality check)", height=320)


def fig_residual_vs_pred(Y_true, Y_pred, model: str) -> go.Figure:
    R = relative_residuals(Y_true, Y_pred)
    fig = make_subplots(rows=1, cols=5, subplot_titles=[SHORT[t] for t in config.TARGETS], horizontal_spacing=0.05)
    for j in range(5):
        fig.add_trace(go.Scatter(x=Y_pred[:, j], y=R[:, j], mode="markers", showlegend=False,
                                 marker=dict(size=5, color=MODEL_COLORS.get(model, SERIES[0]), opacity=0.7),
                                 hovertemplate="pred %{x:.3g}<br>err %{y:.2f} %%<extra></extra>"), 1, j + 1)
        fig.add_hline(y=0, line=dict(color=AXIS, dash="dash"), row=1, col=j + 1)
        fig.update_xaxes(title_text="Predicted", row=1, col=j + 1)
    fig.update_yaxes(title_text="Error [%]", row=1, col=1)
    return _layout(fig, "Residuals vs. prediction (heteroscedasticity check)", height=320)


# ============================================================================ trust
def fig_learning_curves(extras: dict, metrics: dict | None = None, key: str = "stress_mape_pct") -> go.Figure:
    lc = extras["learning_curves"]
    fig = go.Figure()
    for m in MODEL_ORDER:
        if m not in lc["models"]:
            continue
        fig.add_trace(go.Scatter(x=lc["sizes"], y=[r[key] for r in lc["models"][m]], mode="lines+markers",
                                 name=model_label(m, metrics), line=dict(color=MODEL_COLORS[m], width=2),
                                 marker=dict(size=8, line=dict(width=2, color=SURFACE)),
                                 hovertemplate="%{x} samples<br>%{y:.2f}<extra>" + m + "</extra>"))
    fig.update_xaxes(type="log", title_text="Training samples")
    ytitle = {"stress_mape_pct": "Stress MAPE vs FE [%]", "mean_r2": "Mean R² vs FE"}[key]
    fig.update_yaxes(title_text=ytitle, type="log" if key == "stress_mape_pct" else "linear")
    return _layout(fig, "Learning curves: how much data each surrogate needs", height=420)


def fig_calibration(extras: dict) -> go.Figure:
    cal = extras.get("calibration", {}).get("gaussian_process")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.45, 1.0], y=[0.45, 1.0], mode="lines", name="Perfect calibration",
                             line=dict(color=AXIS, dash="dash", width=1.5)))
    if cal:
        for j, t in enumerate(config.TARGETS):
            fig.add_trace(go.Scatter(x=cal["levels"], y=cal["coverage"][t], mode="lines+markers", name=SHORT[t],
                                     line=dict(color=SERIES[j], width=2), marker=dict(size=8),
                                     hovertemplate="nominal %{x:.0%}<br>observed %{y:.1%}<extra>" + SHORT[t] + "</extra>"))
    fig.update_xaxes(title_text="Nominal interval coverage", tickformat=".0%")
    fig.update_yaxes(title_text="Observed coverage (test)", tickformat=".0%")
    return _layout(fig, "Gaussian-process uncertainty calibration", height=420)


def fig_extrapolation(extras: dict, metrics: dict | None = None) -> go.Figure:
    ex = extras["extrapolation"]
    models = [m for m in ["closed_form"] + MODEL_ORDER if m in ex["models"]]
    fig = go.Figure()
    for j, t in enumerate(["peak_von_mises_mpa", "max_deflection_mm", "first_mode_hz"]):
        fig.add_trace(go.Bar(x=[model_label(m, metrics) for m in models],
                             y=[ex["models"][m][t]["mape_pct"] for m in models], name=SHORT[t],
                             marker=dict(color=SERIES[j], line=dict(color=SURFACE, width=2)),
                             hovertemplate="%{x}<br>%{y:.2f} %%<extra>" + SHORT[t] + "</extra>"))
    fig.update_yaxes(title_text="MAPE in unseen corner [%]", type="log")
    fig.update_layout(barmode="group", bargap=0.25)
    return _layout(fig, f"Extrapolation: trained without {ex['region']} ({ex['n_corner_test']} designs)",
                   height=440)


# ============================================================================ insight
def fig_sobol(extras: dict, target: str) -> go.Figure:
    s = extras["sensitivity"]["sobol"]
    labels = [config.FEATURE_LABELS[f].split(" [")[0] for f in s["features"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=s["S1"][target], name="First-order S1", orientation="h",
                         marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=2))))
    fig.add_trace(go.Bar(y=labels, x=s["ST"][target], name="Total-order ST", orientation="h",
                         marker=dict(color=SERIES[1], line=dict(color=SURFACE, width=2))))
    fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"))
    fig.update_xaxes(title_text="Sobol index (share of output variance)", range=[0, 1])
    return _layout(fig, f"Global sensitivity (Sobol) - {config.TARGET_LABELS[target]}", height=380)


def fig_permutation(extras: dict, target: str) -> go.Figure:
    p = extras["sensitivity"]["permutation"]
    labels = [config.FEATURE_LABELS[f].split(" [")[0] for f in p["features"]]
    vals = np.array(p["importance"][target])
    order = np.argsort(vals)
    fig = go.Figure(go.Bar(y=np.array(labels)[order], x=vals[order], orientation="h",
                           marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=2)),
                           hovertemplate="%{y}<br>%{x:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Increase in MAE / target std when shuffled")
    return _layout(fig, f"Permutation importance - {config.TARGET_LABELS[target]}", height=360)


def fig_pdp(extras: dict, target: str) -> go.Figure:
    pd_ = extras["sensitivity"]["partial_dependence"]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[config.FEATURE_LABELS[f] for f in config.FEATURES],
                        vertical_spacing=0.18, horizontal_spacing=0.07)
    for i, f in enumerate(config.FEATURES):
        r, c = divmod(i, 3)
        fig.add_trace(go.Scatter(x=pd_[f]["grid"], y=pd_[f][target], mode="lines", showlegend=False,
                                 line=dict(color=SERIES[0], width=2),
                                 hovertemplate="%{x:.3g} -> %{y:.3g}<extra></extra>"), r + 1, c + 1)
    fig.update_yaxes(title_text=SHORT[target], col=1)
    return _layout(fig, f"Partial dependence - {config.TARGET_LABELS[target]}", height=520)


# ============================================================================ model comparison
def model_table(metrics: dict, selection: dict | None = None) -> pd.DataFrame:
    rows = []
    for name, m in metrics["models"].items():
        pt = m["test_vs_clean_fe"]["per_target"]
        rows.append({
            "Model": m["label"] + ("  [selected]" if selection and name == selection["selected_model"] else ""),
            "key": name,
            "CV min R²": m["cv"].get("min_mean_r2", np.nan),
            "Test min R²": m["test"]["min_r2"],
            **{f"R² {SHORT[t]}": m["test"]["per_target"][t]["r2"] for t in config.TARGETS},
            "Stress MAE [MPa]": m["test"]["per_target"]["peak_von_mises_mpa"]["mae"],
            "Stress max err vs FE [%]": pt["peak_von_mises_mpa"]["max_rel_err_pct"],
            "Mass max err vs FE [%]": pt["mass_kg"]["max_rel_err_pct"],
            "UQ": "yes" if m["supports_uncertainty"] else "no",
            "Fit [s]": m["fit_seconds"],
            "Latency 1 design [ms]": m["latency_ms_single"],
            "Speedup vs FE (batch)": m["speedup_vs_fe_batch"],
        })
    return pd.DataFrame(rows)


def fig_model_scores(selection: dict, metrics: dict | None = None) -> go.Figure:
    """Stacked weighted-criterion bars explaining the composite model selection."""
    bd = selection.get("score_breakdown")
    fig = go.Figure()
    if not bd:
        return _layout(fig, "Model scorecard unavailable (v0 selection rule)", height=200)
    order = sorted(bd, key=lambda n: selection["scores"][n])
    crit = list(selection["rule"]["weights"])
    labels = {"stress_max_err_pct": "worst stress error", "stress_mape_pct": "typical stress error",
              "mean_mape_pct": "typical error, all outputs", "mean_max_err_pct": "worst error, all outputs",
              "extrapolation_stress_mape_pct": "extrapolation", "cv_min_r2": "CV R²",
              "uncertainty": "provides 95% CI", "batch_latency_ms": "speed"}
    for i, c in enumerate(crit):
        fig.add_trace(go.Bar(y=[model_label(n, metrics) for n in order], x=[bd[n][c] for n in order],
                             name=f"{labels.get(c, c)} (w={selection['rule']['weights'][c]:.2f})", orientation="h",
                             marker=dict(color=SERIES[i % len(SERIES)], line=dict(color=SURFACE, width=2)),
                             hovertemplate="%{y}<br>%{x:.3f}<extra>" + labels.get(c, c) + "</extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="Weighted score (0-1)", range=[0, 1])
    return _layout(fig, "Surrogate scorecard - why the selected model wins", height=420)


def fig_accuracy_vs_latency(metrics: dict) -> go.Figure:
    fig = go.Figure()
    for name in ["closed_form"] + MODEL_ORDER:
        if name not in metrics["models"]:
            continue
        m = metrics["models"][name]
        err = m["test_vs_clean_fe"]["per_target"]["peak_von_mises_mpa"]["max_rel_err_pct"]
        fig.add_trace(go.Scatter(
            x=[m["latency_ms_single"]], y=[err], mode="markers+text", name=m["label"],
            text=[m["label"].split(" (")[0]], textposition="top center", textfont=dict(color=TEXT_2, size=11),
            marker=dict(size=14, color=MODEL_COLORS[name], symbol="diamond" if m["supports_uncertainty"] else "circle",
                        line=dict(width=2, color=SURFACE)),
            hovertemplate=f"{m['label']}<br>latency %{{x:.2f}} ms<br>stress max err %{{y:.1f}} %%<br>"
                          f"min R² {m['test']['min_r2']:.4f}<extra></extra>"))
    fig.update_xaxes(type="log", title_text="Single-design inference latency [ms] (log)")
    fig.update_yaxes(type="log", title_text="Worst-case stress error vs FE [%] (log)")
    fig.add_annotation(text="better", x=0.02, y=0.04, xref="paper", yref="paper", showarrow=False,
                       font=dict(color=TEXT_2))
    return _layout(fig, "Accuracy vs. speed trade-off (diamond = provides 95% CI)", height=460, showlegend=False)


def fig_model_radar(metrics: dict, extras: dict | None = None) -> go.Figure:
    names = [n for n in MODEL_ORDER if n in metrics["models"]]
    axes = ["Test R² (min)", "CV R² (min)", "Stress accuracy", "Speed", "Extrapolation"]

    def raw(n):
        m = metrics["models"][n]
        ex = extras["extrapolation"]["models"][n]["peak_von_mises_mpa"]["mape_pct"] if extras else np.nan
        return [m["test"]["min_r2"], m["cv"]["min_mean_r2"],
                -np.log10(m["test_vs_clean_fe"]["per_target"]["peak_von_mises_mpa"]["max_rel_err_pct"]),
                -np.log10(m["latency_ms_single"]), -np.log10(max(ex, 1e-3))]

    M = np.array([raw(n) for n in names], dtype=float)
    span = np.nanmax(M, 0) - np.nanmin(M, 0)
    S = np.where(span > 0, (M - np.nanmin(M, 0)) / np.where(span > 0, span, 1), 1.0)
    S = 0.1 + 0.9 * np.nan_to_num(S)
    fig = go.Figure()
    for n, row in zip(names, S):
        fig.add_trace(go.Scatterpolar(r=np.r_[row, row[0]], theta=axes + [axes[0]], name=model_label(n, metrics),
                                      line=dict(color=MODEL_COLORS[n], width=2), fill="none"))
    fig.update_layout(polar=dict(bgcolor=SURFACE, radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID,
                                                                  showticklabels=False),
                                 angularaxis=dict(gridcolor=GRID)))
    return _layout(fig, "Surrogate scorecard (normalised; outer = better)", height=520)


def fig_speedup(metrics: dict) -> go.Figure:
    names = [n for n in ["closed_form"] + MODEL_ORDER if n in metrics["models"]]
    labels = [metrics["models"][n]["label"] for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=[metrics["models"][n]["speedup_vs_fe_single"] for n in names],
                         name="Single design", orientation="h",
                         marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=2))))
    fig.add_trace(go.Bar(y=labels, x=[metrics["models"][n]["speedup_vs_fe_batch"] for n in names],
                         name="Batch of 1,000 (per design)", orientation="h",
                         marker=dict(color=SERIES[1], line=dict(color=SURFACE, width=2))))
    fig.add_vline(x=1, line=dict(color=AXIS, dash="dash"))
    fig.update_xaxes(type="log", title_text=f"Speedup vs in-repo beam FE solver "
                                            f"({metrics['fe_baseline_ms_per_design']:.1f} ms/design), log")
    fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"))
    return _layout(fig, "Inference speedup vs. physics baseline", height=440)


def fig_training_history(extras: dict) -> go.Figure:
    hist = extras.get("training_history", {})
    fig = go.Figure()
    for n in MODEL_ORDER:
        if n not in hist:
            continue
        h = hist[n]
        fig.add_trace(go.Scatter(y=h["train"], mode="lines", name=f"{n} train",
                                 line=dict(color=MODEL_COLORS[n], width=2)))
        fig.add_trace(go.Scatter(y=h["val"], mode="lines", name=f"{n} validation",
                                 line=dict(color=MODEL_COLORS[n], width=2, dash="dot")))
    fig.update_yaxes(type="log", title_text="MSE (standardised targets)")
    fig.update_xaxes(title_text="Epoch")
    return _layout(fig, "Neural-network training history", height=380)


# ============================================================================ optimisation
def fig_pareto(pareto_df: pd.DataFrame, candidates: pd.DataFrame | None, algorithms: list[str] | None = None,
               color_by: str = "first_mode_hz", highlight: dict | None = None) -> go.Figure:
    fig = go.Figure()
    if candidates is not None and len(candidates):
        inf = candidates[~candidates.feasible]
        fig.add_trace(go.Scattergl(x=inf.mass_kg, y=inf.peak_von_mises_mpa, mode="markers",
                                   name="Infeasible candidates", marker=dict(color=GRID, size=4),
                                   hoverinfo="skip"))
        fe = candidates[candidates.feasible]
        fig.add_trace(go.Scattergl(
            x=fe.mass_kg, y=fe.peak_von_mises_mpa, mode="markers", name="Feasible candidates",
            marker=dict(color=fe[color_by], colorscale=[[i / 6, c] for i, c in enumerate(BLUE_RAMP)], size=5,
                        colorbar=dict(title=config.TARGET_LABELS.get(color_by, color_by), thickness=12)),
            customdata=fe[config.GEOMETRY_FEATURES + ["safety_factor"]].to_numpy(),
            hovertemplate="mass %{x:.3f} kg<br>stress %{y:.1f} MPa<br>OD %{customdata[0]:.1f} | t %{customdata[1]:.2f}"
                          " | gusset %{customdata[2]:.2f} | R %{customdata[3]:.0f}<br>SF %{customdata[4]:.2f}"
                          "<extra></extra>"))
    algos = [a for a in ALGO_ORDER if (algorithms is None or a in algorithms) and a in set(pareto_df.algorithm)]
    for a in algos:
        sub = pareto_df[pareto_df.algorithm == a].sort_values("mass_kg")
        fig.add_trace(go.Scatter(
            x=sub.mass_kg, y=sub.peak_von_mises_mpa, mode="lines+markers", name=_algo_label(a),
            line=dict(color=ALGO_COLORS[a], width=2),
            marker=dict(symbol=ALGO_SYMBOLS[a], size=9, color=ALGO_COLORS[a], line=dict(width=1.5, color=SURFACE)),
            customdata=sub[config.GEOMETRY_FEATURES + ["safety_factor", "first_mode_hz"]].to_numpy(),
            hovertemplate=f"<b>{_algo_label(a)}</b><br>mass %{{x:.3f}} kg<br>stress %{{y:.1f}} MPa<br>"
                          "OD %{customdata[0]:.1f} | t %{customdata[1]:.2f} | gusset %{customdata[2]:.2f} | "
                          "R %{customdata[3]:.0f}<br>SF %{customdata[4]:.2f} | f1 %{customdata[5]:.1f} Hz<extra></extra>"))
    if highlight:
        fig.add_trace(go.Scatter(x=[highlight["fe"]["mass_kg"]], y=[highlight["fe"]["peak_von_mises_mpa"]],
                                 mode="markers+text", name="Recommended (knee, FE)", text=["Recommended"],
                                 textposition="top right", textfont=dict(color=TEXT),
                                 marker=dict(symbol="star", size=18, color=SERIES[3], line=dict(width=1.5, color=TEXT)),
                                 hovertemplate="Recommended design (FE)<br>mass %{x:.3f} kg<br>stress %{y:.1f} MPa"
                                               "<extra></extra>"))
    fig.update_xaxes(title_text="Mass [kg]")
    fig.update_yaxes(title_text="Peak von Mises [MPa]")
    return _layout(fig, "Pareto frontier: mass vs. peak stress", height=560)


def _algo_label(a: str) -> str:
    from src.optimizer import ALGORITHM_LABELS

    return ALGORITHM_LABELS.get(a, a)


def fig_hv_convergence(opt: dict) -> go.Figure:
    fig = go.Figure()
    for a in ALGO_ORDER:
        if a not in opt["algorithms"]:
            continue
        conv = opt["algorithms"][a]["convergence"]
        fig.add_trace(go.Scatter(x=conv["evaluations"], y=conv["hypervolume"], mode="lines+markers",
                                 name=_algo_label(a), line=dict(color=ALGO_COLORS[a], width=2),
                                 marker=dict(symbol=ALGO_SYMBOLS[a], size=8)))
    fig.update_xaxes(type="log", title_text="Surrogate evaluations (log)")
    fig.update_yaxes(title_text="Hypervolume (normalised)")
    return _layout(fig, "Convergence: hypervolume vs. evaluation budget", height=420)


def optimizer_table(opt: dict) -> pd.DataFrame:
    rows = []
    for a in ALGO_ORDER:
        if a not in opt["algorithms"]:
            continue
        r = opt["algorithms"][a]
        rows.append({
            "Optimiser": r["label"] + ("  [selected]" if a == opt["selected_optimizer"] else ""),
            "key": a, "Score": r["score"], "Hypervolume": r["hypervolume"],
            "HV std (seeds)": r.get("std", {}).get("hypervolume", 0.0),
            "FE-verified HV": r["hypervolume_fe_verified"], "IGD": r["igd"], "Spacing": r["spacing"],
            "FE-feasible rate": r["fe_feasible_rate"], "Pareto points": r["n_pareto"],
            "Evaluations": r["n_evaluations"], "Evals to 99% HV": r["evals_to_99pct_hv"],
            "Wall time [s]": r["wall_seconds"], "Stress err vs FE [%]": r["fe_stress_mape_pct"],
        })
    return pd.DataFrame(rows).sort_values("Score", ascending=False)


def fig_optimizer_scores(opt: dict) -> go.Figure:
    order = sorted(opt["algorithms"], key=lambda a: opt["algorithms"][a]["score"])
    metrics_ = list(opt["score_weights"])
    fig = go.Figure()
    for i, k in enumerate(metrics_):
        fig.add_trace(go.Bar(y=[_algo_label(a) for a in order],
                             x=[opt["algorithms"][a]["score_breakdown"][k] for a in order],
                             name=f"{k.replace('_', ' ')} (w={opt['score_weights'][k]:.2f})", orientation="h",
                             marker=dict(color=SERIES[i], line=dict(color=SURFACE, width=2))))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="Weighted score (0-1)", range=[0, 1])
    return _layout(fig, "Optimiser scorecard - why the selected method wins", height=420)


def fig_fe_verification(pareto_df: pd.DataFrame) -> go.Figure:
    v = pareto_df[pareto_df.get("fe_verified", False) == True]  # noqa: E712
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Mass [kg]", "Peak von Mises [MPa]"])
    for c, (t, ft) in enumerate((("mass_kg", "fe_mass_kg"), ("peak_von_mises_mpa", "fe_peak_von_mises_mpa"))):
        for ok, name, color, sym in ((True, "FE-feasible", STATUS["good"], "circle"),
                                     (False, "Violates a constraint in FE", STATUS["critical"], "x")):
            sub = v[v.fe_feasible == ok]
            fig.add_trace(go.Scatter(x=sub[ft], y=sub[t], mode="markers", name=name, showlegend=c == 0,
                                     marker=dict(color=color, symbol=sym, size=8, line=dict(width=1, color=SURFACE)),
                                     text=sub.algorithm,
                                     hovertemplate="%{text}<br>FE %{x:.3g}<br>surrogate %{y:.3g}<extra></extra>"),
                          1, c + 1)
        if len(v):
            lim = [float(min(v[ft].min(), v[t].min())), float(max(v[ft].max(), v[t].max()))]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=AXIS, dash="dash"),
                                     showlegend=False, hoverinfo="skip"), 1, c + 1)
        fig.update_xaxes(title_text="FE ground truth", row=1, col=c + 1)
    fig.update_yaxes(title_text="Surrogate", row=1, col=1)
    return _layout(fig, "Pareto designs re-checked with the FE solver", height=400)


def fig_parallel(pareto_df: pd.DataFrame, algorithm: str) -> go.Figure:
    sub = pareto_df[pareto_df.algorithm == algorithm]
    cols = config.GEOMETRY_FEATURES + ["mass_kg", "peak_von_mises_mpa", "first_mode_hz", "safety_factor"]
    labels = {**{f: config.FEATURE_LABELS[f] for f in config.GEOMETRY_FEATURES},
              **{t: config.TARGET_LABELS[t] for t in config.TARGETS}, "safety_factor": "Safety factor"}
    fig = go.Figure(go.Parcoords(
        line=dict(color=sub.mass_kg, colorscale=[[i / 6, c] for i, c in enumerate(BLUE_RAMP)], showscale=True,
                  colorbar=dict(title="Mass [kg]", thickness=12)),
        dimensions=[dict(label=labels[c], values=sub[c]) for c in cols]))
    return _layout(fig, f"Pareto-optimal designs ({_algo_label(algorithm)}) - parallel coordinates", height=460)


def fig_mass_saving(opt: dict) -> go.Figure | None:
    h = opt.get("headline", {})
    if "optimised_fe" not in h:
        return None
    sy = opt["scenario"]["yield_strength_mpa"]
    items = [("Mass [kg]", "mass_kg"), ("Peak stress [MPa]", "peak_von_mises_mpa"),
             ("First mode [Hz]", "first_mode_hz"), ("Deflection [mm]", "max_deflection_mm")]
    fig = make_subplots(rows=1, cols=5, subplot_titles=[i[0] for i in items] + ["Safety factor"])
    for c, (lab, key) in enumerate(items):
        fig.add_trace(go.Bar(x=["Baseline", "Optimised"], y=[h["baseline_fe"][key], h["optimised_fe"][key]],
                             marker=dict(color=[NEUTRAL, SERIES[0]], line=dict(color=SURFACE, width=2)),
                             showlegend=False, texttemplate="%{y:.3g}", textposition="outside"), 1, c + 1)
    fig.add_trace(go.Bar(x=["Baseline", "Optimised"],
                         y=[sy / h["baseline_fe"]["peak_von_mises_mpa"], sy / h["optimised_fe"]["peak_von_mises_mpa"]],
                         marker=dict(color=[NEUTRAL, SERIES[0]], line=dict(color=SURFACE, width=2)),
                         showlegend=False, texttemplate="%{y:.2f}", textposition="outside"), 1, 5)
    return _layout(fig, f"FE-verified result: {h['mass_saving_pct']:.1f}% lighter than baseline "
                        f"(stress {h['stress_change_pct']:+.1f}%)", height=360)
