"""Streamlit dashboard: physics-guided structural surrogate & Pareto weight optimiser.

Run:  .\\.venv\\Scripts\\python.exe -m streamlit run app.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from src import config, plots
from src.artifacts import load_all
from src.guardrails import apply_guardrails, clip_inputs, guard_interval, safety_factor
from src.optimizer import ALGORITHM_LABELS, non_dominated_mask, screen_candidates
from src.physics import frame_fe_solve
from src.surrogate_zoo import load_surrogate

st.set_page_config(page_title="Chassis Surrogate & Pareto Optimiser", layout="wide")


# ----------------------------------------------------------------------------- helpers
def show(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", theme=None, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, theme=None, key=key)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


@st.cache_data(show_spinner=False)
def artifacts():
    return load_all()


@st.cache_resource(show_spinner=False)
def surrogate(name: str):
    return load_surrogate(name)


art = artifacts()
if art["metrics"] is None or art["selection"] is None:
    st.error("Model artifacts not found. Run `python -m src.data_generator`, `python -m src.train` and "
             "`python -m src.optimizer` first.")
    st.stop()

M, SEL, EX, OPT = art["metrics"], art["selection"], art["extras"], art["optimization"]
BEST = SEL["selected_model"]
TRAINED = [n for n, v in M["models"].items() if v["trainable"]]
HAS_GP = "gaussian_process" in M["models"]

st.title("Physics-guided structural surrogate & Pareto weight optimiser")
st.caption("Tubular two-wheeler rear sub-frame | beam-FE ground truth, 7 benchmarked surrogates, "
           "5 benchmarked optimisers | all displayed outputs pass physical guardrails "
           "(non-negative mass/stress/deflection, bounded safety factor and fatigue life).")

with st.sidebar:
    st.header("Surrogate")
    model_name = st.selectbox("Model", TRAINED, index=TRAINED.index(BEST),
                              format_func=lambda n: M["models"][n]["label"] + ("  (selected)" if n == BEST else ""))
    st.caption(f"Selected by the benchmark: **{M['models'][BEST]['label']}**")
    st.divider()
    st.header("Constraints")
    c_sf = st.slider("Min safety factor", 1.0, 3.0, float(config.CONSTRAINTS["min_safety_factor"]), 0.05)
    c_f1 = st.slider("Min first mode [Hz]", 30.0, 70.0, float(config.CONSTRAINTS["min_first_mode_hz"]), 0.5)
    c_def = st.slider("Max deflection [mm]", 0.5, 5.0, float(config.CONSTRAINTS["max_deflection_mm"]), 0.1)
    c_fat = st.slider("Min fatigue life [log10 cycles]", 4.0, 8.0, float(config.CONSTRAINTS["min_log10_fatigue_life"]), 0.1)
    st.caption("Manufacturability (tube bending)")
    c_dt = st.slider("Max tube D/t", 10.0, 35.0, float(config.CONSTRAINTS["max_d_over_t"]), 0.5)
    c_br = st.slider("Min bend radius / OD", 1.0, 3.0, float(config.CONSTRAINTS["min_bend_ratio"]), 0.1)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1 · Real-time surrogate explorer", "2 · Pareto weight optimisation",
                                  "3 · Model validation & sensitivity", "4 · Benchmark & model selection", "5 · FRAMED benchmark (published data)"])

# ============================================================================ TAB 1
with tab1:
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Design & load")
        x = []
        defaults = {"tube_od_mm": 32.0, "wall_thickness_mm": 2.5, "gusset_thickness_mm": 4.0, "bend_radius_mm": 60.0,
                    **config.DEFAULT_SCENARIO}
        for f in config.FEATURES:
            lo, hi = config.BOUNDS[f]
            step = 0.05 if hi - lo < 10 else 0.5
            x.append(st.slider(config.FEATURE_LABELS[f], float(lo), float(hi), float(defaults[f]), step, key=f"x_{f}"))
    X, outside = clip_inputs(np.array([x]))
    if outside.any():
        st.warning("Inputs outside the trained design space were clipped to the bounds.")
    model = surrogate(model_name)
    t0 = time.perf_counter()
    y = model.predict(X)[0]
    t_sur = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    fe = frame_fe_solve(*X[0])
    t_fe = (time.perf_counter() - t0) * 1e3
    y_fe = apply_guardrails(np.array([[fe[t] for t in config.TARGETS]]))[0]

    ci = None
    if HAS_GP:
        gp = surrogate("gaussian_process")
        lo_, mu_, hi_ = gp.predict_interval(X)
        ci = (lo_[0], mu_[0], hi_[0])
    sy = X[0, 4]
    sf = float(safety_factor(sy, y[1]))

    with right:
        st.subheader(f"Prediction - {M['models'][model_name]['label']}")
        k = st.columns(3)
        stress_ci = ""
        sf_ci = ""
        if ci is not None:
            lo_s, _, hi_s = guard_interval(ci[0][1], ci[1][1], ci[2][1], "peak_von_mises_mpa")
            stress_ci = f"GP 95% CI {float(lo_s):.0f}-{float(hi_s):.0f} MPa"
            sf_ci = f"GP 95% CI {float(safety_factor(sy, hi_s)):.2f}-{float(safety_factor(sy, lo_s)):.2f}"
        k[0].metric("Mass", f"{y[0]:.3f} kg")
        k[1].metric("Peak von Mises", f"{y[1]:.1f} MPa", stress_ci, delta_color="off")
        k[2].metric("Safety factor", f"{sf:.2f}", sf_ci, delta_color="off")
        k = st.columns(3)
        k[0].metric("Max deflection", f"{y[2]:.2f} mm")
        k[1].metric("First mode", f"{y[3]:.1f} Hz")
        k[2].metric("Fatigue life", f"1e{y[4]:.2f} cycles")

        checks = [("Safety factor", sf >= c_sf, f"{sf:.2f} >= {c_sf:.2f}"),
                  ("First mode", y[3] >= c_f1, f"{y[3]:.1f} >= {c_f1:.1f} Hz"),
                  ("Deflection", y[2] <= c_def, f"{y[2]:.2f} <= {c_def:.2f} mm"),
                  ("Fatigue", y[4] >= c_fat, f"1e{y[4]:.2f} >= 1e{c_fat:.1f}"),
                  ("D/t", X[0, 0] / X[0, 1] <= c_dt, f"{X[0, 0] / X[0, 1]:.1f} <= {c_dt:.1f}"),
                  ("Bend ratio", X[0, 3] / X[0, 0] >= c_br, f"{X[0, 3] / X[0, 0]:.2f} >= {c_br:.2f}")]
        st.markdown(" | ".join(f"{'✅ PASS' if ok else '❌ FAIL'} **{n}** ({txt})" for n, ok, txt in checks))

        k = st.columns(3)
        k[0].metric("Surrogate latency", f"{t_sur:.2f} ms")
        k[1].metric("FE solver latency", f"{t_fe:.1f} ms")
        k[2].metric("Live speedup", f"{t_fe / max(t_sur, 1e-6):.1f}x",
                    f"batch: {M['models'][model_name]['speedup_vs_fe_batch']:,.0f}x per design", delta_color="off")

        st.markdown("**Live verification against the FE ground truth**")
        rows = []
        for j, t in enumerate(config.TARGETS):
            r = {"Output": config.TARGET_LABELS[t], "Surrogate": y[j], "FE": y_fe[j],
                 "Error [%]": 100 * (y[j] - y_fe[j]) / max(abs(y_fe[j]), 1.0 if j == 4 else 1e-9)}
            if ci is not None:
                lo_j, _, hi_j = guard_interval(ci[0][j], ci[1][j], ci[2][j], t)
                r["GP 95% CI"] = f"{float(lo_j):.3g} - {float(hi_j):.3g}"
            rows.append(r)
        table(pd.DataFrame(rows).round(4))

# ============================================================================ TAB 2
with tab2:
    if OPT is None or art["pareto"] is None:
        st.info("Run `python -m src.optimizer` to generate the optimisation benchmark.")
    else:
        sc = OPT["scenario"]
        st.caption(f"Benchmark scenario: yield {sc['yield_strength_mpa']:.0f} MPa, DLF {sc['dynamic_load_factor']:.1f} "
                   f"| surrogate `{OPT['surrogate']}` | selected optimiser: "
                   f"**{ALGORITHM_LABELS[OPT['selected_optimizer']]}**")
        with st.expander("Re-run 5,000-candidate screening for a custom material / load case"):
            cc = st.columns(3)
            sy_new = cc[0].slider("Yield strength [MPa]", 250.0, 480.0, float(sc["yield_strength_mpa"]), 5.0)
            dlf_new = cc[1].slider("Dynamic load factor", 1.5, 4.0, float(sc["dynamic_load_factor"]), 0.05)
            if cc[2].button("Run screening", type="primary"):
                with st.spinner("Screening 5,000 candidates..."):
                    st.session_state["custom"] = screen_candidates(
                        surrogate(model_name), {"yield_strength_mpa": sy_new, "dynamic_load_factor": dlf_new})
            if "custom" in st.session_state and st.button("Back to benchmark results"):
                del st.session_state["custom"]

        def meets(df, prefix=""):
            sf_ = safety_factor(df["yield_strength_mpa"], df[f"{prefix}peak_von_mises_mpa"])
            return ((sf_ >= c_sf) & (df[f"{prefix}first_mode_hz"] >= c_f1) & (df[f"{prefix}max_deflection_mm"] <= c_def)
                    & (df[f"{prefix}log10_fatigue_life_cycles"] >= c_fat)
                    & (df["tube_od_mm"] / df["wall_thickness_mm"] <= c_dt)
                    & (df["bend_radius_mm"] / df["tube_od_mm"] >= c_br))

        cands = st.session_state.get("custom", art["candidates"]).copy()
        cands["safety_factor"] = safety_factor(cands["yield_strength_mpa"], cands["peak_von_mises_mpa"])
        cands["feasible"] = meets(cands)
        if "custom" in st.session_state:
            fe_c = cands[cands.feasible]
            front = fe_c[non_dominated_mask(fe_c[["mass_kg", "peak_von_mises_mpa"]].to_numpy())].assign(algorithm="lhs_screening")
        else:
            par = art["pareto"].copy()
            par = par[meets(par)]
            front = pd.concat([g[non_dominated_mask(g[["mass_kg", "peak_von_mises_mpa"]].to_numpy())]
                               for _, g in par.groupby("algorithm")]) if len(par) else par

        cc = st.columns([2, 1])
        algos = cc[0].multiselect("Optimisers", list(ALGORITHM_LABELS), default=list(ALGORITHM_LABELS),
                                  format_func=ALGORITHM_LABELS.get)
        color_by = cc[1].selectbox("Colour candidates by", ["first_mode_hz", "safety_factor", "log10_fatigue_life_cycles"],
                                   format_func=lambda c: config.TARGET_LABELS.get(c, "Safety factor"))
        k = st.columns(3)
        k[0].metric("Feasible candidates", f"{int(cands.feasible.sum()):,} / {len(cands):,}")
        k[1].metric("Pareto designs shown", f"{int(front.algorithm.isin(algos).sum()) if len(front) else 0}")
        if len(front):
            lightest = front[front.algorithm.isin(algos)].sort_values("mass_kg").head(1)
            if len(lightest):
                k[2].metric("Lightest feasible", f"{float(lightest.mass_kg.iloc[0]):.3f} kg")
        rec = OPT.get("headline", {}).get("recommended") if "custom" not in st.session_state else None
        show(plots.fig_pareto(front, cands, algos, color_by, highlight=rec), key="pareto")
        if rec:
            g = rec["geometry"]
            st.info(f"**Recommended design (knee point, FE-verified):** OD {g['tube_od_mm']:.1f} mm, wall "
                    f"{g['wall_thickness_mm']:.2f} mm, gusset {g['gusset_thickness_mm']:.2f} mm, bend radius "
                    f"{g['bend_radius_mm']:.0f} mm -> {rec['fe']['mass_kg']:.3f} kg "
                    f"({rec['mass_saving_pct']:.1f}% lighter than baseline), SF {rec['safety_factor']:.2f}, "
                    f"f1 {rec['fe']['first_mode_hz']:.1f} Hz. The star on the plot marks it.")

        st.markdown("**Optimal dimensions**")
        cols = ["algorithm"] + config.GEOMETRY_FEATURES + ["mass_kg", "peak_von_mises_mpa", "safety_factor",
                                                           "first_mode_hz", "max_deflection_mm", "log10_fatigue_life_cycles"]
        cols += [c for c in ("fe_mass_kg", "fe_peak_von_mises_mpa", "fe_feasible") if c in front.columns]
        out = front[front.algorithm.isin(algos)][cols].sort_values(["algorithm", "mass_kg"]).round(4) if len(front) else front
        table(out, height=320)
        st.download_button("Download optimal designs (CSV)", out.to_csv(index=False).encode(), "pareto_optimal_designs.csv",
                           "text/csv")
        if "custom" not in st.session_state:
            fig = plots.fig_mass_saving(OPT)
            if fig is not None:
                show(fig, key="mass_saving")

# ============================================================================ TAB 3
with tab3:
    P = art["preds"]
    if P is None:
        st.info("Run `python -m src.train` to generate validation data.")
    else:
        vm = st.selectbox("Model to validate", TRAINED + ["closed_form"], index=TRAINED.index(model_name),
                          format_func=lambda n: M["models"][n]["label"], key="val_model")
        truth = st.radio("Compare against", ["Noise-free FE", "Noisy test labels"], horizontal=True)
        Yt = P["Y_test_clean_fe"] if truth == "Noise-free FE" else P["Y_test"]
        Yp = P[f"pred__{vm}"]
        show(plots.fig_parity(Yt, Yp, vm), key="parity")
        c1, c2 = st.columns(2)
        with c1:
            show(plots.fig_residual_hist(Yt, Yp, vm), key="hist")
        with c2:
            show(plots.fig_qq(Yt, Yp, vm), key="qq")
        show(plots.fig_residual_vs_pred(Yt, Yp, vm), key="rvp")
        st.subheader(f"Sensitivity (computed on the selected model: {M['models'][EX['sensitivity']['model']]['label']})")
        tgt = st.selectbox("Output", config.TARGETS, index=1, format_func=config.TARGET_LABELS.get)
        c1, c2 = st.columns(2)
        with c1:
            show(plots.fig_sobol(EX, tgt), key="sobol")
        with c2:
            show(plots.fig_permutation(EX, tgt), key="perm")
        show(plots.fig_pdp(EX, tgt), key="pdp")

# ============================================================================ TAB 4
with tab4:
    st.subheader("Surrogate benchmark")
    st.success(f"Selected surrogate: **{M['models'][BEST]['label']}**")
    with st.expander("Decision rule & trace", expanded=False):
        r = SEL["rule"]
        st.markdown(f"- Gate: CV R² >= {r['r2_gate']} on every output")
        if "weights" in r:
            st.markdown("- Weighted composite score: " + ", ".join(f"{k} {w}" for k, w in r["weights"].items()))
        for line in SEL["trace"]:
            st.markdown(f"- {line}")
    table(plots.model_table(M, SEL).drop(columns="key").round(4))
    show(plots.fig_model_scores(SEL, M), key="model_scores")
    c1, c2 = st.columns(2)
    with c1:
        show(plots.fig_accuracy_vs_latency(M), key="acc_lat")
    with c2:
        show(plots.fig_model_radar(M, EX), key="radar")
    c1, c2 = st.columns(2)
    with c1:
        show(plots.fig_learning_curves(EX, M), key="lc")
    with c2:
        show(plots.fig_calibration(EX), key="cal")
    show(plots.fig_extrapolation(EX, M), key="extrap")
    show(plots.fig_speedup(M), key="speed")
    if EX.get("training_history"):
        show(plots.fig_training_history(EX), key="nn_hist")

    if OPT is not None:
        st.subheader("Optimiser benchmark")
        st.success(f"Selected optimiser: **{ALGORITHM_LABELS[OPT['selected_optimizer']]}**")
        table(plots.optimizer_table(OPT).drop(columns="key").round(4))
        c1, c2 = st.columns(2)
        with c1:
            show(plots.fig_optimizer_scores(OPT), key="opt_scores")
        with c2:
            show(plots.fig_hv_convergence(OPT), key="hv")
        show(plots.fig_fe_verification(art["pareto"]), key="fe_ver")
        show(plots.fig_parallel(art["pareto"], OPT["selected_optimizer"]), key="parcoords")


with tab5:
    from src.framed import figures as FF
    from src.framed.run import load_artifact
    st.caption("Benchmark on the peer-reviewed FRAMED dataset (Regenwetter et al., Computer-Aided Design 2023).")
    repro, ours_, opt_ = load_artifact("reproduction"), load_artifact("ours"), load_artifact("optimize")
    if repro is None:
        st.info("No FRAMED artefacts yet. Run: `python -m src.framed.run --stage all`")
    else:
        st.subheader("Table 4 reproduction")
        t = pd.DataFrame(repro["table"]).set_index("model")
        table(t[["paper_r2", "ours_r2", "delta_r2", "paper_mae", "ours_mae", "status"]].round(3))
        show(FF.fig_table4_comparison(t), key="framed_t4")
        if ours_:
            st.subheader("This project's models (same folds)")
            table(pd.DataFrame({k: {"R2": v["r2"]["mean"], "R2 std": v["r2"]["std"], "MAE": v["mae"]["mean"],
                                    "folds": v["n_folds"]} for k, v in ours_["models"].items()}).T.round(3))
        if opt_:
            st.subheader("Lightweighting of frame 3853 (surrogate-predicted, not FE re-verified)")
            for mat, stt in opt_["studies"].items():
                st.markdown(f"**{mat}**")
                table(pd.DataFrame(stt["table"])[["design", "mass_kg", "dropout_y_mm", "sf1_lo", "sf3_lo",
                                                   "mass_change_pct", "feasible"]].round(3))
        st.markdown("Full report: `reports/framed/index.html`")
