"""Export portfolio-ready figures and selection write-ups.

Run:  .\\.venv\\Scripts\\python.exe -m src.report
Outputs:
  reports/figures/*.html     self-contained interactive Plotly figures (embed in a website via <iframe>)
  reports/figures/*.png      static images, only if `kaleido` is installed
  reports/index.html         gallery page linking every figure
  reports/model_selection.md, reports/optimizer_selection.md   decision write-ups
"""
from __future__ import annotations

import html

import numpy as np

from src import config, plots
from src.artifacts import load_all, missing

FIGURE_CAPTIONS: dict[str, str] = {}


def build_figures(art: dict) -> dict:
    m, ex, preds, opt = art["metrics"], art["extras"], art["preds"], art["optimization"]
    best = art["selection"]["selected_model"]
    Yc = preds["Y_test_clean_fe"]
    Pb = preds[f"pred__{best}"]
    figs = {
        "01_doe_coverage": (plots.fig_doe_coverage(art["dataset"]),
                            "1,500-design DOE: space-filling LHS plus a face-centred CCD that pins corners and axes."),
        "02_parity_selected": (plots.fig_parity(Yc, Pb, best),
                               "Selected surrogate vs. noise-free FE on 300 held-out designs (±10 % band)."),
        "03_residual_hist": (plots.fig_residual_hist(Yc, Pb, best), "Relative error distribution per output."),
        "04_qq": (plots.fig_qq(Yc, Pb, best), "Residual normality check."),
        "05_residual_vs_pred": (plots.fig_residual_vs_pred(Yc, Pb, best), "Error vs. magnitude (heteroscedasticity)."),
        "06_accuracy_vs_latency": (plots.fig_accuracy_vs_latency(m), "Every surrogate on one accuracy-speed map."),
        "06b_model_scorecard": (plots.fig_model_scores(art["selection"], m),
                                "Weighted composite score used to select the surrogate."),
        "07_model_radar": (plots.fig_model_radar(m, ex), "Normalised multi-criteria comparison of the surrogates."),
        "08_speedup": (plots.fig_speedup(m), "Inference speedup over the physics (beam FE) baseline."),
        "09_learning_curves": (plots.fig_learning_curves(ex, m), "Data efficiency: stress error vs. training size."),
        "10_calibration": (plots.fig_calibration(ex), "Do the GP's 95 % intervals contain 95 % of the truth?"),
        "11_extrapolation": (plots.fig_extrapolation(ex, m), "Behaviour outside the training region."),
        "12_sobol_stress": (plots.fig_sobol(ex, "peak_von_mises_mpa"), "Which inputs drive peak stress."),
        "13_sobol_frequency": (plots.fig_sobol(ex, "first_mode_hz"), "Which inputs drive the first mode."),
        "14_pdp_stress": (plots.fig_pdp(ex, "peak_von_mises_mpa"), "Partial dependence of peak stress."),
    }
    if ex.get("training_history"):
        figs["15_nn_training"] = (plots.fig_training_history(ex), "Neural-network learning curves.")
    if opt is not None and art["pareto"] is not None:
        figs.update({
            "20_pareto": (plots.fig_pareto(art["pareto"], art["candidates"],
                                           highlight=opt.get("headline", {}).get("recommended")),
                          "Pareto fronts of all optimisers over the 5,000-candidate cloud (colour = first mode)."),
            "21_hv_convergence": (plots.fig_hv_convergence(opt), "Hypervolume vs. evaluation budget."),
            "22_optimizer_scores": (plots.fig_optimizer_scores(opt), "Weighted scorecard used to pick the optimiser."),
            "23_fe_verification": (plots.fig_fe_verification(art["pareto"]),
                                   "Claimed-optimal designs re-solved with FE: who survives?"),
            "24_parallel_coordinates": (plots.fig_parallel(art["pareto"], opt["selected_optimizer"]),
                                        "Geometry of the optimal family."),
        })
        ms = plots.fig_mass_saving(opt)
        if ms is not None:
            figs["25_mass_saving"] = (ms, "Baseline vs. optimised design, both solved with FE.")
    return figs


def _md_table(df) -> str:
    df = df.drop(columns=[c for c in ("key",) if c in df])
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v) for v in r.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_model_md(art: dict) -> str:
    m, sel = art["metrics"], art["selection"]
    tbl = plots.model_table(m, sel)
    keep = ["Model", "CV min R²", "Test min R²", "Stress max err vs FE [%]", "Stress MAE [MPa]", "UQ",
            "Latency 1 design [ms]", "Speedup vs FE (batch)", "Fit [s]"]
    floor = m["label_noise_floor_max_rel_err_pct"]["peak_von_mises_mpa"]
    best = m["models"][sel["selected_model"]]
    return "\n".join([
        "# Surrogate model selection", "",
        f"**Selected: {best['label']}**", "",
        "## Decision rule", "",
        f"1. Gate: every target must reach cross-validated R² >= {sel['rule']['r2_gate']}.",
        "2. A weighted composite score ranks the eligible models (each criterion min-max normalised, 1 = best):", "",
        *[f"   - {k}: weight {w}" for k, w in sel["rule"].get("weights", {}).items()], "",
        "## Decision trace", "", *[f"- {t}" for t in sel["trace"]], "",
        "## Comparison (test set, 300 designs)", "", _md_table(tbl[keep]), "",
        f"Label-noise floor: the 3 % measurement scatter alone produces a worst-case stress deviation of "
        f"{floor:.1f} % against the noise-free FE, so errors near that level are at the data's resolution.", "",
        f"Measured speedup is against the in-repo beam FE solver ({m['fe_baseline_ms_per_design']:.1f} ms/design). "
        f"Against a nominal {m['nominal_industrial_fe_seconds']:.0f} s production FE run the projected speedup of "
        f"the selected model is ~{best['projected_speedup_vs_industrial_fe']:,.0f}x.", "",
    ])


def write_optimizer_md(art: dict) -> str:
    opt = art["optimization"]
    tbl = plots.optimizer_table(opt)
    h = opt["headline"]
    lines = [
        "# Optimiser selection", "",
        f"Surrogate: `{opt['surrogate']}` | scenario: yield {opt['scenario']['yield_strength_mpa']:.0f} MPa, "
        f"DLF {opt['scenario']['dynamic_load_factor']:.1f}", "",
        f"**Selected: {opt['algorithms'][opt['selected_optimizer']]['label']}**", "",
        "## Score weights", "", *[f"- {k}: {w}" for k, w in opt["score_weights"].items()], "",
        "Every claimed Pareto design (up to 60 per optimiser) is re-solved with the FE model; the FE-verified "
        "hypervolume and FE-feasible rate measure what survives contact with the physics.",
        f"Indicators are means over seeds {opt.get('seeds', ['-'])}.", "",
        "## Comparison", "", _md_table(tbl), "",
    ]
    if "recommended" in h:
        r = h["recommended"]
        lines += ["## Recommended design (knee point)", "",
                  f"{r['rule']}: **{r['mass_saving_pct']:.1f} % lighter** than baseline, peak stress "
                  f"{r['stress_change_pct']:+.1f} %, safety factor {r['safety_factor']:.2f} (FE).", "",
                  *[f"- {config.FEATURE_LABELS[k]}: {v:.2f}" for k, v in r["geometry"].items()], ""]
    if "mass_saving_pct" in h:
        lines += ["## Headline", "",
                  f"Lightest FE-feasible design: **{h['mass_saving_pct']:.1f} % lighter** than the baseline "
                  f"({h['baseline_fe']['mass_kg']:.3f} -> {h['optimised_fe']['mass_kg']:.3f} kg), "
                  f"peak stress {h['stress_change_pct']:+.1f} %.", "",
                  "| Geometry | Baseline | Optimised |", "|---|---|---|",
                  *[f"| {config.FEATURE_LABELS[k]} | {h['baseline_geometry'][k]:.2f} | {h['optimised_geometry'][k]:.2f} |"
                    for k in config.GEOMETRY_FEATURES], ""]
    return "\n".join(lines)


def write_gallery(figs: dict) -> None:
    cards = "\n".join(
        f'<figure><iframe src="figures/{k}.html" loading="lazy"></iframe>'
        f"<figcaption><b>{k[3:].replace('_', ' ').title()}</b> - {html.escape(cap)}</figcaption></figure>"
        for k, (_, cap) in figs.items())
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Structural surrogate & Pareto optimiser - verification gallery</title>
<style>body{{font-family:Inter,Segoe UI,Arial,sans-serif;background:#fcfcfb;color:#0b0b0b;margin:0;padding:32px;max-width:1280px;margin:auto}}
h1{{font-size:24px}}p{{color:#52514e}}figure{{margin:0 0 40px}}iframe{{width:100%;height:620px;border:1px solid #e1e0d9;border-radius:8px;background:#fcfcfb}}
figcaption{{color:#52514e;margin-top:8px}}</style></head><body>
<h1>Physics-guided structural surrogate &amp; Pareto weight optimiser</h1>
<p>Verification and benchmark figures. See model_selection.md and optimizer_selection.md for the decision write-ups.</p>
{cards}</body></html>"""
    (config.REPORTS_DIR / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    art = load_all()
    need = [k for k in ("dataset", "metrics", "selection", "extras", "preds") if k in missing(art)]
    if need:
        raise SystemExit(f"Missing artifacts {need}: run src.data_generator and src.train first.")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figs = build_figures(art)
    png_ok = True
    for name, (fig, _) in figs.items():
        fig.write_html(config.FIGURES_DIR / f"{name}.html", include_plotlyjs="cdn", full_html=True)
        if png_ok:
            try:
                fig.write_image(config.FIGURES_DIR / f"{name}.png", scale=2)
            except Exception:
                png_ok = False
    (config.REPORTS_DIR / "model_selection.md").write_text(write_model_md(art), encoding="utf-8")
    if art["optimization"] is not None:
        (config.REPORTS_DIR / "optimizer_selection.md").write_text(write_optimizer_md(art), encoding="utf-8")
    write_gallery(figs)
    print(f"Wrote {len(figs)} figures to {config.FIGURES_DIR}"
          + ("" if png_ok else " (HTML only - install 'kaleido' for PNG export)"))


if __name__ == "__main__":
    main()
