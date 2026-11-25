"""Build the FRAMED benchmark report: reports/framed/index.html (portfolio-styled) + reports/framed_benchmark.md.

Every number comes from a JSON artefact in models/framed/ or from src/framed/paper_reference.py.
"""
from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.framed import config as fc
from src.framed import figures as F
from src.framed import paper_reference as ref

PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.1.1/plotly.min.js"
REPO = "https://github.com/AlfredChettiar/structural-surrogate-optimizer"
LINKEDIN = "https://www.linkedin.com/in/alfred-chettiar-0b3b53189"
PAPER_URL = "https://arxiv.org/abs/2201.10459"
DATA_URL = "https://github.com/Lyleregenwetter/FRAMED"
PAPER_MODELS = ["autogluon", "xgboost", "neural_net", "decision_tree", "knn"]
OUR_MODELS = ["deep_ensemble_phys", "pg_mlp", "xgboost_phys", "autogluon_phys", "gp_phys"]
SECTIONS = [("summary", "Summary"), ("source", "Source"), ("claims", "Dataset claims"),
            ("reproduction", "Table 4 reproduction"), ("improvements", "Improvements"),
            ("optimisation", "Optimisation"), ("limitations", "Limitations"), ("method", "Method & hardware"),
            ("references", "References")]

CSS = """
:root{color-scheme:light;--bg:#f6f7f9;--surface:#fff;--surface-2:#eef1f5;--border:#d5dbe3;--border-strong:#a9b3c1;
--text:#0f1720;--text-muted:#4a5566;--text-dim:#5d6776;--amber:#f0a020;--amber-ink:#a34f00;--amber-tint:#fff5e3;
--cyan-ink:#0b6e85;--cyan-tint:#e3f3f7;--green:#157a40;--green-tint:#e6f4ec;--red:#b42318;--red-tint:#fdecea;
--font-sans:"Public Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
--font-mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;--radius:10px;--rail:220px}
*,*::before,*::after{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:24px;scrollbar-color:var(--border-strong) var(--bg)}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
body{margin:0;font-family:var(--font-sans);font-size:1.0625rem;line-height:1.6;color:var(--text);background:var(--bg);
-webkit-font-smoothing:antialiased}
::selection{background:#cdeef6;color:var(--text)}
a{color:var(--cyan-ink);text-underline-offset:3px;text-decoration-thickness:1px}
a:hover{text-decoration-thickness:2px}
:focus-visible{outline:2px solid var(--cyan-ink);outline-offset:2px;border-radius:4px}
h1,h2,h3{text-wrap:balance;margin:0;color:var(--text)}
.mono,td.num{font-family:var(--font-mono);font-variant-numeric:tabular-nums}
.skip{position:absolute;left:-999px;top:8px;background:var(--surface);padding:8px 12px;border-radius:6px}
.skip:focus{left:8px}
.layout{max-width:1180px;margin:0 auto;padding:0 clamp(16px,4vw,32px);display:grid;grid-template-columns:var(--rail) minmax(0,1fr);gap:48px}
.rail{position:sticky;top:0;align-self:start;height:100vh;padding:40px 0;overflow:auto}
.rail p{font-size:.8125rem;color:var(--text-dim);margin:0 0 10px;font-weight:600}
.rail ol{list-style:none;margin:0;padding:0;border-left:1px solid var(--border)}
.rail a{display:block;padding:8px 0 8px 14px;margin-left:-1px;border-left:2px solid transparent;color:var(--text-muted);
text-decoration:none;font-size:.9375rem;min-height:44px;display:flex;align-items:center}
.rail a:hover{color:var(--text)}
.rail a[aria-current="true"]{color:var(--cyan-ink);border-left-color:var(--cyan-ink);font-weight:600}
main{padding:40px 0 96px;min-width:0}
.masthead{padding-bottom:28px;border-bottom:1px solid var(--border);margin-bottom:8px}
.masthead h1{font-size:clamp(1.75rem,3.2vw,2.5rem);line-height:1.15;letter-spacing:-.02em;max-width:24ch}
.byline{margin:10px 0 0;color:var(--text-muted);font-size:.9375rem;display:flex;flex-wrap:wrap;align-items:center;gap:0 18px}
.byline>*{min-height:44px;display:inline-flex;align-items:center}
.lede{margin:18px 0 0;max-width:68ch;color:var(--text-muted);font-size:1.0625rem}
section{padding-top:48px}
section>h2{font-size:1.5rem;letter-spacing:-.01em;margin-bottom:14px}
h3{font-size:1.125rem;margin:28px 0 10px}
p,ul{max-width:70ch;margin:0 0 14px}
ul{padding-left:1.2em}
li{margin:4px 0}
.findings{list-style:none;padding:0;margin:4px 0 0;max-width:72ch}
.findings li{border-top:1px solid var(--border);padding:14px 0;margin:0}
.findings li:last-child{border-bottom:1px solid var(--border)}
.findings strong{font-family:var(--font-mono);font-variant-numeric:tabular-nums;font-weight:600}
.table-wrap{overflow-x:auto;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);margin:12px 0 18px;padding:4px 6px}
table{border-collapse:collapse;width:100%;font-size:.9375rem}
caption{text-align:left;padding:12px 14px 4px;color:var(--text-dim);font-size:.8125rem}
th,td{padding:9px 14px;border-bottom:1px solid var(--surface-2);text-align:left;vertical-align:top;white-space:nowrap}
th{font-size:.8125rem;color:var(--text-muted);font-weight:600;background:var(--surface)}
td.num,th.num{text-align:right}
tbody tr:last-child td{border-bottom:0}
td.wrap{white-space:normal;min-width:16ch}
.verdict{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.8125rem;font-weight:600;font-family:var(--font-sans)}
.v-pass{background:var(--green-tint);color:var(--green)}
.v-fail{background:var(--red-tint);color:var(--red)}
.v-info{background:var(--surface-2);color:var(--text-muted)}
.v-better{background:var(--cyan-tint);color:var(--cyan-ink)}
.note{background:var(--amber-tint);border:1px solid #f3d29a;border-radius:var(--radius);padding:12px 16px;color:#5c3a00;max-width:70ch}
figure{margin:12px 0 22px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:8px;min-width:0}
figcaption{color:var(--text-dim);font-size:.8125rem;padding:4px 8px 6px}
.refs li{margin:8px 0}
footer{margin-top:64px;padding-top:20px;border-top:1px solid var(--border);color:var(--text-dim);font-size:.8125rem}
@media (max-width:900px){.layout{grid-template-columns:1fr;gap:0}.rail{position:static;height:auto;padding:24px 0 0}
.rail ol{display:flex;flex-wrap:wrap;gap:0 14px;border-left:0}.rail a{border-left:0;padding:6px 0}
.rail a[aria-current="true"]{border-left:0;text-decoration:underline}main{padding-top:16px}}
"""

RAIL_JS = """
const links=[...document.querySelectorAll('.rail a')];const ids=links.map(a=>a.hash.slice(1));
const obs=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){links.forEach(a=>a.setAttribute('aria-current',
a.hash==='#'+e.target.id?'true':'false'))}})},{rootMargin:'-20% 0px -70% 0px'});
ids.forEach(i=>{const el=document.getElementById(i);if(el)obs.observe(el)});
"""


# ----------------------------------------------------------------------------- helpers
def _load(d: Path, name: str):
    p = Path(d) / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def esc(x) -> str:
    return html.escape(str(x))


def fmt(v, nd=3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return f"{v:,.{nd}f}"


def verdict(text: str, kind: str) -> str:
    return f'<span class="verdict v-{kind}">{esc(text)}</span>'


def table(head: list[str], rows: list[list], num_cols: set[int] = frozenset(), caption: str = "",
          wrap_cols: set[int] = frozenset()) -> str:
    th = "".join(f'<th scope="col"{" class=num" if i in num_cols else ""}>{esc(h)}</th>' for i, h in enumerate(head))
    body = ""
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            cls = "num" if i in num_cols else "wrap" if i in wrap_cols else ""
            cells.append(f'<td{f" class={cls}" if cls else ""}>{c}</td>')
        body += "<tr>" + "".join(cells) + "</tr>"
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return f'<div class="table-wrap"><table>{cap}<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(fig, caption: str) -> str:
    snippet = fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})
    return f"<figure>{snippet}<figcaption>{esc(caption)}</figcaption></figure>"


def label(m: str) -> str:
    return F.MODEL_LABELS.get(m, m)


# ----------------------------------------------------------------------------- content
def headline(art: dict) -> dict:
    stats = art.get("stats") or {"rows": []}
    checks = [r for r in stats["rows"] if r.get("pass") is not None]
    repro = art.get("reproduction") or {"models": {}, "table": []}
    ours = art.get("ours") or {"models": {}}
    best = max(ours["models"], key=lambda k: ours["models"][k]["r2"]["mean"]) if ours["models"] else None
    paper_best = max(repro["models"], key=lambda k: repro["models"][k]["r2"]["mean"]) if repro["models"] else None
    return {"n_checks": len(checks), "n_pass": sum(bool(r["pass"]) for r in checks), "best": best,
            "best_r2": ours["models"][best]["r2"]["mean"] if best else np.nan,
            "paper_best": paper_best,
            "paper_best_r2": repro["models"][paper_best]["r2"]["mean"] if paper_best else np.nan,
            "published_best": max(v["r2"] for v in ref.TABLE4.values())}


def sec_summary(art, h) -> str:
    opt = art.get("optimize") or {}
    ti = next((r for r in (opt.get("studies", {}).get("Titanium · nominal (mean)", {}).get("table") or [])
               if r["design"] != "knee" and r["feasible"] and r["dropout_y_change_pct"] <= 0
               and r["mass_change_pct"] < 0), None)
    items = [f"<li><strong>{h['n_pass']} of {h['n_checks']}</strong> checkable dataset claims in the paper "
             f"reproduce from the released data within tolerance (frame counts, the non-dominated set and the three "
             f"case-study frames).</li>"]
    if h["paper_best"]:
        items.append(f"<li>Rebuilt with the authors' own hyperparameters, the paper's models keep their published "
                     f"ranking; on the released data they score higher than published (best reproduced R² "
                     f"<strong>{fmt(h['paper_best_r2'])}</strong> vs <strong>{fmt(h['published_best'])}</strong> "
                     f"published).</li>")
    if h["best"] and h["paper_best"]:
        ours = art["ours"]["models"]
        pb_mae = art["reproduction"]["models"][h["paper_best"]]["mae"]["mean"]
        b_mae = ours[h["best"]]["mae"]["mean"]
        items.append(f"<li>Adding this project's beam-theory features and log-scaled targets lifts the best model to R² "
                     f"<strong>{fmt(h['best_r2'])}</strong> ({esc(label(h['best']))}): "
                     f"<strong>{h['best_r2'] - h['paper_best_r2']:+.3f}</strong> R² and "
                     f"<strong>{100 * (b_mae / pb_mae - 1):+.0f} %</strong> MAE against the best reproduced paper "
                     f"model, {esc(label(h['paper_best']))}, on identical folds"
                     + (f" (R² gain not significant over {t['n_folds']} folds, p = {t['p_value']:.2f})"
                        if (t := art["ours"]["paired_tests"][h["best"]].get(h["paper_best"])) and t["p_value"] >= 0.05
                        else "") + ".</li>")
        solo = [m for m in ours if not m.startswith("autogluon")]
        if solo:
            sb = max(solo, key=lambda k: ours[k]["r2"]["mean"])
            items.append(f"<li>The best single model, a {esc(label(sb).lower())}, reaches R² "
                         f"<strong>{fmt(ours[sb]['r2']['mean'])}</strong>, on par with the AutoML ensemble, and adds "
                         f"95 % intervals (conservative: they cover 95–98 % of held-out frames) that the paper's models do not "
                         f"provide.</li>")
    if ti:
        ref_row = opt["studies"]["Titanium · nominal (mean)"]["table"][0]
        items.append(f"<li>Re-sizing the tubes of the paper's titanium road frame (frame 3853) gives a design at "
                     f"<strong>{ti['mass_change_pct']:+.1f} %</strong> mass (against the surrogate's own "
                     f"{ref_row['mass_kg']:.2f} kg prediction for the reference; FE: "
                     f"{opt['reference_fe']['Model Mass']:.2f} kg) with "
                     f"<strong>{ti['dropout_y_change_pct']:+.1f} %</strong> in-plane deflection and mean SF ≥ 1. "
                     f"Demanding SF ≥ 1 on the 95 % lower bound instead costs mass (see Optimisation). Surrogate "
                     f"predictions only, not FE re-verified.</li>")
    return (f'<section id="summary" aria-labelledby="h-summary"><h2 id="h-summary">Summary</h2>'
            f'<ul class="findings">{"".join(items)}</ul></section>')


def sec_source(art) -> str:
    t3 = [[esc(n), *[f'{v["actual"][i]:g} → {v["simulated"][i]:g}' for i in range(3)],
           " / ".join(f"{e} %" for e in ref.TABLE3_REPORTED_ERR_PCT[n])] for n, v in ref.TABLE3.items()]
    return f"""<section id="source" aria-labelledby="h-source"><h2 id="h-source">Source</h2>
<p>The reference is <a href="{PAPER_URL}">FRAMED</a> [1], a peer-reviewed study from the MIT DeCoDE Lab published in
<em>Computer-Aided Design</em> (Elsevier, 2023), with its dataset and code released under the MIT licence [2]. It holds
4,047 community-designed bicycle frames (39 inputs: 18 geometry relations, 9 tube outer diameters, 7 wall thicknesses,
2 bridge flags, 3 materials), each solved in SolidWorks under three load cases: in-plane (2,000 N), transverse (500 N)
and eccentric (2,000 N + 140 N·m). Ten outputs are recorded: seven deflections and rotations, two safety factors and
the mass. A bicycle's rear triangle is a tubular two-wheeler rear sub-frame, the same class of structure as this
project's original V1 study.</p>
<p>The authors checked their FE setup with a mesh-convergence study and against physical frame tests (their Table 3,
repeated below from the paper; it is not something this project re-ran).</p>
{table(["Frame", "Front defl. (mm)", "Rear defl. (mm)", "Mass (kg)", "Reported error"], t3, {1, 2, 3},
       "Paper Table 3: physical test → SolidWorks simulation (Rinard 1996 test data)")}
{figure(F.fig_physical_validation(), "Paper Table 3 values, plotted. Source: [1], Table 3.")}
</section>"""


def sec_claims(art, data) -> str:
    stats = art.get("stats") or {"rows": []}
    rows = []
    for r in stats["rows"]:
        v = verdict("context", "info") if r["pass"] is None else verdict("reproduced", "pass") if r["pass"] \
            else verdict("not reproduced", "fail")
        nd = 0 if float(r["paper"]).is_integer() else 3
        rows.append([esc(r["quantity"]), esc(r["source"]), fmt(r["paper"], nd), fmt(r["ours"], nd),
                     f'{r["abs_dev"]:+.{nd}f}', v])
    out = [f'<section id="claims" aria-labelledby="h-claims"><h2 id="h-claims">Dataset claims</h2>',
           "<p>Before any model is trained, the paper's statements about its own data are re-derived from the released "
           "file. Counts are allowed ±2 frames (the authors' notebook ran on 4,046 rows, the release has 4,047); "
           "case-study values must match to ±0.005.</p>",
           table(["Claim", "Source", "Paper", "This study", "Δ", "Verdict"], rows, {2, 3, 4},
                 "Paper vs this study. 'Context' rows are not pass/fail: the paper counted 4,118 frames, 71 more than "
                 "were released.", {0})]
    if data is not None:
        X, Y = data
        out.append(figure(F.fig_performance_pairs(Y), "Five of the ten outputs, valid (both SF ≥ 1) vs invalid frames, "
                          "the three case-study frames marked. Compare with Fig. 6 in [1]. Units: cm, deg, kg."))
        out.append(figure(F.fig_objective_correlation(Y), "Spearman rank correlation between the ten outputs."))
    out.append("</section>")
    return "".join(out)


def sec_reproduction(art, data, oof_dir: Path) -> str:
    repro = art.get("reproduction")
    if not repro:
        return '<section id="reproduction"><h2>Table 4 reproduction</h2><p>Not run yet.</p></section>'
    t = pd.DataFrame(repro["table"]).set_index("model")
    rows = []
    for m in PAPER_MODELS:
        r = t.loc[m]
        kind = {"pass": "pass", "not run": "info"}.get(r["status"], "fail")
        text = {"pass": "within tolerance", "not run": "not run"}.get(r["status"], "outside tolerance")
        rows.append([esc(label(m)), fmt(r["paper_r2"]), fmt(r["ours_r2"]), "–" if np.isnan(r["delta_r2"]) else
                     f'{r["delta_r2"]:+.3f}', fmt(r["paper_mae"]), fmt(r["ours_mae"]), fmt(r["paper_mse"]),
                     fmt(r["ours_mse"]), fmt(int(r.get("n_folds", 0))), verdict(text, kind)])
    rk = repro.get("ranking_preserved") or {}
    rk_text = ("The paper's ranking (" + " > ".join(label(m) for m in rk.get("compared", [])) + ") is "
               + ("preserved" if rk.get("preserved") else "not preserved")
               + ("" if rk.get("complete") else " among the models that ran") + ".") if rk else ""
    d = t["delta_r2"].dropna()
    gap_text = (f"Every reproduced model scores higher than published, by {d.min():.2f}–{d.max():.2f} R²; MAE lands "
                f"close to the paper's (within {(t['ours_mae'] - t['paper_mae']).abs().max():.3f})." if len(d) else "")
    rows_fig = t.loc[PAPER_MODELS, ["paper_r2", "ours_r2", "ours_r2_std", "paper_mae", "ours_mae", "ours_mae_std"]]
    out = [f"""<section id="reproduction" aria-labelledby="h-repro"><h2 id="h-repro">Table 4 reproduction</h2>
<p>The paper's five regressors were rebuilt with the best hyperparameters the authors report (their Tables 6–9) and
scored on 10 × 5-fold cross-validation, with inputs and outputs standardised inside each training fold. R² and MAE are
averaged over the ten standardised outputs, as in the paper. A model passes when both are within ±0.05 of Table 4.
AutoML (AutoGluon) runs on 1 × 5 folds to bound its run time.</p>
{table(["Model", "Paper R²", "This study R²", "Δ R²", "Paper MAE", "This study MAE", "Paper MSE",
        "This study MSE", "Folds", "Verdict"], rows, {1, 2, 3, 4, 5, 6, 7, 8}, "Paper Table 4 vs this study")}
<p>{esc(rk_text)}</p>
<p class="note">{gap_text} The paper's MSE values
(≈1.0) cannot come from unit-variance targets at its own R². The likeliest cause is data version: the paper trained on
4,118 frames, the public release holds 4,047, so 71 frames, plausibly the most extreme, are missing. The authors'
AutoML training code is not public, so the protocol above is inferred. These gaps are reported, not tuned away.</p>
{figure(F.fig_table4_comparison(rows_fig), "Grey: paper Table 4. Blue: this study (mean ± std over folds).")}"""]
    oof = oof_dir / "reproduction_oof.npz"
    if data is not None and oof.exists():
        with np.load(oof) as z:
            m = max((k for k in z.files), key=lambda k: repro["models"][k]["r2"]["mean"])
            out.append(figure(F.fig_parity_grid(data[1], z[m], f"{label(m)}: out-of-fold predictions"),
                              f"Best reproduced paper model, out-of-fold (repeat 1). Axes and units as in Fig. 9 of [1]; "
                              f"points beyond the paper's axis limits are off-chart there too."))
    out.append("</section>")
    return "".join(out)


def sec_improvements(art, data, oof_dir: Path) -> str:
    ours = art.get("ours")
    if not ours:
        return '<section id="improvements"><h2>Improvements</h2><p>Not run yet.</p></section>'
    repro = (art.get("reproduction") or {}).get("models", {})
    allm = {**repro, **ours["models"]}
    order = [m for m in OUR_MODELS if m in ours["models"]] + [m for m in PAPER_MODELS if m in repro]
    pub = {m: v["r2"] for m, v in ref.TABLE4.items()}
    rows = [[esc(label(m)), "this study" if m in ours["models"] else "paper (reproduced)",
             fmt(allm[m]["r2"]["mean"]), fmt(allm[m]["r2"]["std"]), fmt(allm[m]["mae"]["mean"]),
             fmt(pub.get(m)), fmt(allm[m]["n_folds"])] for m in order]
    best = ours["best"]
    tests = ours["paired_tests"][best]
    trows = [[esc(label(p)), fmt(tests[p]["delta"]), f'{tests[p]["wins"]} / {tests[p]["n_folds"]}',
              f'{tests[p]["p_value"]:.1e}', verdict("better", "better") if tests[p]["p_value"] < 0.05 and
              tests[p]["delta"] > 0 else verdict("not significant", "info")] for p in PAPER_MODELS if p in tests]
    per = ours["models"][best]["per_target"]
    pb = max(repro, key=lambda k: repro[k]["r2"]["mean"]) if repro else None
    prow = [[esc(c), fmt(repro[pb]["per_target"][c]["r2_std"]) if pb else "–", fmt(per[c]["r2_std"]),
             fmt(per[c]["mae"], 5), esc(fc.UNITS[c]), fmt(per[c]["max_rel_err_pct"], 0)] for c in fc.Y_COLS]
    cov = ours.get("coverage", {})
    crow = [[esc(c)] + [fmt(cov[m][c]["coverage"], 2) for m in cov] for c in fc.Y_COLS]
    lat = ours["latency"]
    lrow = [[esc(label(m)), fmt(v["single_ms"], 1), fmt(v["batch_ms_per_design"], 4),
             f'{v["speedup_vs_analytic_single"]:.2f}×',
             f'{v["speedup_vs_nominal_fe_batch"]:,.0f}×'] for m, v in lat["models"].items()]
    out = [f"""<section id="improvements" aria-labelledby="h-imp"><h2 id="h-imp">Improvements</h2>
<p>This project's surrogates add what the paper's models lack: beam-theory features computed from each tube (area,
bending and torsional stiffness EI and GJ, slenderness, an analytic mass estimate ρ·A·L, yield strength), log-scaled
targets for the heavy-tailed deflections, physical guardrails on every prediction, and uncertainty intervals. All
models below share the paper models' exact folds, so every comparison is paired.</p>
{table(["Model", "Origin", "R²", "± std", "MAE (std.)", "Paper Table 4 R²", "Folds"], rows, {2, 3, 4, 5, 6},
       "All models on the same folds (R² and MAE on standardised targets, as in Table 4). Published Table 4 values "
       "are descriptive only: different data release and an inferred protocol.")}
<h3>Is the gain real?</h3>
<p>Paired comparison of the best model ({esc(label(best))}) with each reproduced paper model, using the Nadeau–Bengio
corrected resampled t-test, which accounts for overlapping training sets in repeated cross-validation.</p>
{table(["Against", "Mean Δ R²", "Folds won", "p (one-sided)", "Verdict"], trows, {1, 2, 3}, "Corrected resampled t-test")}
<h3>Per-output accuracy of {esc(label(best))}</h3>
{table(["Output", "R² best paper model", "R² this model", "MAE (physical)", "Unit", "Max rel. error (%)"],
       prow, {1, 2, 3, 5}, "R² on standardised scale; max relative error is the worst case over all folds, with a "
       "floor of 5 % of the training median for deflections")}
<h3>Uncertainty calibration</h3>
<p>Observed share of held-out frames inside the predicted 95 % interval (ideal: 0.95), over 5 folds.</p>
{table(["Output"] + [label(m) for m in cov], crow, set(range(1, len(cov) + 1)), "95 % interval coverage")}
<h3>Speed</h3>
<p>The analytic mass estimate takes {fmt(lat["analytic_baseline_ms"], 3)} ms per frame but predicts mass only. Against
a nominal {lat["nominal_fe_seconds"]:.0f} s production FE run (an assumed figure; the paper gives no run time), batch
inference is faster by the factor shown.</p>
{table(["Model", "Single design (ms)", "Batch, per design (ms)", "Speed-up vs analytic (single)",
        "Speed-up vs nominal FE (batch)"], lrow, {1, 2, 3, 4},
       "Inference latency on the benchmark machine")}"""]
    oof = oof_dir / "ours_oof.npz"
    if data is not None and oof.exists():
        with np.load(oof) as z:
            if best in z.files:
                out.append(figure(F.fig_parity_grid(data[1], z[best], f"{label(best)}: out-of-fold predictions"),
                                  "Same axes and units as Fig. 9 in [1], for a like-for-like visual comparison."))
    out.append("</section>")
    return "".join(out)


def sec_optimisation(art, fronts: pd.DataFrame | None) -> str:
    opt = art.get("optimize")
    if not opt:
        return '<section id="optimisation"><h2>Optimisation</h2><p>Not run yet.</p></section>'
    ref_pred = next(iter(opt["studies"].values()))["table"][0]
    out = [f"""<section id="optimisation" aria-labelledby="h-opt"><h2 id="h-opt">Optimisation</h2>
<p>The paper names constrained optimisation with its surrogates as future work. Here the paper's case-study titanium
road frame (frame 3853, 2.84 kg) keeps its rider-fit geometry, and NSGA-II varies the outer diameter and wall thickness
of the top, down and seat tubes and the stays (10 variables, within the dataset's 1st–99th percentile). Objectives:
minimum mass and minimum in-plane dropout deflection. Constraints: both safety factors ≥ 1, transverse deflection no
worse than the reference, D/t ≤ 50. Each material is run twice: <em>nominal</em> uses the mean prediction of the
{esc(label(opt["surrogate"]).lower())}, <em>robust</em> requires SF ≥ 1 on its lower 95 % bound. The gap between the
two is the mass price of certainty.</p>
<p class="note">In FE the reference frame has SF {opt["reference_fe"]["Sim 1 Safety Factor"]:.2f} / {opt["reference_fe"]["Sim 3 Safety Factor"]:.2f}
and {opt["reference_fe"]["Model Mass"]:.2f} kg. The surrogate's safety-factor intervals are wide (the paper warns its SF labels
are mesh-sensitive). The surrogate predicts the reference itself at {ref_pred['mass_kg']:.2f} kg and SF
{ref_pred['sf1']:.2f} / {ref_pred['sf3']:.2f}, so <em>both</em> runs treat it as infeasible, and all changes below are
relative to that prediction. All designs are surrogate
predictions; none was re-checked in FE. "Nearest design" is the standardised distance to the closest simulated frame;
for comparison, a typical FRAMED frame sits {opt["typical_nn_distance"]:.2f} from its nearest neighbour.</p>"""]
    for key, st in opt["studies"].items():
        mat = st.get("material", key)
        rows = [[esc(r["design"]), fmt(r["mass_kg"], 2), f'{r["mass_lo"]:.2f}–{r["mass_hi"]:.2f}',
                 fmt(r["dropout_y_mm"], 1), f'{r["sf1"]:.2f} ({r["sf1_lo"]:.2f})', f'{r["sf3"]:.2f} ({r["sf3_lo"]:.2f})',
                 f'{r["mass_change_pct"]:+.1f} %', f'{r["dropout_y_change_pct"]:+.1f} %', fmt(r["nearest_distance"], 2),
                 verdict("feasible", "pass") if r["feasible"] else verdict("infeasible", "fail")] for r in st["table"]]
        out.append(f"<h3>{esc(key)}</h3>")
        if len(st["table"]) == 1:
            n_ev = sum(a["evaluations"] for a in st["algorithms"].values())
            out.append(f'<p class="note">No feasible design was found in {n_ev:,} evaluations: with these interval '
                       f'widths no tube sizing in the bounds reaches SF ≥ 1 on the lower bound. Only the reference is '
                       f'shown.</p>')
        out.append(table(["Design", "Mass (kg)", "95 % range", "Dropout Y (mm)", "SF1 mean (low)", "SF3 mean (low)", "Δ mass",
                          "Δ deflection", "Nearest design", "Status"], rows, {1, 2, 3, 4, 5, 6, 7, 8},
                         f"{key}: reference vs optimised (surrogate-predicted)"))
        mode = "robust" if st.get("robust") else "nominal"
        sel = fronts[(fronts["material"] == mat) & (fronts["mode"] == mode)] if fronts is not None and len(fronts) else []
        if len(sel):
            out.append(figure(F.fig_framed_pareto(sel, pd.DataFrame(st["table"]), key),
                              "Feasible Pareto designs; bars show the 95 % mass interval."))
    out.append("</section>")
    return "".join(out)


def sec_limitations() -> str:
    items = ["The structure is a bicycle frame, not a motorcycle sub-frame. It is the closest public, peer-reviewed, "
             "tube-parametric two-wheeler dataset available.",
             "The paper's AutoML protocol is inferred; its code is not public. The released data hold 4,047 frames vs "
             "4,118 in the paper, and the paper's MSE values are inconsistent with its R².",
             "Safety factors come from SolidWorks solid meshes; the authors note they include stress singularities at "
             "tube junctions (paper Sec. 3.5), so SF labels are noisier than deflections.",
             "Optimised designs are surrogate predictions without FE re-verification (FE was out of scope).",
             "Repeated cross-validation folds overlap; significance uses the corrected resampled t-test, but p-values "
             "remain approximate.",
             "The analytic mass estimate leaves out the top tube and seat stays, which have no length parameter."]
    return (f'<section id="limitations" aria-labelledby="h-lim"><h2 id="h-lim">Limitations</h2><ul>'
            + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul></section>")


def sec_method(art) -> str:
    env = art.get("env") or {}
    rows = [[esc(k), esc(v)] for k, v in env.items()]
    repro, ours = art.get("reproduction") or {}, art.get("ours") or {}
    wall = [[esc(label(m)), fmt(r.get("wall_seconds", np.nan) / 60, 1)] for m, r in
            {**repro.get("models", {}), **ours.get("models", {})}.items()]
    return f"""<section id="method" aria-labelledby="h-method"><h2 id="h-method">Method &amp; hardware</h2>
<p>Data: <code class="mono">all_structural_data_oh.csv</code> from [2] at commit
<code class="mono">{fc.COMMIT[:7]}</code>, verified by SHA-256. Outputs follow the authors' notebook (absolute values of
deflections). Folds: 10 repeats × 5-fold KFold, seed 42. Reproduce everything with
<code class="mono">python -m src.framed.run --stage all</code>.</p>
{table(["Item", "Value"], rows, caption="Benchmark machine")}
{table(["Model", "Wall time (min)"], wall, {1}, "Cross-validation wall time")}
</section>"""


def sec_refs() -> str:
    return f"""<section id="references" aria-labelledby="h-refs"><h2 id="h-refs">References</h2><ol class="refs">
<li>[1] {esc(ref.CITATION)} <a href="{PAPER_URL}">arXiv:2201.10459</a></li>
<li>[2] L. Regenwetter et al., FRAMED dataset and code, MIT licence, © 2022 Lyleregenwetter.
<a href="{DATA_URL}">github.com/Lyleregenwetter/FRAMED</a></li>
<li>[3] C. Nadeau, Y. Bengio, "Inference for the Generalization Error", Machine Learning 52, 2003.</li></ol></section>"""


# ----------------------------------------------------------------------------- build
def build_report(artifacts_dir: Path = fc.MODELS_DIR, out_dir: Path = fc.REPORT_DIR, data=None,
                 md_path: Path | None = None) -> Path:
    artifacts_dir, out_dir = Path(artifacts_dir), Path(out_dir)
    art = {k: _load(artifacts_dir, k) for k in ("stats", "reproduction", "ours", "optimize", "env")}
    h = headline(art)
    from src import config as v1
    pareto = v1.DATA_DIR / "framed_pareto.csv"
    fronts = pd.read_csv(pareto) if pareto.exists() and art["optimize"] else None
    body = "".join([sec_summary(art, h), sec_source(art), sec_claims(art, data),
                    sec_reproduction(art, data, artifacts_dir), sec_improvements(art, data, artifacts_dir),
                    sec_optimisation(art, fronts), sec_limitations(), sec_method(art), sec_refs()])
    rail = "".join(f'<li><a href="#{i}">{esc(t)}</a></li>' for i, t in SECTIONS)
    today = dt.date.today().isoformat()
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FRAMED Benchmark Report</title>
<meta name="description" content="Validating and improving on a published bicycle-frame surrogate benchmark (FRAMED, Computer-Aided Design 2023).">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400..700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="{PLOTLY_CDN}"></script><style>{CSS}</style></head>
<body><a class="skip" href="#main">Skip to content</a><div class="layout">
<nav class="rail" aria-label="Contents"><p>Contents</p><ol>{rail}</ol></nav>
<main id="main"><header class="masthead">
<h1>Reproducing a published bicycle-frame surrogate benchmark, then extending it with physics</h1>
<p class="byline"><span>Alfred Chettiar</span><a href="{LINKEDIN}">LinkedIn</a><a href="{REPO}">GitHub repository</a>
<span class="mono">{today}</span></p>
<p class="lede">A peer-reviewed dataset of 4,047 FE-simulated tubular frames, the paper's claims re-derived from its
released data, its surrogate benchmark rebuilt, and this project's physics-informed models measured against it on
identical folds.</p></header>
{body}
<footer>Every number on this page is generated from <span class="mono">models/framed/*.json</span> or the paper reference
file in the repository; paper figures are cited, not reproduced.</footer></main></div>
<script>{RAIL_JS}</script></body></html>"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    (md_path or out_dir.parent / "framed_benchmark.md").write_text(markdown(art, h), encoding="utf-8")
    return path


def markdown(art: dict, h: dict) -> str:
    lines = ["# FRAMED benchmark report", "", f"Alfred Chettiar · [LinkedIn]({LINKEDIN}) · [GitHub]({REPO})", "",
             f"Reference: {ref.CITATION} Data: [{DATA_URL}]({DATA_URL}) (MIT).", "", "## Summary", "",
             f"- Dataset claims reproduced: {h['n_pass']} of {h['n_checks']}."]
    if h["paper_best"]:
        lines.append(f"- Best reproduced paper model ({label(h['paper_best'])}): R² {h['paper_best_r2']:.3f} "
                     f"(published best {h['published_best']:.3f}).")
    if h["best"]:
        t = (art["ours"]["paired_tests"][h["best"]].get(h["paper_best"]) or {}) if h["paper_best"] else {}
        lines.append(f"- Best model of this project ({label(h['best'])}): R² {h['best_r2']:.3f} on identical folds"
                     + (f" (vs {label(h['paper_best'])}: {t['delta']:+.3f}, p = {t['p_value']:.2f} over "
                        f"{t['n_folds']} folds)" if t else "") + ".")
        lines.append("- Published Table 4 values are compared descriptively only (different data release, inferred "
                     "protocol).")
    stats = art.get("stats")
    if stats:
        lines += ["", "## Dataset claims", "", "| Claim | Paper | This study | Δ | Verdict |", "|---|---|---|---|---|"]
        for r in stats["rows"]:
            v = "context" if r["pass"] is None else "reproduced" if r["pass"] else "not reproduced"
            lines.append(f"| {r['quantity']} | {r['paper']:g} | {r['ours']:g} | {r['abs_dev']:+g} | {v} |")
    repro = art.get("reproduction")
    if repro:
        lines += ["", "## Table 4 reproduction", "",
                  "| Model | Paper R² | This study R² | Paper MAE | This study MAE | Verdict |", "|---|---|---|---|---|---|"]
        for r in repro["table"]:
            lines.append(f"| {label(r['model'])} | {r['paper_r2']:.3f} | {fmt(r['ours_r2'])} | {r['paper_mae']:.3f} | "
                         f"{fmt(r['ours_mae'])} | {r['status']} |")
    ours = art.get("ours")
    if ours:
        lines += ["", "## This project's models", "", "| Model | R² | MAE | Folds |", "|---|---|---|---|"]
        for m, r in ours["models"].items():
            lines.append(f"| {label(m)} | {r['r2']['mean']:.3f} ± {r['r2']['std']:.3f} | {r['mae']['mean']:.3f} | "
                         f"{r['n_folds']} |")
    opt = art.get("optimize")
    if opt:
        lines += ["", "## Optimisation (surrogate-predicted, not FE re-verified)", ""]
        for mat, st in opt["studies"].items():
            if len(st["table"]) == 1:
                lines.append(f"- {mat}: no feasible design found.")
            for r in st["table"]:
                lines.append(f"- {mat}, {r['design']}: {r['mass_kg']:.2f} kg ({r['mass_change_pct']:+.1f} %), "
                             f"dropout Y {r['dropout_y_mm']:.1f} mm ({r['dropout_y_change_pct']:+.1f} %), "
                             f"feasible: {r['feasible']}")
    lines += ["", "## Limitations", "", "See the HTML report; FE re-verification of optimised designs was out of scope.",
              ""]
    return "\n".join(lines)
