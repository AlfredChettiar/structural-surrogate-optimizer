# Optimiser selection

Surrogate: `gaussian_process` | scenario: yield 355 MPa, DLF 3.0

**Selected: NSGA-II**

## Score weights

- hypervolume_fe_verified: 0.3
- hypervolume: 0.2
- igd: 0.15
- fe_feasible_rate: 0.2
- evals_to_99pct_hv: 0.1
- wall_seconds: 0.05

Every claimed Pareto design (up to 60 per optimiser) is re-solved with the FE model; the FE-verified hypervolume and FE-feasible rate measure what survives contact with the physics.
Indicators are means over seeds [42, 43, 44].

## Comparison

| Optimiser | Score | Hypervolume | HV std (seeds) | FE-verified HV | IGD | Spacing | FE-feasible rate | Pareto points | Evaluations | Evals to 99% HV | Wall time [s] | Stress err vs FE [%] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NSGA-II  [selected] | 0.9135 | 1.01 | 0.0009803 | 1.001 | 0.001472 | 0.0009825 | 1 | 705.7 | 5000 | 2459 | 0.8615 | 1.385 |
| epsilon-constraint + DE | 0.7795 | 0.993 | 0.001643 | 0.9883 | 0.01235 | 0.02177 | 1 | 104.3 | 1.248e+04 | 3385 | 1.016 | 1.424 |
| Weighted-sum SLSQP | 0.7638 | 0.9859 | 0.006881 | 0.9834 | 0.01738 | 0.001848 | 1 | 308.3 | 1646 | 726.3 | 1.468 | 1.757 |
| ParEGO (Bayesian) | 0.3029 | 0.9293 | 0.01093 | 0.9259 | 0.05902 | 0.03776 | 1 | 18 | 150 | 113.3 | 6.98 | 1.047 |
| LHS screening (5,000) | 0.2753 | 0.9281 | 0.003558 | 0.9266 | 0.05042 | 0.05114 | 1 | 24.67 | 5000 | 4097 | 0.8351 | 1.308 |

## Recommended design (knee point)

knee point of the FE-verified front (best-balanced mass/stress compromise): **1.1 % lighter** than baseline, peak stress -33.7 %, safety factor 2.29 (FE).

- Tube OD [mm]: 38.00
- Wall thickness [mm]: 2.00
- Gusset thickness [mm]: 5.00
- Bend radius [mm]: 89.05

## Headline

Lightest FE-feasible design: **14.6 % lighter** than the baseline (3.742 -> 3.197 kg), peak stress -19.3 %.

| Geometry | Baseline | Optimised |
|---|---|---|
| Tube OD [mm] | 32.00 | 35.77 |
| Wall thickness [mm] | 2.50 | 1.79 |
| Gusset thickness [mm] | 4.00 | 4.83 |
| Bend radius [mm] | 60.00 | 84.23 |
