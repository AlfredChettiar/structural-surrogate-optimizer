# Surrogate model selection

**Selected: Gaussian process (Matern 5/2 ARD)**

## Decision rule

1. Gate: every target must reach cross-validated R² >= 0.98.
2. A weighted composite score ranks the eligible models (each criterion min-max normalised, 1 = best):

   - stress_max_err_pct: weight 0.25
   - stress_mape_pct: weight 0.2
   - mean_mape_pct: weight 0.15
   - mean_max_err_pct: weight 0.1
   - extrapolation_stress_mape_pct: weight 0.15
   - cv_min_r2: weight 0.05
   - uncertainty: weight 0.05
   - batch_latency_ms: weight 0.05

## Decision trace

- ridge_poly: CV min R2 = 0.9865 -> eligible
- gaussian_process: CV min R2 = 0.9919 -> eligible
- residual_mlp: CV min R2 = 0.9876 -> eligible
- physics_guided_mlp: CV min R2 = 0.9853 -> eligible
- random_forest: CV min R2 = 0.9346 -> rejected (below gate)
- hist_gradient_boosting: CV min R2 = 0.9745 -> rejected (below gate)
- svr: CV min R2 = 0.9556 -> rejected (below gate)
- gaussian_process: composite score 0.946
- ridge_poly: composite score 0.390
- residual_mlp: composite score 0.334
- physics_guided_mlp: composite score 0.184
- Highest composite score -> gaussian_process

## Comparison (test set, 300 designs)

| Model | CV min R² | Test min R² | Stress max err vs FE [%] | Stress MAE [MPa] | UQ | Latency 1 design [ms] | Speedup vs FE (batch) | Fit [s] |
|---|---|---|---|---|---|---|---|---|
| Closed-form hand calc | -0.03367 | -0.03367 | 61.95 | 139.2 | no | 0.1068 | 1.458e+04 | 0 |
| Ridge polynomial (deg 3) | 0.9865 | 0.9905 | 18.5 | 9.416 | no | 0.4513 | 4278 | 0.02213 |
| Gaussian process (Matern 5/2 ARD)  [selected] | 0.9919 | 0.9925 | 12.44 | 8.475 | yes | 0.6827 | 32.39 | 66.76 |
| Residual MLP (PyTorch) | 0.9876 | 0.9865 | 14.41 | 10.69 | no | 0.6141 | 1728 | 9.502 |
| Physics-guided residual MLP | 0.9853 | 0.9899 | 20.34 | 9.439 | no | 0.501 | 2534 | 6.199 |
| Random forest | 0.9346 | 0.9413 | 31.04 | 22.75 | no | 6.129 | 236.6 | 0.1979 |
| Hist gradient boosting | 0.9745 | 0.9778 | 19.01 | 13.94 | no | 11.36 | 106.6 | 1.487 |
| Support vector regression (RBF) | 0.9556 | 0.9181 | 43.33 | 18.2 | no | 0.7803 | 25.35 | 4.222 |

Label-noise floor: the 3 % measurement scatter alone produces a worst-case stress deviation of 8.7 % against the noise-free FE, so errors near that level are at the data's resolution.

Measured speedup is against the in-repo beam FE solver (4.4 ms/design). Against a nominal 180 s production FE run the projected speedup of the selected model is ~263,678x.
