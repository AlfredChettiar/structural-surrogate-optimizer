# FRAMED benchmark report

Alfred Chettiar · [LinkedIn](https://www.linkedin.com/in/alfred-chettiar-0b3b53189) · [GitHub](https://github.com/AlfredChettiar/structural-surrogate-optimizer)

Reference: L. Regenwetter, C. Weaver, F. Ahmed, "FRAMED: An AutoML Approach for Structural Performance Prediction of Bicycle Frames", Computer-Aided Design (Elsevier), 2023. arXiv:2201.10459. Data: [https://github.com/Lyleregenwetter/FRAMED](https://github.com/Lyleregenwetter/FRAMED) (MIT).

## Summary

- Dataset claims reproduced: 11 of 11.
- Best reproduced paper model (AutoML (AutoGluon)): R² 0.737 (published best 0.605).
- Best model of this project (AutoGluon + physics): R² 0.747 on identical folds.

## Dataset claims

| Claim | Paper | This study | Δ | Verdict |
|---|---|---|---|---|
| Frames: both fail | 1966 | 1967 | +1 | reproduced |
| Frames: sf1 only | 1272 | 1272 | +0 | reproduced |
| Frames: sf3 only | 182 | 182 | +0 | reproduced |
| Frames: valid | 626 | 626 | +0 | reproduced |
| Non-dominated designs (5 objectives) | 76 | 74 | -2 | reproduced |
| Frames failing SF >= 1 in any case | 3419 | 3421 | +2 | context |
| Geometrically valid frames simulated | 4118 | 4047 | -71 | context |
| Frame 3853 mass (kg) | 2.84 | 2.84 | +0 | reproduced |
| Frame 3853 SF1 | 1.29 | 1.29 | +0 | reproduced |
| Frame 2108 mass (kg) | 22.83 | 22.83 | +0 | reproduced |
| Frame 2108 SF1 | 0.09256 | 0.09256 | +0 | reproduced |
| Frame 857 mass (kg) | 1.6 | 1.6 | +0 | reproduced |
| Frame 857 SF1 | 2.416 | 2.416 | +0 | reproduced |

## Table 4 reproduction

| Model | Paper R² | This study R² | Paper MAE | This study MAE | Verdict |
|---|---|---|---|---|---|
| Decision tree | 0.440 | 0.514 | 0.297 | 0.338 | outside tolerance |
| K-nearest neighbours | 0.306 | 0.378 | 0.412 | 0.447 | outside tolerance |
| XGBoost | 0.595 | 0.691 | 0.208 | 0.227 | outside tolerance |
| Neural network | 0.545 | 0.592 | 0.263 | 0.309 | pass |
| AutoML (AutoGluon) | 0.605 | 0.737 | 0.182 | 0.182 | outside tolerance |

## This project's models

| Model | R² | MAE | Folds |
|---|---|---|---|
| XGBoost + physics | 0.705 ± 0.055 | 0.198 | 50 |
| Physics-guided MLP | 0.717 ± 0.048 | 0.203 | 50 |
| Deep ensemble + physics | 0.734 ± 0.048 | 0.189 | 50 |
| Gaussian process + physics | 0.651 ± 0.084 | 0.236 | 50 |
| AutoGluon + physics | 0.747 ± 0.052 | 0.157 | 5 |

## Optimisation (surrogate-predicted, not FE re-verified)

- Titanium · nominal (mean), reference (frame 3853 geometry): 2.55 kg (+0.0 %), dropout Y 27.8 mm (+0.0 %), feasible: False
- Titanium · nominal (mean), knee: 3.51 kg (+37.6 %), dropout Y 4.5 mm (-83.7 %), feasible: True
- Titanium · nominal (mean), lightest feasible: 1.95 kg (-23.8 %), dropout Y 19.3 mm (-30.5 %), feasible: True
- Titanium · robust (95 % lower bound), reference (frame 3853 geometry): 2.55 kg (+0.0 %), dropout Y 27.8 mm (+0.0 %), feasible: False
- Titanium · robust (95 % lower bound), knee: 5.24 kg (+105.5 %), dropout Y 2.6 mm (-90.6 %), feasible: True
- Titanium · robust (95 % lower bound), lightest feasible: 3.07 kg (+20.4 %), dropout Y 5.4 mm (-80.5 %), feasible: True
- Steel · nominal (mean), reference (frame 3853 geometry): 4.48 kg (+0.0 %), dropout Y 12.1 mm (+0.0 %), feasible: False
- Steel · nominal (mean), knee: 8.09 kg (+80.5 %), dropout Y 1.7 mm (-86.3 %), feasible: True
- Steel · nominal (mean), lightest feasible: 3.76 kg (-16.0 %), dropout Y 3.9 mm (-68.1 %), feasible: True
- Steel · robust (95 % lower bound), reference (frame 3853 geometry): 4.48 kg (+0.0 %), dropout Y 12.1 mm (+0.0 %), feasible: False
- Steel · robust (95 % lower bound), knee: 12.30 kg (+174.7 %), dropout Y 1.3 mm (-89.4 %), feasible: True
- Steel · robust (95 % lower bound), lightest feasible: 11.01 kg (+145.8 %), dropout Y 1.5 mm (-87.6 %), feasible: True
- Aluminum · nominal (mean), reference (frame 3853 geometry): 1.81 kg (+0.0 %), dropout Y 37.8 mm (+0.0 %), feasible: False
- Aluminum · nominal (mean), knee: 4.41 kg (+144.1 %), dropout Y 2.8 mm (-92.5 %), feasible: True
- Aluminum · nominal (mean), lightest feasible: 2.29 kg (+26.8 %), dropout Y 6.4 mm (-83.1 %), feasible: True
- Aluminum · robust (95 % lower bound), reference (frame 3853 geometry): 1.81 kg (+0.0 %), dropout Y 37.8 mm (+0.0 %), feasible: False

## Limitations

See the HTML report; FE re-verification of optimised designs was out of scope.
