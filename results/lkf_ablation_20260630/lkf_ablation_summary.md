# LKF Ablation Summary

Accuracy is the mean ± sample standard deviation over five training seeds.
FLOPs use the existing paper convention plus estimated basis-construction operations.
These are server-side complexity estimates, not RK3588 deployment measurements.

| Dataset | Variant | n | Params (M) | FLOPs (M) | Basis ops (M) | Low SNR | High SNR | Overall | Delta vs Full (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2016.10a | none | 5 | 0.1109 | 27.5439 | 0.0000 | 39.01 ± 0.07 | 92.01 ± 0.29 | 62.96 ± 0.13 | -0.22 |
| 2016.10a | conv | 5 | 0.1205 | 29.9831 | 0.0000 | 39.19 ± 0.40 | 92.68 ± 0.39 | 63.36 ± 0.36 | +0.18 |
| 2016.10a | base_only | 5 | 0.1205 | 29.9831 | 0.0000 | 39.39 ± 0.19 | 92.12 ± 0.26 | 63.20 ± 0.07 | +0.02 |
| 2016.10a | rbf_only | 5 | 0.1265 | 31.5119 | 0.0051 | 39.13 ± 0.37 | 92.25 ± 0.33 | 63.15 ± 0.18 | -0.04 |
| 2016.10a | bspline | 5 | 0.1344 | 33.6111 | 0.0727 | 38.26 ± 0.22 | 90.97 ± 0.27 | 62.10 ± 0.05 | -1.09 |
| 2016.10a | full | 5 | 0.1285 | 32.0198 | 0.0051 | 39.11 ± 0.23 | 92.36 ± 0.45 | 63.18 ± 0.22 | +0.00 |
| 2016.10b | conv | 5 | 0.1205 | 29.9830 | 0.0000 | 41.82 ± 0.11 | 93.58 ± 0.04 | 65.18 ± 0.06 | -0.33 |
| 2016.10b | rbf_only | 5 | 0.1264 | 31.5118 | 0.0051 | 42.39 ± 0.15 | 93.79 ± 0.04 | 65.58 ± 0.09 | +0.07 |
| 2016.10b | full | 5 | 0.1284 | 32.0197 | 0.0051 | 42.28 ± 0.22 | 93.75 ± 0.05 | 65.51 ± 0.13 | +0.00 |
