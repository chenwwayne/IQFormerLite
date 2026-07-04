# Conv(k=3) Five-Seed Comparison

| Dataset | Variant | Params (M) | FLOPs (M) | Low SNR | High SNR | Overall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2016.10a | conv_k3 | 0.1187 | 29.5244 | 39.10 ± 0.36 | 92.12 ± 0.34 | 63.04 ± 0.24 |
| 2016.10b | conv_k3 | 0.1187 | 29.5242 | 47.14 ± 0.57 | 93.56 ± 0.03 | 68.09 ± 0.32 |
| 2016.10a | conv_k31 | 0.1205 | 29.9831 | 39.19 ± 0.40 | 92.68 ± 0.39 | 63.36 ± 0.36 |
| 2016.10a | full_k31 | 0.1285 | 32.0198 | 39.11 ± 0.23 | 92.36 ± 0.45 | 63.18 ± 0.22 |
| 2016.10b | conv_k31 | 0.1205 | 29.9830 | 41.82 ± 0.11 | 93.58 ± 0.04 | 65.18 ± 0.06 |
| 2016.10b | full_k31 | 0.1284 | 32.0197 | 42.28 ± 0.22 | 93.75 ± 0.05 | 65.51 ± 0.13 |

## Paired differences

| Dataset | Variant A | Variant B | Delta A-B (pp) | 95% CI (pp) |
| --- | --- | --- | ---: | ---: |
| 2016.10a | conv_k3 | conv_k31 | -0.316 | [-0.620, -0.013] |
| 2016.10a | conv_k3 | full_k31 | -0.136 | [-0.590, +0.318] |
| 2016.10b | conv_k3 | conv_k31 | +2.909 | [+2.520, +3.297] |
| 2016.10b | conv_k3 | full_k31 | +2.576 | [+2.132, +3.020] |

Conv(k=3) is a conventional small-receptive-field baseline. It is not a receptive-field-matched replacement for the existing Conv(k=31) control.
