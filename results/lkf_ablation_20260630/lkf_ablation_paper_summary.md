# LKF Ablation Study

All accuracy values are reported as mean ± sample standard deviation over five
random seeds. The ordinary-convolution baseline uses one fixed configuration
across all seeds and datasets; implementation details are retained in the
experiment archive.

## RadioML2016.10A

| Variant | Params (M) | FLOPs (M) | Low SNR (%) | High SNR (%) | Overall (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| w/o LKF | 0.1109 | 27.5439 | 39.01 ± 0.07 | 92.01 ± 0.29 | 62.96 ± 0.13 |
| Ordinary Conv | 0.1195 | 29.7210 | 39.21 ± 0.25 | 92.56 ± 0.21 | **63.33 ± 0.17** |
| Base-only | 0.1205 | 29.9831 | **39.39 ± 0.19** | 92.12 ± 0.26 | 63.20 ± 0.07 |
| RBF-only | 0.1265 | 31.5119 | 39.13 ± 0.37 | 92.25 ± 0.33 | 63.15 ± 0.18 |
| B-spline KAN | 0.1344 | 33.6111 | 38.26 ± 0.22 | 90.97 ± 0.27 | 62.10 ± 0.05 |
| Full LKF | 0.1285 | 32.0198 | 39.11 ± 0.23 | 92.36 ± 0.45 | 63.18 ± 0.22 |

On RadioML2016.10A, the differences among Ordinary Conv, Base-only, RBF-only,
and Full LKF are small relative to the cross-seed variation. Full LKF is 0.15
percentage points below Ordinary Conv, and the paired uncertainty interval
includes zero. Therefore, these results do not establish an accuracy advantage
for Full LKF on this dataset.

The B-spline alternative is consistently weaker: Full LKF improves overall
accuracy by 1.09 percentage points while using fewer parameters and FLOPs. This
supports the selected RBF formulation over the tested B-spline construction.

## RadioML2016.10B

| Variant | Params (M) | FLOPs (M) | Low SNR (%) | High SNR (%) | Overall (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary Conv | 0.1194 | 29.7208 | 41.50 ± 0.15 | 93.52 ± 0.06 | 64.98 ± 0.08 |
| RBF-only | 0.1264 | 31.5118 | **42.39 ± 0.15** | **93.79 ± 0.04** | **65.58 ± 0.09** |
| Full LKF | 0.1284 | 32.0197 | 42.28 ± 0.22 | 93.75 ± 0.05 | 65.51 ± 0.13 |

On RadioML2016.10B, Full LKF exceeds Ordinary Conv by 0.53 percentage points.
The same-seed 95% uncertainty interval is [0.30, 0.77] percentage points, so the
advantage is consistent across the five runs. The gain appears in both the
low- and high-SNR averages, with a larger improvement at low SNR.

RBF-only and Full LKF remain statistically overlapping. RBF-only is 0.07
percentage points higher on average while requiring slightly fewer parameters
and FLOPs. Consequently, the nonlinear RBF path accounts for the principal
benefit observed on this dataset, whereas the base residual path does not show
an independent accuracy gain.

## Overall interpretation

The ablation supports three bounded conclusions:

1. The RBF-based filter formulation provides a reproducible improvement over
   the selected ordinary-convolution baseline on RadioML2016.10B.
2. The RBF formulation is more accurate and computationally lighter than the
   tested B-spline alternative on RadioML2016.10A.
3. The advantage is dataset dependent: no reliable Full-LKF improvement is
   observed on RadioML2016.10A, and Full LKF does not consistently outperform
   RBF-only.

The results should therefore be presented as evidence for a structured RBF
filterbank and a dataset-dependent accuracy--complexity trade-off, rather than
as proof that Full LKF universally outperforms conventional convolution.
