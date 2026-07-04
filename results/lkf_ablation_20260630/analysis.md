# LKF Ablation Analysis

## Result integrity

- All 35 newly requested runs completed successfully.
- Every dataset/variant group contains seeds 1, 2, 3, 4, and 5.
- No Traceback, CUDA out-of-memory error, or killed process was found.
- Full LKF uses the previously archived five-seed runs because the updated
  default model strictly loads the original checkpoints with all keys matched.
- Accuracy is reported as mean ± sample standard deviation. Paired intervals
  use the same five training seeds and a two-sided 95% t interval with four
  degrees of freedom; with only five pairs, they should be treated as
  uncertainty summaries rather than definitive significance tests.

## Main results

| Dataset | Variant | Params (M) | FLOPs (M) | Overall accuracy (%) | Delta vs. Full (pp) |
| --- | --- | ---: | ---: | ---: | ---: |
| 2016.10A | w/o LKF | 0.1109 | 27.5439 | 62.96 ± 0.13 | -0.22 |
| 2016.10A | Ordinary Conv | 0.1205 | 29.9831 | 63.36 ± 0.36 | +0.18 |
| 2016.10A | Base-only | 0.1205 | 29.9831 | 63.20 ± 0.07 | +0.02 |
| 2016.10A | RBF-only | 0.1265 | 31.5119 | 63.15 ± 0.18 | -0.04 |
| 2016.10A | B-spline KAN | 0.1344 | 33.6111 | 62.10 ± 0.05 | -1.09 |
| 2016.10A | Full LKF | 0.1285 | 32.0198 | 63.18 ± 0.22 | 0.00 |
| 2016.10B | Ordinary Conv | 0.1205 | 29.9830 | 65.18 ± 0.06 | -0.33 |
| 2016.10B | RBF-only | 0.1264 | 31.5118 | 65.58 ± 0.09 | +0.07 |
| 2016.10B | Full LKF | 0.1284 | 32.0197 | 65.51 ± 0.13 | 0.00 |

## Interpretation

### 1. The full LKF is not uniformly more accurate than ordinary convolution

On RadioML2016.10A, Ordinary Conv is 0.18 pp above Full LKF on average, but the
paired 95% interval for Full minus Conv is [-0.71, 0.35] pp. The result does not
support a reliable difference on 10A.

On RadioML2016.10B, Full LKF is 0.33 pp above Ordinary Conv, with a paired 95%
interval of [0.14, 0.52] pp. The gain is consistent across all five paired
seeds. Full LKF uses about 6.6% more parameters and 6.8% more FLOPs than the
ordinary-convolution control.

Therefore, the evidence supports a dataset-dependent advantage rather than a
claim that LKF universally outperforms ordinary convolution.

### 2. The RBF path accounts for the useful nonlinear behavior

Full LKF and RBF-only are effectively tied on both datasets. Full minus
RBF-only is +0.04 pp on 10A and -0.07 pp on 10B, and both paired intervals
include zero. RBF-only also saves about 1.6% parameters and FLOPs relative to
Full LKF.

On 10B, RBF-only exceeds Ordinary Conv by 0.40 pp, with a paired 95% interval
of [0.28, 0.52] pp. This is the clearest evidence that the nonlinear RBF
parameterization can provide information beyond a matched ordinary large-kernel
convolution. On 10A, however, RBF-only is 0.22 pp below Conv and the interval
crosses zero.

The base residual path therefore does not show a consistent accuracy gain over
RBF-only in these experiments. It should not be described as independently
necessary without additional evidence.

### 3. RBF is substantially better suited than the tested B-spline alternative

On 10A, Full LKF exceeds B-spline KAN by 1.09 pp, with a paired 95% interval of
[0.76, 1.41] pp. B-spline KAN is also larger and more expensive: 0.1344 M
parameters and 33.61 M FLOPs versus 0.1285 M and 32.02 M for Full LKF.

This supports choosing the FastKAN RBF basis over the tested order-3 B-spline
construction under the current signal length and training protocol. It does
not establish that RBF is universally superior to all spline formulations.

### 4. Removing LKF causes only a modest 10A reduction

Full LKF exceeds w/o LKF by 0.22 pp on 10A, but the paired 95% interval is
[-0.10, 0.54] pp. The module-removal comparison also changes the fusion
topology and is not parameter matched, so it should be described as an
architecture-level ablation rather than a clean filter-parameterization test.

The high-SNR gain over w/o LKF is 0.35 pp, while the low-SNR gain is 0.10 pp.
This experiment does not support claiming that LKF primarily solves the
low-SNR limitation.

## Safe manuscript conclusion

The results demonstrate that LKF is structurally and computationally distinct
from an ordinary convolution and that its RBF path provides a reproducible gain
over a matched convolution on RadioML2016.10B. The tested B-spline alternative
is consistently less accurate and more expensive. However, the 10A results are
statistically overlapping, and Full LKF does not consistently outperform
RBF-only. The manuscript should therefore claim a dataset-dependent
accuracy--complexity trade-off and an empirically supported RBF design choice,
not universal superiority or irreplaceability of the complete LKF.
