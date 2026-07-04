# Ordinary Convolution Kernel Sweep Analysis

## Configuration

Every convolution control uses `Conv1d(2, 32)`, stride 1, groups 1, no bias,
and odd-kernel same padding. The exact kernel/padding pairs are `(3,1)`,
`(7,3)`, `(15,7)`, `(31,15)`, and `(51,25)`. All controls share the same
log-absolute compression, BatchNorm, GELU, 1x1 projection, and IQ-feature fusion
path. Therefore, kernel size is the only changed convolution hyperparameter.

All results are mean ± sample standard deviation over seeds 1--5. Paired 95%
t intervals use the same seeds and four degrees of freedom and should be
interpreted as small-sample uncertainty summaries.

## Accuracy summary

| Dataset | Variant | Overall accuracy (%) | Conv minus LKF (pp) | Paired 95% CI (pp) |
| --- | --- | ---: | ---: | ---: |
| 2016.10A | Conv(k=3) | 63.04 ± 0.24 | -0.136 | [-0.590, 0.318] |
| 2016.10A | Conv(k=7) | 63.11 ± 0.16 | -0.068 | [-0.369, 0.232] |
| 2016.10A | Conv(k=15) | 63.33 ± 0.17 | +0.145 | [-0.214, 0.505] |
| 2016.10A | Conv(k=31) | 63.36 ± 0.36 | +0.180 | [-0.350, 0.710] |
| 2016.10A | Conv(k=51) | 63.15 ± 0.10 | -0.027 | [-0.389, 0.334] |
| 2016.10A | Full LKF(k=31) | 63.18 ± 0.22 | 0.000 | reference |
| 2016.10B | Conv(k=3) | 68.09 ± 0.32 | +2.576 | [2.132, 3.020] |
| 2016.10B | Conv(k=7) | 65.06 ± 0.11 | -0.455 | [-0.560, -0.349] |
| 2016.10B | Conv(k=15) | 64.98 ± 0.08 | -0.534 | [-0.773, -0.295] |
| 2016.10B | Conv(k=31) | 65.18 ± 0.06 | -0.333 | [-0.524, -0.141] |
| 2016.10B | Conv(k=51) | 65.24 ± 0.16 | -0.277 | [-0.413, -0.141] |
| 2016.10B | Full LKF(k=31) | 65.51 ± 0.13 | 0.000 | reference |

## Configurations reliably exceeded by LKF

On RadioML2016.10B, Full LKF(k=31) reliably exceeds:

- Conv(k=7, padding=3) by 0.45 pp.
- Conv(k=15, padding=7) by 0.53 pp.
- Conv(k=31, padding=15) by 0.33 pp.
- Conv(k=51, padding=25) by 0.28 pp.

All four paired intervals exclude zero. The strongest mean advantage is against
Conv(k=15). LKF does not exceed Conv(k=3); Conv(k=3) is 2.58 pp higher.

On RadioML2016.10A, LKF has slightly higher means than k=3, k=7, and k=51,
but every interval crosses zero. Therefore, no tested convolution configuration
is reliably exceeded by LKF on 10A.

## SNR behavior

On 10B, LKF improves both low- and high-SNR averages over k=7, k=15, k=31,
and k=51. The low-SNR gains are approximately 0.55, 0.78, 0.46, and 0.29 pp,
respectively. The corresponding high-SNR gains are approximately 0.33, 0.23,
0.17, and 0.25 pp.

Conv(k=3) is the exception: its low-SNR accuracy is 47.14%, compared with
42.28% for LKF, while high-SNR accuracy is similar. This indicates a strong
dataset-specific preference for a very short local kernel on 10B.

## Complexity and claim boundary

Full LKF uses 0.1284 M parameters and 32.02 M FLOPs on 10B. The k=7--51
convolution controls use 0.1189--0.1217 M parameters and 29.59--30.31 M FLOPs.
Thus, LKF's advantage over k=7/15/31/51 is an accuracy advantage, not an
efficiency advantage.

The manuscript may state that, on RadioML2016.10B, LKF outperforms matched
ordinary convolution over medium-to-large tested receptive fields, including
the directly matched k=31 control. It must also disclose that Conv(k=3) is
substantially better and that the advantage does not reproduce on 10A. It would
be misleading to report only the weaker convolution kernels.
