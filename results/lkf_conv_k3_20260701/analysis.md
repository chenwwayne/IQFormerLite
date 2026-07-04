# Conv(k=3) Control Analysis

## Integrity checks

- All 10 requested runs completed successfully with seeds 1--5 for both datasets.
- No traceback, out-of-memory event, killed process, or missing `Test_ACC.csv` was found.
- All comparisons use the same data split, optimizer, learning rate, batch size,
  validation criterion, and early-stopping rule.
- Training was configured for a maximum of 60 epochs with patience-10 early
  stopping. Conv(k=3) stopped after 32--38 epochs on 10A and 43--52 epochs on
  10B; Conv(k=31) stopped after 32--46 and 29--36 epochs, respectively.
- Paired intervals use the same five seeds and a two-sided 95% t interval with
  four degrees of freedom. They are uncertainty summaries based on a small
  number of runs, not definitive significance claims.

## Main comparison

| Dataset | Variant | Params (M) | FLOPs (M) | Low SNR | High SNR | Overall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2016.10A | Conv(k=3) | 0.1187 | 29.5244 | 39.10 ± 0.36 | 92.12 ± 0.34 | 63.04 ± 0.24 |
| 2016.10A | Conv(k=31) | 0.1205 | 29.9831 | 39.19 ± 0.40 | 92.68 ± 0.39 | 63.36 ± 0.36 |
| 2016.10A | Full LKF(k=31) | 0.1285 | 32.0198 | 39.11 ± 0.23 | 92.36 ± 0.45 | 63.18 ± 0.22 |
| 2016.10B | Conv(k=3) | 0.1187 | 29.5242 | 47.14 ± 0.57 | 93.56 ± 0.03 | 68.09 ± 0.32 |
| 2016.10B | Conv(k=31) | 0.1205 | 29.9830 | 41.82 ± 0.11 | 93.58 ± 0.04 | 65.18 ± 0.06 |
| 2016.10B | Full LKF(k=31) | 0.1284 | 32.0197 | 42.28 ± 0.22 | 93.75 ± 0.05 | 65.51 ± 0.13 |

## Paired results

- On 10A, Conv(k=3) is 0.32 pp below Conv(k=31), with a paired 95% interval
  of [-0.62, -0.01] pp.
- On 10A, Conv(k=3) is 0.14 pp below Full LKF(k=31), with an interval of
  [-0.59, 0.32] pp; the difference is not reliably separated from seed noise.
- On 10B, Conv(k=3) is 2.91 pp above Conv(k=31), with an interval of
  [2.52, 3.30] pp.
- On 10B, Conv(k=3) is 2.58 pp above Full LKF(k=31), with an interval of
  [2.13, 3.02] pp.

## Where the 10B gain comes from

The 10B gain is almost entirely a low-SNR effect. Compared with Conv(k=31),
Conv(k=3) improves the low-SNR average by 5.32 pp while the high-SNR average is
0.02 pp lower. Compared with Full LKF(k=31), the corresponding differences are
+4.86 pp and -0.19 pp.

The largest mean improvements over Conv(k=31) occur at -18 dB (+12.53 pp),
-16 dB (+11.95 pp), -14 dB (+8.75 pp), and -20 dB (+7.65 pp). Above 0 dB,
the three variants are nearly indistinguishable.

This pattern suggests that a short local receptive field is particularly well
matched to low-SNR artifacts or local signal structure in RadioML2016.10B.
Because the same improvement is absent on 10A, it should not be presented as a
general low-SNR solution without validation on an additional dataset or channel
condition.

## Efficiency

Conv(k=3) is also the smallest tested filtered variant. Relative to Conv(k=31),
it reduces parameters by about 1.5% and FLOPs by about 1.5%. Relative to Full
LKF(k=31), it reduces parameters by about 7.6% and FLOPs by about 7.8%.

## Implication for the LKF claim

These results do not support claiming that Full LKF is more accurate or more
necessary than ordinary convolution. The matched Conv(k=31) experiment shows a
10B advantage for the RBF parameterization at the same receptive field, but a
simple Conv(k=3) obtains substantially higher 10B accuracy with lower
complexity. Therefore, using only Conv(k=31) as the conventional-convolution
baseline would omit an important kernel-size sensitivity.

The manuscript-safe conclusion is that LKF offers a structured, interpretable
RBF filterbank and a dataset-dependent trade-off, not universally superior
accuracy. Conv(k=3) and Full LKF(k=31) have different receptive fields, so the
k=3 comparison must be identified as a tuned small-kernel baseline rather than
a parameterization-matched replacement.
