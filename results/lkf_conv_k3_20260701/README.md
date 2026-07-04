# Ordinary Conv(k=3) Control

This experiment compares a conventional small-kernel filter branch against the
existing Conv(k=31) and Full LKF(k=31) five-seed results.

## Conv(k=3) configuration

- Conv1d: 2 input channels, 32 output channels, kernel size 3, stride 1,
  padding 1, groups 1, no bias.
- Common processing: log-absolute compression, BatchNorm, GELU, 1x1 projection,
  BatchNorm, and the same IQ-feature fusion path.
- Datasets: RadioML2016.10A and RadioML2016.10B.
- Seeds: 1, 2, 3, 4, 5.
- Training: 60 epochs, batch size 1024, learning rate 0.001, SNR -20 to 18.

Conv(k=3) is not receptive-field matched to Full LKF(k=31). The original
Conv(k=31) result remains the fair parameterization control.

## Completed outputs

- `conv_k3_per_seed.csv`: five-seed results for both datasets.
- `conv_k3_comparison_summary.csv` and `.md`: comparison with Conv(k=31) and
  Full LKF(k=31).
- `conv_k3_pairwise.csv`: paired seed differences and 95% t intervals.
- `analysis.md`: result interpretation, limitations, and manuscript-safe claims.
- `raw_runs/`: the 10 completed server run directories.
