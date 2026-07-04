# LKF Ablation Results

This directory contains the completed five-seed LKF mechanism ablation.

## Experiment matrix

- RadioML2016.10A: `none`, `conv`, `base_only`, `rbf_only`, `bspline`, `full`.
- RadioML2016.10B: `conv`, `rbf_only`, `full`.
- Seeds: 1, 2, 3, 4, 5.
- Shared setup: 60 epochs, batch size 1024, learning rate 0.001, SNR -20 to 18.
- LKF setup: 32 bands, kernel size 31, grid size 4, grid range [-2, 2].

`conv` uses the same kernel size, output channels, log-absolute compression,
normalization, projection, and fusion topology as the complete LKF. The only
intended difference is the filter parameterization.

## Complexity scope

The complexity profiler counts Conv1d/Linear multiplication and addition as
separate operations and counts four operations per BatchNorm element, matching
the approximately 32.06 M convention used by the existing paper table. It then
adds estimated RBF/B-spline basis-construction operations. This is a consistent
software complexity estimate, not an RK3588 deployment measurement. No board
latency or throughput should be inferred from these files.

The existing five-seed Full LKF accuracy archive is reused after confirming that
the default `full` forward path is unchanged. The default runner therefore
launches 35 new jobs. Set `VARIANTS_10A` and `VARIANTS_10B` explicitly if a full
rerun is required.

Run `scripts/profile_lkf_ablation.py` once, then run
`scripts/summarize_lkf_ablation.py --fail-on-missing` after all jobs finish.

## Final outputs

- `lkf_ablation_per_seed.csv`: all 45 records, including the 10 archived Full
  LKF runs used as the reference.
- `lkf_ablation_summary.csv` and `.md`: mean ± sample standard deviation,
  complexity, and deltas relative to Full LKF.
- `lkf_ablation_pairwise.csv`: paired seed differences and 95% t intervals.
- `analysis.md`: interpretation, limitations, and manuscript-safe conclusions.
- `raw_runs/`: the 35 newly completed server run directories.
- `server_logs/`: compressed master execution logs retained for audit.
