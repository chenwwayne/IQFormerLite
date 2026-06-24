# 5-seed experiment archive

This archive stores the 5-seed offline evaluation results used for the IQFormerLite revision.

## Experiment scope

- Remote source: `/data/iqformer-lite` on `A800-2`
- Local archive: `code/seed_sweep_results_20260624`
- Datasets: `2016.10a`, `2016.10b`
- Models: `IQFormer`, `IQFormerLite`
- Seeds: `1, 2, 3, 4, 5`
- IQFormer setting: `aux_mode=stft`
- IQFormerLite setting: `aux_mode=kan`, `kernel_size=31`, `grid_size=4`, `grid_range=[-2, 2]`
- Training setup: `num_epochs=60`, `batch_size=1024`, `lr=0.001`, `test_size=0.2`, `minSNR=-20`, `maxSNR=18`

## Contents

- `summary/seed_results_per_run.csv`: per-seed overall accuracy.
- `summary/seed_summary.csv`: mean, standard deviation, min, and max accuracy.
- `summary/seed_model_delta.csv`: IQFormerLite minus IQFormer mean accuracy difference.
- `summary/seed_summary.md`: Markdown summary table.
- `raw/<dataset>/<model>/seed*/`: raw output files for each run.

Each raw run directory contains:

- `Test_ACC.csv`
- `Test_mod_SNR.csv`
- `Train_Epoch.csv`
- `Val_Epoch.csv`
- `model_report.txt`

## Verified summary

| Dataset | Model | n | Mean ± Std (%) |
| --- | --- | ---: | ---: |
| 2016.10a | IQFormer | 5 | 63.60 ± 0.11 |
| 2016.10a | IQFormerLite | 5 | 63.18 ± 0.22 |
| 2016.10b | IQFormer | 5 | 65.42 ± 0.18 |
| 2016.10b | IQFormerLite | 5 | 65.51 ± 0.13 |

The standard deviation is the sample standard deviation over five random seeds (`ddof=1`).
