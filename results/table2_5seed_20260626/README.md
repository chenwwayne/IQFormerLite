# Table II 5-Seed Results Organization

Date: 2026-06-26

This directory is the organized local archive for the Table II 5-seed revision results.

## Directory Roles

| Directory | Role | Status | Notes |
| --- | --- | --- | --- |
| `baseline/` | Baseline Table II summary for MCLDNN, MCFormer, PET-CGDNN, AMC-Net, and FEA-T | Canonical | Contains per-seed and 5-seed summary files. |
| `iqformer_vs_lite/` | IQFormer and IQFormerLite 5-seed archive | Canonical | Contains summary files and raw run outputs. |

## Source Files

### Baseline Results

- `baseline/baseline_per_seed_accuracy.csv`
  - 50 per-run records.
  - Five baseline models across two datasets and five seeds.
- `baseline/baseline_summary_accuracy.csv`
  - 10 summary records.
  - Mean and sample standard deviation over five seeds.
- `baseline/baseline_summary_table.md`
  - Markdown summary for paper Table II.

### IQFormer / IQFormerLite Results

- `iqformer_vs_lite/iqformer_per_seed_overall_accuracy.csv`
  - 20 per-run records.
  - Two models across two datasets and five seeds.
- `iqformer_vs_lite/iqformer_summary_overall_accuracy.csv`
  - Overall accuracy mean, standard deviation, min, and max.
- `iqformer_vs_lite/iqformer_lite_delta_vs_iqformer.csv`
  - IQFormerLite minus IQFormer mean accuracy difference.
- `iqformer_vs_lite/raw_runs/`
  - 20 raw run directories.
  - 100 core raw files in total.
  - Includes 20 `Test_ACC.csv` and 20 `Test_mod_SNR.csv` files.

## Experiment Settings

| Group | Datasets | Seeds | Training setup | Notes |
| --- | --- | --- | --- | --- |
| Baselines | RadioML2016.10A, RadioML2016.10B | 1, 2, 3, 4, 5 | 60 epochs, batch size 1024, lr 0.001, SNR -20 to 18 | Uses the existing `main.py` split with `test_size=0.2` and `random_state=233`. |
| IQFormer | RadioML2016.10A, RadioML2016.10B | 1, 2, 3, 4, 5 | 60 epochs, batch size 1024, lr 0.001, SNR -20 to 18 | `aux_mode=stft`. |
| IQFormerLite | RadioML2016.10A, RadioML2016.10B | 1, 2, 3, 4, 5 | 60 epochs, batch size 1024, lr 0.001, SNR -20 to 18 | `aux_mode=kan`, `kernel_size=31`, `grid_size=4`, `grid_range=[-2, 2]`. |

## Paper Table II Values

Use the following values for Table II. Accuracy is reported as mean ± sample standard deviation over five random seeds.

| Model | Dataset | n | -20..0 | 0..18 | Overall |
| --- | --- | ---: | ---: | ---: | ---: |
| MCLDNN | A | 5 | 32.09 ± 4.67 | 78.50 ± 8.41 | 53.23 ± 6.24 |
| MCLDNN | B | 5 | 34.84 ± 13.89 | 76.44 ± 37.14 | 53.64 ± 24.40 |
| MCFormer | A | 5 | 34.38 ± 0.16 | 80.47 ± 1.85 | 55.29 ± 0.80 |
| MCFormer | B | 5 | 39.63 ± 0.45 | 91.73 ± 0.58 | 63.22 ± 0.46 |
| PET-CGDNN | A | 5 | 32.82 ± 3.47 | 81.22 ± 3.34 | 54.87 ± 3.25 |
| PET-CGDNN | B | 5 | 39.44 ± 0.22 | 92.34 ± 0.09 | 63.39 ± 0.15 |
| AMC-Net | A | 5 | 36.79 ± 0.36 | 87.65 ± 0.78 | 59.79 ± 0.52 |
| AMC-Net | B | 5 | 41.38 ± 0.12 | 93.13 ± 0.09 | 64.75 ± 0.10 |
| FEA-T | A | 5 | 35.44 ± 1.68 | 85.82 ± 2.26 | 58.37 ± 1.84 |
| FEA-T | B | 5 | 34.39 ± 13.64 | 76.31 ± 37.07 | 53.35 ± 24.23 |
| IQFormer | A | 5 | 39.50 ± 0.30 | 92.81 ± 0.27 | 63.60 ± 0.11 |
| IQFormer | B | 5 | 42.24 ± 0.31 | 93.62 ± 0.07 | 65.42 ± 0.18 |
| IQFormerLite | A | 5 | 39.11 ± 0.23 | 92.36 ± 0.45 | 63.18 ± 0.22 |
| IQFormerLite | B | 5 | 42.28 ± 0.22 | 93.75 ± 0.05 | 65.51 ± 0.13 |

## Ranking Notes For Table II

### RadioML2016.10A

- Low SNR (-20..0): best `IQFormer` (39.50), second `IQFormerLite` (39.11).
- High SNR (0..18): best `IQFormer` (92.81), second `IQFormerLite` (92.36).
- Overall: best `IQFormer` (63.60), second `IQFormerLite` (63.18).

### RadioML2016.10B

- Low SNR (-20..0): best `IQFormerLite` (42.28), second `IQFormer` (42.24).
- High SNR (0..18): best `IQFormerLite` (93.75), second `IQFormer` (93.62).
- Overall: best `IQFormerLite` (65.51), second `IQFormer` (65.42).

## Notable Points

- IQFormerLite is slightly below IQFormer on RadioML2016.10A overall by 0.42 percentage points.
- IQFormerLite is slightly above IQFormer on RadioML2016.10B overall by 0.09 percentage points.
- Baseline results for RadioML2016.10B MCLDNN and FEA-T have large standard deviations because one retained seed for each model produced a low-accuracy run:
  - `2016.10b`, `MCLDNN`, seed 2: overall accuracy 10.00%.
  - `2016.10b`, `FEA_T128`, seed 1: overall accuracy 10.00%.
- These low-accuracy runs are documented in `baseline/README.md` and retained in the five-seed averages.

## Cleanup Applied

- Replaced `baseline_table2_summary/` with `results/table2_5seed_20260626/baseline/`.
- Replaced `seed_sweep_results_20260624/` with `results/table2_5seed_20260626/iqformer_vs_lite/`.
- Removed the redundant `seed_sweep_summary/` copy because it was identical to the IQFormer/IQFormerLite summary files.
