# Seed Sweep for IQFormerLite and IQFormer

This directory contains the reproducible 5-seed experiment workflow for the paper revision.

## Run

```bash
cd /Users/chenww/develop/IQFormerLite/code
bash scripts/run_seed_sweep.sh
```

Defaults:

- datasets: `2016.10a 2016.10b`
- models: `IQFormerLite IQFormer`
- seeds: `1 2 3 4 5`
- IQFormerLite: `--aux_mode kan --kernel_size 31 --grid_size 4 --grid_range -2 2`
- IQFormer: `--aux_mode stft`

## Summarize

```bash
cd /Users/chenww/develop/IQFormerLite/code
python scripts/summarize_seed_sweep.py
```

Outputs:

- `results/table2_5seed_20260626/iqformer_vs_lite/iqformer_per_seed_overall_accuracy.csv`
- `results/table2_5seed_20260626/iqformer_vs_lite/iqformer_summary_overall_accuracy.csv`
- `results/table2_5seed_20260626/iqformer_vs_lite/iqformer_summary_overall_accuracy.md`

The reported mean and standard deviation are computed from the `Avg` row in each run's `Test_ACC.csv`.
