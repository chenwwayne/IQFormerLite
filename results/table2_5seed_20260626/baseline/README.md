# Baseline Table II Runs

This directory stores the 5-seed baseline results generated on A800-2 for Table II.

- Remote project: `/data/iqformer-lite`
- Remote datasets:
  - `/data/iqformer-lite/dataset/RML2016.10a.pkl`
  - `/data/iqformer-lite/dataset/RML2016.10b.dat`
- Split policy: the existing `main.py` RadioML split, using `train_test_split(..., random_state=233)` and `test_size=0.2`
- Seeds: `1 2 3 4 5`
- Models: `MCLDNN`, `MCFormer`, `PETCGDNN`, `AMCNET`, `FEA_T128`
- Training setup: `num_epochs=60`, `batch_size=1024`, `lr=0.001`, `minSNR=-20`, `maxSNR=18`
- Completed runs: 50/50

Files:

- `baseline_per_seed_accuracy.csv`: per-seed low-SNR, high-SNR, and overall accuracy.
- `baseline_summary_accuracy.csv`: mean and sample standard deviation over five seeds.
- `baseline_summary_table.md`: Markdown summary for Table II review.

Notable low-accuracy runs retained in the 5-seed average after rerun:

- `2016.10b`, `MCLDNN`, seed `2`: rerun overall accuracy `10.00%`
- `2016.10b`, `FEA_T128`, seed `1`: rerun overall accuracy `10.00%`

The previous low-accuracy outputs were archived on A800-2 under:

- `/data/iqformer-lite/rerun_archive/20260625_201121`
