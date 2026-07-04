# LKF Interpretability Analysis

This folder contains the analysis script for the reviewer-response LKF experiment.

## Purpose

Use IQFormerLite with `aux_mode=kan` to show whether the learnable KAN filterbank develops stable spectral structure beyond accuracy gains.

## Important Interpretation Boundary

- `FilterbankKANStem` uses `FastKANConv1DLayer`, which has a linear base convolution branch and a nonlinear spline branch.
- The direct frequency-response heatmaps are computed from the learned convolution weights.
- The SNR/family curves are not dynamic filter kernels. They are input-driven spectra of the `kanstem` output for grouped samples.
- Do not use these results as RK3588 deployment evidence; the server is not connected to the board.

## Recommended Command

Use five IQFormerLite `aux_mode=kan` checkpoints from RadioML2016.10B:

```bash
python notebook/plot_lkf_interpretability/extract_lkf_responses.py \
  --database_path ./dataset \
  --database_choose 2016.10b \
  --include_random_init \
  --checkpoint /path/to/seed1/weight.pt --checkpoint_label seed1 \
  --checkpoint /path/to/seed2/weight.pt --checkpoint_label seed2 \
  --checkpoint /path/to/seed3/weight.pt --checkpoint_label seed3 \
  --checkpoint /path/to/seed4/weight.pt --checkpoint_label seed4 \
  --checkpoint /path/to/seed5/weight.pt --checkpoint_label seed5 \
  --output_dir results/lkf_interpretability_202606xx
```

If only a representative model is available:

```bash
python notebook/plot_lkf_interpretability/extract_lkf_responses.py \
  --database_path ./dataset \
  --database_choose 2016.10b \
  --include_random_init \
  --checkpoint save_models/model_2016.10b_60_1024_0.001_IQFormerLite_seed1/weight.pt \
  --checkpoint_label seed1 \
  --output_dir results/lkf_interpretability_202606xx
```

## Optional Stage Checkpoints

To save stage checkpoints during a representative training run:

```bash
python main.py \
  --model IQFormerLite \
  --database_choose 2016.10b \
  --aux_mode kan \
  --band_k 32 \
  --kernel_size 31 \
  --grid_size 4 \
  --grid_range -2 2 \
  --seed 1 \
  --comment IQFormerLite_lkf_stage_seed1 \
  --save_stage_checkpoints \
  --stage_epochs 0,5,15,best,final
```

The extra files are saved under:

```text
save_models/<model_tag>/stage_checkpoints/
```

## Outputs

- `lkf_weight_response.csv`
- `*_base_response.pdf/png`
- `*_spline_response.pdf/png`
- `*_snr_family_response.csv`
- `*_snr_family_response.pdf/png`
- `aggregate_base_response_mean_std.csv/pdf/png`
- `aggregate_spline_response_mean_std.csv/pdf/png`
- `aggregate_snr_family_response_mean_std.csv/pdf/png`
- `run_config.json`
- `README.md`

For the paper, prefer:

- `aggregate_base_response_mean_std.pdf` for the learned filter response stability.
- `aggregate_snr_family_response_mean_std.pdf` for SNR/family input-response stability.

Keep per-seed heatmaps and spline-response plots as audit or supplementary material unless they are visually cleaner.
