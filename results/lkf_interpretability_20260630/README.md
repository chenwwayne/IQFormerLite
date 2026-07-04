# LKF Interpretability Results

This directory stores IQFormerLite LKF interpretability artifacts.

## Interpretation Boundary

- The KAN filterbank is nonlinear; direct learned-filter frequency responses are computed from stored convolution weights.
- Different-SNR curves are not dynamic kernel responses. They are spectra of LKF outputs for grouped input samples.
- Use these figures to support stable spectral-structure claims, not real-device deployment claims.

## Files

- `lkf_weight_response.csv`: normalized frequency response for base and spline convolution weights.
- `*_base_response.pdf/png`: paper-ready base-branch filter response heatmaps.
- `*_spline_response.pdf/png`: spline-branch weight response heatmaps for audit/supplement use.
- `*_snr_family_response.csv`: SNR/family grouped LKF output spectra.
- `*_snr_family_response.pdf/png`: paper-ready grouped output-response curves.
- `aggregate_*`: checkpoint-level mean/std summaries when multiple trained checkpoints are provided.
- `run_config.json`: exact analysis inputs and parameters.
