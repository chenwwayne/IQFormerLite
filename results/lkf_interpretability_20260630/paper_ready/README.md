# Paper-Ready LKF Interpretability Figures

Recommended use:

1. Main paper:
   - `lkf_5seed_summary.pdf`
   - Use this as the primary LKF interpretability figure because it only reports aggregate results over five random seeds.

2. Optional main-paper single-panel figures:
   - `aggregate_base_response_mean_std.pdf`
   - `aggregate_snr_family_response_mean_std.pdf`
   - Use only if the journal layout prefers separate figures instead of a multi-panel figure.

3. Supplementary material:
   - Per-seed CSV files from the parent directory, if auditability is needed.
   - Per-seed images were removed from the curated result directory to keep the output focused on five-seed aggregate results.

Interpretation boundary:

- The learned-filter frequency response comes from the KAN base-branch convolution weights.
- The SNR/modulation-family curves are spectra of the LKF output, not dynamic SNR-conditioned filter kernels.
- These results are server-side model interpretability results, not RK3588 board deployment validation.
