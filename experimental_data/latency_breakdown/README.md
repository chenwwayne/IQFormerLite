# Minimal RK3588 latency-breakdown experiment

This supplementary record supports the revised deployment discussion in `Main.tex`.

- Hardware: Orange Pi 5 Plus with Rockchip RK3588; runtime: `rknn-toolkit-lite2 2.3.2`.
- Models: the existing INT8 RKNN checkpoints for IQFormer and IQFormerLite; no retraining or architecture change was performed.
- Data: RadioML2016.10A, the fixed test split generated with seed 233; the first 1,024 samples were used for this timing-only measurement.
- Protocol: batch size 16, five repeated runs after warm-up. The IQFormer CPU-side STFT was timed separately. IQFormerLite consumes raw I/Q input and therefore has no separately measured CPU STFT stage.
- The RKNN timing includes the runtime call and associated input handling. Host-to-device transfer was not separately instrumented, so the result is reported as RKNN model-execution time rather than a causal compiler or transfer decomposition.
- The experiment does not infer whether any individual operator internally falls back to the CPU; no such fallback was separately measurable through the Runtime API.

The numerical results are stored in `iqformer_r2_breakdown.csv`.
