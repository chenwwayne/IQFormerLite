# Ordinary Convolution Kernel Sweep

This experiment evaluates ordinary Conv1d filterbanks with kernel sizes
`3, 7, 15, 31, 51` against Full LKF(k=31). All convolution controls use 2 input
channels, 32 output channels, stride 1, odd-kernel same padding, groups 1,
no bias, and the same log-absolute, BatchNorm, 1x1 projection, and fusion path.

Existing five-seed k=3 and k=31 results are reused. New training is required
only for k=7, k=15, and k=51 on RadioML2016.10A and RadioML2016.10B.

All kernel configurations must be reported. The sweep must not be used to
selectively present only a weak convolution baseline.
