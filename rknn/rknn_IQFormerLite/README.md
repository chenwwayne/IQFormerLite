# RKNN 推理与评测说明

本目录包含 RK3588 上 IQFormerLite 的 RKNN 推理脚本、权重与评测结果。核心入口为 inference_on_rk3588.py，可批量评测不同精度的 RKNN 模型并导出 CSV。

## 目录结构
- inference_on_rk3588.py：主推理/评测脚本
- inference_on_rk3588.sh：示例启动脚本
- weights/：多精度 RKNN 权重（如 weight_fp32.rknn、weight_fp16.rknn、weight_int8.rknn）
- benchmark.csv：评测输出结果
- calib/、pt2rknn/：RKNN 校准与转换相关内容

## 运行环境
激活 conda 环境:
```bash
conda activate rknn
```

## 依赖包
必需：
- rknn_toolkit_lite2
- numpy
- psutil
- torch
- einops
- timm

可选（用于 FLOPs/模型统计更完整）：
- torchinfo

安装示例：
```bash
pip install rknn_toolkit_lite2 numpy psutil torch einops timm torchinfo
```

## 使用方法
批量评测并输出 CSV
```bash
python inference_on_rk3588.py --models_dir /home/orangepi/IQFormerLite/rknn
                              --output_csv /home/orangepi/IQFormerLite/rknn/benchmark.csv
```

## 输出字段说明（benchmark.csv）
- params_m / flops_g / cpu_model_size_kb：CPU 侧模型复杂度（按模型名映射 fp32/fp16/int8）
- rknn_model_size_kb：RKNN 文件大小
- cpu_latency_ms / cpu_throughput：CPU 侧动态性能
- npu_latency_batch_ms / npu_latency_sample_ms / npu_throughput / npu_accuracy：NPU 侧动态性能
- speedup：CPU/NPU 延迟比
- npu_load_mean / npu_load_max：NPU 负载（依赖 /sys/kernel/debug/rknpu/load 可读）
- memory_baseline_mb / memory_peak_mb / memory_delta_mb：进程 RSS 基线/峰值/增量

## NPU 负载权限
如 npu_load_mean / npu_load_max 为空，请先设置权限：
```bash
sudo chmod a+rx /sys/kernel/debug
sudo chmod a+rx /sys/kernel/debug/rknpu
sudo chmod a+r /sys/kernel/debug/rknpu/load
```
