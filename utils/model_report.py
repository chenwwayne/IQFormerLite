import time
import copy
import torch

def get_report_batch(train_loader, aux_mode, device):
    batch_x, batch_stft, _, _ = next(iter(train_loader))
    batch_x = batch_x.to(device)
    if aux_mode == 'stft':
        batch_stft = batch_stft.to(device)
    else:
        batch_stft = None
    return batch_x, batch_stft

def adjust_inputs(batch_x, batch_stft, batch_size=1, length=None):
    x0 = batch_x[:1]
    if length is not None and x0.size(-1) != length:
        L = min(length, x0.size(-1))
        x0 = x0[..., :L]
    if batch_size > 1:
        reps = [batch_size, 1, 1]
        x0 = x0.repeat(*reps)
    if batch_stft is not None:
        s0 = batch_stft[:1]
        if batch_size > 1:
            reps_s = [batch_size] + [1] * (s0.dim() - 1)
            s0 = s0.repeat(*reps_s)
    else:
        s0 = None
    return x0, s0

def get_model_bytes(model):
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return param_bytes, buffer_bytes

def format_bytes(num_bytes):
    if num_bytes is None:
        return None
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"

def try_torchinfo(model, batch_x, batch_stft, device, aux_mode):
    try:
        from torchinfo import summary
    except Exception:
        return None
    if aux_mode == 'stft':
        return summary(model, input_data=(batch_x, batch_stft), device=device, verbose=0)
    return summary(model, input_data=(batch_x,), device=device, verbose=0)

def measure_latency(model, batch_x, batch_stft, device, aux_mode, warmup=10, iters=30):
    model.eval()
    with torch.no_grad():
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            for _ in range(warmup):
                if aux_mode == 'stft':
                    model(batch_x, batch_stft)
                else:
                    model(batch_x)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                if aux_mode == 'stft':
                    model(batch_x, batch_stft)
                else:
                    model(batch_x)
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
            return elapsed_ms / iters
        for _ in range(warmup):
            if aux_mode == 'stft':
                model(batch_x, batch_stft)
            else:
                model(batch_x)
        start = time.perf_counter()
        for _ in range(iters):
            if aux_mode == 'stft':
                model(batch_x, batch_stft)
            else:
                model(batch_x)
        end = time.perf_counter()
        return (end - start) * 1000 / iters

def measure_peak_memory(model, batch_x, batch_stft, device, aux_mode, param_bytes):
    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        return None, None
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        if aux_mode == 'stft':
            model(batch_x, batch_stft)
        else:
            model(batch_x)
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    peak_activation = max(0, peak_alloc - param_bytes)
    return peak_activation, peak_reserved

def build_model_report(model, batch_x, batch_stft, device, aux_mode):
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes, buffer_bytes = get_model_bytes(model)
    total_bytes = param_bytes + buffer_bytes
    info = try_torchinfo(model, batch_x, batch_stft, device, aux_mode)
    macs = getattr(info, "total_mult_adds", None) if info is not None else None
    flops = macs * 2 if macs is not None else None
    latency_ms = measure_latency(model, batch_x, batch_stft, device, aux_mode)
    throughput = batch_x.size(0) / (latency_ms / 1000) if latency_ms and latency_ms > 0 else None
    peak_activation, peak_total = measure_peak_memory(model, batch_x, batch_stft, device, aux_mode, param_bytes)
    return {
        "params": param_count,
        "trainable_params": trainable_count,
        "model_size": total_bytes,
        "macs": macs,
        "flops": flops,
        "latency_ms": latency_ms,
        "throughput": throughput,
        "peak_activation": peak_activation,
        "peak_total": peak_total,
    }

def cast_model_and_inputs(model, batch_x, batch_stft, device, dtype):
    # Always work on a copy to avoid side effects (especially for half() which is in-place)
    try:
        m = copy.deepcopy(model)
    except Exception:
        # Fallback if deepcopy fails (e.g. some complex objects), though nn.Module usually copies fine
        m = model

    if dtype == "fp32":
        m = m.to(device)
        x = batch_x.to(device, dtype=torch.float32)
        s = batch_stft.to(device, dtype=torch.float32) if batch_stft is not None else None
        return m, x, s, device
    if dtype == "fp16":
        if not (str(device).startswith("cuda") and torch.cuda.is_available()):
            return None, None, None, None
        m = m.to(device).half()
        x = batch_x.to(device, dtype=torch.float16)
        s = batch_stft.to(device, dtype=torch.float16) if batch_stft is not None else None
        return m, x, s, device
    if dtype == "int8":
        try:
            from torch.ao.quantization import quantize_dynamic
        except Exception:
            return None, None, None, None
        modules = {torch.nn.Linear}
        # Dynamic quantization is typically for CPU inference
        effective_device = torch.device("cpu")
        m = m.to(effective_device)
        m = quantize_dynamic(m, modules, dtype=torch.qint8)
        x = batch_x.to(effective_device, dtype=torch.float32)
        s = batch_stft.to(effective_device, dtype=torch.float32) if batch_stft is not None else None
        return m, x, s, effective_device
    return None, None, None, None

def build_model_report_for_dtype(model, batch_x, batch_stft, device, aux_mode, dtype):
    m, x, s, effective_device = cast_model_and_inputs(model, batch_x, batch_stft, device, dtype)
    if m is None:
        return {"dtype": dtype, "available": False}
    r = build_model_report(m, x, s, effective_device, aux_mode)
    r["dtype"] = dtype
    r["available"] = True
    return r

def build_multi_dtype_report(model, batch_x, batch_stft, device, aux_mode, dtypes):
    reports = []
    for dt in dtypes:
        reports.append(build_model_report_for_dtype(model, batch_x, batch_stft, device, aux_mode, dt))
    return reports

def format_multi_dtype_report(reports):
    lines = []
    for r in reports:
        lines.append(f"=== DType: {r.get('dtype')} ===")
        if not r.get("available", False):
            lines.append("报告不可用")
            continue
        lines.append(f"参数量: {r['params']:,}")
        lines.append(f"可训练参数量: {r['trainable_params']:,}")
        lines.append(f"模型大小: {format_bytes(r['model_size'])}")
        if r["macs"] is None:
            lines.append("MACs: N/A")
        else:
            lines.append(f"MACs: {r['macs']:,}")
        if r["flops"] is None:
            lines.append("FLOPs: N/A")
        else:
            lines.append(f"FLOPs: {r['flops']:,}")
        lines.append(f"Latency: {r['latency_ms']:.3f} ms")
        if r["throughput"] is None:
            lines.append("Throughput: N/A")
        else:
            lines.append(f"Throughput: {r['throughput']:.2f} samples/s")
        if r["peak_activation"] is None:
            lines.append("Peak Activation Memory: N/A")
            lines.append("Peak Total Memory: N/A")
        else:
            lines.append(f"Peak Activation Memory: {format_bytes(r['peak_activation'])}")
            lines.append(f"Peak Total Memory: {format_bytes(r['peak_total'])}")
    return "\n".join(lines) + "\n"

def print_multi_dtype_report(reports):
    for r in reports:
        print(f"=== DType: {r.get('dtype')} ===")
        if not r.get("available", False):
            print("报告不可用")
            continue
        print(f"参数量: {r['params']:,}")
        print(f"可训练参数量: {r['trainable_params']:,}")
        print(f"模型大小: {format_bytes(r['model_size'])}")
        if r["macs"] is None:
            print("MACs: N/A")
        else:
            print(f"MACs: {r['macs']:,}")
        if r["flops"] is None:
            print("FLOPs: N/A")
        else:
            print(f"FLOPs: {r['flops']:,}")
        print(f"Latency: {r['latency_ms']:.3f} ms")
        if r["throughput"] is None:
            print("Throughput: N/A")
        else:
            print(f"Throughput: {r['throughput']:.2f} samples/s")
        if r["peak_activation"] is None:
            print("Peak Activation Memory: N/A")
            print("Peak Total Memory: N/A")
        else:
            print(f"Peak Activation Memory: {format_bytes(r['peak_activation'])}")
            print(f"Peak Total Memory: {format_bytes(r['peak_total'])}")

def print_model_report(report):
    print("Model Report")
    print(f"参数量: {report['params']:,}")
    print(f"可训练参数量: {report['trainable_params']:,}")
    print(f"模型大小: {format_bytes(report['model_size'])}")
    if report["macs"] is None:
        print("MACs: N/A")
    else:
        print(f"MACs: {report['macs']:,}")
    if report["flops"] is None:
        print("FLOPs: N/A")
    else:
        print(f"FLOPs: {report['flops']:,}")
    print(f"Latency: {report['latency_ms']:.3f} ms")
    if report["throughput"] is None:
        print("Throughput: N/A")
    else:
        print(f"Throughput: {report['throughput']:.2f} samples/s")
    if report["peak_activation"] is None:
        print("Peak Activation Memory: N/A")
        print("Peak Total Memory: N/A")
    else:
        print(f"Peak Activation Memory: {format_bytes(report['peak_activation'])}")
        print(f"Peak Total Memory: {format_bytes(report['peak_total'])}")
