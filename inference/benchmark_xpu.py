"""
Intel XPU (PyTorch) GEMM & Conv2d Benchmark
测试矩阵乘法和卷积在 XPU 上的实际 FLOPS 性能
"""

import time
import torch
import numpy as np

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def sync(dev: torch.device):
    if dev.type == "xpu":
        torch.xpu.synchronize()


def format_flops(n: float) -> str:
    """将 FLOPS 数值格式化为可读字符串"""
    for unit in ("FLOPS", "TFLOPS", "PFLOPS"):
        if abs(n) < 1000:
            return f"{n:.2f} {unit}"
        n /= 1000
    return f"{n:.2f} EFLOPS"


def format_time(sec: float) -> str:
    if sec < 1e-3:
        return f"{sec*1e6:.2f} us"
    elif sec < 1:
        return f"{sec*1e3:.2f} ms"
    return f"{sec:.4f} s"


# ---------------------------------------------------------------------------
# GEMM Benchmark
# ---------------------------------------------------------------------------

def benchmark_gemm(
    M: int, N: int, K: int,
    dtype: torch.dtype = torch.float32,
    num_warmup: int = 10,
    num_iter: int = 100,
    device: torch.device = None,
) -> dict:
    """测试矩阵乘法 C = A @ B 的 FLOPS

    A: (M, K), B: (K, N), C: (M, N)
    FLOPS = 2 * M * N * K
    """
    if device is None:
        device = get_device()

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    # warmup
    for _ in range(num_warmup):
        C = A @ B
    sync(device)

    # benchmark
    sync(device)
    start = time.perf_counter()
    for _ in range(num_iter):
        C = A @ B
    sync(device)
    elapsed = time.perf_counter() - start

    avg_time = elapsed / num_iter
    flops = 2 * M * N * K / avg_time

    return {
        "op": "GEMM",
        "shape": f"({M},{N},{K})",
        "dtype": str(dtype),
        "avg_time_s": avg_time,
        "avg_time_str": format_time(avg_time),
        "flops": flops,
        "flops_str": format_flops(flops),
    }


# ---------------------------------------------------------------------------
# Conv2d Benchmark
# ---------------------------------------------------------------------------

def benchmark_conv2d(
    N: int, C: int, H: int, W: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dtype: torch.dtype = torch.float32,
    num_warmup: int = 10,
    num_iter: int = 100,
    device: torch.device = None,
) -> dict:
    """测试 Conv2d 的 FLOPS

    输入: (N, C, H, W)
    输出: (N, out_channels, H_out, W_out)
    FLOPS = 2 * N * out_channels * H_out * W_out * C * kernel_size * kernel_size
    """
    if device is None:
        device = get_device()

    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1

    conv = torch.nn.Conv2d(
        in_channels=C,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
        dtype=dtype,
    ).to(device)

    x = torch.randn(N, C, H, W, dtype=dtype, device=device)

    # warmup
    for _ in range(num_warmup):
        y = conv(x)
    sync(device)

    # benchmark
    sync(device)
    start = time.perf_counter()
    for _ in range(num_iter):
        y = conv(x)
    sync(device)
    elapsed = time.perf_counter() - start

    avg_time = elapsed / num_iter
    flops = (
        2 * N * out_channels * H_out * W_out * C * kernel_size * kernel_size
    ) / avg_time

    return {
        "op": "Conv2d",
        "shape": f"({N},{C},{H},{W}) -> ({out_channels},{H_out},{W_out})",
        "kernel": f"{kernel_size}x{kernel_size}",
        "stride": stride,
        "dtype": str(dtype),
        "avg_time_s": avg_time,
        "avg_time_str": format_time(avg_time),
        "flops": flops,
        "flops_str": format_flops(flops),
    }


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

GEMM_CONFIGS = [
    # (M, N, K)  常见 LLM / CV 中的矩阵尺寸
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (1, 4096, 4096),       # 推理典型: batch=1
    (4, 4096, 4096),
    (16384, 4096, 4096),   # 大矩阵
    (4096, 16384, 4096),
]

CONV2D_CONFIGS = [
    # (N, C, H, W, out_channels, kernel_size, stride, padding)
    # ResNet18 典型层
    (1, 3, 224, 224, 64, 7, 2, 3),       # conv1
    (1, 64, 56, 56, 64, 3, 1, 1),        # res2
    (1, 64, 56, 56, 128, 3, 2, 1),       # res3 (downsample)
    (1, 128, 28, 28, 128, 3, 1, 1),      # res3
    (1, 128, 28, 28, 256, 3, 2, 1),      # res4 (downsample)
    (1, 256, 14, 14, 256, 3, 1, 1),      # res4
    (1, 256, 14, 14, 512, 3, 2, 1),      # res5 (downsample)
    (1, 512, 7, 7, 512, 3, 1, 1),        # res5
    # 大 batch
    (4, 64, 56, 56, 64, 3, 1, 1),
    (8, 64, 56, 56, 64, 3, 1, 1),
    # 大分辨率
    (1, 64, 128, 128, 128, 3, 1, 1),
    (1, 128, 64, 64, 256, 3, 1, 1),
]


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def run_gemm_benchmarks(device: torch.device, dtype: torch.dtype = torch.float32):
    print("\n" + "=" * 80)
    print(f"GEMM Benchmark  |  device={device}  dtype={dtype}")
    print("=" * 80)
    print(f"{'Shape (M,N,K)':<24} {'Time':<14} {'FLOPS':<18} {'TFLOPS':<10}")
    print("-" * 80)

    results = []
    for M, N, K in GEMM_CONFIGS:
        r = benchmark_gemm(M, N, K, dtype=dtype, device=device)
        results.append(r)
        tflops = r["flops"] / 1e12
        print(f"{r['shape']:<24} {r['avg_time_str']:<14} {r['flops_str']:<18} {tflops:>8.2f}")
    return results


def run_conv2d_benchmarks(device: torch.device, dtype: torch.dtype = torch.float32):
    print("\n" + "=" * 100)
    print(f"Conv2d Benchmark  |  device={device}  dtype={dtype}")
    print("=" * 100)
    print(f"{'Shape':<36} {'Kernel':<10} {'Time':<14} {'FLOPS':<18} {'TFLOPS':<10}")
    print("-" * 100)

    results = []
    for cfg in CONV2D_CONFIGS:
        N, C, H, W, out_channels, kernel_size, stride, padding = cfg
        r = benchmark_conv2d(
            N, C, H, W, out_channels, kernel_size,
            stride=stride, padding=padding,
            dtype=dtype, device=device,
        )
        results.append(r)
        tflops = r["flops"] / 1e12
        print(f"{r['shape']:<36} {r['kernel']:<10} {r['avg_time_str']:<14} {r['flops_str']:<18} {tflops:>8.2f}")
    return results


def main():
    device = get_device()
    print(f"PyTorch version: {torch.__version__}")
    print(f"Benchmark device: {device}")
    if device.type == "xpu":
        print(f"XPU count: {torch.xpu.device_count()}")
        print(f"XPU name: {torch.xpu.get_device_name(0)}")

    # FP32 benchmark
    run_gemm_benchmarks(device, torch.float32)
    run_conv2d_benchmarks(device, torch.float32)

    # FP16 benchmark (if supported)
    if device.type == "xpu":
        try:
            run_gemm_benchmarks(device, torch.float16)
            run_conv2d_benchmarks(device, torch.float16)
        except Exception as e:
            print(f"\nFP16 benchmark skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
