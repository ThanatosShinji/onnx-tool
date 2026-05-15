"""
动态分辨率推理测试：PyTorch vs GraphInfer

从第一个分辨率开始依次推理全部图像，统计从第一个到最后一个的
总耗时和峰值 XPU 内存，模拟真实动态分辨率推理场景。
"""

import time
import torch
import torch.nn as nn
import torchvision
import numpy as np
import psutil
import os


def get_memory_usage() -> float:
    """获取当前进程内存使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_xpu_memory() -> dict:
    """获取 XPU 设备内存使用量（MB）"""
    if not torch.xpu.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}
    return {
        "allocated": torch.xpu.memory_allocated() / 1024 / 1024,
        "reserved": torch.xpu.memory_reserved() / 1024 / 1024,
        "max_allocated": torch.xpu.max_memory_allocated() / 1024 / 1024,
    }


def reset_xpu_memory_stats():
    """重置 XPU 峰值内存统计"""
    if torch.xpu.is_available():
        torch.xpu.reset_peak_memory_stats()
        torch.xpu.reset_accumulated_memory_stats()


def main():
    device = "xpu" if torch.xpu.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"{'='*70}")

    # ===== 测试集：多种分辨率（按顺序推理） =====
    test_images = [
        {"name": "HD (720p)", "shape": (1, 3, 720, 1280)},
        {"name": "Full HD (1080p)", "shape": (1, 3, 1080, 1920)},
        {"name": "2K (1440p)", "shape": (1, 3, 1440, 2560)},
        {"name": "4K (2160p)", "shape": (1, 3, 2160, 3840)},
        {"name": "Square Large", "shape": (1, 3, 2048, 2048)},
        {"name": "Mobile (360p)", "shape": (1, 3, 360, 640)},
        {"name": "Small", "shape": (1, 3, 224, 224)},
    ]

    # ===== 1. 导出 ONNX 模型 =====
    onnx_path = "resnet18.onnx"
    print("\n=== Exporting ResNet18 to ONNX ===")
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval().to(device)

    dummy = torch.randn(1, 3, 1080, 1920).to('cpu')
    torch.onnx.export(
        model.to('cpu'), dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17, dynamo=False,
    )
    print(f"ONNX model exported to {onnx_path}")
    model = model.to(device)

    # ===== 2. 初始化 GraphInfer =====
    from infer import GraphInfer

    all_heights = [img["shape"][2] for img in test_images]
    all_widths = [img["shape"][3] for img in test_images]

    infer_engine = GraphInfer(
        onnx_path,
        {'input': ('batch', 3, 'height', 'width')},
        {
            'batch': (1, 1),
            'height': (min(all_heights), max(all_heights)),
            'width': (min(all_widths), max(all_widths)),
        },
        dtype=torch.float32,
        device=device,
    )
    infer_engine.print_summary()

    # 预生成所有输入
    inputs = [torch.randn(img["shape"]).to(device) for img in test_images]

    # ===== 3. 预热 =====
    print("\n=== Warming up ===")
    # PyTorch warmup（用第一个分辨率）
    x0 = inputs[0]
    for _ in range(3):
        with torch.no_grad():
            _ = model(x0)
    torch.xpu.synchronize()
    # GraphInfer warmup（遍历所有分辨率，触发 kernel 编译）
    for x in inputs:
        infer_engine.forward({'input': x}, debug=False)
    torch.xpu.synchronize()
    print("Warmup done.")

    # ===== 4. PyTorch 顺序推理 =====
    print("\n" + "=" * 70)
    print("PyTorch: 顺序推理所有分辨率")
    print("=" * 70)

    reset_xpu_memory_stats()
    pt_mem_before = get_xpu_memory()["allocated"]

    pt_start = time.perf_counter()
    pt_per_image = []
    for i, (img, x) in enumerate(zip(test_images, inputs)):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.xpu.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        pt_per_image.append(elapsed)
        mem_now = get_xpu_memory()
        print(f"  [{i+1}/{len(test_images)}] {img['name']:<22} {elapsed:>8.2f} ms  "
              f"XPU={mem_now['allocated']:>8.1f} MB  peak={mem_now['max_allocated']:>8.1f} MB")

    pt_total_time = (time.perf_counter() - pt_start) * 1000
    pt_peak_mem = get_xpu_memory()["max_allocated"]
    pt_final_mem = get_xpu_memory()["allocated"]

    # ===== 5. GraphInfer 顺序推理 =====
    print("\n" + "=" * 70)
    print("GraphInfer: 顺序推理所有分辨率")
    print("=" * 70)

    reset_xpu_memory_stats()
    gi_mem_before = get_xpu_memory()["allocated"]

    gi_start = time.perf_counter()
    gi_per_image = []
    for i, (img, x) in enumerate(zip(test_images, inputs)):
        t0 = time.perf_counter()
        infer_engine.forward({'input': x}, debug=False)
        torch.xpu.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        gi_per_image.append(elapsed)
        mem_now = get_xpu_memory()
        print(f"  [{i+1}/{len(test_images)}] {img['name']:<22} {elapsed:>8.2f} ms  "
              f"XPU={mem_now['allocated']:>8.1f} MB  peak={mem_now['max_allocated']:>8.1f} MB")

    gi_total_time = (time.perf_counter() - gi_start) * 1000
    gi_peak_mem = get_xpu_memory()["max_allocated"]
    gi_final_mem = get_xpu_memory()["allocated"]

    # ===== 6. 数值对比 =====
    print("\n" + "=" * 70)
    print("数值对比（逐张验证）")
    print("=" * 70)
    all_match = True
    for i, (img, x) in enumerate(zip(test_images, inputs)):
        with torch.no_grad():
            pt_out = model(x)
        torch.xpu.synchronize()
        gi_out = infer_engine.forward({'input': x}, debug=False)['output']
        torch.xpu.synchronize()

        pt_np = pt_out.cpu().numpy()
        gi_np = gi_out.cpu().numpy()
        max_diff = np.max(np.abs(pt_np - gi_np))
        match = max_diff < 1e-4
        if not match:
            all_match = False
        print(f"  [{i+1}] {img['name']:<22} max_diff={max_diff:.2e} {'✓' if match else '✗'}")

    # ===== 7. 汇总 =====
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'':<30} {'PyTorch':>16} {'GraphInfer':>16}")
    print("-" * 62)
    print(f"{'Total time (ms)':<30} {pt_total_time:>16.2f} {gi_total_time:>16.2f}")
    print(f"{'Speedup':<30} {'':>16} {pt_total_time/gi_total_time:>15.2f}x")
    print(f"{'Peak XPU memory (MB)':<30} {pt_peak_mem:>16.1f} {gi_peak_mem:>16.1f}")
    print(f"{'Final XPU memory (MB)':<30} {pt_final_mem:>16.1f} {gi_final_mem:>16.1f}")
    print(f"{'GraphInfer pool (MB)':<30} {'':>16} {infer_engine.pool.compress_size/1024/1024:>16.1f}")
    print(f"{'Host process Δ (MB)':<30} {'':>16} {get_memory_usage() - 1299:>16.1f}")
    print("-" * 62)

    # 逐张耗时明细
    print(f"\n{'':<30} {'PT(ms)':>10} {'GI(ms)':>10} {'Ratio':>8}")
    print("-" * 58)
    for i, img in enumerate(test_images):
        ratio = pt_per_image[i] / gi_per_image[i] if gi_per_image[i] > 0 else 0
        print(f"{img['name']:<30} {pt_per_image[i]:>10.2f} {gi_per_image[i]:>10.2f} {ratio:>7.2f}x")

    print(f"\nAll outputs match: {'✓' if all_match else '✗'}")
    print(f"GraphInfer pool vs peak XPU: pool={infer_engine.pool.compress_size/1024/1024:.1f} MB, "
          f"peak={gi_peak_mem:.1f} MB "
          f"({'⚠ exceeds' if gi_peak_mem > infer_engine.pool.compress_size/1024/1024 else '✓ within pool'})")


if __name__ == "__main__":
    main()
