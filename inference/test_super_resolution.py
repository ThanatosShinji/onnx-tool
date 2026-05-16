"""
超分模型推理测试：GraphInfer vs PyTorch

支持 RealESRGAN / SRGAN / 任意超分 ONNX 模型的多分辨率测试。
测试内容包括：
  1. 多分辨率顺序推理（模拟真实超分场景）
  2. 单分辨率性能 Profile（op 级别耗时分析）
  3. 数值精度对比（与 PyTorch 原始模型对比）
  4. XPU 内存统计

用法：
  # 使用默认的 RealESRGAN x4 模型
  python inference/test_super_resolution.py

  # 指定自定义 ONNX 模型
  python inference/test_super_resolution.py --onnx-path /path/to/model.onnx

  # 仅运行 GraphInfer（无 PyTorch 对比）
  python inference/test_super_resolution.py --no-pytorch

  # 启用 profile 模式
  python inference/test_super_resolution.py --profile
"""

import argparse
import time
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

# 添加父目录到 path，确保可以 import inference 下的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def parse_args():
    parser = argparse.ArgumentParser(description="超分模型 GraphInfer 测试")
    parser.add_argument("--onnx-path", type=str, default=None,
                        help="ONNX 模型路径（默认自动查找 realesrgan-x4.onnx）")
    parser.add_argument("--no-pytorch", action="store_true",
                        help="跳过 PyTorch 对比（仅运行 GraphInfer）")
    parser.add_argument("--profile", action="store_true",
                        help="启用 op 级别 profile 分析")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备（xpu/cpu，默认自动选择）")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16"],
                        help="推理精度")
    parser.add_argument("--warmup-iters", type=int, default=2,
                        help="预热迭代次数")
    return parser.parse_args()


def find_default_onnx() -> Optional[str]:
    """查找默认的超分 ONNX 模型"""
    # 优先查找 realesrgan-x4.onnx
    search_paths = [
        os.path.join(os.path.dirname(__file__), "..", "realesrgan-x4.onnx"),
        os.path.join(os.path.dirname(__file__), "..", "benchmark", "realesrgan-x4.onnx"),
    ]
    for p in search_paths:
        full = os.path.abspath(p)
        if os.path.isfile(full):
            return full
    return None


def get_model_info(onnx_path: str) -> dict:
    """从 ONNX 模型路径推断模型信息"""
    basename = os.path.basename(onnx_path).lower()
    info = {
        "name": basename,
        "scale": 4,  # 默认 4x 超分
        "description": basename,
    }
    if "x2" in basename or "2x" in basename:
        info["scale"] = 2
    elif "x3" in basename or "3x" in basename:
        info["scale"] = 3
    elif "x4" in basename or "4x" in basename:
        info["scale"] = 4
    return info


# ===========================================================================
# 测试配置：多种输入分辨率
# ===========================================================================
TEST_IMAGES = [
    {"name": "Mobile (540p)",   "shape": (1, 3, 540, 960)},
    {"name": "VGA (480p)",      "shape": (1, 3, 480, 640)},
    {"name": "Square 512",      "shape": (1, 3, 512, 512)},
    {"name": "Small (360p)",    "shape": (1, 3, 360, 640)},
    {"name": "Tiny (180p)",     "shape": (1, 3, 180, 320)},
    {"name": "Min (128p)",      "shape": (1, 3, 128, 128)},
]


def build_input_desc_and_range(test_images: List[dict], onnx_path: str) -> Tuple[dict, dict]:
    """根据测试图像列表构建 input_desc 和 input_range"""
    import onnx
    onnx_model = onnx.load(onnx_path)
    input_name = onnx_model.graph.input[0].name

    all_heights = [img["shape"][2] for img in test_images]
    all_widths = [img["shape"][3] for img in test_images]
    input_desc = {input_name: (1, 3, "height", "width")}
    input_range = {
        "height": (min(all_heights), max(all_heights)),
        "width": (min(all_widths), max(all_widths)),
    }
    return input_desc, input_range, input_name


# ===========================================================================
# PyTorch 超分模型（basicsr RRDBNet）
# ===========================================================================
class SRModel(nn.Module):
    """
    基于 basicsr 库的 RRDBNet 超分模型（RealESRGAN 架构）。

    默认参数与 realesrgan-x4.onnx 完全一致：
      num_in_ch=3, num_out_ch=3, num_feat=64,
      num_block=23, num_grow_ch=32, scale=4
    """

    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale
        from basicsr.archs.rrdbnet_arch import RRDBNet
        self.model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ===========================================================================
# 主测试流程
# ===========================================================================
def main():
    args = parse_args()

    # ---- 设备选择 ----
    if args.device:
        device = args.device
    else:
        device = "xpu" if torch.xpu.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    print(f"Using device: {device}, dtype: {args.dtype}")
    print(f"{'='*70}")

    # ---- 查找 ONNX 模型 ----
    onnx_path = args.onnx_path
    if onnx_path is None:
        onnx_path = find_default_onnx()
    if onnx_path is None or not os.path.isfile(onnx_path):
        print(f"错误：找不到 ONNX 模型文件！")
        print(f"请通过 --onnx-path 指定模型路径，或确保以下路径存在：")
        print(f"  - realesrgan-x4.onnx（在工作区根目录）")
        sys.exit(1)

    onnx_path = os.path.abspath(onnx_path)
    model_info = get_model_info(onnx_path)
    print(f"Model: {model_info['name']}")
    print(f"Scale: {model_info['scale']}x")
    print(f"{'='*70}")

    # ---- 选择测试分辨率 ----
    # 超分模型对显存需求大，根据 scale 过滤掉过大的分辨率
    scale = model_info["scale"]
    test_images = [
        img for img in TEST_IMAGES
        if img["shape"][2] * scale <= 4320 and img["shape"][3] * scale <= 7680
    ]
    print(f"Testing {len(test_images)} resolutions (output capped at 4K):")
    for img in test_images:
        out_h = img["shape"][2] * scale
        out_w = img["shape"][3] * scale
        print(f"  {img['name']:<22} -> {img['shape'][2]}x{img['shape'][3]} -> {out_h}x{out_w}")
    print()

    # ---- 构建 input_desc / input_range ----
    input_desc, input_range, input_name = build_input_desc_and_range(test_images, onnx_path)

    # ---- 初始化 GraphInfer ----
    print("=== Initializing GraphInfer ===")
    from inference.infer import GraphInfer

    infer_engine = GraphInfer(
        onnx_path,
        input_desc,
        input_range,
        dtype=dtype,
        device=device,
    )
    infer_engine.print_summary()

    # 预生成所有输入
    inputs = [torch.randn(img["shape"]).to(device=device, dtype=dtype)
              for img in test_images]

    # ---- 预热 ----
    print("\n=== Warming up ===")
    for i in range(args.warmup_iters):
        for x in inputs:
            infer_engine.forward({input_name: x}, debug=False)
        if device == "xpu":
            torch.xpu.synchronize()
        print(f"  Warmup iter {i + 1}/{args.warmup_iters} done")
    print("Warmup done.")

    # =====================================================================
    # GraphInfer 顺序推理
    # =====================================================================
    print("\n" + "=" * 70)
    print("GraphInfer: 顺序推理所有分辨率")
    print("=" * 70)

    reset_xpu_memory_stats()
    gi_mem_before = get_xpu_memory()["allocated"]

    gi_start = time.perf_counter()
    gi_per_image = []
    for i, (img, x) in enumerate(zip(test_images, inputs)):
        t0 = time.perf_counter()
        infer_engine.forward({input_name: x}, debug=False)
        if device == "xpu":
            torch.xpu.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        gi_per_image.append(elapsed)
        mem_now = get_xpu_memory()
        out_h = img["shape"][2] * scale
        out_w = img["shape"][3] * scale
        print(f"  [{i + 1}/{len(test_images)}] {img['name']:<22} "
              f"{img['shape'][2]}x{img['shape'][3]:<9} -> {out_h}x{out_w:<9} "
              f"{elapsed:>8.2f} ms  "
              f"XPU={mem_now['allocated']:>8.1f} MB  "
              f"peak={mem_now['max_allocated']:>8.1f} MB")

    gi_total_time = (time.perf_counter() - gi_start) * 1000
    gi_peak_mem = get_xpu_memory()["max_allocated"]
    gi_final_mem = get_xpu_memory()["allocated"]

    # =====================================================================
    # PyTorch 对比（如果可用）
    # =====================================================================
    pt_total_time = 0.0
    pt_peak_mem = 0.0
    pt_final_mem = 0.0
    pt_per_image = []

    if not args.no_pytorch:
        print("\n" + "=" * 70)
        print("PyTorch (basicsr RRDBNet): 顺序推理所有分辨率")
        print("=" * 70)

        pt_model = SRModel(scale=scale).to(device)
        pt_model.eval()

        # PyTorch warmup
        x0 = inputs[0]
        for _ in range(args.warmup_iters):
            with torch.no_grad():
                _ = pt_model(x0)
        if device == "xpu":
            torch.xpu.synchronize()

        reset_xpu_memory_stats()
        pt_mem_before = get_xpu_memory()["allocated"]

        pt_start = time.perf_counter()
        for i, (img, x) in enumerate(zip(test_images, inputs)):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = pt_model(x)
            if device == "xpu":
                torch.xpu.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            pt_per_image.append(elapsed)
            mem_now = get_xpu_memory()
            out_h = img["shape"][2] * scale
            out_w = img["shape"][3] * scale
            print(f"  [{i + 1}/{len(test_images)}] {img['name']:<22} "
                  f"{img['shape'][2]}x{img['shape'][3]:<9} -> {out_h}x{out_w:<9} "
                  f"{elapsed:>8.2f} ms  "
                  f"XPU={mem_now['allocated']:>8.1f} MB  "
                  f"peak={mem_now['max_allocated']:>8.1f} MB")

        pt_total_time = (time.perf_counter() - pt_start) * 1000
        pt_peak_mem = get_xpu_memory()["max_allocated"]
        pt_final_mem = get_xpu_memory()["allocated"]

        # ---- 数值精度对比（仅第一个分辨率） ----
        print("\n" + "-" * 70)
        print("数值精度对比（GraphInfer vs PyTorch RRDBNet）")
        print("-" * 70)
        x = inputs[0]
        with torch.no_grad():
            pt_out = pt_model(x)
        if device == "xpu":
            torch.xpu.synchronize()
        gi_out = infer_engine.forward({input_name: x}, debug=False)[list(
            infer_engine.forward({input_name: x}, debug=False).keys())[0]]
        # 重新跑一次拿到正确的 output
        gi_result = infer_engine.forward({input_name: x}, debug=False)
        gi_out = gi_result[list(gi_result.keys())[0]]
        if device == "xpu":
            torch.xpu.synchronize()

        pt_np = pt_out.cpu().numpy()
        gi_np = gi_out.cpu().numpy()
        max_diff = np.max(np.abs(pt_np - gi_np))
        mean_diff = np.mean(np.abs(pt_np - gi_np))
        print(f"  Resolution: {test_images[0]['shape'][2]}x{test_images[0]['shape'][3]}")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        if max_diff < 0.1:
            print(f"  ✓ 精度合格 (max_diff < 0.1)")
        else:
            print(f"  ⚠ 精度偏差较大 (max_diff >= 0.1)")

    # =====================================================================
    # Profile 分析（单分辨率）
    # =====================================================================
    if args.profile:
        print("\n" + "=" * 70)
        print("Profile 分析（使用中等分辨率）")
        print("=" * 70)

        # 选一个中等分辨率做 profile
        profile_idx = len(test_images) // 2
        profile_img = test_images[profile_idx]
        profile_x = inputs[profile_idx]

        print(f"Profile resolution: {profile_img['name']} "
              f"({profile_img['shape'][2]}x{profile_img['shape'][3]})")

        result = infer_engine.forward({input_name: profile_x}, debug=False, profile=True)
        if device == "xpu":
            torch.xpu.synchronize()

        infer_engine.print_profile(result["__profile__"])

    # =====================================================================
    # 汇总
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    if pt_total_time > 0:
        print(f"\n{'':<35} {'PyTorch':>14} {'GraphInfer':>14}")
        print("-" * 63)
        print(f"{'Total time (ms)':<35} {pt_total_time:>14.2f} {gi_total_time:>14.2f}")
        print(f"{'Speedup':<35} {'':>14} {pt_total_time / gi_total_time:>13.2f}x")
        print(f"{'Peak XPU memory (MB)':<35} {pt_peak_mem:>14.1f} {gi_peak_mem:>14.1f}")
        print(f"{'Final XPU memory (MB)':<35} {pt_final_mem:>14.1f} {gi_final_mem:>14.1f}")
    else:
        print(f"\n{'':<35} {'GraphInfer':>14}")
        print("-" * 49)
        print(f"{'Total time (ms)':<35} {gi_total_time:>14.2f}")
        print(f"{'Peak XPU memory (MB)':<35} {gi_peak_mem:>14.1f}")
        print(f"{'Final XPU memory (MB)':<35} {gi_final_mem:>14.1f}")

    print(f"{'GraphInfer pool (MB)':<35} {'':>14} "
          f"{infer_engine.pool.compress_size / 1024 / 1024:>14.1f}")
    print("-" * 63)

    # 逐张耗时明细
    print(f"\n{'':<35} {'GI(ms)':>10} {'Output Size':>18}")
    print("-" * 63)
    for i, img in enumerate(test_images):
        out_h = img["shape"][2] * scale
        out_w = img["shape"][3] * scale
        print(f"{img['name']:<35} {gi_per_image[i]:>10.2f}  {out_h}x{out_w:<10}")

    if pt_per_image:
        print(f"\n{'':<35} {'PT(ms)':>10} {'GI(ms)':>10} {'Ratio':>8}")
        print("-" * 63)
        for i, img in enumerate(test_images):
            ratio = pt_per_image[i] / gi_per_image[i] if gi_per_image[i] > 0 else 0
            print(f"{img['name']:<35} {pt_per_image[i]:>10.2f} {gi_per_image[i]:>10.2f} {ratio:>7.2f}x")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
