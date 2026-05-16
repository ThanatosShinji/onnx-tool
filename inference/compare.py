"""
GraphInfer 推理对比工具

ONNX Runtime 始终在 CPU 上运行（无 XPU/OpenVINO 后端）。
PyTorch 和 GraphInfer 在 XPU 上运行。

对比项目:
  - ONNX Runtime (CPU) vs GraphInfer (XPU)  — 跨设备数值一致性
  - PyTorch (XPU) vs GraphInfer (XPU)        — 同设备数值一致性
"""

import time
import torch
import torch.nn as nn
import torch.onnx
import onnxruntime as ort
import numpy as np
import torchvision
import psutil
import os


def get_memory_usage() -> float:
    """获取当前进程内存使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def run_pytorch(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 10,
    num_iter: int = 50,
    use_fp16: bool = False,
) -> tuple:
    """使用 PyTorch 推理，返回 (输出, 耗时秒, 内存增量MB)

    Args:
        use_fp16: 如果为 True，在 XPU 上使用 autocast(fp16) 进行半精度推理。
    """
    is_xpu = input_tensor.device.type == 'xpu'

    def _run_inference():
        if use_fp16 and is_xpu:
            if hasattr(torch.xpu, 'amp') and hasattr(torch.xpu.amp, 'autocast'):
                with torch.xpu.amp.autocast(dtype=torch.float16):
                    return model(input_tensor)
            else:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    return model(input_tensor)
        else:
            return model(input_tensor)

    for _ in range(num_warmup):
        with torch.no_grad():
            _ = _run_inference()
    if is_xpu:
        torch.xpu.synchronize()

    mem_before = get_memory_usage()
    start = time.perf_counter()
    for _ in range(num_iter):
        with torch.no_grad():
            output = _run_inference()
    if is_xpu:
        torch.xpu.synchronize()
    elapsed = (time.perf_counter() - start) / num_iter

    mem_after = get_memory_usage()
    return output.cpu().numpy(), elapsed, mem_after - mem_before


def run_onnx(
    session: ort.InferenceSession,
    input_tensor: np.ndarray,
    num_warmup: int = 10,
    num_iter: int = 50,
) -> tuple:
    """使用 ONNX Runtime 推理，返回 (输出, 耗时秒, 内存增量MB)"""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    for _ in range(num_warmup):
        _ = session.run([output_name], {input_name: input_tensor})

    mem_before = get_memory_usage()
    start = time.perf_counter()
    for _ in range(num_iter):
        result = session.run([output_name], {input_name: input_tensor})
    elapsed = (time.perf_counter() - start) / num_iter
    mem_after = get_memory_usage()

    return result[0], elapsed, mem_after - mem_before


def run_graphinfer(
    engine,
    input_tensor: torch.Tensor,
    device: str,
    num_warmup: int = 10,
    num_iter: int = 50,
) -> tuple:
    """使用 GraphInfer 推理，返回 (输出, 耗时秒, 内存增量MB)"""
    for _ in range(num_warmup):
        engine.forward({'input': input_tensor}, debug=False)
    if device == 'xpu':
        torch.xpu.synchronize()

    mem_before = get_memory_usage()
    start = time.perf_counter()
    for _ in range(num_iter):
        outputs = engine.forward({'input': input_tensor}, debug=False)
    if device == 'xpu':
        torch.xpu.synchronize()
    elapsed = (time.perf_counter() - start) / num_iter
    mem_after = get_memory_usage()

    return outputs['output'].cpu().numpy(), elapsed, mem_after - mem_before


def profile_graphinfer_overhead(
    engine,
    input_tensor: torch.Tensor,
    device: str,
    num_iter: int = 20,
) -> dict:
    """使用 profile 模式运行 GraphInfer，分析 overhead 占比。

    Returns:
        dict: 包含 overhead_total, kernel_total, total_time, overhead_pct 等统计
    """
    # warmup
    for _ in range(5):
        engine.forward({'input': input_tensor}, debug=False)
    if device == 'xpu':
        torch.xpu.synchronize()

    # profile run
    outputs = engine.forward({'input': input_tensor}, debug=False, profile=True)
    if device == 'xpu':
        torch.xpu.synchronize()

    profile_data = outputs['__profile__']
    engine.print_profile(profile_data)

    overhead_total = profile_data['overhead_total']
    kernel_total = sum(profile_data['op_time'].values())
    total_time = overhead_total + kernel_total
    ob = profile_data.get('overhead_breakdown', {})

    result = {
        'overhead_total': overhead_total,
        'kernel_total': kernel_total,
        'total_time': total_time,
        'overhead_pct': overhead_total / total_time * 100 if total_time > 0 else 0,
        'kernel_pct': kernel_total / total_time * 100 if total_time > 0 else 0,
        'num_nodes': len(profile_data['node_times']),
        'avg_overhead_per_node': overhead_total / len(profile_data['node_times']) * 1000 if profile_data['node_times'] else 0,
        'overhead_breakdown': {
            'resolve_input': ob.get('resolve_input', 0),
            'prepare_output': ob.get('prepare_output', 0),
            'kernel_lookup': ob.get('kernel_lookup', 0),
        },
    }
    return result


def compare_outputs(
    ref_output: np.ndarray,
    test_output: np.ndarray,
    label_a: str = "Reference",
    label_b: str = "GraphInfer",
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """比较两个输出是否一致"""
    if ref_output.shape != test_output.shape:
        print(f"Shape mismatch: {ref_output.shape} vs {test_output.shape}")
        return False

    max_diff = np.max(np.abs(ref_output - test_output))
    mean_diff = np.mean(np.abs(ref_output - test_output))
    cosine_sim = np.dot(ref_output.flatten(), test_output.flatten()) / (
        np.linalg.norm(ref_output.flatten()) * np.linalg.norm(test_output.flatten())
    )

    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    print(f"Cosine similarity: {cosine_sim:.8f}")

    is_close = np.allclose(ref_output, test_output, rtol=rtol, atol=atol)
    if is_close:
        print(f"✓ {label_a} vs {label_b}: match within rtol={rtol}, atol={atol}")
    else:
        print(f"✗ {label_a} vs {label_b}: differ beyond rtol={rtol}, atol={atol}")

    return is_close


def main():
    resolution = 640
    onnx_path = "resnet18.onnx"

    print(f"PyTorch/GraphInfer device: XPU")
    print(f"ONNX Runtime device: CPU")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"ONNX Runtime available providers: {ort.get_available_providers()}")
    print()

    # ===== 1. 导出 ONNX 模型 =====
    print("=== 1. Exporting ResNet18 to ONNX ===")
    # CPU 模型
    pytorch_model_cpu = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    pytorch_model_cpu.eval()
    # XPU 模型（独立实例，避免设备冲突）
    pytorch_model_xpu = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    pytorch_model_xpu.eval().to('xpu')

    # XPU FP16 模型（半精度）
    pytorch_model_xpu_fp16 = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    pytorch_model_xpu_fp16.eval().to('xpu').to(torch.float16)

    # 导出 ONNX（用 CPU 模型导出）
    dummy = torch.randn(1, 3, resolution, resolution)
    torch.onnx.export(
        pytorch_model_cpu, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17, dynamo=False,
    )
    print(f"ONNX model exported to {onnx_path}")

    # ===== 2. 创建 ONNX Runtime session =====
    print("\n=== 2. Creating ONNX Runtime session ===")
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # ===== 3. 初始化 GraphInfer =====
    print("\n=== 3. Initializing GraphInfer ===")
    from infer import GraphInfer
    from kernels import KernelRegistry
    print(f"Registered kernels: {len(KernelRegistry.registered_ops())} ops")

    # GraphInfer XPU
    infer_engine_xpu = GraphInfer(
        onnx_path,
        {'input': ('batch', 3, 'height', 'width')},
        {'batch': (1, 1), 'height': (resolution, resolution), 'width': (resolution, resolution)},
        dtype=torch.float32,
        device='xpu',
    )
    infer_engine_xpu.print_summary()

    # GraphInfer CPU（用于与 ONNX Runtime 同设备对比）
    print("\n--- GraphInfer (CPU) ---")
    infer_engine_cpu = GraphInfer(
        onnx_path,
        {'input': ('batch', 3, 'height', 'width')},
        {'batch': (1, 1), 'height': (resolution, resolution), 'width': (resolution, resolution)},
        dtype=torch.float32,
        device='cpu',
    )
    infer_engine_cpu.print_summary()

    # GraphInfer XPU FP16
    print("\n--- GraphInfer (XPU FP16) ---")
    infer_engine_xpu_fp16 = GraphInfer(
        onnx_path,
        {'input': ('batch', 3, 'height', 'width')},
        {'batch': (1, 1), 'height': (resolution, resolution), 'width': (resolution, resolution)},
        dtype=torch.float16,
        device='xpu',
    )
    infer_engine_xpu_fp16.print_summary()

    # ===== 4. 准备输入数据 =====
    print("\n=== 4. Preparing input data ===")
    np.random.seed(42)
    torch.manual_seed(42)

    input_np = np.random.randn(1, 3, resolution, resolution).astype(np.float32)
    input_tensor = torch.from_numpy(input_np)

    # ===== 5. 运行推理 =====
    print("\n=== 5. Running inference ===")

    # ONNX Runtime — CPU
    print("\n--- ONNX Runtime (CPU) ---")
    onnx_output, ort_time, ort_mem = run_onnx(ort_session, input_np)

    # PyTorch CPU
    print("\n--- PyTorch (CPU) ---")
    pt_cpu_output, pt_cpu_time, pt_cpu_mem = run_pytorch(pytorch_model_cpu, input_tensor)

    # GraphInfer CPU
    print("\n--- GraphInfer (CPU) ---")
    gi_cpu_output, gi_cpu_time, gi_cpu_mem = run_graphinfer(infer_engine_cpu, input_tensor, 'cpu')

    # PyTorch XPU
    print("\n--- PyTorch (XPU) ---")
    pt_xpu_output, pt_xpu_time, pt_xpu_mem = run_pytorch(pytorch_model_xpu, input_tensor.to('xpu'))

    # GraphInfer XPU
    print("\n--- GraphInfer (XPU) ---")
    gi_xpu_output, gi_xpu_time, gi_xpu_mem = run_graphinfer(infer_engine_xpu, input_tensor.to('xpu'), 'xpu')

    # PyTorch XPU FP16
    print("\n--- PyTorch (XPU FP16) ---")
    pt_xpu_fp16_output, pt_xpu_fp16_time, pt_xpu_fp16_mem = run_pytorch(
        pytorch_model_xpu_fp16, input_tensor.to('xpu').to(torch.float16), use_fp16=False)

    # GraphInfer XPU FP16
    print("\n--- GraphInfer (XPU FP16) ---")
    gi_xpu_fp16_output, gi_xpu_fp16_time, gi_xpu_fp16_mem = run_graphinfer(
        infer_engine_xpu_fp16, input_tensor.to('xpu').to(torch.float16), 'xpu')

    # ===== 6. 比较输出 =====
    print("\n=== 6. Comparing outputs ===")

    print("\n--- CPU: ONNX Runtime vs GraphInfer ---")
    match_gi_cpu_ort = compare_outputs(onnx_output, gi_cpu_output, "ONNX Runtime(CPU)", "GraphInfer(CPU)")

    print("\n--- CPU: ONNX Runtime vs PyTorch ---")
    match_pt_cpu_ort = compare_outputs(onnx_output, pt_cpu_output, "ONNX Runtime(CPU)", "PyTorch(CPU)")

    print("\n--- CPU: PyTorch vs GraphInfer ---")
    match_gi_cpu_pt = compare_outputs(pt_cpu_output, gi_cpu_output, "PyTorch(CPU)", "GraphInfer(CPU)")

    print("\n--- XPU: PyTorch vs GraphInfer ---")
    match_gi_xpu_pt = compare_outputs(pt_xpu_output, gi_xpu_output, "PyTorch(XPU)", "GraphInfer(XPU)")

    print("\n--- Cross: ONNX Runtime (CPU) vs GraphInfer (XPU) ---")
    match_gi_xpu_ort = compare_outputs(onnx_output, gi_xpu_output, "ONNX Runtime(CPU)", "GraphInfer(XPU)")

    print("\n--- XPU FP16: PyTorch vs GraphInfer ---")
    match_gi_xpu_fp16_pt = compare_outputs(
        pt_xpu_fp16_output, gi_xpu_fp16_output, "PyTorch(XPU FP16)", "GraphInfer(XPU FP16)",
        rtol=1e-2, atol=1e-3)

    print("\n--- Cross: ONNX Runtime (CPU) vs GraphInfer (XPU FP16) ---")
    match_gi_xpu_fp16_ort = compare_outputs(
        onnx_output, gi_xpu_fp16_output, "ONNX Runtime(CPU)", "GraphInfer(XPU FP16)",
        rtol=1e-2, atol=1e-3)

    print("\n--- Cross: ONNX Runtime (CPU) vs PyTorch (XPU FP16) ---")
    match_pt_xpu_fp16_ort = compare_outputs(
        onnx_output, pt_xpu_fp16_output, "ONNX Runtime(CPU)", "PyTorch(XPU FP16)",
        rtol=1e-2, atol=1e-3)

    # ===== 7. Overhead Profiling =====
    print("\n=== 7. Overhead Profiling (GraphInfer XPU FP32) ===")
    prof_result = profile_graphinfer_overhead(
        infer_engine_xpu, input_tensor.to('xpu'), 'xpu', num_iter=20)

    print(f"\nOverhead Summary:")
    print(f"  Total nodes: {prof_result['num_nodes']}")
    print(f"  Total time: {prof_result['total_time']*1000:.4f} ms")
    print(f"  Kernel time: {prof_result['kernel_total']*1000:.4f} ms ({prof_result['kernel_pct']:.1f}%)")
    print(f"  Overhead time: {prof_result['overhead_total']*1000:.4f} ms ({prof_result['overhead_pct']:.1f}%)")
    print(f"  Avg overhead per node: {prof_result['avg_overhead_per_node']:.4f} ms")
    ob = prof_result['overhead_breakdown']
    print(f"\n  Overhead Breakdown:")
    print(f"    Resolve Input:   {ob['resolve_input']*1000:.4f} ms ({ob['resolve_input']/prof_result['overhead_total']*100:.1f}%)")
    print(f"    Prepare Output:  {ob['prepare_output']*1000:.4f} ms ({ob['prepare_output']/prof_result['overhead_total']*100:.1f}%)")
    print(f"    Kernel Lookup:   {ob['kernel_lookup']*1000:.4f} ms ({ob['kernel_lookup']/prof_result['overhead_total']*100:.1f}%)")

    # ===== 8. 汇总 =====
    # CPU 表
    print(f"\n{'='*75}")
    print("PERFORMANCE SUMMARY — CPU")
    print("=" * 75)
    print(f"{'':<20} {'ONNX Runtime':>14} {'PyTorch':>12} {'GraphInfer':>12} {'Speedup':>10}")
    print(f"{'':<20} {'(CPU)':>14} {'(CPU)':>12} {'(CPU)':>12} {'':>10}")
    print("-" * 68)
    print(f"{'Time (s)':<20} {ort_time:>14.4f} {pt_cpu_time:>12.4f} {gi_cpu_time:>12.4f} {pt_cpu_time/gi_cpu_time:>9.2f}x")
    print(f"{'Mem Δ (MB)':<20} {ort_mem:>14.2f} {pt_cpu_mem:>12.2f} {gi_cpu_mem:>12.2f} {'':>10}")
    print("=" * 75)

    # XPU 表
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY — XPU")
    print("=" * 80)
    print(f"{'':<22} {'PyTorch':>12} {'GraphInfer':>12} {'Speedup':>10} {'PyTorch':>12} {'GraphInfer':>12}")
    print(f"{'':<22} {'(FP32)':>12} {'(FP32)':>12} {'':>10} {'(FP16)':>12} {'(FP16)':>12}")
    print("-" * 80)
    print(f"{'Time (s)':<22} {pt_xpu_time:>12.4f} {gi_xpu_time:>12.4f} {pt_xpu_time/gi_xpu_time:>9.2f}x {pt_xpu_fp16_time:>12.4f} {gi_xpu_fp16_time:>12.4f}")
    print(f"{'Mem Δ (MB)':<22} {pt_xpu_mem:>12.2f} {gi_xpu_mem:>12.2f} {'':>10} {pt_xpu_fp16_mem:>12.2f} {gi_xpu_fp16_mem:>12.2f}")
    print("=" * 80)

    all_pass = (match_gi_cpu_ort and match_gi_cpu_pt and match_gi_xpu_pt
                and match_gi_xpu_fp16_pt)
    if all_pass:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
