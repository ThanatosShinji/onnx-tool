"""
PyTorch vs ONNX Runtime 推理对比工具
比较 PyTorch 原生模型和导出的 ONNX 模型在 ResNet18 上的输出是否一致。
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

def export_resnet18_onnx(output_path: str = "resnet18.onnx") -> torch.nn.Module:
    """导出 ResNet18 到 ONNX 格式"""
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 1080, 1920)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )

    print(f"ONNX model exported to {output_path}")
    return model


def get_memory_usage() -> float:
    """获取当前进程内存使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def run_pytorch(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 10,
    num_iter: int = 50,
) -> tuple:
    """使用 PyTorch 推理，返回 (输出, 耗时秒, 内存增量MB)"""
    # warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(input_tensor)
    if torch.xpu.is_available():
        torch.xpu.synchronize()

    mem_before = get_memory_usage()
    if torch.xpu.is_available():
        torch.xpu.synchronize()

    start = time.perf_counter()
    for _ in range(num_iter):
        with torch.no_grad():
            output = model(input_tensor)
    if torch.xpu.is_available():
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

    # warmup
    for _ in range(num_warmup):
        _ = session.run([output_name], {input_name: input_tensor})

    mem_before = get_memory_usage()
    start = time.perf_counter()
    for _ in range(num_iter):
        result = session.run([output_name], {input_name: input_tensor})
    elapsed = (time.perf_counter() - start) / num_iter
    mem_after = get_memory_usage()

    return result[0], elapsed, mem_after - mem_before




def compare_outputs(
    pytorch_output: np.ndarray,
    onnx_output: np.ndarray,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """比较两个输出是否一致"""
    if pytorch_output.shape != onnx_output.shape:
        print(f"Shape mismatch: {pytorch_output.shape} vs {onnx_output.shape}")
        return False

    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
    cosine_sim = np.dot(pytorch_output.flatten(), onnx_output.flatten()) / (
        np.linalg.norm(pytorch_output.flatten()) * np.linalg.norm(onnx_output.flatten())
    )

    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    print(f"Cosine similarity: {cosine_sim:.8f}")

    is_close = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)
    if is_close:
        print(f"✓ Outputs match within rtol={rtol}, atol={atol}")
    else:
        print(f"✗ Outputs differ beyond rtol={rtol}, atol={atol}")

    return is_close


def main():
    import torchvision

    device = "xpu" if torch.xpu.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"ONNX Runtime available providers: {ort.get_available_providers()}")

    # 1. 导出 ONNX 模型
    onnx_path = "resnet18.onnx"
    print("\n=== Exporting ResNet18 to ONNX ===")
    pytorch_model = export_resnet18_onnx(onnx_path)
    pytorch_model = pytorch_model.to(device)

    # 2. 创建 ONNX Runtime session
    print("\n=== Creating ONNX Runtime session ===")
    ort_providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"] if device == "xpu" else ["CPUExecutionProvider"]
    print(f"ONNX Runtime providers: {ort_providers}")
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=ort_providers,
    )

    # 3. 准备输入数据（使用相同的随机种子确保一致性）
    np.random.seed(42)
    torch.manual_seed(42)

    input_np = np.random.randn(1, 3, 1080, 1920).astype(np.float32)
    input_tensor = torch.from_numpy(input_np).to(device)

    # 4. 运行推理
    print("\n=== Running PyTorch inference ===")
    pytorch_output, pt_time, pt_mem = run_pytorch(pytorch_model, input_tensor)

    print("\n=== Running ONNX Runtime inference ===")
    onnx_output, ort_time, ort_mem = run_onnx(ort_session, input_np)

    print("\n=== Running GraphInfer inference ===")
    from infer import GraphInfer
    from kernels import KernelRegistry
    print(f"Registered kernels: {KernelRegistry.registered_ops()}")
    infer_engine = GraphInfer(
        onnx_path,
        {'input': ('batch', 3, 'height', 'width')},
        {'batch': (1, 4), 'height': (224, 1080), 'width': (224, 1920)},
        dtype=torch.float32,
        device=device,
    )
    infer_engine.print_summary()
    # warmup forward
    infer_engine.forward({'input': input_tensor}, debug=False)
    if device == 'xpu':
        torch.xpu.synchronize()

    # 计时 forward（包含 XPU 同步）
    gi_start = time.perf_counter()
    gi_outputs = infer_engine.forward({'input': input_tensor}, debug=False)
    if device == 'xpu':
        torch.xpu.synchronize()
    gi_time = (time.perf_counter() - gi_start)
    gi_output = gi_outputs['output'].cpu().numpy()
    print(f"GraphInfer time: {gi_time:.4f}s")

    # profile forward
    print("\n=== GraphInfer Profile ===")
    gi_profile = infer_engine.forward({'input': input_tensor}, debug=False, profile=True)
    infer_engine.print_profile(gi_profile['__profile__'])

    # 5. 比较输出
    print("\n=== Comparing: PyTorch vs ONNX Runtime ===")
    match_ort = compare_outputs(pytorch_output, onnx_output)

    print("\n=== Comparing: PyTorch vs GraphInfer ===")
    match_gi = compare_outputs(pytorch_output, gi_output)

    # 6. 额外：测试批量推理
    print("\n=== Testing batch inference ===")
    batch_np = np.random.randn(4, 3, 1080, 1920).astype(np.float32)
    batch_tensor = torch.from_numpy(batch_np).to(device)

    pytorch_batch, pt_batch_time, pt_batch_mem = run_pytorch(pytorch_model, batch_tensor)
    onnx_batch, ort_batch_time, ort_batch_mem = run_onnx(ort_session, batch_np)

    # warmup + 计时 batch=4
    infer_engine.forward({'input': batch_tensor}, debug=False)
    if device == 'xpu':
        torch.xpu.synchronize()
    gi_batch_start = time.perf_counter()
    gi_batch_outputs = infer_engine.forward({'input': batch_tensor}, debug=False)
    if device == 'xpu':
        torch.xpu.synchronize()
    gi_batch = gi_batch_outputs['output'].cpu().numpy()
    gi_batch_time = time.perf_counter() - gi_batch_start

    batch_match_ort = compare_outputs(pytorch_batch, onnx_batch)
    batch_match_gi = compare_outputs(pytorch_batch, gi_batch)

    # 7. 汇总结果
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'':<20} {'PyTorch':>12} {'ONNX Runtime':>14} {'GraphInfer':>12} {'Speedup':>10}")
    print("-" * 70)
    print(f"{'Batch=1 Time (s)':<20} {pt_time:>12.4f} {ort_time:>14.4f} {gi_time:>12.4f} {pt_time/gi_time:>9.2f}x")
    print(f"{'Batch=4 Time (s)':<20} {pt_batch_time:>12.4f} {ort_batch_time:>14.4f} {gi_batch_time:>12.4f} {pt_batch_time/gi_batch_time:>9.2f}x")
    print(f"{'Batch=1 Mem Δ (MB)':<20} {pt_mem:>12.2f} {ort_mem:>14.2f} {'':>12} {'':>10}")
    print(f"{'Batch=4 Mem Δ (MB)':<20} {pt_batch_mem:>12.2f} {ort_batch_mem:>14.2f} {'':>12} {'':>10}")
    print("=" * 70)
    if match_ort and match_gi:
        print("✓ All tests passed: PyTorch, ONNX Runtime and GraphInfer outputs match!")
    else:
        print("✗ Some tests failed: outputs differ.")


if __name__ == "__main__":
    main()
