"""
GraphInfer + ResNet18 输出对比测试脚本

使用小数据集对比 PyTorch 和 GraphInfer 的逐样本输出差异。
支持两种数据模式:
  - cifar10: 使用 CIFAR-10 测试集（需下载，约 170MB）
  - synthetic: 使用随机生成图片（无需下载，快速测试）

用法:
    python test_accuracy.py --data_mode synthetic          # 随机数据快速测试
    python test_accuracy.py --data_mode cifar10            # CIFAR-10 数据集
    python test_accuracy.py --num_samples 100              # 自定义样本数
    python test_accuracy.py --device cpu                   # 指定设备
    python test_accuracy.py --rtol 1e-3 --atol 1e-4        # 自定义容忍度
"""

import argparse
import time
import torch
import torchvision
import numpy as np
from typing import List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="GraphInfer + ResNet18 输出对比")
    parser.add_argument("--data_mode", type=str, default="synthetic",
                        choices=["cifar10", "synthetic"],
                        help="数据模式: cifar10 / synthetic (default)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="测试样本数 (default: 100)")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备 (cpu/xpu, default: auto)")
    parser.add_argument("--rtol", type=float, default=1e-3,
                        help="相对误差容忍度 (default: 1e-3)")
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="绝对误差容忍度 (default: 1e-4)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="推理 batch size (default: 1)")
    parser.add_argument("--onnx_path", type=str, default="resnet18.onnx",
                        help="ONNX 模型路径 (default: resnet18.onnx)")
    parser.add_argument("--export_onnx", action="store_true", default=True,
                        help="是否导出 ONNX 模型 (default: True)")
    parser.add_argument("--resolution", type=int, default=224,
                        help="输入分辨率 (default: 224)")
    return parser.parse_args()


def get_device(device_arg: str = None) -> str:
    if device_arg is not None:
        return device_arg
    if torch.xpu.is_available():
        return "xpu"
    return "cpu"


def export_resnet18_onnx(onnx_path: str, resolution: int = 224) -> torch.nn.Module:
    """导出 ResNet18 到 ONNX 格式"""
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    dummy_input = torch.randn(1, 3, resolution, resolution)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )

    print(f"ONNX model exported to {onnx_path}")
    return model


def load_synthetic_samples(num_samples: int, resolution: int) -> torch.Tensor:
    """生成随机测试数据"""
    np.random.seed(42)
    torch.manual_seed(42)
    images = torch.randn(num_samples, 3, resolution, resolution) * 0.5 + 0.5
    images = images.clamp(0, 1).float()
    return images


def load_cifar10_samples(num_samples: int, resolution: int) -> torch.Tensor:
    """加载 CIFAR-10 测试集样本"""
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    indices = list(range(min(num_samples, len(testset))))
    images = []
    for i in indices:
        img, _ = testset[i]
        images.append(img.unsqueeze(0))

    return torch.cat(images, dim=0)


def compare_outputs(
    pt_output: np.ndarray,
    gi_output: np.ndarray,
) -> dict:
    """
    比较 PyTorch 和 GraphInfer 的输出差异。

    Returns:
        {
            "max_diff": 最大绝对差异,
            "mean_diff": 平均绝对差异,
            "cosine_sim": 余弦相似度,
            "per_sample_max_diff": 每个样本的最大差异列表,
            "per_sample_cosine": 每个样本的余弦相似度列表,
        }
    """
    num_samples = pt_output.shape[0]
    per_sample_max_diff = []
    per_sample_cosine = []
    for i in range(num_samples):
        diff = np.max(np.abs(pt_output[i] - gi_output[i]))
        per_sample_max_diff.append(diff)
        a = pt_output[i].flatten()
        b = gi_output[i].flatten()
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        per_sample_cosine.append(cos)

    max_diff = np.max(per_sample_max_diff)
    mean_diff = np.mean(per_sample_max_diff)

    pt_flat = pt_output.flatten()
    gi_flat = gi_output.flatten()
    cosine_sim = np.dot(pt_flat, gi_flat) / (
        np.linalg.norm(pt_flat) * np.linalg.norm(gi_flat) + 1e-12
    )

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cosine_sim": cosine_sim,
        "per_sample_max_diff": per_sample_max_diff,
        "per_sample_cosine": per_sample_cosine,
    }


def print_diff_summary(diff_result: dict):
    """打印输出差异汇总"""
    print(f"\n{'='*50}")
    print("输出数值差异汇总")
    print("=" * 50)
    print(f"最大绝对差异 (Max diff):     {diff_result['max_diff']:.6e}")
    print(f"平均绝对差异 (Mean diff):    {diff_result['mean_diff']:.6e}")
    print(f"余弦相似度 (Cosine sim):     {diff_result['cosine_sim']:.8f}")
    print("=" * 50)


def print_worst_samples(diff_result: dict, top_n: int = 5):
    """打印差异最大的 top-N 样本"""
    per_sample = diff_result["per_sample_max_diff"]
    worst_indices = np.argsort(-np.array(per_sample))[:top_n]

    print(f"\n差异最大的 {top_n} 个样本:")
    print(f"{'Idx':<6} {'Max Diff':<14} {'Cosine Sim':<14}")
    print("-" * 34)
    for idx in worst_indices:
        print(f"{idx:<6} {per_sample[idx]:<14.6e} {diff_result['per_sample_cosine'][idx]:<14.8f}")


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Test samples: {args.num_samples}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Tolerance: rtol={args.rtol}, atol={args.atol}")
    print()

    # ===== 1. 导出 ONNX 模型 =====
    if args.export_onnx:
        print("=== 1. 导出 ResNet18 到 ONNX ===")
        pytorch_model = export_resnet18_onnx(args.onnx_path, args.resolution)
    else:
        print("=== 1. 加载预训练 ResNet18 ===")
        pytorch_model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )
        pytorch_model.eval()

    pytorch_model = pytorch_model.to(device)

    # ===== 2. 加载测试数据 =====
    if args.data_mode == "cifar10":
        print(f"\n=== 2. 加载 CIFAR-10 测试集 ({args.num_samples} 张) ===")
        images = load_cifar10_samples(args.num_samples, args.resolution)
        print(f"Loaded {len(images)} samples from CIFAR-10 test set")
    else:
        print(f"\n=== 2. 生成随机测试数据 ({args.num_samples} 张) ===")
        images = load_synthetic_samples(args.num_samples, args.resolution)
        print(f"Generated {len(images)} synthetic samples")
    print(f"Image shape: {images.shape}")

    # ===== 3. 初始化 GraphInfer =====
    print("\n=== 3. 初始化 GraphInfer ===")
    from infer import GraphInfer
    from kernels import KernelRegistry
    print(f"Registered kernels: {len(KernelRegistry.registered_ops())} ops")

    infer_engine = GraphInfer(
        args.onnx_path,
        {'input': ('batch', 3, 'height', 'width')},
        {
            'batch': (args.batch_size, args.batch_size),
            'height': (args.resolution, args.resolution),
            'width': (args.resolution, args.resolution),
        },
        dtype=torch.float32,
        device=device,
    )
    infer_engine.print_summary()

    # ===== 4. 预热 =====
    print("\n=== 4. 预热 ===")
    warmup_input = images[:args.batch_size].to(device)
    with torch.no_grad():
        _ = pytorch_model(warmup_input)
    if device == "xpu":
        torch.xpu.synchronize()

    infer_engine.forward({'input': warmup_input}, debug=False)
    if device == "xpu":
        torch.xpu.synchronize()
    print("Warmup done.")

    # ===== 5. 推理所有样本 =====
    print("\n=== 5. 推理所有样本 ===")

    # PyTorch 推理
    print("\n--- PyTorch 推理 ---")
    pt_start = time.perf_counter()
    all_pt_outputs = []
    for i in range(0, len(images), args.batch_size):
        batch = images[i:i + args.batch_size].to(device)
        with torch.no_grad():
            out = pytorch_model(batch)
        all_pt_outputs.append(out.cpu().numpy().copy())
    if device == "xpu":
        torch.xpu.synchronize()
    pt_elapsed = time.perf_counter() - pt_start
    pt_outputs = np.concatenate(all_pt_outputs, axis=0)
    print(f"PyTorch inference done: {pt_elapsed:.4f}s ({pt_elapsed/len(images)*1000:.2f} ms/sample)")

    # GraphInfer 推理
    print("\n--- GraphInfer 推理 ---")
    gi_start = time.perf_counter()
    all_gi_outputs = []
    for i in range(0, len(images), args.batch_size):
        batch = images[i:i + args.batch_size].to(device)
        out = infer_engine.forward({'input': batch}, debug=False)['output']
        # 必须深拷贝！out 是 memory pool 的 view，后续 forward 会覆盖
        all_gi_outputs.append(out.cpu().numpy().copy())
    if device == "xpu":
        torch.xpu.synchronize()
    gi_elapsed = time.perf_counter() - gi_start
    gi_outputs = np.concatenate(all_gi_outputs, axis=0)
    print(f"GraphInfer inference done: {gi_elapsed:.4f}s ({gi_elapsed/len(images)*1000:.2f} ms/sample)")

    # ===== 6. 比较输出差异 =====
    print("\n=== 6. 比较输出数值差异 ===")
    diff_result = compare_outputs(pt_outputs, gi_outputs)
    print_diff_summary(diff_result)

    # ===== 7. 打印差异最大的样本 =====
    print("\n=== 7. 差异最大的样本详情 ===")
    print_worst_samples(diff_result, top_n=5)

    # ===== 8. 预测一致性 =====
    pt_preds = np.argmax(pt_outputs, axis=1)
    gi_preds = np.argmax(gi_outputs, axis=1)
    pred_match_count = np.sum(pt_preds == gi_preds)
    pred_match_ratio = pred_match_count / len(images)
    print(f"\n预测一致: {pred_match_count}/{len(images)} ({pred_match_ratio:.2%})")

    # ===== 9. 汇总 =====
    print(f"\n{'='*60}")
    print("对比测试汇总")
    print("=" * 60)
    dataset_name = "CIFAR-10" if args.data_mode == "cifar10" else "Synthetic"
    print(f"数据集: {dataset_name} ({len(images)} 张)")
    print(f"模型: ResNet18 (ImageNet 预训练)")
    print(f"分辨率: {args.resolution}x{args.resolution}")
    print(f"设备: {device}")
    print(f"容忍度: rtol={args.rtol}, atol={args.atol}")
    print()
    print(f"输出数值差异:")
    print(f"  最大绝对差异: {diff_result['max_diff']:.6e}")
    print(f"  平均绝对差异: {diff_result['mean_diff']:.6e}")
    print(f"  余弦相似度:   {diff_result['cosine_sim']:.8f}")
    print(f"  预测一致率:   {pred_match_ratio:.2%}")
    print()
    print(f"性能:")
    print(f"  PyTorch:    {pt_elapsed:.4f}s ({pt_elapsed/len(images)*1000:.2f} ms/sample)")
    print(f"  GraphInfer: {gi_elapsed:.4f}s ({gi_elapsed/len(images)*1000:.2f} ms/sample)")
    if gi_elapsed > 0:
        print(f"  速度比: {pt_elapsed/gi_elapsed:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
