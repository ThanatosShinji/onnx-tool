"""
XPU vs CPU 逐层对比：用 XPU（精度已验证）校准 CPU 每层输出。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from infer import GraphInfer
from kernels import KernelRegistry


def main():
    resolution = 224

    # 加载一张 CIFAR-10 图片
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    img, label = testset[0]
    x = img.unsqueeze(0)  # [1, 3, 224, 224]

    # ---- 分别在 XPU 和 CPU 上初始化 GraphInfer ----
    gi_xpu = GraphInfer(
        'resnet18.onnx',
        {'input': ('batch', 3, 'height', 'width')},
        {'batch': (1, 1), 'height': (resolution, resolution), 'width': (resolution, resolution)},
        dtype=torch.float32, device='xpu',
    )
    gi_cpu = GraphInfer(
        'resnet18.onnx',
        {'input': ('batch', 3, 'height', 'width')},
        {'batch': (1, 1), 'height': (resolution, resolution), 'width': (resolution, resolution)},
        dtype=torch.float32, device='cpu',
    )

    cg = gi_xpu.compute_graph
    node_names = list(cg.nodemap.keys())

    def forward_collect(engine, input_tensor):
        """执行 forward 并收集每层输出"""
        acts = {}
        engine._update_shape_from_input(input_tensor)

        for input_name, input_tensor_ in {'input': input_tensor}.items():
            if input_name in engine._tensor_views:
                flat = engine._tensor_views[input_name]
                needed = input_tensor_.numel()
                t = flat[:needed].reshape(input_tensor_.shape)
                t.copy_(input_tensor_.to(device=engine.device, dtype=engine.dtype))

        for idx, node_name in enumerate(node_names):
            node = cg.nodemap[node_name]

            input_tensors = []
            for tname in node.input:
                t = engine._resolve_tensor(tname)
                input_tensors.append(t)

            output_tensors = []
            for tname in node.output:
                shape = engine.get_tensor_shape(tname)
                t = engine._reshape_view(tname, shape)
                output_tensors.append(t)

            kernel_cls = KernelRegistry.get(node.op_type)
            if kernel_cls is not None:
                kernel_cls.run(input_tensors, output_tensors, node.attr)

            for tname in node.output:
                if tname in engine._tensor_views:
                    shape = engine.get_tensor_shape(tname)
                    acts[node_name] = engine._reshape_view(tname, shape).clone()

        return acts

    xpu_acts = forward_collect(gi_xpu, x.to('xpu'))
    cpu_acts = forward_collect(gi_cpu, x.to('cpu'))

    # ---- 逐层对比 ----
    print(f"{'Node':<55} {'Op':<12} {'XPU Shape':<22} {'CPU Shape':<22} {'MaxDiff':<14} {'CosSim':<12} {'Status'}")
    print("=" * 150)

    max_diffs = []
    first_bad = None

    for node_name in node_names:
        if node_name not in xpu_acts or node_name not in cpu_acts:
            continue

        node = cg.nodemap[node_name]
        xpu_t = xpu_acts[node_name]
        cpu_t = cpu_acts[node_name]

        xpu_np = xpu_t.cpu().numpy()
        cpu_np = cpu_t.cpu().numpy()

        max_diff = np.max(np.abs(xpu_np - cpu_np))
        max_diffs.append(max_diff)

        cos_sim = np.dot(xpu_np.flatten(), cpu_np.flatten()) / (
            np.linalg.norm(xpu_np.flatten()) * np.linalg.norm(cpu_np.flatten()) + 1e-12
        )

        status = '✓' if max_diff < 1e-4 else '✗'
        if status == '✗' and first_bad is None:
            first_bad = (node_name, node.op_type, max_diff, cos_sim)

        print(f"{node_name:<55} {node.op_type:<12} {str(xpu_np.shape):<22} {str(cpu_np.shape):<22} "
              f"{max_diff:<14.6e} {cos_sim:<12.8f} {status}")

    print("=" * 150)
    print(f"Total nodes: {len([n for n in node_names if n in xpu_acts and n in cpu_acts])}")
    if max_diffs:
        print(f"Overall max diff: {max(max_diffs):.6e}")
        print(f"Overall mean diff: {np.mean(max_diffs):.6e}")

    if first_bad:
        print(f"\n⚠ 第一个问题节点:")
        print(f"  Node: {first_bad[0]} ({first_bad[1]})")
        print(f"  MaxDiff: {first_bad[3]:.6e}")
        print(f"  CosSim: {first_bad[4]:.8f}")

        # 打印该节点的输入输出详情
        node = cg.nodemap[first_bad[0]]
        print(f"\n--- 节点详情 ---")
        print(f"  Attrs: {node.attr}")
        for tname in node.input:
            t_xpu = gi_xpu._resolve_tensor(tname)
            t_cpu = gi_cpu._resolve_tensor(tname)
            if t_xpu is not None and t_cpu is not None:
                xpu_np = t_xpu.cpu().numpy()
                cpu_np = t_cpu.cpu().numpy()
                diff = np.max(np.abs(xpu_np - cpu_np))
                print(f"  Input '{tname}': XPU={t_xpu.shape}, CPU={t_cpu.shape}, diff={diff:.6e}")
            else:
                print(f"  Input '{tname}': XPU={t_xpu}, CPU={t_cpu}")
        for tname in node.output:
            shape_xpu = gi_xpu.get_tensor_shape(tname)
            shape_cpu = gi_cpu.get_tensor_shape(tname)
            print(f"  Output '{tname}': XPU={shape_xpu}, CPU={shape_cpu}")
    else:
        print("\n✓ 所有节点 XPU vs CPU 完全一致！")


if __name__ == '__main__':
    main()
