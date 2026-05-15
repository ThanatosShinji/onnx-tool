# ONNX Graph Inference Engine

基于 `onnx_tool` 的 compute graph 和 MemoryPool 的图级别推理引擎，使用 PyTorch 作为后端执行算子。

## 架构

```
inference/
├── memory_pool.py          # MemoryPool — 基于 compress_memory() 的 PyTorch 内存池
├── kernels.py              # 注册式算子内核（Conv, Add, Relu, Gemm 等 40+ 算子）
├── infer.py                # GraphInfer — 计算图推理引擎
├── compare.py              # PyTorch vs ONNX Runtime vs GraphInfer 三端对比
├── test_multi_resolution.py  # 多分辨率顺序推理测试
├── README.md               # 英文文档
└── README_CN.md            # 中文文档
```

### 核心组件

**MemoryPool** — 基于 `cg.compress_memory()` 的结果，分配一块连续 `torch.Tensor` 作为内存池。每个 tensor 通过 `[offset, size]` 映射到池中的一个视图（view），实现零拷贝复用。内存池大小固定，不随输入分辨率变化。

**Kernel Registry** — 注册式算子内核系统。每个算子注册为一个 `Kernel` 子类，实现 `run(inputs, outputs, attrs)` 静态方法。已注册 40+ 算子（Conv, Add, Relu, Gemm, MatMul, BatchNormalization 等）。

**GraphInfer** — 计算图推理引擎。初始化时完成模型加载、节点重排、shape regression、compute graph 提取、内存压缩和 MemoryPool 创建。`forward` 遍历所有节点，通过注册的 kernel 执行推理。

## 性能测试结果

### 测试环境

- **设备**: Intel XPU
- **模型**: ResNet18
- **输入**: 多种分辨率，单 batch，float32
- **ONNX Runtime**: CPUExecutionProvider

### 内存优化

GraphInfer 使用**两遍内存压缩算法**：

1. **第一遍**：运行原始 `compress_memory()` 获取 tensor 生命周期信息
2. **第二遍**：在每个 node 内按大小降序重新分配 tensor，大 tensor 优先，减少碎片和尾部浪费

4K（2160x3840）输入下的优化效果：

| 指标 | 原始算法 | 优化后 | 提升 |
|------|---------|-------|------|
| Pool 大小 | 2214.8 MB | **1012.5 MB** | **↓ 54.3%** |
| 尾部浪费 | 506.2 MB (22.9%) | **0.0 MB (0%)** | 完全消除 |
| 最大同时存活 | 1012.5 MB | 1012.5 MB | 不变 |
| 利用率 | 77.1% | **100%** | +22.9% |

优化后的 pool（1012.5 MB）已接近 PyTorch 的峰值内存（1152 MB），差距仅 12%。

### 单分辨率对比（warmup 后）

| 分辨率 | PyTorch | ONNX Runtime | GraphInfer | 加速比 |
|--------|---------|-------------|------------|--------|
| Batch=1 (1080p) | 0.0151s | 0.1013s | **0.0167s** | **0.91x** |
| Batch=4 (1080p) | 0.0623s | 0.4373s | **0.0681s** | **0.91x** |

GraphInfer 在 warmup 后与 PyTorch 性能非常接近（~91%），远超 ONNX Runtime CPU。

### Profile 分析（单次 forward 耗时分布）

| Op Type | 次数 | 总耗时 | 平均每次 | 占比 |
|---------|------|--------|---------|------|
| Conv | 20 | 14.0ms | 0.70ms | **70.0%** |
| Relu | 17 | 3.0ms | 0.18ms | 15.2% |
| Add | 8 | 1.3ms | 0.16ms | 6.4% |
| MaxPool | 1 | 0.6ms | 0.55ms | 2.7% |
| Gemm | 1 | 0.2ms | 0.15ms | 0.8% |
| GlobalAveragePool | 1 | 0.1ms | 0.13ms | 0.7% |
| Flatten | 1 | 0.04ms | 0.04ms | 0.2% |
| **Overhead（prepare）** | — | **0.8ms** | — | **4.1%** |
| **Total** | 49 | **20.0ms** | — | **100%** |

- Conv 占 70% 以上耗时，是主要优化目标
- Overhead（tensor prepare/reshape）仅占 4.1%，非常低

### 多分辨率顺序推理

模拟真实动态分辨率场景：依次推理 7 张不同分辨率的图像，统计总耗时和峰值 XPU 内存。

| 分辨率 | PyTorch | GraphInfer | 加速比 |
|--------|---------|------------|--------|
| HD (720p) | 6.68ms | 8.06ms | 0.83x |
| Full HD (1080p) | 311.70ms | 17.16ms | **18.17x** |
| 2K (1440p) | 286.09ms | 28.88ms | **9.91x** |
| 4K (2160p) | 366.22ms | 66.91ms | **5.47x** |
| Square Large | 322.74ms | 32.90ms | **9.81x** |
| Mobile (360p) | 262.82ms | 3.49ms | **75.29x** |
| Small (224x224) | 262.05ms | 2.25ms | **116.55x** |
| **Total** | **1820.35ms** | **161.54ms** | **11.27x** |

| 指标 | PyTorch | GraphInfer |
|------|---------|------------|
| 总耗时 | 1820.35 ms | **161.54 ms** |
| 峰值 XPU 内存 | 2338.0 MB | **1926.7 MB** |
| 最终 XPU 内存 | 1325.5 MB | 1325.5 MB |
| Memory Pool | — | **1012.5 MB** |

**GraphInfer 在动态分辨率场景下比 PyTorch 快 11.3x**，原因：
- PyTorch 在分辨率切换时触发 XPU kernel 重编译（~300ms/张）
- GraphInfer 的 memory pool 固定，kernel 在 warmup 时已编译，后续只需 reshape 视图
- 小分辨率下优势更明显（360p: 75x, 224x224: 117x）

### XPU 内存对比

| 分辨率 | PyTorch XPU 峰值 | GraphInfer XPU 峰值 |
|--------|-----------------|-------------------|
| HD (720p) | 1438.0 MB | 1392.3 MB |
| Full HD (1080p) | 1578.6 MB | 1475.8 MB |
| 2K (1440p) | 1776.5 MB | 1593.7 MB |
| 4K (2160p) | **2338.0 MB** | **1926.7 MB** |
| Square Large | 2338.0 MB | 1926.7 MB |

GraphInfer 峰值 XPU 内存比 PyTorch 低 **17.6%**（1926 MB vs 2338 MB），pool 仅 **1012.5 MB**——比原始算法减少 **54%**。

> **注意：** 顺序推理时峰值 XPU 内存（1926 MB）大于 pool（1012.5 MB），是因为 XPU 在分辨率切换时会保留 kernel 编译缓存和临时 buffer。在**固定分辨率的单次推理**中，XPU 内存 ≈ **pool + 常量**（1057 MB = 1012.5 MB pool + 44.6 MB 权重），几乎没有额外临时 buffer（~0 MB）。

### Profiling 分析（固定分辨率）

**时间：** Kernel 总和（19.2 ms）+ Overhead（0.8 ms）= **20.0 ms**，无隐藏开销。4.1% 的 overhead 来自 `_resolve_tensor` 的 shape 查询和 `_reshape_view`。

**内存：** XPU allocated（1057 MB）≈ Pool（1012.5 MB）+ Constants（44.6 MB），无额外临时 buffer。全部 49 个 node 已验证：**输入输出 tensor 在 pool 中零重叠**。

## 使用方法

```python
from infer import GraphInfer

# 初始化
infer_engine = GraphInfer(
    'model.onnx',
    {'input': ('batch', 3, 'height', 'width')},
    {'batch': (1, 4), 'height': (224, 1080), 'width': (224, 1920)},
    dtype=torch.float32,
    device='xpu',  # 或 'cpu'
)

# 推理（输入输出均为 {name: tensor} 格式）
outputs = infer_engine.forward({'input': input_tensor})

# 获取输出
result = outputs['output']

# Profile 模式
result = infer_engine.forward({'input': input_tensor}, profile=True)
infer_engine.print_profile(result['__profile__'])
```

## 注册自定义 Kernel

```python
from kernels import KernelRegistry, Kernel

@KernelRegistry.register("CustomOp")
class CustomOpKernel(Kernel):
    @staticmethod
    def run(inputs, outputs, attrs):
        # inputs: List[torch.Tensor] — 输入 tensor 列表
        # outputs: List[torch.Tensor] — 预分配的输出视图，结果用 copy_ 写入
        # attrs: Dict — 算子属性（kernel_shape, strides 等）
        outputs[0].copy_(custom_function(inputs[0]))
```

## 测试脚本

```bash
# 三端对比（PyTorch vs ONNX Runtime vs GraphInfer）
python inference/compare.py

# 多分辨率顺序推理测试
python inference/test_multi_resolution.py
```
