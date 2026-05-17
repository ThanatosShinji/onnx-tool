# ONNX Graph Inference Engine

基于 `onnx_tool` 的 compute graph 和 MemoryPool 的图级别推理引擎，使用 PyTorch 作为后端执行算子。

支持 CV（ResNet、RealESRGAN 等）和 LLM（Qwen3、GPT2 等）模型的推理。

## 架构

```
inference/
├── memory_pool.py              # MemoryPool — 基于 compress_memory() 的 PyTorch 内存池
├── kernels.py                  # 注册式算子内核（Conv, Add, Relu, Gemm 等 50 算子）
├── infer.py                    # GraphInfer — 计算图推理引擎
├── llm_infer.py                # LLMInfer — LLM 自回归推理引擎（KV-cache 支持）
├── compare.py                  # PyTorch vs ONNX Runtime vs GraphInfer 多端对比
├── test_accuracy.py            # 精度验证（CIFAR-10 / 合成数据）
├── test_multi_resolution.py    # 多分辨率顺序推理测试
├── test_super_resolution.py    # 超分模型推理测试
├── benchmark_xpu.py            # XPU GEMM / Conv2d 性能基准
├── plot_xpu_memory.py          # XPU 内存曲线绘制
├── sr_gui.py                   # 超分模型 GUI 工具
├── README.md                   # 英文文档
└── README_CN.md                # 中文文档
```

### 核心组件

**MemoryPool** — 基于 `cg.compress_memory()` 的结果，分配一块连续 `torch.Tensor` 作为内存池。每个 tensor 通过 `[offset, size]` 映射到池中的一个视图（view），实现零拷贝复用。内存池大小固定，不随输入分辨率变化。LLM 模型使用双池设计（activation pool + kv_cache pool），将 KV-cache 与普通 activation 隔离。

**Kernel Registry** — 注册式算子内核系统。每个算子注册为一个 `Kernel` 子类，实现 `run(inputs, outputs, attrs)` 静态方法。支持 `preprocess_weight` 对 weight 做前处理（如 Gemm 预转置）。已注册 50 算子，包括 LLM 专用融合算子（Layernrom、Mad、Gelu、RangeGather、Rope、SDPA 等）。

**GraphInfer** — 计算图推理引擎。初始化时完成模型加载、节点重排、shape regression、compute graph 提取、内存压缩和 MemoryPool 创建。支持 safetensors 外部权重加载。`forward` 遍历所有节点，通过注册的 kernel 执行推理，支持 profile 模式。

**LLMInfer** — LLM 自回归推理引擎。基于 GraphInfer，支持 KV-cache 的 prefill + decode 两阶段推理，与 HuggingFace Transformers 对比验证。支持 Qwen3-0.6B、GPT2 等模型。

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

| 分辨率 | PyTorch (XPU) | ONNX Runtime (CPU) | GraphInfer (XPU) | 加速比 vs PT |
|--------|---------------|--------------------|------------------|-------------|
| Batch=1 (1080p) | 14.92ms | 101.36ms | **16.43ms** | **0.91x** |
| Batch=4 (1080p) | 61.22ms | 430.88ms | **65.88ms** | **0.93x** |

GraphInfer 在 warmup 后达到 PyTorch 性能的 ~91-93%，远超 ONNX Runtime CPU。Kernel 使用 `out=` 参数实现零拷贝输出，消除了早期版本的 `copy_` 开销。

### FP16 性能（640x640，XPU）

| 精度 | PyTorch (XPU) | GraphInfer (XPU) | 加速比 vs PT |
|------|---------------|------------------|-------------|
| FP32 | 3.3ms | 3.5ms | 0.95x |
| FP16 | 1.8ms | **1.3ms** | **1.38x** |

GraphInfer 在 XPU 上同时支持 FP32 和 FP16 精度。FP16 模式下 GraphInfer 比 PyTorch 快 38%，得益于内存池在半精度下的开销更低。

### Profile 分析（1080p 单次 forward 耗时分布）

| Op Type | 次数 | 总耗时 | 平均每次 | 占比 |
|---------|------|--------|---------|------|
| Conv | 20 | 13.76ms | 0.69ms | **75.1%** |
| Relu | 17 | 2.75ms | 0.16ms | 15.0% |
| Add | 8 | 0.95ms | 0.12ms | 5.2% |
| MaxPool | 1 | 0.54ms | 0.54ms | 3.0% |
| Gemm | 1 | 0.12ms | 0.12ms | 0.6% |
| GlobalAveragePool | 1 | 0.10ms | 0.10ms | 0.6% |
| Flatten | 1 | 0.04ms | 0.04ms | 0.2% |
| **Overhead（prepare）** | — | **0.06ms** | — | **0.3%** |
| **Total** | 49 | **18.33ms** | — | **100%** |

- Conv 占 75% 以上耗时，是主要优化目标
- Overhead（tensor prepare/reshape）仅占 0.3%，得益于 `_node_tensor_cache` 预构建和批量 shape 刷新

### 多分辨率顺序推理

模拟真实动态分辨率场景：依次推理 7 张不同分辨率的图像，统计总耗时和峰值 XPU 内存。

| 分辨率 | PyTorch | GraphInfer | 加速比 |
|--------|---------|------------|--------|
| HD (720p) | 6.96ms | 7.60ms | 0.92x |
| Full HD (1080p) | 308.41ms | 16.48ms | **18.71x** |
| 2K (1440p) | 281.98ms | 27.94ms | **10.09x** |
| 4K (2160p) | 364.25ms | 63.79ms | **5.71x** |
| Square Large | 323.70ms | 31.77ms | **10.19x** |
| Mobile (360p) | 265.37ms | 2.39ms | **111.08x** |
| Small (224x224) | 264.04ms | 2.15ms | **122.99x** |
| **Total** | **1816.79ms** | **154.15ms** | **11.79x** |

| 指标 | PyTorch | GraphInfer |
|------|---------|------------|
| 总耗时 | 1816.79 ms | **154.15 ms** |
| 峰值 XPU 内存 | 2337.9 MB | **1926.6 MB** |
| 最终 XPU 内存 | 1325.4 MB | 1325.4 MB |
| Memory Pool | — | **1012.5 MB** |

**GraphInfer 在动态分辨率场景下比 PyTorch 快 11.8x**，原因：
- PyTorch 在分辨率切换时触发 XPU kernel 重编译（~300ms/张）
- GraphInfer 的 memory pool 固定，kernel 在 warmup 时已编译，后续只需 reshape 视图
- 小分辨率下优势更明显（360p: 111x, 224x224: 123x）

### XPU 内存对比

| 分辨率 | PyTorch XPU 峰值 | GraphInfer XPU 峰值 |
|--------|-----------------|-------------------|
| HD (720p) | 1437.9 MB | 1392.3 MB |
| Full HD (1080p) | 1578.6 MB | 1475.8 MB |
| 2K (1440p) | 1776.4 MB | 1593.7 MB |
| 4K (2160p) | **2337.9 MB** | **1926.6 MB** |
| Square Large | 2337.9 MB | 1926.6 MB |

GraphInfer 峰值 XPU 内存比 PyTorch 低 **17.6%**（1926.6 MB vs 2337.9 MB），pool 仅 **1012.5 MB**——比原始算法减少 **54%**。

> **注意：** 顺序推理时峰值 XPU 内存（1926.6 MB）大于 pool（1012.5 MB），是因为 XPU 在分辨率切换时会保留 kernel 编译缓存和临时 buffer。在**固定分辨率的单次推理**中，XPU 内存 ≈ **pool + 常量**（1057 MB = 1012.5 MB pool + 44.6 MB 权重），几乎没有额外临时 buffer（~0 MB）。

### Profiling 分析（1080p 固定分辨率）

**时间：** Kernel 总和（18.27 ms）+ Overhead（0.06 ms）= **18.33 ms**，无隐藏开销。0.3% 的 overhead 来自 `_node_tensor_cache`（预构建 tensor 视图）和批量 shape 刷新——比早期 4.1% 的 overhead 降低了 13.7x。

**内存：** XPU allocated（1057 MB）≈ Pool（1012.5 MB）+ Constants（44.6 MB），无额外临时 buffer。全部 49 个 node 已验证：**输入输出 tensor 在 pool 中零重叠**。

### 精度验证

使用 `test_accuracy.py` 在 CIFAR-10 测试集（100 张，224x224）上验证 GraphInfer 精度：

| 指标 | 结果 |
|------|------|
| 最大绝对差异 | 2.00e-05 |
| 平均绝对差异 | 9.52e-06 |
| 余弦相似度 | 0.99999994 |
| 预测一致率 | **100%** |
| 性能 | PyTorch 2.25ms vs GraphInfer 2.15ms（1.05x） |

### LLM 推理支持

支持 Qwen3-0.6B、GPT2 等模型的 KV-cache 自回归推理：

- **双池内存压缩**：activation 和 kv_cache 使用独立内存池，避免相互干扰
- **safetensors 外部权重**：支持 Builder 导出的 ONNX（无内嵌权重）+ safetensors 文件
- **Weight 前处理**：Gemm 的 `preprocess_weight` 预转置 weight 矩阵，消除运行时开销
- **融合算子**：Layernrom、Mad、Gelu、RangeGather、Rope、SDPA 等 LLM 专用融合算子

## 使用方法

### GraphInfer（通用 CV 模型）

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

### LLMInfer（LLM 自回归推理）

```python
from llm_infer import LLMInfer

# 初始化（Qwen3-0.6B，safetensors 外部权重）
llm = LLMInfer(
    onnx_path='qwen3_0_6b_kvcache.onnx',
    model_name='Qwen/Qwen3-0.6B-Base',
    safetensors_path='model.safetensors',
    weight_map={'model.embed_tokens.weight': 'model-00001-of-00002.safetensors', ...},
    device='xpu',
    max_seq_len=128,
)

# 自回归生成
output = llm.generate("Hello, how are you?", max_length=50, verbose=True)
print(output)
```

### 超分模型 GUI

```bash
# 启动图形界面
python inference/sr_gui.py --onnx-path ./realesrgan-x4.onnx --device xpu
```

## 注册自定义 Kernel

```python
from kernels import KernelRegistry, Kernel

@KernelRegistry.register("CustomOp")
class CustomOpKernel(Kernel):
    @staticmethod
    def run(inputs, outputs, attrs):
        # inputs: List[torch.Tensor] — 输入 tensor 列表
        # outputs: List[torch.Tensor] — 预分配的输出视图，结果用 copy_ 或 out= 写入
        # attrs: Dict — 算子属性（kernel_shape, strides 等）
        torch.add(inputs[0], inputs[1], out=outputs[0])

    @staticmethod
    def preprocess_weight(src_name, tensor, attrs):
        """可选：对 weight 常量做前处理（如格式转换、预转置等）"""
        return tensor
```

## 测试脚本

```bash
# 多端对比（PyTorch vs ONNX Runtime vs GraphInfer，含 FP32/FP16）
python inference/compare.py

# 精度验证（CIFAR-10 / 合成数据）
python inference/test_accuracy.py --data_mode cifar10 --num_samples 100

# 多分辨率顺序推理测试
python inference/test_multi_resolution.py

# 超分模型推理测试
python inference/test_super_resolution.py --onnx-path ./realesrgan-x4.onnx

# LLM 推理
python inference/llm_infer.py --model-name Qwen/Qwen3-0.6B-Base --prompt "Hello" --device xpu

# XPU 性能基准
python inference/benchmark_xpu.py
```
