# ONNX Graph Inference Engine

A graph-level inference engine built on `onnx_tool`'s compute graph and MemoryPool, using PyTorch as the backend operator executor.

## Architecture

```
inference/
├── memory_pool.py              # MemoryPool — PyTorch memory pool based on compress_memory()
├── kernels.py                  # Registered op kernels (Conv, Add, Relu, Gemm, 40+ ops)
├── infer.py                    # GraphInfer — compute graph inference engine
├── compare.py                  # PyTorch vs ONNX Runtime vs GraphInfer comparison
├── test_multi_resolution.py    # Multi-resolution sequential inference test
├── README.md                   # English documentation
└── README_CN.md                # Chinese documentation
```

### Core Components

**MemoryPool** — Allocates a contiguous `torch.Tensor` as a memory pool based on `cg.compress_memory()`. Each tensor maps to a view within the pool via `[offset, size]`, achieving zero-copy memory reuse. Pool size is fixed and does not change with input resolution.

**Kernel Registry** — A registration-based op kernel system. Each op is registered as a `Kernel` subclass implementing the `run(inputs, outputs, attrs)` static method. 40+ ops are registered (Conv, Add, Relu, Gemm, MatMul, BatchNormalization, etc.).

**GraphInfer** — The compute graph inference engine. Initialization performs model loading, node reordering, shape regression, compute graph extraction, memory compression, and MemoryPool creation. `forward` traverses all nodes and executes inference through registered kernels.

## Benchmark Results

### Test Environment

- **Device**: Intel XPU
- **Model**: ResNet18
- **Input**: Multiple resolutions, single batch, float32
- **ONNX Runtime**: CPUExecutionProvider

### Memory Optimization

GraphInfer uses a **two-pass memory compression algorithm**:

1. **Pass 1**: Run the original `compress_memory()` to obtain tensor lifetime information
2. **Pass 2**: Re-allocate tensors sorted by size (largest first) within each node, reducing fragmentation and tail waste

Result for 4K (2160x3840) input:

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Pool size | 2214.8 MB | **1012.5 MB** | **↓ 54.3%** |
| Tail waste | 506.2 MB (22.9%) | **0.0 MB (0%)** | Eliminated |
| Max concurrent alive | 1012.5 MB | 1012.5 MB | Unchanged |
| Utilization | 77.1% | **100%** | +22.9% |

The optimized pool (1012.5 MB) is now close to PyTorch's peak memory (1152 MB), with only a 12% gap due to PyTorch's allocator internal caching.

### Single Resolution Comparison (after warmup)

| Resolution | PyTorch | ONNX Runtime | GraphInfer | Speedup |
|------------|---------|-------------|------------|---------|
| Batch=1 (1080p) | 0.0151s | 0.1013s | **0.0167s** | **0.91x** |
| Batch=4 (1080p) | 0.0623s | 0.4373s | **0.0681s** | **0.91x** |

GraphInfer achieves ~91% of PyTorch performance after warmup, significantly faster than ONNX Runtime CPU.

### Profile Analysis (single forward time breakdown)

| Op Type | Count | Total | Avg | % |
|---------|-------|-------|-----|---|
| Conv | 20 | 14.0ms | 0.70ms | **70.0%** |
| Relu | 17 | 3.0ms | 0.18ms | 15.2% |
| Add | 8 | 1.3ms | 0.16ms | 6.4% |
| MaxPool | 1 | 0.6ms | 0.55ms | 2.7% |
| Gemm | 1 | 0.2ms | 0.15ms | 0.8% |
| GlobalAveragePool | 1 | 0.1ms | 0.13ms | 0.7% |
| Flatten | 1 | 0.04ms | 0.04ms | 0.2% |
| **Overhead (prepare)** | — | **0.8ms** | — | **4.1%** |
| **Total** | 49 | **20.0ms** | — | **100%** |

- Conv accounts for over 70% of total time — the primary optimization target
- Overhead (tensor prepare/reshape) is only 4.1%

### Multi-Resolution Sequential Inference

Simulates a real dynamic resolution scenario: sequentially infer 7 images at different resolutions, measuring total time and peak XPU memory.

| Resolution | PyTorch | GraphInfer | Ratio |
|------------|---------|------------|-------|
| HD (720p) | 6.68ms | 8.06ms | 0.83x |
| Full HD (1080p) | 311.70ms | 17.16ms | **18.17x** |
| 2K (1440p) | 286.09ms | 28.88ms | **9.91x** |
| 4K (2160p) | 366.22ms | 66.91ms | **5.47x** |
| Square Large | 322.74ms | 32.90ms | **9.81x** |
| Mobile (360p) | 262.82ms | 3.49ms | **75.29x** |
| Small (224x224) | 262.05ms | 2.25ms | **116.55x** |
| **Total** | **1820.35ms** | **161.54ms** | **11.27x** |

| Metric | PyTorch | GraphInfer |
|--------|---------|------------|
| Total time | 1820.35 ms | **161.54 ms** |
| Peak XPU memory | 2338.0 MB | **1926.7 MB** |
| Final XPU memory | 1325.5 MB | 1325.5 MB |
| Memory Pool | — | **1012.5 MB** |

**GraphInfer is 11.3x faster than PyTorch in dynamic resolution scenarios** because:
- PyTorch triggers XPU kernel recompilation on resolution changes (~300ms/image)
- GraphInfer's memory pool is fixed; kernels are compiled during warmup — subsequent runs only need view reshaping
- The advantage is more pronounced at smaller resolutions (360p: 75x, 224x224: 117x)

### XPU Memory Comparison

| Resolution | PyTorch XPU Peak | GraphInfer XPU Peak |
|------------|-----------------|-------------------|
| HD (720p) | 1438.0 MB | 1392.3 MB |
| Full HD (1080p) | 1578.6 MB | 1475.8 MB |
| 2K (1440p) | 1776.5 MB | 1593.7 MB |
| 4K (2160p) | **2338.0 MB** | **1926.7 MB** |
| Square Large | 2338.0 MB | 1926.7 MB |

GraphInfer's peak XPU memory is **17.6% lower** than PyTorch (1926 MB vs 2338 MB), and the pool is only **1012.5 MB** — a **54% reduction** from the original algorithm.

> **Note:** The peak XPU memory (1926 MB) is larger than the pool (1012.5 MB) because XPU retains kernel compilation caches and temporary buffers across resolution switches during sequential inference. In a **single fixed-resolution inference**, the XPU memory usage is approximately **pool + constants** (1057 MB = 1012.5 MB pool + 44.6 MB weights), with negligible temporary buffers (~0 MB).

### Profiling Details (Single Fixed Resolution)

**Time:** Kernel total (19.2 ms) + Overhead (0.8 ms) = **20.0 ms**. No hidden overhead. The 4.1% overhead comes from `_resolve_tensor` shape queries and `_reshape_view`.

**Memory:** XPU allocated (1057 MB) ≈ Pool (1012.5 MB) + Constants (44.6 MB). No extra temporary buffers. All 49 nodes verified: **zero input/output overlap** in the memory pool.

## Usage

```python
from infer import GraphInfer

# Initialize
infer_engine = GraphInfer(
    'model.onnx',
    {'input': ('batch', 3, 'height', 'width')},
    {'batch': (1, 4), 'height': (224, 1080), 'width': (224, 1920)},
    dtype=torch.float32,
    device='xpu',  # or 'cpu'
)

# Inference (inputs/outputs are {name: tensor} dicts)
outputs = infer_engine.forward({'input': input_tensor})

# Get output
result = outputs['output']

# Profile mode
result = infer_engine.forward({'input': input_tensor}, profile=True)
infer_engine.print_profile(result['__profile__'])
```

## Registering Custom Kernels

```python
from kernels import KernelRegistry, Kernel

@KernelRegistry.register("CustomOp")
class CustomOpKernel(Kernel):
    @staticmethod
    def run(inputs, outputs, attrs):
        # inputs: List[torch.Tensor]
        # outputs: List[torch.Tensor] (pre-allocated views, write results via copy_)
        # attrs: Dict (op attributes)
        outputs[0].copy_(custom_function(inputs[0]))
```

## Test Scripts

```bash
# Three-way comparison (PyTorch vs ONNX Runtime vs GraphInfer)
python inference/compare.py

# Multi-resolution sequential inference test
python inference/test_multi_resolution.py
```
