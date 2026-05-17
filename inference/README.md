# ONNX Graph Inference Engine

A graph-level inference engine built on `onnx_tool`'s compute graph and MemoryPool, using PyTorch as the backend operator executor.

Supports CV models (ResNet, RealESRGAN, etc.) and LLM models (Qwen3, GPT2, etc.).

## Architecture

```
inference/
├── memory_pool.py              # MemoryPool — PyTorch memory pool based on compress_memory()
├── kernels.py                  # Registered op kernels (Conv, Add, Relu, Gemm, 50 ops)
├── infer.py                    # GraphInfer — compute graph inference engine
├── llm_infer.py                # LLMInfer — LLM autoregressive inference engine (KV-cache)
├── compare.py                  # PyTorch vs ONNX Runtime vs GraphInfer comparison
├── test_accuracy.py            # Accuracy validation (CIFAR-10 / synthetic data)
├── test_multi_resolution.py    # Multi-resolution sequential inference test
├── test_super_resolution.py    # Super-resolution model inference test
├── benchmark_xpu.py            # XPU GEMM / Conv2d performance benchmark
├── plot_xpu_memory.py          # XPU memory curve plotting
├── sr_gui.py                   # Super-resolution GUI tool
├── README.md                   # English documentation
└── README_CN.md                # Chinese documentation
```

### Core Components

**MemoryPool** — Allocates a contiguous `torch.Tensor` as a memory pool based on `cg.compress_memory()`. Each tensor maps to a view within the pool via `[offset, size]`, achieving zero-copy memory reuse. Pool size is fixed and does not change with input resolution. LLM models use a dual-pool design (activation pool + kv_cache pool) to isolate KV-cache from regular activations.

**Kernel Registry** — A registration-based op kernel system. Each op is registered as a `Kernel` subclass implementing the `run(inputs, outputs, attrs)` static method. Supports `preprocess_weight` for weight preprocessing (e.g., Gemm pre-transpose). 50 ops are registered, including LLM-specific fused ops (Layernrom, Mad, Gelu, RangeGather, Rope, SDPA, etc.).

**GraphInfer** — The compute graph inference engine. Initialization performs model loading, node reordering, shape regression, compute graph extraction, memory compression, and MemoryPool creation. Supports safetensors external weight loading. `forward` traverses all nodes and executes inference through registered kernels, with profile mode support.

**LLMInfer** — LLM autoregressive inference engine. Built on GraphInfer, supports KV-cache prefill + decode two-stage inference, with HuggingFace Transformers comparison. Supports Qwen3-0.6B, GPT2, and other models.

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

| Resolution | PyTorch (XPU) | ONNX Runtime (CPU) | GraphInfer (XPU) | Speedup vs PT |
|------------|---------------|--------------------|------------------|---------------|
| Batch=1 (1080p) | 14.92ms | 101.36ms | **16.43ms** | **0.91x** |
| Batch=4 (1080p) | 61.22ms | 430.88ms | **65.88ms** | **0.93x** |

GraphInfer achieves ~91-93% of PyTorch performance after warmup, significantly faster than ONNX Runtime CPU. Kernels use `out=` parameter for zero-copy output, eliminating the `copy_` overhead from earlier versions.

### FP16 Performance (640x640, XPU)

| Precision | PyTorch (XPU) | GraphInfer (XPU) | Speedup vs PT |
|-----------|---------------|------------------|---------------|
| FP32 | 3.3ms | 3.5ms | 0.95x |
| FP16 | 1.8ms | **1.3ms** | **1.38x** |

GraphInfer supports both FP32 and FP16 precision on XPU. In FP16 mode, GraphInfer outperforms PyTorch by 38% due to the memory pool's reduced overhead at half precision.

### Profile Analysis (1080p single forward time breakdown)

| Op Type | Count | Total | Avg | % |
|---------|-------|-------|-----|---|
| Conv | 20 | 13.76ms | 0.69ms | **75.1%** |
| Relu | 17 | 2.75ms | 0.16ms | 15.0% |
| Add | 8 | 0.95ms | 0.12ms | 5.2% |
| MaxPool | 1 | 0.54ms | 0.54ms | 3.0% |
| Gemm | 1 | 0.12ms | 0.12ms | 0.6% |
| GlobalAveragePool | 1 | 0.10ms | 0.10ms | 0.6% |
| Flatten | 1 | 0.04ms | 0.04ms | 0.2% |
| **Overhead (prepare)** | — | **0.06ms** | — | **0.3%** |
| **Total** | 49 | **18.33ms** | — | **100%** |

- Conv accounts for over 75% of total time — the primary optimization target
- Overhead (tensor prepare/reshape) is only 0.3%, thanks to `_node_tensor_cache` and batched shape refresh

### Multi-Resolution Sequential Inference

Simulates a real dynamic resolution scenario: sequentially infer 7 images at different resolutions, measuring total time and peak XPU memory.

| Resolution | PyTorch | GraphInfer | Ratio |
|------------|---------|------------|-------|
| HD (720p) | 6.96ms | 7.60ms | 0.92x |
| Full HD (1080p) | 308.41ms | 16.48ms | **18.71x** |
| 2K (1440p) | 281.98ms | 27.94ms | **10.09x** |
| 4K (2160p) | 364.25ms | 63.79ms | **5.71x** |
| Square Large | 323.70ms | 31.77ms | **10.19x** |
| Mobile (360p) | 265.37ms | 2.39ms | **111.08x** |
| Small (224x224) | 264.04ms | 2.15ms | **122.99x** |
| **Total** | **1816.79ms** | **154.15ms** | **11.79x** |

| Metric | PyTorch | GraphInfer |
|--------|---------|------------|
| Total time | 1816.79 ms | **154.15 ms** |
| Peak XPU memory | 2337.9 MB | **1926.6 MB** |
| Final XPU memory | 1325.4 MB | 1325.4 MB |
| Memory Pool | — | **1012.5 MB** |

**GraphInfer is 11.8x faster than PyTorch in dynamic resolution scenarios** because:
- PyTorch triggers XPU kernel recompilation on resolution changes (~300ms/image)
- GraphInfer's memory pool is fixed; kernels are compiled during warmup — subsequent runs only need view reshaping
- The advantage is more pronounced at smaller resolutions (360p: 75x, 224x224: 117x)

### XPU Memory Comparison

| Resolution | PyTorch XPU Peak | GraphInfer XPU Peak |
|------------|-----------------|-------------------|
| HD (720p) | 1437.9 MB | 1392.3 MB |
| Full HD (1080p) | 1578.6 MB | 1475.8 MB |
| 2K (1440p) | 1776.4 MB | 1593.7 MB |
| 4K (2160p) | **2337.9 MB** | **1926.6 MB** |
| Square Large | 2337.9 MB | 1926.6 MB |

GraphInfer's peak XPU memory is **17.6% lower** than PyTorch (1926.6 MB vs 2337.9 MB), and the pool is only **1012.5 MB** — a **54% reduction** from the original algorithm.

> **Note:** The peak XPU memory (1926.6 MB) is larger than the pool (1012.5 MB) because XPU retains kernel compilation caches and temporary buffers across resolution switches during sequential inference. In a **single fixed-resolution inference**, the XPU memory usage is approximately **pool + constants** (1057 MB = 1012.5 MB pool + 44.6 MB weights), with negligible temporary buffers (~0 MB).

### Profiling Details (1080p Single Forward)

**Time:** Kernel total (18.27 ms) + Overhead (0.06 ms) = **18.33 ms**. No hidden overhead. The 0.3% overhead comes from `_node_tensor_cache` (pre-built tensor views) and batched shape refresh — a 13.7x improvement over the earlier 4.1% overhead.

**Memory:** XPU allocated (1057 MB) ≈ Pool (1012.5 MB) + Constants (44.6 MB). No extra temporary buffers. All 49 nodes verified: **zero input/output overlap** in the memory pool.

### Accuracy Validation

Validated on CIFAR-10 test set (100 images, 224x224) using `test_accuracy.py`:

| Metric | Result |
|--------|--------|
| Max absolute difference | 2.00e-05 |
| Mean absolute difference | 9.52e-06 |
| Cosine similarity | 0.99999994 |
| Prediction agreement | **100%** |
| Performance | PyTorch 2.25ms vs GraphInfer 2.15ms (1.05x) |

### LLM Inference Support

Supports KV-cache autoregressive inference for Qwen3-0.6B, GPT2, and other models:

- **Dual-pool memory compression**: Activation and kv_cache use separate memory pools to avoid interference
- **safetensors external weights**: Supports Builder-exported ONNX (no embedded weights) + safetensors files
- **Weight preprocessing**: Gemm's `preprocess_weight` pre-transposes weight matrices, eliminating runtime overhead
- **Fused ops**: Layernrom, Mad, Gelu, RangeGather, Rope, SDPA — LLM-specific fused operators

## Usage

### GraphInfer (General CV Models)

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

### LLMInfer (LLM Autoregressive Inference)

```python
from llm_infer import LLMInfer

# Initialize (Qwen3-0.6B with safetensors external weights)
llm = LLMInfer(
    onnx_path='qwen3_0_6b_kvcache.onnx',
    model_name='Qwen/Qwen3-0.6B-Base',
    safetensors_path='model.safetensors',
    weight_map={'model.embed_tokens.weight': 'model-00001-of-00002.safetensors', ...},
    device='xpu',
    max_seq_len=128,
)

# Autoregressive generation
output = llm.generate("Hello, how are you?", max_length=50, verbose=True)
print(output)
```

### Super-Resolution GUI

```bash
# Launch GUI
python inference/sr_gui.py --onnx-path ./realesrgan-x4.onnx --device xpu
```

## Registering Custom Kernels

```python
from kernels import KernelRegistry, Kernel

@KernelRegistry.register("CustomOp")
class CustomOpKernel(Kernel):
    @staticmethod
    def run(inputs, outputs, attrs):
        # inputs: List[torch.Tensor]
        # outputs: List[torch.Tensor] (pre-allocated views, write via copy_ or out=)
        # attrs: Dict (op attributes)
        torch.add(inputs[0], inputs[1], out=outputs[0])

    @staticmethod
    def preprocess_weight(src_name, tensor, attrs):
        """Optional: preprocess weight constants (format conversion, pre-transpose, etc.)"""
        return tensor
```

## Test Scripts

```bash
# Multi-way comparison (PyTorch vs ONNX Runtime vs GraphInfer, FP32/FP16)
python inference/compare.py

# Accuracy validation (CIFAR-10 / synthetic data)
python inference/test_accuracy.py --data_mode cifar10 --num_samples 100

# Multi-resolution sequential inference test
python inference/test_multi_resolution.py

# Super-resolution model inference test
python inference/test_super_resolution.py --onnx-path ./realesrgan-x4.onnx

# LLM inference
python inference/llm_infer.py --model-name Qwen/Qwen3-0.6B-Base --prompt "Hello" --device xpu

# XPU performance benchmark
python inference/benchmark_xpu.py
```
