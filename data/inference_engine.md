# Inference Engine

This document covers two approaches for running inference using onnx-tool's compute graph and shape engine:

1. **Python Inference Engine** (new, recommended) — Full PyTorch-backed inference with memory pool
2. **C++ Integration** (legacy) — Serialize compute graph for external C++ engines

## Python Inference Engine (Recommended)

The [`inference/`](../inference/) module provides a complete PyTorch-backed inference engine built on the compressed memory pool:

- **MemoryPool**: Zero-copy tensor views into a pre-allocated contiguous buffer
- **Kernel Registry**: 40+ registered op kernels (Conv, Add, Relu, Gemm, MatMul, etc.)
- **GraphInfer**: Graph-level inference with shape engine integration

### Quick Start

```python
from inference.infer import GraphInfer
import torch

# Initialize with model and dynamic shape ranges
infer_engine = GraphInfer(
    'model.onnx',
    {'input': ('batch', 3, 'height', 'width')},
    {'batch': (1, 4), 'height': (224, 1080), 'width': (224, 1920)},
    dtype=torch.float32,
    device='xpu',  # or 'cpu'
)

# Run inference (inputs/outputs are {name: tensor} dicts)
outputs = infer_engine.forward({'input': input_tensor})
result = outputs['output']

# Profile mode
result = infer_engine.forward({'input': input_tensor}, profile=True)
infer_engine.print_profile(result['__profile__'])
```

### Performance Highlights (ResNet18 on Intel XPU)

| Metric | PyTorch | GraphInfer | Improvement |
|--------|---------|------------|-------------|
| Single inference (1080p) | 0.0151s | **0.0167s** | 0.91x (on par) |
| Sequential 7 resolutions | 1820ms | **162ms** | **11.3x faster** |
| Peak XPU memory (4K) | 2338 MB | **1926 MB** | **17.6% less** |
| Memory pool (4K) | — | **1012 MB** | Fixed size |

> 📌 See [`inference/README.md`](../inference/README.md) for full benchmark details

### Registering Custom Kernels

```python
from inference.kernels import KernelRegistry, Kernel

@KernelRegistry.register("CustomOp")
class CustomOpKernel(Kernel):
    @staticmethod
    def run(inputs, outputs, attrs):
        # inputs: List[torch.Tensor]
        # outputs: List[torch.Tensor] (pre-allocated views, write via copy_)
        # attrs: Dict (op attributes)
        outputs[0].copy_(custom_function(inputs[0]))
```

## Legacy: C++ Integration via Serialization

For integrating compute graph & shape engine into a C++ based inference engine:

### Python Serialization

```python
import onnx_tool

resnetinfo = {
    'name': 'data/public/resnet18-v1-7.onnx',
    'input_desc': {'data': [1, 3, 'h', 'w']},
    'input_range': {'h': (224, 299), 'w': (224, 299)},
}
shape_engine, compute_graph = onnx_tool.model_shape_regress(
    resnetinfo['name'], resnetinfo['input_desc'], resnetinfo['input_range'])
onnx_tool.serialize_graph(compute_graph, 'resnet18.cg')
onnx_tool.serialize_shape_engine(shape_engine, 'resnet18.se')
```

The file `resnet18.cg` contains compute graph information.  
The file `resnet18.se` contains shape engine structure. They will be used by the C++ graph loader and shape engine loader.  
Tested models: **ResNet18, GPT-2, Bert**

### C++ Integration Example

For the C++ implementation, please refer to: [shape-engine-cpp](https://github.com/ThanatosShinji/shape-engine-cpp.git)