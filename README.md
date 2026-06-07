<a href="README_CN.md">📄 简体中文</a> | <a href="https://github.com/ThanatosShinji/AI-Enhancement-Filter">✨ New Project: AI-Enhancement-Filter</a> (powered by onnx-tool)

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.6%2B-blue?logo=python" alt="Python 3.6+">
  <img src="https://img.shields.io/pypi/v/onnx-tool?color=green" alt="PyPI Version">
  <img src="https://img.shields.io/github/license/ThanatosShinji/onnx-tool?color=orange" alt="License">
</p> 

# onnx-tool

**A comprehensive toolkit for analyzing, optimizing, and transforming ONNX models** with advanced capabilities for LLMs, diffusion models, and computer vision architectures.

- **LLM Optimization**: Build and profile large language models with KV cache analysis ([example](#build-profile))
- **Graph Transformation**: 
  - Constant folding ([docs](data/ConstantFolding.md))
  - Operator fusion ([docs](data/GraphFusion.md))
- **Advanced Profiling**: 
  - Rapid shape inference
  - MACs/parameter statistics with sparsity awareness
- **Compute Graph Engine**: Runtime shape computation with minimal overhead ([details](#compute_graph-header))
- **Inference Engine**: PyTorch-backed graph inference with memory pool ([details](inference/README.md))
  - 40+ registered op kernels (Conv, Add, Relu, Gemm, etc.)
  - Up to **11.3x faster** than PyTorch in dynamic resolution scenarios
  - **54% memory reduction** via two-pass compression algorithm
- **Memory Compression**:
  - Activation memory optimization (up to 95% reduction)
  - Weight quantization (FP16, INT8/INT4 with per-tensor/channel/block schemes)
- **Quantization & Sparsity**: Full support for quantized and sparse model analysis
## 🤖 Supported Model Architectures

| Domain      | Models                                                                 |
|-------------|------------------------------------------------------------------------|
| **NLP**     | BERT, T5, GPT, LLaMa, MPT, Qwen3, Qwen3.5 (Dense & MoE), DeepSeek-V4 (Flash/Pro, MLA+MoE), MiniMax-M2.7 (MoE) ([TransformerModel](benchmark/transfomer_models.py)) |
| **Diffusion** | Stable Diffusion (TextEncoder, VAE, UNet)                            |
| **CV**      | Detic, BEVFormer, SSD300_VGG16, ConvNeXt, Mask R-CNN, Silero VAD         |
| **Audio**   | Sovits, LPCNet                                                        |

> 🆕 **Qwen3.5 Series**: Full support for Qwen3.5 hybrid architecture including:
> - **Gated DeltaNet (GDN)** layers with linear attention
> - **QKV Gating** (Q projection with built-in gate, applied before O-projection)
> - **Sparse Mixture-of-Experts (MoE)** with routed + shared experts
> - **Mixed layer types** (linear_attention / full_attention) per config

---

## ⚡ Build & Profile LLMs in Seconds
<a id="build-profile"></a>
Profile 10 Hugging Face models in under one second. Export ONNX models with llama.cpp-like simplicity ([code](benchmark/llm_test.py)).

### Model Statistics (1k token input)
model name(1k input)                 | MACs(G) | Parameters(G) |   KV Cache(G)
------------------------------------ |---------|---------------| ----------
Phi-3-mini-4k                        | 4083    | 3.82108       |  0.201327
Phi-3-small-8k-instruct              | 7912    | 7.80167       |  0.0671089
Phi-3-medium-4k-instruct             | 14665   | 13.9602       |  0.104858
Llama3-8B                            | 8029    | 8.03026       |  0.0671089
Llama-3.1-70B-Japanese-Instruct-2407 | 72888   | 70.5537       |  0.167772
**Qwen3.5-4B-Instruct** 🆕           | 4807    | 4.651         |  0.067109
**Qwen3.5-35B-A3B-Instruct** 🆕 (MoE)| 3574    | 34.705        |  0.041943
**DeepSeek-V4-Flash** 🆕 (MoE/MLA)   | 15681   | 283.811       |  0.045089
**DeepSeek-V4-Pro** 🆕 (MoE/MLA)     | 55701   | 1571.742      |  0.063963
**MiniMax-M2.7** 🆕 (MoE)            | 12554   | 230.315       |  0.130023

### MoE Activated Parameters vs Sequence Length
> *Activated parameters = sum of all nodes' `static_params` (weights actually accessed during forward pass).*
> *S=32: Qwen3.5-35B-A3B & MiniMax-M2.7 experts fully activated (gap ≈ embedding); DeepSeek-V4 not yet saturated (top-6 × 32 = 192 < num_experts).*

| model                   | Total(G) | S=1    | S=2    | S=4    | S=8    | S=16    | S=32    |
|-------------------------|----------|--------|--------|--------|--------|---------|---------|
| Qwen3.5-35B-A3B (MoE)   | 34.3     | 2.44   | 3.44   | 5.46   | 9.48   | 17.54   | 33.64   |
| MiniMax-M2.7 (MoE)      | 230.3    | 10.42  | 17.44  | 31.48  | 59.56  | 115.73  | 228.08  |
| DeepSeek-V4-Flash (MoE)  | 283.8    | 12.75  | 19.24  | 32.23  | 58.20  | 110.14  | 214.03  |
| DeepSeek-V4-Pro (MoE)    | 1571.7   | 47.60  | 71.77  | 120.13 | 216.84 | 410.27  | 797.12  |

### Latency Estimation (4-bit weights, 16-bit KV cache)

**Prefill Throughput (tokens/s, 1k input)**

model                                | Ultra-358H | Arc-B70 | RTX-4090 | RTX-5090
------------------------------------ |-----------|---------|----------|----------
Phi-3-mini-4k                        | 6658.9    | 21279.8 | 19896.0  | 25567.0
Phi-3-small-8k-instruct              | 3599.5    | 11316.9 | 10429.8  | 13334.3
Phi-3-medium-4k-instruct             | 1964.8    | 6151.8  | 5649.1   | 7213.2
Llama3-8B                            | 3499.9    | 11053.2 | 10226.7  | 13092.7
Llama-3.1-70B-Japanese-Instruct-2407 | 401.2     | 1249.3  | 1141.8   | 1455.5
**Qwen3.5-4B-Instruct** 🆕           | 5705.0    | 18162.0 | 16923.1  | 21720.3
**Qwen3.5-35B-A3B-Instruct** 🆕 (MoE)| 4660.9    | 18458.3 | 21970.0  | 29367.3
**MiniMax-M2.7** 🆕 (MoE)            | 970.5     | 4285.1  | 6090.4   | 9083.3

**Decode Throughput (tokens/s)**

model                                | Ultra-358H | Arc-B70 | RTX-4090 | RTX-5090
------------------------------------ |-----------|---------|----------|----------
Phi-3-mini-4k                        | 56.4      | 267.9   | 444.2    | 789.7
Phi-3-small-8k-instruct              | 33.4      | 158.5   | 262.8    | 467.2
Phi-3-medium-4k-instruct             | 18.0      | 85.4    | 141.6    | 251.8
Llama3-8B                            | 32.9      | 156.1   | 258.9    | 460.2
Llama-3.1-70B-Japanese-Instruct-2407 | 3.6       | 17.3    | 28.7     | 51.0
**Qwen3.5-4B-Instruct** 🆕           | 56.5      | 268.4   | 444.9    | 791.0
**Qwen3.5-35B-A3B-Instruct** 🆕 (MoE)| 79.0      | 375.0   | 621.8    | 1105.4
**MiniMax-M2.7** 🆕 (MoE)            | 23.4      | 111.0   | 184.1    | 327.3
> 💡 *Latencies computed from hardware specs – no actual inference required. Uses BF16/FP16 compute with FP32 accumulate as the standard.*
> 
---

## 🔧 Basic Parsing & Editing

<a id="basic-parse-edit"></a>
Intuitive API for model manipulation:
```python
from onnx_tool import Model

model = Model('model.onnx')          # Load any ONNX file
graph = model.graph                  # Access computation graph
node = graph.nodemap['Conv_0']       # Modify operator attributes
tensor = graph.tensormap['weight']   # Edit tensor data/types
model.save_model('modified.onnx')    # Persist changes
```
See comprehensive examples in [`benchmark/examples.py`](benchmark/examples.py).

---

## 📊 Shape Inference & Profiling
<a id="shapeinfer-profile"></a>
All profiling relies on precise shape inference:

<p align="center">
  <img src="data/shape_inference.jpg" width="600" alt="Shape inference visualization">
</p>

### Profiling Capabilities
- **Standard profiling**: MACs, parameters, memory footprint
- **Sparse-aware profiling**: Quantify sparsity impact on compute
  
<p align="center">
  <img src="data/macs_counting.png" width="450" alt="MACs profiling table">
  <img src="data/sparse_model.png" width="450" alt="Sparse model profiling">
</p>

📚 **Learn more**: 
- [Profiling Guide](data/Profile.md)
- [PyTorch Integration](data/PytorchUsage.md)
- [TensorFlow Integration](data/TensorflowUsage.md)


---

## ⚙️ Compute Graph & Shape Engine
<a id="compute_graph-header"></a>

Transform exported ONNX graphs into efficient *Compute Graphs* by removing shape-calculation overhead:

<p align="center">
  <img src="data/compute_graph.png" width="700" alt="Compute graph transformation">
</p>

- **Compute Graph**: Minimal graph containing only compute operations
- **Shape Engine**: Runtime shape resolver for dynamic models

**Use Cases**:
- Integration with custom inference engines ([guide](data/inference_engine.md))
- Shape regression testing ([example](benchmark/shape_regress.py))


---

## 💾 Memory Compression
<a id="memory-compression"></a>

### Activation Memory Compression
Reuses temporary buffers to minimize peak memory usage – critical for LLMs and high-res CV models.

 model                         | Native Memory Size(MB) | Compressed Memory Size(MB) | Compression Ratio(%) 
-------------------------------|------------------------|----------------------------|----------------------
 StableDiffusion(VAE_encoder)  | 14,245                 | 540                        | 3.7                  
 StableDiffusion(VAE_decoder)  | 25,417                 | 1,140                      | 4.48                 
 StableDiffusion(Text_encoder) | 215                    | 5                          | 2.5                  
 StableDiffusion(UNet)         | 36,135                 | 2,232                      | 6.2                  
 GPT2                          | 40                     | 2                          | 6.9                  
 BERT                          | 2,170                  | 27                         | 1.25                 

> ✅ Typical models achieve **>90% activation memory reduction**  
> 📌 Implementation: [`benchmark/compression.py`](benchmark/compression.py)

The `compress_memory()` algorithm has been patched with two improvements (see [`onnx_tool/graph.py`](onnx_tool/graph.py)):
1. **Size-sorted allocation**: New tensors within each node are allocated in descending size order, reducing fragmentation
2. **Tail compression**: Unused space at the end of the memory pool is trimmed
3. **List reference fix**: Each tensor's `[offset, size]` is stored as an independent copy, preventing accidental cross-tensor aliasing

Benchmark results across models:

 model                         | Native(MB) | Compressed(MB) | Ratio(%)
-------------------------------|------------|----------------|----------
 VAE encoder                   | 11,313.6   | **512.0**      | 4.53
 VAE decoder                   | 19,816.2   | **896.1**      | 4.52
 Text encoder                  | 172.5      | **3.7**        | 2.12
 GPT2                          | 381.1      | **16.1**       | 4.23
 ResNet50                      | 279.3      | **10.7**       | 3.84

> ✅ Optimized algorithm achieves **up to 54% additional pool reduction** (ResNet50: 21.4→10.7 MB vs original)

### Inference Engine

The [`inference/`](inference/) module provides a complete PyTorch-backed inference engine built on the compressed memory pool:

- **MemoryPool**: Zero-copy tensor views into a pre-allocated contiguous buffer
- **Kernel Registry**: 40+ registered op kernels (Conv, Add, Relu, Gemm, MatMul, etc.)
- **GraphInfer**: Graph-level inference with shape engine integration

**Performance highlights** (ResNet18 on Intel XPU):

| Metric | PyTorch | GraphInfer | Improvement |
|--------|---------|------------|-------------|
| Single inference (1080p) | 0.0151s | **0.0167s** | 0.91x (on par) |
| Sequential 7 resolutions | 1820ms | **162ms** | **11.3x faster** |
| Peak XPU memory (4K) | 2338 MB | **1926 MB** | **17.6% less** |
| Memory pool (4K) | — | **1012 MB** | Fixed size |

> 📌 See [`inference/README.md`](inference/README.md) for full benchmark details
> 
### Weight Compression
Essential for deploying large models on memory-constrained devices:

| Quantization Scheme                     | Size vs FP32 | Example (7B model) |
|-----------------------------------------|--------------|--------------------|
| FP32 (baseline)                         | 1.00×        | 28 GB              |
| FP16                                    | 0.50×        | 14 GB              |
| INT8 (per-channel)                      | 0.25×        | 7 GB               |
| INT4 (block=32, symmetric) – llama.cpp  | 0.156×       | 4.4 GB             |

**Supported schemes**:
- ✅ FP16
- ✅ INT8: symmetric/asymmetric × per-tensor/channel/block
- ✅ INT4: symmetric/asymmetric × per-tensor/channel/block

📌 See [`benchmark/examples.py`](benchmark/examples.py) for implementation examples.


---

## 🚀 Installation
    
```bash
# PyPI (recommended)
pip install onnx-tool

# Latest development version
pip install --upgrade git+https://github.com/ThanatosShinji/onnx-tool.git
```

**Requirements**: Python ≥ 3.6

> ⚠️ **Troubleshooting**: If ONNX installation fails, try:
> ```bash
> pip install onnx==1.8.1 && pip install onnx-tool
> ```


---

## Known Issues
* Loop op is not supported
* Sequence type is not supported
  
---

## 📈 Model Zoo Results
<a id='models'></a>
Comprehensive profiling of [ONNX Model Zoo](https://github.com/onnx/models) and SOTA models. Input shapes defined in [`data/public/config.py`](data/public/config.py).

📥 **Download pre-profiled models** (with full tensor shapes):
- [Baidu Drive](https://pan.baidu.com/s/1eebBP-n-wXvOhSmIH-NUZQ) (code: `p91k`)
- [Google Drive](https://drive.google.com/drive/folders/1H-ya1wTvjIMg2pMcMITWDIfWNSnjYxTn?usp=sharing)
<p id="results" align="center">
<table>
<tr>
<td>

Model | Params(M) | MACs(M)
---|---|---
<a href="benchmark/transfomer_models.py">GPT-J 1 layer</a> | 464 | 173,398  
<a href="benchmark/transfomer_models.py">MPT 1 layer</a> | 261 | 79,894
[text_encoder](https://huggingface.co/bes-dev/stable-diffusion-v1-4-onnx/tree/main)| 123.13 | 6,782
[UNet2DCondition](https://huggingface.co/bes-dev/stable-diffusion-v1-4-onnx/tree/main)| 859.52 | 888,870
[VAE_encoder](https://huggingface.co/bes-dev/stable-diffusion-v1-4-onnx/tree/main) | 34.16 | 566,371
[VAE_decoder](https://huggingface.co/bes-dev/stable-diffusion-v1-4-onnx/tree/main) | 49.49 | 1,271,959
[SqueezeNet 1.0](https://github.com/onnx/models/tree/main/vision/classification/squeezenet) | 1.23 | 351
[AlexNet](https://github.com/onnx/models/tree/main/vision/classification/alexnet) | 60.96 | 665
[GoogleNet](https://github.com/onnx/models/tree/main/vision/classification/inception_and_googlenet/googlenet) | 6.99 | 1,606
[googlenet_age](https://github.com/onnx/models/tree/main/vision/body_analysis/age_gender) | 5.98 | 1,605
[LResNet100E-IR](https://github.com/onnx/models/tree/main/vision/body_analysis/arcface) | 65.22 | 12,102
[BERT-Squad](https://github.com/onnx/models/tree/main/text/machine_comprehension/bert-squad) | 113.61 | 22,767
[BiDAF](https://github.com/onnx/models/tree/main/text/machine_comprehension/bidirectional_attention_flow) | 18.08 | 9.87
[EfficientNet-Lite4](https://github.com/onnx/models/tree/main/vision/classification/efficientnet-lite4) | 12.96 | 1,361
[Emotion](https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus) | 12.95 | 877
[Mask R-CNN](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/mask-rcnn) | 46.77 | 92,077
</td>

<td>

Model | Params(M) | MACs(M)
---|-----------|---
<a href="benchmark/transfomer_models.py">LLaMa 1 layer</a> | 618       | 211,801  
[BEVFormer Tiny](https://github.com/DerryHub/BEVFormer_tensorrt) | 33.7      | 210,838
[rvm_mobilenetv3](https://github.com/PeterL1n/RobustVideoMatting) | 3.73      | 4,289
[yolov4](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/yolov4) | 64.33     | 3,319
[ConvNeXt-L](https://github.com/facebookresearch/ConvNeXt) | 229.79    | 34,872
[edgenext_small](https://github.com/mmaaz60/EdgeNeXt) | 5.58      | 1,357
[SSD](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/ssd) | 19.98     | 216,598
[RealESRGAN](https://github.com/xinntao/Real-ESRGAN) | 16.69     | 73,551
[ShuffleNet](https://github.com/onnx/models/tree/main/vision/classification/shufflenet) | 2.29      | 146
[GPT-2](https://github.com/onnx/models/tree/main/text/machine_comprehension/gpt-2) | 137.02    | 1,103
[T5-encoder](https://github.com/onnx/models/tree/main/text/machine_comprehension/t5) | 109.62    | 686
[T5-decoder](https://github.com/onnx/models/tree/main/text/machine_comprehension/t5) | 162.62    | 1,113
[RoBERTa-BASE](https://github.com/onnx/models/tree/main/text/machine_comprehension/roberta) | 124.64    | 688
[Faster R-CNN](https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/faster-rcnn) | 44.10     | 46,018
[FCN ResNet-50](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/fcn) | 35.29     | 37,056
[ResNet50](https://github.com/onnx/models/tree/main/vision/classification/resnet) | 25        | 3,868

</td>
</tr>
</table> 

## 🤝 Contributing

Contributions are welcome! Please open an issue or PR for:
- Bug reports
- Feature requests
- Documentation improvements
- New model support