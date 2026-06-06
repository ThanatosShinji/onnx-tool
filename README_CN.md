<a href="README_CN.md">📄 简体中文</a> | <a href="https://github.com/ThanatosShinji/AI-Enhancement-Filter">✨ 新项目：AI-Enhancement-Filter</a> (由 onnx-tool 提供支持)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.6%2B-blue?logo=python"">
  <img src="https://img.shields.io/pypi/v/onnx-tool?color=green"">
  <img src="https://img.shields.io/github/license/ThanatosShinji/onnx-tool?color=orange"">
</p> 

# onnx-tool

**一个用于分析、优化和转换ONNX模型的综合性工具包**，具备针对大语言模型（LLM）、扩散模型和计算机视觉架构的高级功能。

- **LLM优化**：构建和分析大语言模型，支持KV缓存分析 ([示例](#build-profile))
- **图变换**： 
    - 常量折叠 ([文档](data/ConstantFolding.md))
    - 算子融合 ([文档](data/GraphFusion.md))
- **高级性能分析**： 
    - 快速形状推断
    - 支持稀疏性感知的MACs/参数量统计
- **计算图引擎**：运行时形状计算，开销极小 ([详情](#compute_graph-header))
- **推理引擎**：基于 PyTorch 的图推理引擎，集成内存池 ([详情](inference/README_CN.md))
    - 40+ 注册算子内核（Conv, Add, Relu, Gemm 等）
    - 动态分辨率场景下比 PyTorch **快 11.3 倍**
    - 两遍压缩算法实现 **54% 内存减少**
- **内存压缩**：
    - 激活内存优化（最高减少95%）
    - 权重量化（支持FP16、INT8/INT4，含逐张量/逐通道/逐块方案）
- **量化与稀疏性**：全面支持量化和稀疏模型的分析
## 🤖 支持的模型架构

| 领域 | 模型 |
| --- | --- |
| **NLP** | BERT, T5, GPT, LLaMa, MPT, Qwen3, Qwen3.5 (Dense & MoE), DeepSeek-V4 (Flash/Pro, MLA+MoE) ([TransformerModel](benchmark/transfomer_models.py)) |
| **扩散模型** | Stable Diffusion (TextEncoder, VAE, UNet) |
| **计算机视觉** | Detic, BEVFormer, SSD300_VGG16, ConvNeXt, Mask R-CNN, Silero VAD |
| **音频** | Sovits, LPCNet |

<p align="center">
  <img src="data/shape_inference.jpg"">
</p>

## ⚡ 几秒内构建并分析LLM
<a id="build-profile"></a>
在一秒钟内分析10个Hugging Face模型。以类似llama.cpp的简洁方式导出ONNX模型 ([代码](benchmark/llm_test.py))。

### 模型统计（1k token输入）
model name(1k input)                 | MACs(G) | Parameters(G) |   KV Cache(G)
------------------------------------ |---------|---------------| -------
gpt-j-6b                             | 6277    | 6.05049       |  0.234881
yi-1.5-34B                           | 35862   | 34.3889       |  0.125829
microsoft/phi-2                      | 2948    | 2.77944       |  0.167772
Phi-3-mini-4k                        | 4083    | 3.82108       |  0.201327
Phi-3-small-8k-instruct              | 7912    | 7.80167       |  0.0671089
Phi-3-medium-4k-instruct             | 14665   | 13.9602       |  0.104858
Llama3-8B                            | 8029    | 8.03026       |  0.0671089
Llama-3.1-70B-Japanese-Instruct-2407 | 72888   | 70.5537       |  0.167772
QWen-7B                              | 7509    | 7.61562       |  0.0293601
Qwen2_72B_Instruct                   | 74895   | 72.7062       |  0.167772
**Qwen3.5-4B-Instruct** 🆕           | 4807    | 4.401         |  0.067109
**Qwen3.5-35B-A3B-Instruct** 🆕 (MoE)| 3574    | 34.161        |  0.041943
**DeepSeek-V4-Flash** 🆕 (MoE/MLA)   | 15681   | 283.811       |  0.045089
**DeepSeek-V4-Pro** 🆕 (MoE/MLA)     | 55701   | 1571.742      |  0.063963

### 延迟估计（4-bit权重，16-bit KV缓存）

**Prefill 吞吐量 (tokens/s, 1k 输入)**

model                                | Ultra-358H | Arc-B70 | RTX-4090 | RTX-5090
------------------------------------ |-----------|---------|----------|----------
gpt-j-6b                             | 4688.3    | 14585.1 | 13319.9  | 16966.2
yi-1.5-34B                           | 826.8     | 2562.7  | 2333.0   | 2969.2
microsoft/phi-2                      | 9781.9    | 30676.2 | 28209.0  | 35996.7
Phi-3-mini-4k                        | 6885.3    | 21761.2 | 20147.4  | 25777.3
Phi-3-small-8k-instruct              | 3680.5    | 11484.2 | 10514.9  | 13409.2
Phi-3-medium-4k-instruct             | 2003.1    | 6230.3  | 5688.8   | 7247.3
Llama3-8B                            | 3622.8    | 11308.4 | 10357.2  | 13209.5
Llama-3.1-70B-Japanese-Instruct-2407 | 407.5     | 1262.0  | 1148.2   | 1461.4
QWen-7B                              | 3851.9    | 12046.8 | 11051.8  | 14107.0
Qwen2_72B_Instruct                   | 397.9     | 1230.9  | 1118.8   | 1423.3

**Decode 吞吐量 (tokens/s)**

model                                | Ultra-358H | Arc-B70 | RTX-4090 | RTX-5090
------------------------------------ |-----------|---------|----------|----------
gpt-j-6b                             | 37.4      | 177.6   | 294.4    | 523.4
yi-1.5-34B                           | 7.4       | 35.2    | 58.4     | 103.8
microsoft/phi-2                      | 76.8      | 364.8   | 604.7    | 1075.1
Phi-3-mini-4k                        | 56.2      | 266.7   | 442.2    | 786.1
Phi-3-small-8k-instruct              | 33.1      | 157.2   | 260.7    | 463.5
Phi-3-medium-4k-instruct             | 17.9      | 85.2    | 141.3    | 251.2
Llama3-8B                            | 32.6      | 154.9   | 256.9    | 456.7
Llama-3.1-70B-Japanese-Instruct-2407 | 3.5       | 16.7    | 27.7     | 49.2
QWen-7B                              | 34.5      | 163.7   | 271.3    | 482.4
Qwen2_72B_Instruct                   | 3.5       | 16.7    | 27.7     | 49.2
> 💡 *延迟基于硬件规格计算——无需实际推理。以 BF16/FP16 计算 + FP32 累加作为算力标准。*

## 🔧 基本解析与编辑

<a id="basic-parse-edit"></a>
直观的API用于模型操作：
```python
from onnx_tool import Model

model = Model('model.onnx')          # 加载任意ONNX文件
graph = model.graph                  # 访问计算图
node = graph.nodemap['Conv_0']       # 修改算子属性
tensor = graph.tensormap['weight']   # 编辑张量数据/类型
model.save_model('modified.onnx')    # 保存更改
```
详见 [`benchmark/examples.py`](benchmark/examples.py) 中的完整示例。

## 📊 形状推断与性能分析
<a id="shapeinfer-profile"></a>
所有性能分析都依赖于精确的形状推断：

<p align="center">
  <img src="data/macs_counting.png"">
  <img src="data/sparse_model.png"">
</p>

### 性能分析能力

- **标准分析**：MACs、参数量、内存占用
- **稀疏性感知分析**：量化稀疏性对计算的影响

📚 **了解更多**: 

- [性能分析指南](data/Profile.md)
- [PyTorch集成](data/PytorchUsage.md)
- [TensorFlow集成](data/TensorflowUsage.md)

## ⚙️ 计算图与形状引擎
<a id="compute_graph-header"></a>

通过消除形状计算开销，将导出的ONNX图转换为高效的*计算图*：

<p align="center">
  <img src="data/compute_graph.png"">
</p>

- **计算图**：仅包含计算操作的最小图
- **形状引擎**：用于动态模型的运行时形状解析器

**使用场景**：

- 与自定义推理引擎集成 ([指南](data/inference_engine.md))
- 形状回归测试 ([示例](benchmark/shape_regress.py))

## 💾 内存压缩
<a id="memory-compression"></a>

### 激活内存压缩
重用临时缓冲区以最小化峰值内存使用——对大语言模型和高分辨率CV模型至关重要。

 model                         | Native Memory Size(MB) | Compressed Memory Size(MB) | Compression Ratio(%) 
-------------------------------|------------------------|----------------------------|-------------------
 StableDiffusion(VAE_encoder)  | 14,245                 | 540                        | 3.7                  
 StableDiffusion(VAE_decoder)  | 25,417                 | 1,140                      | 4.48                 
 StableDiffusion(Text_encoder) | 215                    | 5                          | 2.5                  
 StableDiffusion(UNet)         | 36,135                 | 2,232                      | 6.2                  
 GPT2                          | 40                     | 2                          | 6.9                  
 BERT                          | 2,170                  | 27                         | 1.25                 
> ✅ 典型模型实现 **>90% 的激活内存减少**  
> 📌 实现代码：[`benchmark/compression.py`](benchmark/compression.py)

`compress_memory()` 算法已通过以下改进进行了补丁更新（参见 [`onnx_tool/graph.py`](onnx_tool/graph.py)）：
1. **大小降序分配**：每个节点内的新张量按大小降序分配，减少碎片
2. **尾部压缩**：裁剪内存池末尾未使用的空间
3. **列表引用修复**：每个张量的 `[offset, size]` 存储为独立副本，防止意外的跨张量别名

各模型的基准测试结果：

 model                         | Native(MB) | Compressed(MB) | Ratio(%)
-------------------------------|------------|----------------|----------
 VAE encoder                   | 11,313.6   | **512.0**      | 4.53
 VAE decoder                   | 19,816.2   | **896.1**      | 4.52
 Text encoder                  | 172.5      | **3.7**        | 2.12
 GPT2                          | 381.1      | **16.1**       | 4.23
 ResNet50                      | 279.3      | **10.7**       | 3.84

> ✅ 优化算法实现 **最高 54% 的额外池减少**（ResNet50：21.4→10.7 MB，相比原始算法）

### 推理引擎

[`inference/`](inference/) 模块提供了一个完整的基于 PyTorch 的推理引擎，构建在压缩内存池之上：

- **MemoryPool**：预分配连续缓冲区的零拷贝张量视图
- **Kernel Registry**：40+ 注册算子内核（Conv, Add, Relu, Gemm, MatMul 等）
- **GraphInfer**：集成形状引擎的图级别推理

**性能亮点**（ResNet18 在 Intel XPU 上）：

| 指标 | PyTorch | GraphInfer | 提升 |
|------|---------|------------|------|
| 单次推理 (1080p) | 0.0151s | **0.0167s** | 0.91x（持平） |
| 顺序推理 7 种分辨率 | 1820ms | **162ms** | **快 11.3 倍** |
| 峰值 XPU 内存 (4K) | 2338 MB | **1926 MB** | **减少 17.6%** |
| 内存池 (4K) | — | **1012 MB** | 固定大小 |

> 📌 完整基准测试详情参见 [`inference/README_CN.md`](inference/README_CN.md)

### 权重压缩
对于在内存受限设备上部署大型模型至关重要：

| 量化方案 | 相对于FP32大小 | 示例（7B模型） |
| --- | --- | --- |
| FP32 (基线) | 1.00× | 28 GB |
| FP16 | 0.50× | 14 GB |
| INT8 (逐通道) | 0.25× | 7 GB |
| INT4 (block=32, 对称) – llama.cpp | 0.156× | 4.4 GB |

**支持的方案**：

- ✅ FP16
- ✅ INT8: 对称/非对称 × 逐张量/逐通道/逐块
- ✅ INT4: 对称/非对称 × 逐张量/逐通道/逐块

📌 参见 [`benchmark/examples.py`](benchmark/examples.py) 获取实现示例。

## 🚀 安装

```bash
# PyPI（推荐）
pip install onnx-tool

# 最新开发版本
pip install --upgrade git+https://github.com/ThanatosShinji/onnx-tool.git
```

**要求**：Python ≥ 3.6

> ⚠️ **故障排除**：如果ONNX安装失败，请尝试：
> ```bash
> pip install onnx==1.8.1 && pip install onnx-tool
> ```

## 已知问题

* 不支持 Loop 算子
* 不支持 Sequence 类型

## 📈 模型动物园结果
<a id='models'></a>
对 [ONNX Model Zoo](https://github.com/onnx/models) 和前沿模型进行全面性能分析。输入形状定义于 [`data/public/config.py`](data/public/config.py) 中。

📥 **下载预分析模型**（含完整张量形状）:

- [百度网盘](https://pan.baidu.com/s/1eebBP-n-wXvOhSmIH-NUZQ) (提取码: `p91k`)
- [Google Drive](https://drive.google.com/drive/folders/1H-ya1wTvjIMg2pMcMITWDIfWNSnjYxTn?usp=sharing)
<p id="results" align="center">
<table>
<tr>
<td>

模型 | 参数量(M) | MACs(M)
---|---|---|
<a href="benchmark/transfomer_models.py">GPT-J 1层</a> | 464 | 173,398  
<a href="benchmark/transfomer_models.py">MPT 1层</a> | 261 | 79,894
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

模型 | 参数量(M) | MACs(M)
---|-----------|---|
<a href="benchmark/transfomer_models.py">LLaMa 1层</a> | 618       | 211,801  
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

## 🤝 贡献

欢迎贡献！请提交 issue 或 PR 以进行：

- 错误报告
- 功能请求
- 文档改进
- 新模型支持