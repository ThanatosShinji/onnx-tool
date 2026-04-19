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
- **内存压缩**：
    - 激活内存优化（最高减少95%）
    - 权重量化（支持FP16、INT8/INT4，含逐张量/逐通道/逐块方案）
- **量化与稀疏性**：全面支持量化和稀疏模型的分析
## 🤖 支持的模型架构

| 领域 | 模型 |
| --- | --- |
| **NLP** | BERT, T5, GPT, LLaMa, MPT ([TransformerModel](benchmark/transfomer_models.py)) |
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

### 延迟估计（4-bit权重，16-bit KV缓存）

model_type_4bit_kv16bit              | memory_size(GB) | Ultra-155H_TTFT | Ultra-155H_TPOT |  Arc-A770_TTFT |   Arc-A770_TPOT | H100-PCIe_TTFT |   H100-PCIe_TPOT
------------------------------------ |-----------------|-----------------|-----------------|------------------------ | ----------------------- |----------------| ---------------------
gpt-j-6b                             | 3.75678         | 1.0947          | 0.041742        |               0.0916882 |              0.00670853 | 0.0164015      |              0.00187839
yi-1.5-34B                           | 19.3369         | 5.77095         | 0.214854        |               0.45344   |              0.0345302  | 0.0747854      |              0.00966844
microsoft/phi-2                      | 1.82485         | 0.58361         | 0.0202761       |               0.0529628 |              0.00325866 | 0.010338       |              0.000912425
Phi-3-mini-4k                        | 2.49649         | 0.811173        | 0.0277388       |               0.0745356 |              0.00445802 | 0.0147274      |              0.00124825
Phi-3-small-8k-instruct              | 4.2913          | 1.38985         | 0.0476811       |               0.117512  |              0.00766303 | 0.0212535      |              0.00214565
Phi-3-medium-4k-instruct             | 7.96977         | 2.4463          | 0.088553        |               0.198249  |              0.0142317  | 0.0340576      |              0.00398489
Llama3-8B                            | 4.35559         | 1.4354          | 0.0483954       |               0.123333  |              0.00777784 | 0.0227182      |              0.00217779
Llama-3.1-70B-Japanese-Instruct-2407 | 39.4303         | 11.3541         | 0.438114        |               0.868475  |              0.0704112  | 0.137901       |              0.0197151
QWen-7B                              | 4.03576         | 1.34983         | 0.0448417       |               0.11722   |              0.00720671 | 0.0218461      |              0.00201788
Qwen2_72B_Instruct                   | 40.5309         | 11.6534         | 0.450343        |               0.890816  |              0.0723766  | 0.14132        |              0.0202654
> 💡 *延迟基于硬件规格计算——无需实际推理*

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
> 
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