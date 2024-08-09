# onnx-tool

**A tool for ONNX model:**
* *[构建LLM模型](benchmark/llm_test.py)*
* *解析ONNX模型并且编辑: [常量层折叠](data/ConstantFolding_CN.md), Ops fusion.*
* *模型分析：Tensor形状推理，每个Op的MACs统计*
* *Compute Graph 和 Shape Engine.*
* *内存压缩：激活Tenosr的内存压缩和权重的内存压缩*
* *支持量化模型和稀疏模型.*

支持的模型有:

* NLP: BERT, T5, GPT, LLaMa, MPT([TransformerModel](benchmark/transfomer_models.py))
* Diffusion: Stable Diffusion(TextEncoder, VAE, UNET)
* CV: [Detic](https://github.com/ThanatosShinji/onnx-tool/issues/63), [BEVFormer](benchmark/compression.py), [SSD300_VGG16](https://github.com/ThanatosShinji/onnx-tool/issues/66), ...
* Audio: sovits, LPCNet

---

## 构建LLM模型并分析
<a id="build-profile"></a>
在1秒内快速分析10个hugging face模型. 将模型保存为和llama.cpp一样简单的ONNX格式.
[code ref](benchmark/llm_test.py)

model name(1k input)                 | MACs(G) | Parameters(G) |   KV Cache(G)
------------------------------------ |---------|---------------| ----------
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

通过硬件参数快速获取每个模型的第一个token延时和后续token延时

model_type_4bit_kv16bit              | memory_size(GB) |   Ultra-155H_first_latency |   Ultra-155H_next_latency |  Arc-A770_first_latency |   Arc-A770_next_latency |   H100-PCIe_first_latency |   H100-PCIe_next_latency
------------------------------------ |-----------------| -------------------------- | ------------------------- |------------------------ | ----------------------- | ------------------------- | ------------------------
gpt-j-6b                             | 3.75678         |                   1.0947   |                 0.041742  |               0.0916882 |              0.00670853 |                 0.0164015 |              0.00187839
yi-1.5-34B                           | 19.3369         |                   5.77095  |                 0.214854  |               0.45344   |              0.0345302  |                 0.0747854 |              0.00966844
microsoft/phi-2                      | 1.82485         |                   0.58361  |                 0.0202761 |               0.0529628 |              0.00325866 |                 0.010338  |              0.000912425
Phi-3-mini-4k                        | 2.49649         |                   0.811173 |                 0.0277388 |               0.0745356 |              0.00445802 |                 0.0147274 |              0.00124825
Phi-3-small-8k-instruct              | 4.2913          |                   1.38985  |                 0.0476811 |               0.117512  |              0.00766303 |                 0.0212535 |              0.00214565
Phi-3-medium-4k-instruct             | 7.96977         |                   2.4463   |                 0.088553  |               0.198249  |              0.0142317  |                 0.0340576 |              0.00398489
Llama3-8B                            | 4.35559         |                   1.4354   |                 0.0483954 |               0.123333  |              0.00777784 |                 0.0227182 |              0.00217779
Llama-3.1-70B-Japanese-Instruct-2407 | 39.4303         |                  11.3541   |                 0.438114  |               0.868475  |              0.0704112  |                 0.137901  |              0.0197151
QWen-7B                              | 4.03576         |                   1.34983  |                 0.0448417 |               0.11722   |              0.00720671 |                 0.0218461 |              0.00201788
Qwen2_72B_Instruct                   | 40.5309         |                  11.6534   |                 0.450343  |               0.890816  |              0.0723766  |                 0.14132   |              0.0202654

---


## 解析与编辑
你可以用onnx_tool.Model类去加载任意ONNX模型，变成易于编辑的python类实例，你可以:  
用onnx_tool.Graph类去改变图结构;  
用onnx_tool.Node类去改变每个Op的属性和输入输出Tensor;  
用onnx_tool.Tensor改变任意Tensor的数据类型和数据内容.  
修改完成后，只需要调用Graph或者Model类的save_model接口可以保存所有的修改内容到新的ONNX模型.

请参考 [benchmark/examples.py](benchmark/examples.py).

---

## 形状推理 和 模型分析
每个模型分析报告需要基于某个特定的输入Tensor的形状。所以在分析模型之前要先进行一次形状推理。
<p align="center">  
  <img src="data/shape_inference.jpg">
</p>  

<p align="center">
  <img src="data/macs_counting.png">
</p>
浮点乘加数（等于2倍的浮点操作数）, 内存占用(字节数), 参数量(参数个数)<br><br>

<p id="sparsity" align="center">
  <img src="data/sparse_model.png">
</p>
稀疏的块的形状, 稀疏块的稀疏率（全为0的稀疏块的稀疏率）, 参数的稀疏率（数值为0的稀疏率）<br><br>  

how to use: [data/Profile.md](data/Profile.md).  
pytorch usage: [data/PytorchUsage.md](data/PytorchUsage.md).  
tensorflow
usage: [data/TensorflowUsage.md](data/TensorflowUsage.md).  
examples: [benchmark/examples.py](benchmark/examples.py).

---

## Compute Graph with Shape Engine

<p id="compute_graph" align="center">
  <img src="data/compute_graph.png">
</p>  

移除了所有的Tensor形状计算op， 更新动态Tensor的形状可以用Shape Engine来替代。推理引擎只需要负责计算图的计算，不需要考虑Tensor的形状更新。   
examples:   
[benchmark/shape_regress.py](benchmark/shape_regress.py).  
[benchmark/examples.py](benchmark/examples.py).  
如何集成 *Compute Graph* 和 *Shape Engine* 到cpp推理引擎中: [data/inference_engine.md](data/inference_engine.md)

---

## 多OP融合为新OP

MHA and Layernorm Fusion for Transformers
<p align="center">
  <img src="data/mha_fusion.png">
</p>
<p align="center">
  <img src="data/layernorm_fusion.png">
</p>
Resnet18 fusion
<p align="center">
  <img src="data/resnet18_fused.png">
</p>

how to use: [data/Subgraph.md](data/GraphFusion.md).  
BERT examples: [benchmark/examples.py](benchmark/examples.py).  
Pattern fusion: [benchmark/do_fusion.py](benchmark/do_fusion.py).

---

## 从模型中提取一个子模型
可以帮助实现model parallel。
<p align="center">
  <img src="data/resnet18_subgraph.png">
</p>

how to use: [data/Subgraph.md](data/GraphFusion.md).

---

## Memory Compression

对于LLM和高分辨CV模型, 激活内存的压缩可以帮助节省整个模型的内存使用.  
压缩方法可以在大多数模型上实现 5% 内存压缩率.   
例如:

 model                         | Native Memory Size(MB) | Compressed Memory Size(MB) | Compression Ratio(%) 
-------------------------------|------------------------|----------------------------|----------------------
 StableDiffusion(VAE_encoder)  | 14,245                 | 540                        | 3.7                  
 StableDiffusion(VAE_decoder)  | 25,417                 | 1,140                      | 4.48                 
 StableDiffusion(Text_encoder) | 215                    | 5                          | 2.5                  
 StableDiffusion(UNet)         | 36,135                 | 2,232                      | 6.2                  
 GPT2                          | 40                     | 2                          | 6.9                  
 BERT                          | 2,170                  | 27                         | 1.25                 

code example: [benchmark/compression.py](benchmark/compression.py)

---


## How to install
    
`pip install onnx-tool`

OR

`pip install --upgrade git+https://github.com/ThanatosShinji/onnx-tool.git`  

python>=3.6

If `pip install onnx-tool` failed by onnx's installation, you may try `pip install onnx==1.8.1` (a lower version like this) first.  
Then `pip install onnx-tool` again.


---

## Known Issues
* Loop op is not supported

---

## Results of [ONNX Model Zoo](https://github.com/onnx/models) and SOTA models
注意对于支持动态输入形状的模型，模型的MACs随输入形状的改变而改变。下表中的MACs数据是基于[data/public/config.py](data/public/config.py)中的配置输入形状得到。
带有所有Tensor形状的模型和分析报告可以从下面的网盘中下载: [baidu drive](https://pan.baidu.com/s/1eebBP-n-wXvOhSmIH-NUZQ 
)(code: p91k) [google drive](https://drive.google.com/drive/folders/1H-ya1wTvjIMg2pMcMITWDIfWNSnjYxTn?usp=sharing)
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
</p>