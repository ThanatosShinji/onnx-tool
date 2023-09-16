<a href="README_CN.md">简体中文</a>
# onnx-tool

**A tool for ONNX model:**

* *Parse and edit: <a href="data/ConstantFolding.md">Constant Folding</a>; OPs fusion.*
* *Model profiling: Rapid shape inference; MACs statistics*
* *Compute Graph and Shape Engine.*
* *Model memory compression: activation compression and weight compression.*
* *Quantized models and sparse models are supported.*

Supported Models:

* NLP: BERT, T5, GPT, LLaMa, MPT(<a href="benchmark/transfomer_models.py">TransformerModel</a>)
* Diffusion: Stable Diffusion(TextEncoder, VAE, UNET)
* CV: <a href="benchmark/compression.py">BEVFormer</a>, MobileNet, YOLO, ...
* Audio: sovits, LPCNet

---

## Parse and edit
You can load any onnx file by onnx_tool.Model:  
Change graph structure with onnx_tool.Graph;  
Change op attributes and IO tensors with onnx_tool.Node;  
Change tensor data or type with onnx_tool.Tensor.  
To apply your changes, just call save_model method of onnx_tool.Model or onnx_tool.Graph.

Please refer [benchmark/examples.py](benchmark/examples.py).

---

## Shape inference
All profiling data must be built on shape inference result.
<p align="center">  
  <img src="data/shape_inference.jpg">
</p>  

how to use: [data/Profile.md](data/Profile.md).  
pytorch usage: [data/PytorchUsage.md](data/PytorchUsage.md).  
tensorflow
usage: [data/TensorflowUsage.md](data/TensorflowUsage.md).  
samples: [benchmark/examples.py](benchmark/examples.py).


## Profile Model

<p align="center">
  <img src="data/macs_counting.png">
</p>
Float MultipleAdd Count(1 MAC=2 FLOPs), Memory Usage(in bytes), Parameters(elements number)<br><br>

<p id="sparsity" align="center">
  <img src="data/sparse_model.png">
</p>
Sparse Pattern, Sparse Block Ratio, Sparse Element Ratio<br><br>  

how to use: [data/Profile.md](data/Profile.md).  
pytorch usage: [data/PytorchUsage.md](data/PytorchUsage.md).  
tensorflow
usage: [data/TensorflowUsage.md](data/TensorflowUsage.md).  
samples: [benchmark/examples.py](benchmark/examples.py).

---

## Compute Graph with Shape Engine

<p id="compute_graph" align="center">
  <img src="data/compute_graph.png">
</p>  

Remove shape calculation layers(created by ONNX export) to get a *Compute Graph*. Use *Shape Engine* to update tensor
shapes at runtime.  
Samples: [benchmark/shape_regress.py](benchmark/shape_regress.py).
[benchmark/examples.py](benchmark/examples.py).  
Integrate *Compute Graph* and *Shape Engine* into a cpp inference
engine: [data/inference_engine.md](data/inference_engine.md)

---

## Inplace op fusion

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

how to use: [data/Subgraph.md](data/Subgraph.md).  
BERT samples: [benchmark/examples.py](benchmark/examples.py).  
Pattern fusion: [benchmark/do_fusion.py](benchmark/do_fusion.py).

---

## Extract subgraph from ONNX model
Help implement model parallelism.
<p align="center">
  <img src="data/resnet18_subgraph.png">
</p>

how to use: [data/Subgraph.md](data/Subgraph.md).

---

## Memory Compression

### Activation compression
Activation memory also called temporary memory is created by each OP's output. Only the last activation marked as the
model's output will be kept. So you don't have to prepare memory space for each activation tensor. They better reuse 
an optimized memory size.

For large language models and high-resolution CV models, the activation memory compression is a key to save memory.  
The compression method achieves 5% memory compression on most models.   
For example:

 model                         | Native Memory Size(MB) | Compressed Memory Size(MB) | Compression Ratio(%) 
-------------------------------|------------------------|----------------------------|----------------------
 StableDiffusion(VAE_encoder)  | 14,245                 | 540                        | 3.7                  
 StableDiffusion(VAE_decoder)  | 25,417                 | 1,140                      | 4.48                 
 StableDiffusion(Text_encoder) | 215                    | 5                          | 2.5                  
 StableDiffusion(UNet)         | 36,135                 | 2,232                      | 6.2                  
 GPT2                          | 40                     | 2                          | 6.9                  
 BERT                          | 2,170                  | 27                         | 1.25                 

code sample: [benchmark/compression.py](benchmark/compression.py)

### Weight compression
A fp32 model with 7B parameters will take 28GB disk space and memory space. You can not even run the model if your device
 doesn't have that much memory space. So weight compression is critical to run large language models. As a reference, 7B 
model with int4 symmetric per block(32) quantization(llama.cpp's q4_0 quantization method) only has ~0.156x model size compared with fp32 model. 

Current support:   
* [fp16]
* [int8]x[symmetric/asymmetric]x[per tensor/per channel/per block]  
* [int4]x[symmetric/asymmetric]x[per tensor/per channel/per block]  

code samples:[benchmark/examples.py](benchmark/examples.py).  


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
* Tensor types created by onnx_tool are not correct
  
---

## Results of [ONNX Model Zoo](https://github.com/onnx/models) and SOTA models
Some models have dynamic input shapes. The MACs varies from input shapes. The input shapes used in these results are writen to [data/public/config.py](data/public/config.py).
These onnx models with all tensors' shape can be downloaded: [baidu drive](https://pan.baidu.com/s/1eebBP-n-wXvOhSmIH-NUZQ 
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