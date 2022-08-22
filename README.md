# onnx-tool
**A tool for ONNX model:**  
* *Shape inference.*  
* *MACs(FLOPs) counting for each layer.*   
* *Extract subgraph from ONNX model, or do inplace op fusion.*
...  
**and any operation you can image with ONNX.**

---
## Shape inference
<p align="center">  
  <img src="https://raw.githubusercontent.com/ThanatosShinji/onnx-tool/main/data/shape_inference.jpg">
</p>  

how to use: [data/Profile.md](https://github.com/ThanatosShinji/onnx-tool/blob/main/data/Profile.md).  

---
## MACs counting for each layer (FLOPs=2*MACs)
<p align="center">
  <img src="https://raw.githubusercontent.com/ThanatosShinji/onnx-tool/main/data/macs_counting.jpg">
</p>

how to use: [data/Profile.md](https://github.com/ThanatosShinji/onnx-tool/blob/main/data/Profile.md).  

---
## Extract subgraph from ONNX model
<p align="center">
  <img src="https://raw.githubusercontent.com/ThanatosShinji/onnx-tool/main/data/resnet18_subgraph.png">
</p>

how to use: [data/Subgraph.md](https://github.com/ThanatosShinji/onnx-tool/blob/main/data/Subgraph.md).     

---

## Inplace op fusion
<p align="center">
  <img src="https://raw.githubusercontent.com/ThanatosShinji/onnx-tool/main/data/resnet18_fused.png">
</p>

how to use: [data/Subgraph.md](https://github.com/ThanatosShinji/onnx-tool/blob/main/data/Subgraph.md).  

---

## Add any hidden tensors to model's outputs
<p align="center">
  <img src="https://raw.githubusercontent.com/ThanatosShinji/onnx-tool/main/data/add_otuput_tensors.png">
</p>

how to use: [data/Profile.md](https://github.com/ThanatosShinji/onnx-tool/blob/main/data/Profile.md).  

---
## Tensor operations
* *Export weight tensors to files*  
* *Simplify tensor and node names, convert name from a long string to a short string*  
* *Remove unused tensors, models like vgg19-7.onnx set its static weight tensors as its input tensors*  
* *Set custom input and output tensors' name and dimension, change model from fixed input to dynamic input*  
how to use: [data/Tensors.md](https://github.com/ThanatosShinji/onnx-tool/blob/main/data/Tensors.md).  

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
Some models have dynamic input shapes. The MACs varies from input shapes. The input shapes used in these results are writen to [data/public/config.py](https://github.com/ThanatosShinji/onnx-tool/blob/main/data/public/config.py).
These onnx models with all tensors' shape can be downloaded: [baidu drive](https://pan.baidu.com/s/1eebBP-n-wXvOhSmIH-NUZQ 
)(code: p91k) [google drive](https://drive.google.com/drive/folders/1H-ya1wTvjIMg2pMcMITWDIfWNSnjYxTn?usp=sharing)
<p align="center">
<table>
<tr>

<td>

Model | Params(M) | MACs(M)
---|---|---
[MobileNet v2-1.0-fp32](https://github.com/onnx/models/blob/main/vision/classification/mobilenet) | 3.3 | 300
[ResNet50_fp32](https://github.com/onnx/models/tree/main/vision/classification/resnet) | 25 | 3868
[SqueezeNet 1.0](https://github.com/onnx/models/tree/main/vision/classification/squeezenet) | 1.23 | 351
[VGG 19](https://github.com/onnx/models/tree/main/vision/classification/vgg) | 143.66 | 19643
[AlexNet](https://github.com/onnx/models/tree/main/vision/classification/alexnet) | 60.96 | 665
[GoogleNet](https://github.com/onnx/models/tree/main/vision/classification/inception_and_googlenet/googlenet) | 6.99 | 1606
[googlenet_age_adience](https://github.com/onnx/models/tree/main/vision/body_analysis/age_gender) | 5.98 | 1605
[LResNet100E-IR](https://github.com/onnx/models/tree/main/vision/body_analysis/arcface) | 65.22 | 12102
[BERT-Squad](https://github.com/onnx/models/tree/main/text/machine_comprehension/bert-squad) | 113.61 | 22767
[BiDAF](https://github.com/onnx/models/tree/main/text/machine_comprehension/bidirectional_attention_flow) | 18.08 | 9.87
[EfficientNet-Lite4](https://github.com/onnx/models/tree/main/vision/classification/efficientnet-lite4) | 12.96 | 1361
[Emotion FERPlus](https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus) | 12.95 | 877
[Mask R-CNN R-50-FPN-fp32](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/mask-rcnn) | 46.77 | 92077
</td>

<td>

Model | Params(M) | MACs(M)
---|---|---
[rvm_mobilenetv3_fp32.onnx](https://github.com/PeterL1n/RobustVideoMatting) | 3.73 | 4289
[yolov4](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/yolov4) | 64.33 | 33019
[ConvNeXt-L](https://github.com/facebookresearch/ConvNeXt) | 229.79 | 34872
[edgenext_small](https://github.com/mmaaz60/EdgeNeXt) | 5.58 | 1357
[SSD](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/ssd) | 19.98 | 216598
[RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN) | 16.69 | 73551
[ShuffleNet-v2-fp32](https://github.com/onnx/models/tree/main/vision/classification/shufflenet) | 2.29 | 146
[GPT-2](https://github.com/onnx/models/tree/main/text/machine_comprehension/gpt-2) | 137.02 | 1103
[T5-encoder](https://github.com/onnx/models/tree/main/text/machine_comprehension/t5) | 109.62 | 686
[T5-decoder-with-lm-head](https://github.com/onnx/models/tree/main/text/machine_comprehension/t5) | 162.62 | 1113
[RoBERTa-BASE](https://github.com/onnx/models/tree/main/text/machine_comprehension/roberta) | 124.64 | 688
[Faster R-CNN R-50-FPN-fp32](https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/faster-rcnn) | 44.10 | 46018
[FCN ResNet-50](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/fcn) | 35.29 | 37056


</td>
</tr>
</p>
