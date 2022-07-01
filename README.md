# onnx-tool
A tool for ONNX model's shape inference and MACs counting.

* Shape inference
<p align="center">
  <img src="https://raw.githubusercontent.com/ThanatosShinji/onnx-tool/main/data/shape_inference.jpg">
</p>

---
* MACs counting for each node
<p align="center">
  <img src="https://raw.githubusercontent.com/ThanatosShinji/onnx-tool/main/data/macs_counting.jpg">
</p>

## How to install
    
`pip install onnx-tool`

OR

`pip install --upgrade git+https://github.com/ThanatosShinji/onnx-tool.git`
    
## How to use 
* Basic usage 
    ```python
    import onnx_tool
    modelpath = 'resnet50-v1-12.onnx'
    onnx_tool.model_profile(modelpath, None, None) #pass file name
    ```    
  
    ```python
    import onnx
    import onnx_tool
    modelpath = 'resnet50-v1-12.onnx'
    model = onnx.load_model(modelpath)
    onnx_tool.model_shape_infer(model, None, saveshapesmodel='resnet50_shapes.onnx')  # pass ONNX.ModelProto
    ```    

* Dynamic input shapes and dynamic resize scales('downsample_ratio')
    ```python
    import numpy
    import onnx_tool
    from onnx_tool import create_ndarray_f32 #or use numpy.ones(shape,numpy.float32) is ok
    modelpath = 'rvm_mobilenetv3_fp32.onnx'
    inputs= {'src': create_ndarray_f32((1, 3, 1080, 1920)), 'r1i': create_ndarray_f32((1, 16, 135, 240)),
                                 'r2i':create_ndarray_f32((1,20,68,120)),'r3i':create_ndarray_f32((1,40,34,60)),
                                 'r4i':create_ndarray_f32((1,64,17,30)),'downsample_ratio':numpy.array((0.25,),dtype=numpy.float32)}
    onnx_tool.model_profile(modelpath,inputs,None,saveshapesmodel='rvm_mobilenetv3_fp32_shapes.onnx')
    ```    

* Define your custom op's node profiler.
    ```python
    import numpy
    import onnx
    import onnx_tool
    from onnx_tool import NODEPROFILER_REGISTRY,NodeBase,create_ndarray_f32

    @NODEPROFILER_REGISTRY.register()
    class CropPlugin(NodeBase):
        def __init__(self,nodeproto:onnx.NodeProto):
            super().__init__(nodeproto)
            #parse your attributes here

        def infer_shape(self, intensors: list[numpy.ndarray]):
            #calculate output shapes here
            #this node crops intensors[0] to the shape of intensors[1], just return list of intensors[1]
            #no need to finish the true calculation, just return a ndarray of a right shape
            return [intensors[1]]

        def profile(self,intensors:list[numpy.ndarray],outtensors:list[numpy.ndarray]):
            macs=0
            params=0
            #accumulate macs and params here
            #this node has no calculation
            return macs,params

    onnx_tool.model_profile('./rrdb_new.onnx', {'input': create_ndarray_f32((1, 3, 335, 619))},
                            savenode='rrdb_new_nodemap.txt', saveshapesmodel='rrdb_new_shapes.onnx')
    ```

## Known Issues
* Loop op is not supported
* Shared weight tensor will be counted more than once

## Results of [ONNX Model Zoo](https://github.com/onnx/models) and SOTA models
Some models have dynamic input shapes. The MACs varies from input shapes. The input shapes used in these results are writen to [data/public/config.py](https://github.com/ThanatosShinji/onnx-tool/blob/main/data/public/config.py).
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
