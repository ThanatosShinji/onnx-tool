    
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
    onnx_tool.model_shape_infer(model, None, saveshapesmodel='resnet50_shapes.onnx',shapesonly=True)  # pass ONNX.ModelProto and remove static weights, minimize storage space, but may lead to display problem.
    ```    
    ```python
    import onnx
    import onnx_tool
    modelpath = 'resnet50-v1-12.onnx'
    model = onnx.load_model(modelpath)
    onnx_tool.model_shape_infer(model, None, saveshapesmodel='resnet50_shapes.onnx',shapesonly=True,dump_outputs=['resnetv17_stage1_conv3_fwd','resnetv17_stage1_conv3_fwd'])  
    # add two hidden tensors resnetv17_stage1_conv3_fwd resnetv17_stage1_conv3_fwd to 'resnet50_shapes.onnx' model's output tensors
    ```    
    cli usage (dynamic shapes is not supported)
    ```shell
    python -m onnx_tool -i 'resnet50-v1-12.onnx' -o 'resnet50_shapes.onnx'
    ```    
    ```shell
    python -m onnx_tool -i 'resnet50-v1-12.onnx' --names resnetv17_stage1_conv3_fwd resnetv17_stage1_conv2_fwd  -o 'resnet50_shapes.onnx'#add hidden tensors to graph's output tensors
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
