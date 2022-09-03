## Updates

* v0.2.10:
  * You can pass a list of remove ops for the profile table. This will get a more  
    practical memory usage, e.g. Relu. See hidden_ops of model_profile.
* v0.2.9:
  * Support memory usage profile for each layer
  * Export profile table to a .csv file is valid, you can use excel to operate the table

## How to use

* Basic usage
    ```python
    import onnx_tool
    modelpath = 'resnet50-v1-12.onnx'
    onnx_tool.model_profile(modelpath, None, None) # pass file name
    onnx_tool.model_profile(modelpath, savenode='node_table.txt') # save profile table to txt file
    onnx_tool.model_profile(modelpath, savenode='node_table.csv') # save profile table to csv file
    ```    

    ```python
    import onnx
    import onnx_tool
    modelpath = 'resnet50-v1-12.onnx'
    model = onnx.load_model(modelpath)
    onnx_tool.model_shape_infer(model, None, saveshapesmodel='resnet50_shapes.onnx',shapesonly=True)  
  # pass ONNX.ModelProto and remove static weights, minimize storage space.
    ```    
    ```python
    import onnx
    import onnx_tool
    modelpath = 'resnet50-v1-12.onnx'
    model = onnx.load_model(modelpath)
    onnx_tool.model_shape_infer(model, None, saveshapesmodel='resnet50_shapes.onnx',shapesonly=True,dump_outputs=['resnetv17_stage1_conv3_fwd' \
    ,'resnetv17_stage1_conv3_fwd'])  
    # add two hidden tensors resnetv17_stage1_conv3_fwd resnetv17_stage1_conv3_fwd to 'resnet50_shapes.onnx' model's output tensors
    ```    
    cli usage (dynamic shapes is also supported)
    ```shell
    python -m onnx_tool -i 'resnet50-v1-12.onnx' -o 'resnet50_shapes.onnx'
    ```    
    ```shell
    python -m onnx_tool -i 'resnet50-v1-12.onnx' --names resnetv17_stage1_conv3_fwd resnetv17_stage1_conv2_fwd  -o 'resnet50_shapes.onnx'
  #add hidden tensors to graph's output tensors
    ```    
    ```shell
    python -m onnx_tool -i rvm_mobilenetv3_fp32.onnx --mode profile --dynamic_inputs \
    src:f32:1x3x1080x1920 r1i:f32:1x16x135x240 r2i:f32:1x20x68x120 r3i:f32:1x40x34x60 r4i:f32:1x64x17x30 downsample_ratio:f32:-1:0.25
    #dynamic_inputs string format:  <tensor name>:<data type>:<shape>[:<data>]
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
## Notes
* Parameter's statistics is very accurate now, weight sharing among nodes will be detected, and only count it for the first node.  
  You may see something like this (e.g. 'bidaf-9.onnx' ):
```shell
bidaf-9.onnx
infered all tensor shapes, time cost 0.028 s
profile all nodes, time cost 0.002 s

****************************************************************
Please note that Weight Tensors Sharing is detected:
Tensor:Word_Embedding 
Shared by: 
            Gather_8
            Gather_10

Tensor:Char_Embedding 
Shared by: 
            Gather_9
            Gather_11

Tensor:W_0 
Shared by: 
            Convolution10413
            Convolution10253

Tensor:b 
Shared by: 
            Plus10415
            Plus10255

Tensor:0_WU 
Shared by: 
            0_U_0
            0_U

Tensor:0_bU 
Shared by: 
            Plus10470
            Plus10310

Tensor:0_WT 
Shared by: 
            0_T_0
            0_T

Tensor:0_bT 
Shared by: 
            Plus10463
            Plus10303

Tensor:1_WU 
Shared by: 
            1_U_0
            1_U

Tensor:1_bU 
Shared by: 
            Plus10515
            Plus10355

Tensor:1_WT 
Shared by: 
            1_T_0
            1_T

Tensor:1_bT 
Shared by: 
            Plus10508
            Plus10348

****************************************************************
```

* Profile result's memory value = static tensors' memory + output tensors' memory.  
  The memory of the same tensor will be counted once like the parameter's statistics.