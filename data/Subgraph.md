## Concept

To get a partition of a graph, you may do it in two ways: split the graph into three graphs or replace the subgraph as a new op.

### split the graph

In this case, the onnx model may be split into three onnx models: level 0 model, level 1 model, and level 2 model.  
Level 0 model: need to be executed before the subgraph.  
Level 1 model: the subgraph model.  
Level 2 model: need to be executed after the subgraph.  
You may see this image:
<p align="center">
  <img src="https://raw.githubusercontent.com/ThanatosShinji/onnx-tool/main/data/resnet18_subgraph.png">
</p>

Step1: execute level 0 model to get resnetv15_stage4_batchnorm2_fwd and resnetv15_stage4_conv0_fwd tensors.  
Step2: feed resnetv15_stage4_conv0_fwd to level 1 model and execute it.  
Step3: feed resnetv15_stage4_batchnorm2_fwd and resnetv15_stage4_batchnorm1_fwd from level 0 and level 1 models. Then get the final output tensor.    

### replace the subgraph as a new op
In this case, the onnx model will still be executed as one inference. The subgraph may be fused as one op layer which contains the all attributes of
op layers and all weight tensors from the subgraph .
<p align="center">
  <img src="https://raw.githubusercontent.com/ThanatosShinji/onnx-tool/main/data/resnet18_fused.png">
</p>
You can register your implementation to the inference engine to execute the fused op, for example: create a custom plugin in TensorRT.

## How to use 
* Split subgraph 
    ```python
    import onnx_tool
    modelpath = 'resnet18-v1-7_shapes.onnx'
    onnx_tool.model_subgraph(modelpath,['resnetv15_stage4_conv0_fwd'],['resnetv15_stage4_batchnorm1_fwd']) 
    #get the subgraph by selecting its input and output tensors
  
    onnx_tool.model_subgraph(modelpath,nodenames=['resnetv15_stage1_conv0_fwd','resnetv15_stage1_batchnorm0_fwd', \ 
                                                                                         'resnetv15_stage1_relu0_fwd'])
    #get the subgraph by selecting node names
    
    ```    
  
* Op fusion
    ```python
    import onnx_tool
    modelpath = 'resnet18-v1-7_shapes.onnx'
    onnx_tool.model_opfusion(modelpath,'fused','fused_0','fused.onnx',['resnetv15_stage4_conv0_fwd'],['resnetv15_stage4_batchnorm1_fwd']) 
    #create a new op, its op_type=='fused', name=='fused_0'. The new graph will be saved as 'fused.onnx'. 
  
    onnx_tool.model_opfusion(modelpath,'fused','fused_0','fused.onnx',nodenames=['resnetv15_stage1_conv0_fwd','resnetv15_stage1_batchnorm0_fwd',
                                                                                         'resnetv15_stage1_relu0_fwd'])
    #get the subgraph by selecting node names
      
