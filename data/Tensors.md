    
## How to use 
* Export Tensors
    ```python
    import onnx_tool
    onnx_tool.model_export_tensors_numpy( 'resnet50-v1-12.onnx', tensornames=None, savefolder=None,fp16=True) 
    #export all tensors in fp16 in current directory
    ```    
    ```python
    import onnx_tool
    onnx_tool.model_export_tensors_numpy( 'resnet50-v1-12.onnx', tensornames=['resnetv17_conv0_weight','resnetv17_stage1_conv1_weight'] \
    , savefolder='resnet50',fp16=False) 
    #export two tensors in 'resnet50' folder(create if not exists)
    ```    

    cli usage
    ```shell
    python -m onnx_tool -i 'resnet50-v1-12.onnx' --mode export_tensors --names 410 420 -o 'numpyfolder' 
    #export two tensors to ./numpyfolder
    ```    
    ```shell
    python -m onnx_tool -i 'resnet50-v1-12.onnx' --mode export_tensors -o 'numpyfolder' 
    #export all tensors
    ```
    ```shell
    python -m onnx_tool -i 'resnet50-v1-12.onnx' --mode export_tensors -o 'numpyfolder' --fp16 
    #export all tensors and convert all fp32 and fp64 tensors to fp16
    ```
* Simplify Tensor Names and Node Names
    ```python
    import onnx_tool
    onnx_tool.model_simplify_names('vgg19-7.onnx',savemodel='sim.onnx',renametensor=True,renamelayer=True,remove_unused_tensors=True)
    #rename all tensor names from xxxx/xxx_weight to number in [0, tensor count)  
    #rename all layer names from resnetv17_conv0_fwd to Conv_0 ( optype+num)
    #remove unused tensor proto
    ```    
  
  cli TBD   

  
* Modify input and output tensors
  ```python
    import onnx_tool
  
    #names only
    onnx_tool.model_simplify_names('vgg19-7.onnx',savemodel='sim.onnx',custom_inputs=['input'],custom_outputs=['output'])
    # change input tensor from data => input
    # change output tensor from vgg0_dense2_fwd => output
  
    #names and dimensions, set dynamic input and output dimensions
    onnx_tool.model_simplify_names('vgg19-7.onnx',savemodel='sim.onnx',custom_inputs={'input':'Batchx3x224x224'},custom_outputs={'output':'Batchx1000'})
    # change input tensor from data => input, set its dimension from fixed [1,3,224,224] to dynamic [Batch,3,224,224]
    # change output tensor from vgg0_dense2_fwd => output, set its dimension from fixed [1,1000] to dynamic [Batch,1000]
    ```    
  cli TBD