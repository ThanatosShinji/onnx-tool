    
## How to use 
* Basic usage
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
