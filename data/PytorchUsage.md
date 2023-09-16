    
## This md will show you how to apply onnx-tool to a pytorch model
* Pytorch export to ONNX 
    ```python
    import torchvision
    import torch
    model = torchvision.models.alexnet()
    x = torch.rand(1, 3, 224, 224)
    tmpfile='tmp.onnx'
    with torch.no_grad():
        torch_out = torch.onnx._export(model, x, tmpfile, opset_version=12 )  #opset 12 and opset 7 tested
        #do not use dynamic axes will simplify the process
    ```    

* Pytorch MACs profile and shape inference sample
    ```python
    import torchvision
    import onnx_tool
    import torch
    model = torchvision.models.alexnet()
    x = torch.rand(1, 3, 224, 224)
    tmpfile='tmp.onnx'
    with torch.no_grad():
        torch_out = torch.onnx._export(model, x, tmpfile, opset_version=12 )  #opset 12 and opset 7 tested
        #do not use dynamic axes will simplify the process
        onnx_tool.model_profile(tmpfile,saveshapesmodel='shapes.onnx')
        #you will get the print of MACs of each layer, and the hidden tensor's shapes will be export to shapes.onnx
    ```    
* Pytorch export all weight tensors to folder
    ```python
    import torchvision
    import onnx_tool
    import torch
    model = torchvision.models.alexnet()
    x = torch.rand(1, 3, 224, 224)
    tmpfile='tmp.onnx'
    with torch.no_grad():
        torch_out = torch.onnx._export(model, x, tmpfile, opset_version=12 )  #opset 12 and opset 7 tested
        #do not use dynamic axes will simplify the process
        onnx_tool.model_export_tensors_numpy(tmpfile,savefolder='./alexnet/')
        #all pretrained weight and bias tensors will be exported as a numpy file in './alexnet'
    ```    

* sample code [pytorch_example.py](../benchmark/pytorch_example.py)