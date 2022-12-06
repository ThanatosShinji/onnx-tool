import torchvision
import onnx_tool
import torch

tmpfile = 'tmp.onnx'


def alexnet():
    model = torchvision.models.alexnet()
    model.eval()
    x = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        torch_out = torch.onnx._export(model, x, tmpfile, opset_version=12)  # opset 12 and opset 7 tested
        # do not use dynamic axes will simplify the process
        onnx_tool.model_profile_v2(tmpfile, verbose=False)


def convnext_large():
    model = torchvision.models.convnext_large()
    model.eval()
    x = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        torch_out = torch.onnx._export(model, x, tmpfile, opset_version=12)  # opset 12 and opset 7 tested
        # do not use dynamic axes will simplify the process
        onnx_tool.model_profile_v2(tmpfile, verbose=False)


alexnet()
convnext_large()
