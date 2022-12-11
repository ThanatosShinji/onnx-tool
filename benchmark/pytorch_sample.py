import onnx
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


def Issue11():
    from torch.nn import Module
    import torch

    class Dummy(Module):
        def __init__(self):
            super(Dummy, self).__init__()

        def forward(self, x):
            return torch.unsqueeze(torch.sum(x, dim=1), 1)

    model = Dummy()
    x = torch.zeros((32, 10))
    torch.onnx.export(model, x, "model.onnx")
    m = onnx.load_model('model.onnx')
    graph = onnx_tool.Graph(m.graph)
    graph.shape_infer()
    graph.save_model('shapes.onnx')

alexnet()
convnext_large()
# Issue11()
