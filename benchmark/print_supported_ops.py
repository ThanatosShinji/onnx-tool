import onnx_tool

print(len(onnx_tool.NODE_REGISTRY.keys()))
for key in onnx_tool.NODE_REGISTRY.keys():
    print(key)
