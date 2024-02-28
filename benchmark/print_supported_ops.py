import onnx_tool

print('Total OPs: ',len(onnx_tool.NODE_REGISTRY.keys()))
for key in onnx_tool.NODE_REGISTRY.keys():
    print(key[:-4])
