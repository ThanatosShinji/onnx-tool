from onnx_tool.node_profilers import NODEPROFILER_REGISTRY

print(len(NODEPROFILER_REGISTRY.keys()))
for key in NODEPROFILER_REGISTRY.keys():
    print(key)