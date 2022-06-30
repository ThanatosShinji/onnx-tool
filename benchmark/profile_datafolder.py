from data.public.config import public_models
# from data.private.config import private_models
from onnx_tool.node_profilers import graph_profile,print_node_map
import onnx
import os.path

folder=public_models['folder']
for modelinfo in public_models['models']:
    print('-'*64)
    print(modelinfo['name'])
    model = onnx.load_model(os.path.join(folder, modelinfo['name']))
    basen=os.path.basename(modelinfo['name'])
    name=os.path.splitext(basen)[0]
    macs, params, node_map = graph_profile(model.graph, modelinfo['dynamic_input'])
    print(macs / 1e6, params / 1e6)
    onnx.save_model(model, os.path.join(folder,name+'_shapes.onnx'))
    print_node_map(node_map,os.path.join(folder,name+'_info.log'))
    print('-'*64)

# folder=private_models['folder']
# for modelinfo in private_models['models']:
#     print('-'*64)
#     print(modelinfo['name'])
#     model = onnx.load_model(os.path.join(folder, modelinfo['name']))
#     basen=os.path.basename(modelinfo['name'])
#     name=os.path.splitext(basen)[0]
#     macs, params, node_map = graph_profile(model.graph, modelinfo['dynamic_input'])
#     print(macs / 1e6, params / 1e6)
#     onnx.save_model(model, os.path.join(folder,name+'_shapes.onnx'))
#     print_node_map(node_map,os.path.join(folder,name+'_info.log'))
#     print('-'*64)

