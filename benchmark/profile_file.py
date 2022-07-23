from onnx_tool import infer_shapes,graph_profile,print_node_map,create_ndarray_f32,NODEPROFILER_REGISTRY,create_ndarray_int64,model_profile
import onnx
import os.path
import numpy

models=[
        {
            'name': 'data/public/t5-decoder-with-lm-head-12.onnx',
            'dynamic_input':
                {
                    'input_ids': create_ndarray_f32((1, 8)),
                    'encoder_hidden_states': create_ndarray_f32((1, 8, 768)),

                }
        },
    ]

# for modelinfo in models:
#     print('-'*64)
#     print(modelinfo['name'])
#     model = onnx.load_model(modelinfo['name'])
#     basen=os.path.basename(modelinfo['name'])
#     name=os.path.splitext(basen)[0]
#     macs, params, node_map = graph_profile(model.graph, modelinfo['dynamic_input'])
#     print(int(macs / 1e6), params / 1e6)
#     print_node_map(node_map)
#     onnx.save_model(model,'tmp.onnx')
#     print('-'*64)

for modelinfo in models:
    model_profile(modelinfo['name'],modelinfo['dynamic_input'],saveshapesmodel='tmp.onnx',shapesonly=True)