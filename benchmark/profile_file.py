import onnx
import os.path
import numpy
from onnx_tool import create_ndarray_f32
import onnx_tool

models=[
        # {
        #     'name': 'data/public/rvm_mobilenetv3_fp32.onnx',
        #     'dynamic_input':
        #         {'src': create_ndarray_f32((1, 3, 1080, 1920)), 'r1i': create_ndarray_f32((1, 16, 135, 240)),
        #                          'r2i':create_ndarray_f32((1,20,68,120)),'r3i':create_ndarray_f32((1,40,34,60)),
        #                          'r4i':create_ndarray_f32((1,64,17,30)),'downsample_ratio':numpy.array((0.25,),dtype=numpy.float32)}
        # },
        {
            'name': 'data/public/resnet18-v1-7.onnx',
            'dynamic_input':
                {
                    'data': numpy.zeros((1, 3, 224, 224), numpy.float32)
                }
        },
        # {
        #     'name': 'data/private/paddle_matting.onnx',
        #     'dynamic_input':
        #         {
        #             'image': create_ndarray_f32((1, 3, 256, 256)),
        #         }
        # },
        # {
        #     'name': 'data/public/bidaf-9.onnx',
        #     'dynamic_input':
        #         {
        #             'context_word':create_ndarray_f32((16,1)),
        #             'context_char':create_ndarray_f32((16,1,1,16)),
        #             'query_word':create_ndarray_f32((16,1)),
        #             'query_char':create_ndarray_f32((16,1,1,16)),
        #
        #         }
        # },
        # {
        #     'name': 'data/public/t5-decoder-with-lm-head-12.onnx',
        #     'dynamic_input':
        #         {
        #             'input_ids': create_ndarray_f32((1, 8)),
        #             'encoder_hidden_states': create_ndarray_f32((1, 8, 768)),
        #
        #         }
        # },
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

def set_inputs(modelname,savemodel,dynamicinps):
    model = onnx.load_model(modelname)
    onnx_tool.graph_set_inputs(model.graph, dynamicinps)
    onnx.save_model(model,savemodel)

def add_outputs(modelname,savemodel,newoutputs):
    model = onnx.load_model(modelname)
    onnx_tool.graph_addoutputs(model.graph, newoutputs)
    onnx.save_model(model,savemodel)

for modelinfo in models:
    # onnx_tool.model_profile(modelinfo['name'],modelinfo['dynamic_input'],saveshapesmodel='tmp.onnx',shapesonly=True)
    # set_inputs(modelinfo['name'],'inputs_set.onnx',modelinfo['dynamic_input'])
    # add_outputs(modelinfo['name'],'outputs_set.onnx',['443','586'])
    onnx_tool.model_profile(modelinfo['name'],modelinfo['dynamic_input'],saveshapesmodel='tmp.onnx',shapesonly=True,dump_outputs=['443','586'])
    # onnx_tool.model_export_tensors_numpy(modelinfo['name'],tensornames=['830'],savefolder='rvm',fp16=True)
    # print(onnx_tool.GLOBAL_VARS['tensor_map'].keys())