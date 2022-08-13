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

def set_inputs(modelname,savemodel,dynamicinps):
    model = onnx.load_model(modelname)
    onnx_tool.graph_set_inputs(model.graph, dynamicinps)
    onnx.save_model(model,savemodel)

def add_outputs(modelname,savemodel,newoutputs):
    model = onnx.load_model(modelname)
    onnx_tool.graph_addoutputs(model.graph, newoutputs)
    onnx.save_model(model,savemodel)

for modelinfo in models:
    onnx_tool.model_profile(modelinfo['name'],modelinfo['dynamic_input'],saveshapesmodel='tmp.onnx',shapesonly=True)
    # set_inputs(modelinfo['name'],'inputs_set.onnx',modelinfo['dynamic_input'])
    # add_outputs(modelinfo['name'],'outputs_set.onnx',['443','586'])
    # onnx_tool.model_profile(modelinfo['name'],modelinfo['dynamic_input'] \
    #                         ,saveshapesmodel='tmp.onnx',shapesonly=True,dump_outputs=['443','586'])
    # onnx_tool.model_export_tensors_numpy(modelinfo['name'],tensornames=['830'],savefolder='rvm',fp16=True)
    # print(onnx_tool.GLOBAL_VARS['tensor_map'].keys())
    # onnx_tool.model_simplify_names(modelinfo['name'],savemodel='sim.onnx',renametensor=True,renamelayer=True
    #                                ,custom_inputs={'input':'BatchxChannelxHeightxWidth'},custom_outputs={'output':'BatchxNClass'})