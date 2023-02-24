import onnx
import numpy
import onnx_tool
from onnx_tool import create_ndarray_f32, create_ndarray_int64

models = [
    # {
    #     'name': 'data/public/rvm_mobilenetv3_fp32.onnx',
    #     'dynamic_input':
    #         {'src': create_ndarray_f32((1, 3, 1080, 1920)), 'r1i': create_ndarray_f32((1, 16, 135, 240)),
    #                          'r2i':create_ndarray_f32((1,20,68,120)),'r3i':create_ndarray_f32((1,40,34,60)),
    #                          'r4i':create_ndarray_f32((1,64,17,30)),'downsample_ratio':numpy.array((0.25,),dtype=numpy.float32)}
    # },
    # {
    #     'name': 'data/public/gpt2-10.onnx',
    #     'dynamic_input':
    #         {
    #             'input1': create_ndarray_int64((1, 1, 8)),
    #         }
    # },

    # {
    #     'name': 'data/public/bidaf-9.onnx',
    #     'dynamic_input':
    #         {
    #             'context_word': create_ndarray_f32((16, 1)),
    #             'context_char': create_ndarray_f32((16, 1, 1, 16)),
    #             'query_word': create_ndarray_f32((16, 1)),
    #             'query_char': create_ndarray_f32((16, 1, 1, 16)),
    #
    #         }
    # },
    # {
    #     'name': 'data/public/vae_encoder.onnx',
    #     'dynamic_input': None
    # },
    # {
    #     'name': 'data/public/Inceptionv3.onnx',
    #     'dynamic_input':
    #         {
    #             'image': numpy.zeros((1, 3, 299, 299), numpy.float32)
    #         }
    # },
    # {
    #     'name': 'data/public/gpt2-10.onnx',
    #     'dynamic_input':
    #         {
    #             'input1': create_ndarray_int64((1, 1, 8)),
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
    # {
    #     'name': 'data/public/MobileNetV1_Pruned_Quantized.onnx',
    #     'dynamic_input': None
    # },
    {
        'name': 'data/public/resnet18-v1-7.onnx',
        'dynamic_input':
            {
                'data': create_ndarray_f32((1, 3, 224, 224)),
            },
        'input_desc':
            {
                'data': [1, 3, 'h', 'w']
            },
        'input_range':
            {
                'h': (224, 299),
                'w': (224, 299),
            }
    },
    # {
    #     'name': 'data/public/gpt2-10.onnx',
    #     'dynamic_input':
    #         {
    #             'input1': create_ndarray_int64((1, 1, 8)),
    #         },
    #     'input_desc':
    #         {
    #             'input1': [1,1,'seq']
    #         },
    #     'input_range':
    #         {
    #             'seq': (8,64),
    #         }
    # },
    # {
    #     'name': 'data/public/BERT_quan95.onnx',
    #     'dynamic_input': None,
    #     'input_desc':
    #         {
    #             'input_ids': ('batch', 'seq'),
    #             'attention_mask': ('batch', 'seq'),
    #             'token_type_ids': ('batch', 'seq'),
    #         },
    #     'input_range':
    #         {
    #             'batch': (1, 4),
    #             'seq': (16, 384)
    #         }
    # },

]


for modelinfo in models:
    # onnx_tool.model_simplify_names(modelinfo['name'],'mobilenetv1_quanpruned_sim.onnx',node_reorder=True)
    onnx_tool.model_profile(modelinfo['name'], modelinfo['dynamic_input'], saveshapesmodel='debug.onnx',
                            dump_outputs=['resnetv15_conv0_fwd'])
    # onnx_tool.model_shape_regress(modelinfo['name'], modelinfo['input_desc'], modelinfo['input_range'])
    # onnx_tool.model_profile(modelinfo['name'], modelinfo['dynamic_input'], None, 'tmp.onnx', shapesonly=False)
    # onnx_tool.model_io_modify(modelinfo['name'], 'newio.onnx', {"input": "1x3x111x11","output": '1x3x444x44'})
    # onnx_tool.model_subgraph('tmp.onnx', ['sequential/mobilenetv2_1.00_160/Conv1/Conv2D__7426:0'], ['dense'])
    # onnx_tool.model_opfusion(modelinfo['name'],'fused','fused_0','fused.onnx', ['StatefulPartitionedCall/model/conv2d_101/BiasAdd:0'], ['Identity_1:0'])
