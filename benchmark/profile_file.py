import onnx
import numpy
import onnx_tool
from onnx_tool import create_ndarray_f32, create_ndarray_int64

models = [
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
    # {
    #     'name': 'data/public/EdgeNeXt-small.onnx',
    #     'dynamic_input':
    #         {
    #             'image': create_ndarray_f32((1, 3, 224, 224)),
    #         }
    # },
    # {
    #     'name': 'data/public/text_encoder.onnx',
    #     'dynamic_input': None,
    # },
    # {
    #     'name': 'data/public/so-vits-svc.onnx',
    #     'dynamic_input': {
    #         'c': create_ndarray_f32((1, 10, 768)),
    #         'f0': create_ndarray_f32((1, 10)),
    #         'mel2ph': create_ndarray_int64((1, 10)),
    #         'uv': create_ndarray_f32((1, 10)),
    #         'noise': create_ndarray_f32((1, 192, 10)),
    #         'sid': create_ndarray_int64(1),
    #     }
    # },
    # {
    #     'name': 'data/public/Inceptionv3_rerodered.onnx',
    #     'dynamic_input':
    #         {
    #             'image': numpy.zeros((1, 3, 299, 299), numpy.float32)
    #         }
    # },
    # {
    #     'name': 'm2_subgraph_static_qdq_quant.onnx',
    #     'dynamic_input':None
    # },
    # {
    #     'name': 'data/public/SwiftFormer-S.onnx',
    #     'dynamic_input':{
    #         'input':numpy.zeros((1,3,224,224),numpy.float32)
    #     }
    # },
    {
        'name': 'data/public/model_custom_vocabulary.onnx',
        'dynamic_input': None,
        'mcfg':{
            'constant_folding':True,
            'verbose':True,
            'if_fixed_branch':'else',
            'fixed_topk':1000,
        }
    }
    # {
    #     'name': 'data/public/resnet50-v2-7.onnx',
    #     'dynamic_input':
    #         {
    #             'data': numpy.zeros((1, 3, 224, 224), numpy.float32)
    #         }
    # },
    # {
    #     'name': 'data/public/unet/unet.onnx',
    #     'dynamic_input': None
    # },
]

for modelinfo in models:
    from pathlib import Path
    onnx_tool.model_profile(Path(modelinfo['name']), modelinfo['dynamic_input'], save_model='shape.onnx',
                            mcfg=modelinfo['mcfg'], shape_only=False)
    # onnx_tool.model_constant_folding(Path(modelinfo['name']),'folded.onnx')
    # onnx_tool.model_profile('model_custom_vocabulary.onnx', mcfg={'if_fixed_branch': 'else', 'fixed_topk': 200},
    #                         save_model='detic.onnx')
    # onnx_tool.model_reorder_nodes('yolox_s_lite_640x640_20220221_model.onnx','reordered.onnx')
    # onnx_tool.model_constant_folding('yolox_s_lite_640x640_20220221_model.onnx','folded.onnx')
    # shape_engie, compute_graph = onnx_tool.model_shape_regress(modelinfo['name'], modelinfo['input_desc'],
    #                                                            modelinfo['input_range'])
    # onnx_tool.serialize_graph(compute_graph, 'resnet18.cg')
    # onnx_tool.serialize_shape_engine(shape_engie, 'resnet18.se')
    # onnx_tool.model_profile(modelinfo['name'], modelinfo['dynamic_input'], None, 'tmp.onnx', shapesonly=False)
    # onnx_tool.model_io_modify(modelinfo['name'], 'newio.onnx', {"input": "1x3x111x11","output": '1x3x444x44'})
    # onnx_tool.model_subgraph('data/public/resnet50-v2-7.onnx', ['data'], ['resnetv24_stage3__plus0'])
    # onnx_tool.model_opfusion(modelinfo['name'],'fused','fused_0','fused.onnx', ['StatefulPartitionedCall/model/conv2d_101/BiasAdd:0'], ['Identity_1:0'])
