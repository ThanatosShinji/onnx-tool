import numpy
import onnx
from onnx_tool import Graph
from onnx_tool.fusion import FusionPattern
from onnx_tool.fusion import ConvBNFusion, Fused_Element, Conv_Res
from onnx_tool.serialization import *


def resnet_compress():
    file = 'data/public/resnet50-v2-7.onnx'
    m = onnx.load_model(file)
    g = Graph(m.graph)
    max_input = {'data': numpy.zeros((1, 3, 224, 224), dtype=numpy.float32)}
    g.shape_infer(max_input)
    g.compress_memory()


def resnet_fusion_compression():
    remove_shapeop = [
        {
            'name': 'any',
            'op': 'Any',
            'attrs': [],
            'inport': [],
            'outport': [[0, 'fla', 0]],
        },
        {
            'name': 'fla',
            'op': ['Flatten', 'Reshape'],
            'attrs': [],
            'inport': [[0, 'any', 0]],
            'outport': [],
        }
    ]

    file = 'data/public/resnet50-v2-7.onnx'
    m = onnx.load_model(file)
    g = Graph(m.graph)

    shapeengine = g.shape_regress(
        {
            'data': [1, 3, 'h', 'w']
        },
        {
            'h': (224, 299),
            'w': (224, 299),
        })
    serialize_shape_engine(shapeengine, 'resnet50_fused.se')  # create shape engine before any fusion
    max_shape_key = {'h': 224, 'w': 224}
    max_shape = {'data': numpy.zeros((1, 3, max_shape_key['h'], max_shape_key['w']))}
    g.shape_infer(max_shape)

    cg = g.get_compute_graph()
    ConvBNFusion(cg)
    pattern = FusionPattern(Fused_Element)
    nodes = pattern.find_pattern(cg)
    for names in nodes:
        cg.fuse_postop_node_names(names, True)
    pattern = FusionPattern(Conv_Res)
    nodes = pattern.find_pattern(cg)
    for names in nodes:
        cg.fuse_postop_node_names(names, True)
    # remove flattern
    pattern = FusionPattern(remove_shapeop)
    nodes = pattern.find_pattern(cg)
    for names in nodes:
        cg.fuse_postop_node_names(names, False)
    cg.graph_reorder()
    compress_mem = cg.compress_memory()
    serialize_memory_compression(compress_mem, max_shape_key, 'resnet50_fused.cm')
    serialize_graph(cg, 'resnet50_fused.cg')
    cg.save_model('resnet50_fused.onnx')


# resnet_compress()
resnet_fusion_compression()
