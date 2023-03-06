from onnx_tool import Graph
from onnx_tool.fusion import *
import onnx

ResBlock = [
    {
        'name': 'conv_0',
        'op': 'Conv',
        'attrs': [
        ],
        'inport': [],
        'outport': [[0, 'bn_0', 0]],
    },
    {
        'name': 'bn_0',
        'op': 'BatchNormalization',
        'attrs': [
        ],
        'inport': [[0, 'conv_0', 0]],
        'outport': [[0, 'relu_0', 0]],
    },
    {
        'name': 'relu_0',
        'op': 'Relu',
        'attrs': [
        ],
        'inport': [[0, 'bn_0', 0]],
        'outport': [[0, 'conv_1', 0]],
    },
    {
        'name': 'conv_1',
        'op': 'Conv',
        'attrs': [
        ],
        'inport': [[0, 'relu_0', 0]],
        'outport': [[0, 'bn_1', 0]],
    },
    {
        'name': 'bn_1',
        'op': 'BatchNormalization',
        'attrs': [
        ],
        'inport': [[0, 'conv_1', 0]],
        'outport': [[0, 'add_0', 1]],
    },
    {
        'name': 'add_0',
        'op': 'Add',
        'attrs': [
        ],
        'inport': [[1, 'bn_1', 0]],
        'outport': [[0, 'relu_1', 0]],
    },
    {
        'name': 'relu_1',
        'op': 'Relu',
        'attrs': [
        ],
        'inport': [[0, 'add_0', 0]],
        'outport': [],
    },
]


def test():
    pattern = FusionPattern(ResBlock)
    file = 'data/public/resnet18-v1-7.onnx'
    m = onnx.load_model(file)
    g = Graph(m.graph)
    cg = g.get_compute_graph()
    found_nodes = pattern.find_pattern(cg)
    for nodes in found_nodes:
        cg.fuse_subgraph_node_names(nodes, 'Conv', nodes[0], True)
    cg.save_model('convbn.onnx')



def MHA_test():
    pattern = FusionPattern(MHAint8_Pattern)
    pattern1 = FusionPattern(layernorm_pattern)
    file = 'data/public/BERT_quan95.onnx'
    m = onnx.load_model(file)
    g = Graph(m.graph)
    cg = g.get_compute_graph()

    found_nodes = pattern.find_pattern(cg)
    for nodes in found_nodes:
        cg.fuse_subgraph_node_names(nodes, 'MHA', nodes[0], True)
    found_nodes = pattern1.find_pattern(cg)
    for nodes in found_nodes:
        cg.fuse_subgraph_node_names(nodes, 'Layernorm', nodes[0], True)
    cg.graph_reorder()
    cg.save_model('MHA_Layernorm.onnx')


remove_flattern = [
    {
        'name': 'any',
        'op': 'Any',
        'attrs': [],
        'inport': [],
        'outport': [[0, 'fla', 0]],
    },
    {
        'name': 'fla',
        'op': 'Flatten',
        'attrs': [],
        'inport': [[0, 'any', 0]],
        'outport': [],
    }
]


def resnet_fusion():
    file = 'data/public/resnet18-v1-7.onnx'
    m = onnx.load_model(file)
    g = Graph(m.graph)
    cg = g.get_compute_graph()
    ConvBNFusion(cg)
    pattern = FusionPattern(Conv_Act)
    nodes = pattern.find_pattern(cg)
    for names in nodes:
        cg.fuse_postop_node_names(names, True)
    pattern = FusionPattern(Conv_Res)
    nodes = pattern.find_pattern(cg)
    for names in nodes:
        cg.fuse_postop_node_names(names, True)

    cg.graph_reorder()
    shapeengine = cg.shape_regress(
        {
            'data': [1, 3, 'h', 'w']
        },
        {
            'h': (224, 299),
            'w': (224, 299),
        })
    from onnx_tool.serialization import serialize_shape_engine, serialize_graph
    serialize_shape_engine(shapeengine, 'resnet_fused.se')

    # remove flattern
    pattern = FusionPattern(remove_flattern)
    nodes = pattern.find_pattern(cg)
    for names in nodes:
        cg.fuse_postop_node_names(names, False)
    cg.graph_reorder()
    serialize_graph(cg, 'resnet_fused.cg')
    cg.save_model('convbn_merged.onnx')


# test()
# MHA_test()
resnet_fusion()
