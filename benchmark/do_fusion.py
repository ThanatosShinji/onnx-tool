import numpy

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


def resnet_fusion():
    file = 'data/public/resnet18-v1-7.onnx'
    file = 'data/public/resnet50-v2-7.onnx'
    m = onnx.load_model(file)
    g = Graph(m.graph)
    cg = g.get_compute_graph()
    from onnx_tool.serialization import serialize_shape_engine, serialize_graph
    shapeengine = cg.shape_regress(
        {
            'data': [1, 3, 'h', 'w']
        },
        {
            'h': (224, 299),
            'w': (224, 299),
        })
    serialize_shape_engine(shapeengine, 'resnet50_fused.se')  # create shape engine before any fusion

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
    serialize_graph(cg, 'resnet50_fused.cg')
    cg.save_model('resnet50_fused.onnx')


def asr_fusion():
    file = 'data/private/asr_500G_.onnx'
    m = onnx.load_model(file)
    g = Graph(m.graph)
    g.remove_constant()
    g.update_input_by_map({'input': numpy.zeros((1, 3, 256, 256), dtype=numpy.float32)})

    from onnx_tool.serialization import serialize_shape_engine, serialize_graph
    shapeengine = g.shape_regress(
        {
            'input': [1, 3, 'h', 'w']
        },
        {
            'h': (224, 640),
            'w': (224, 320),
        })
    serialize_shape_engine(shapeengine, 'asr_500G.se')  # create shape engine before any fusion

    cg = g.get_compute_graph()
    Fused_leaky = [
        {
            'name': 'any',
            'op': 'Any',
            'attrs': [],
            'inport': [],
            'outport': [[0, 'act_0', 0]],
        },
        {
            'name': 'act_0',
            'op': ['Relu', 'LeakyRelu', 'Add', 'Clip'],
            'attrs': [
            ],
            'inport': [[0, 'any', 0]],
            'outport': [],
        },
    ]
    pattern = FusionPattern(Fused_leaky, inplace_fusion=True)
    nodes = pattern.find_pattern(cg)
    for names in nodes:
        cg.fuse_postop_node_names(names, True)

    SliceSlice = [
        {
            'name': 'any',
            'op': 'Slice',
            'attrs': [],
            'inport': [],
            'outport': [[0, 'n_0', 0]],
        },
        {
            'name': 'n_0',
            'op': ['Slice', ],
            'attrs': [
            ],
            'inport': [[0, 'any', 0]],
            'outport': [],
        },
    ]
    pattern = FusionPattern(SliceSlice, inplace_fusion=True)
    nodes = pattern.find_pattern(cg)
    for names in nodes:
        cg.fuse_subgraph_node_names(names, 'Slice2D', names[0])

    # remove padding zeros
    rmlist = []
    for key in cg.nodemap:
        node = cg.nodemap[key]
        if node.op_type == 'Pad':
            if len(node.input) == 2:
                pads = cg.tensormap[node.input[1]].numpy
                zeroflag = pads == 0
                if numpy.sum(zeroflag) == pads.size:
                    rmlist.append(key)
    for key in rmlist:
        cg.skip_node(key)

    cg.graph_reorder()
    serialize_graph(cg, 'asr_500G.cg')
    cg.save_model('asr_500G_merged.onnx')


# test()
# MHA_test()
# resnet_fusion()
asr_fusion()
