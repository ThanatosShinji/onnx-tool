from onnx_tool import Graph
from onnx_tool.fusion import *
import onnx


def test():
    pattern = FusionPattern(ResBlock)
    file = 'data/public/resnet18-v1-7.onnx'
    m = onnx.load_model(file)
    g = Graph(m.graph)
    cg = g.get_compute_graph()
    cg = Graph(cg)
    found_nodes = pattern.find_pattern(cg)
    newcg = cg
    for nodes in found_nodes:
        newcg = newcg.fuse_subgraph_node_names(nodes, 'Conv', nodes[0], True)
    newcg.save_model('convbn.onnx')


MHAint8_Pattern = [
    {
        'name': 'MatMul_0',
        'op': 'QLinearMatMul',
        'attrs': [
        ],
        'inport': [],
        'outport': [[0, 'DequantizeLinea_0', 0]],
    },
    {
        'name': 'DequantizeLinea_0',
        'op': 'DequantizeLinear',
        'attrs': [
        ],
        'inport': [[0, 'MatMul_0', 0]],
        'outport': [[0, 'div_0', 0]],
    },
    {
        'name': 'div_0',
        'op': 'Div',
        'attrs': [
        ],
        'inport': [[0, 'DequantizeLinea_0', 0]],
        'outport': [[0, 'Add_0', 0]],
    },
    {
        'name': 'Add_0',
        'op': 'Add',
        'attrs': [
        ],
        'inport': [[0, 'div_0', 0]],
        'outport': [[0, 'Softmax_0', 0]],
    },
    {
        'name': 'Softmax_0',
        'op': 'Softmax',
        'attrs': [
        ],
        'inport': [[0, 'Add_0', 0]],
        'outport': [[0, 'QuantizeLinear_1', 0]],
    },
    {
        'name': 'QuantizeLinear_1',
        'op': 'QuantizeLinear',
        'attrs': [
        ],
        'inport': [[0, 'Add_0', 0]],
        'outport': [[0, 'MatMul_1', 0]],
    },
    {
        'name': 'MatMul_1',
        'op': 'QLinearMatMul',
        'attrs': [
        ],
        'inport': [[0, 'QuantizeLinear_1', 0]],
        'outport': [],
    },
]

layernorm_pattern = [
    {
        'name': 'ReduceMean_196',
        'op': 'ReduceMean',
        'attrs': [],
        'inport': [],
        'outport': [[0, 'Sub_197', 1]]
    },
    {
        'name': 'Sub_197',
        'op': 'Sub',
        'attrs': [],
        'inport': [[1, 'ReduceMean_196', 0]],
        'outport': [[0, 'Pow_0', 0]]
    },
    {
        'name': 'Pow_0',
        'op': 'Pow',
        'attrs': [],
        'inport': [[0, 'Sub_197', 0]],
        'outport': [[0, 'ReduceMean_0', 0]]
    },
    {
        'name': 'ReduceMean_0',
        'op': 'ReduceMean',
        'attrs': [],
        'inport': [[0, 'Pow_0', 0]],
        'outport': [[0, 'Add_0', 0]]
    },
    {
        'name': 'Add_0',
        'op': 'Add',
        'attrs': [],
        'inport': [[0, 'ReduceMean_0', 0]],
        'outport': [[0, 'Sqrt_0', 0]]
    },
    {
        'name': 'Sqrt_0',
        'op': 'Sqrt',
        'attrs': [],
        'inport': [[0, 'Add_0', 0]],
        'outport': [[0, 'Div_0', 1]]
    },
    {
        'name': 'Div_0',
        'op': 'Div',
        'attrs': [],
        'inport': [[1, 'Sqrt_0', 0]],
        'outport': []
    },
]


def MHA_test():
    pattern = FusionPattern(MHAint8_Pattern)
    pattern1 = FusionPattern(layernorm_pattern)
    file = 'data/public/BERT_quan95.onnx'
    m = onnx.load_model(file)
    g = Graph(m.graph)
    cg = g.get_compute_graph()
    cg = Graph(cg)
    newcg = cg

    found_nodes = pattern.find_pattern(cg)
    for nodes in found_nodes:
        newcg = newcg.fuse_subgraph_node_names(nodes, 'MHA', nodes[0], True)
    found_nodes = pattern1.find_pattern(newcg)
    for nodes in found_nodes:
        newcg = newcg.fuse_subgraph_node_names(nodes, 'Layernorm', nodes[0], True)
    newcg.save_model('MHA_Layernorm.onnx')


MHA_test()
