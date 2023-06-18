import onnx
import onnx_tool
import numpy
from onnx_tool.fusion import layernorm_pattern, FusionPattern, createSerialPattern,removeShapeOps, create_descs_from_nodenames
from onnx_tool.serialization import serialize_shape_engine, serialize_graph,serialize_memory_compression



RangeGatherOps=['ConstantOfShape','NonZero','Transpose','Squeeze','Mul','Add','Cast','Unsqueeze','Reshape','Gather']

ShapeOps = ['Any',['Flatten','Reshape']]


MHAPattern = [
    {
        'name': 'Split_60',
        'op': 'Split',
        'attrs': [],
        'inport': [],
        'outport': [[1,'Transpose_118',0],[2,'Transpose_117',0],[1,'Transpose_98',0],[0,'Transpose_79',0],],
    },
    {
        'name': 'Transpose_118',
        'op': 'Transpose',
        'attrs': [],
        'inport': [[0,'Split_60',1]],
        'outport': [[0,'Concat_121',0]],
    },
    {
        'name': 'Transpose_117',
        'op': 'Transpose',
        'attrs': [],
        'inport': [[0,'Split_60',2]],
        'outport': [[0,'Concat_121',1]],
    },
    {
        'name': 'Transpose_98',
        'op': 'Transpose',
        'attrs': [],
        'inport': [[0,'Split_60',1]],
        'outport': [[0,'MatMul_122',1]],
    },
    {
        'name': 'Transpose_79',
        'op': 'Transpose',
        'attrs': [],
        'inport': [[0,'Split_60',0]],
        'outport': [[0,'MatMul_122',0]],
    },
    {
        'name': 'Concat_121',
        'op': 'Concat',
        'attrs': [],
        'inport': [[0,'Transpose_118',0],[1,'Transpose_117',0]],
        'outport': [],
    },
    {
        'name': 'MatMul_122',
        'op': 'MatMul',
        'attrs': [],
        'inport': [[0,'Transpose_79',0],[1,'Transpose_98',0]],
        'outport': [[0,'Div_124',0]],
    },
    {
        'name': 'Div_124',
        'op': 'Div',
        'attrs': [],
        'inport': [[0,'MatMul_122',0]],
        'outport': [[0,'Mul_139',0]],
    },
    {
        'name': 'Mul_139',
        'op': 'Mul',
        'attrs': [],
        'inport': [[0,'Div_124',0],[1,'Slice_138',0]],
        'outport': [[0,'Sub_144',0]],
    },
    {
        'name': 'Slice_135',
        'op': 'Slice',
        'attrs': [],
        'inport': [],
        'outport': [[0,'Slice_138',0]],
    },
    {
        'name': 'Slice_138',
        'op': 'Slice',
        'attrs': [],
        'inport': [[0,'Slice_135',0]],
        'outport': [[0,'Mul_139',1],[0,'Sub_141',1]],
    },
    {
        'name': 'Sub_141',
        'op': 'Sub',
        'attrs': [],
        'inport': [[1,'Slice_138',0]],
        'outport': [[0,'Mul_143',0]],
    },
    {
        'name': 'Mul_143',
        'op': 'Mul',
        'attrs': [],
        'inport': [[0,'Sub_141',0]],
        'outport': [[0,'Sub_144',1]],
    },
    {
        'name': 'Sub_144',
        'op': 'Sub',
        'attrs': [],
        'inport': [[0,'Mul_139',0],[1,'Mul_143',0]],
        'outport': [[0,'Softmax_145',0]],
    },
    {
        'name': 'Softmax_145',
        'op': 'Softmax',
        'attrs': [],
        'inport': [[0,'Sub_144',0]],
        'outport': [[0,'MatMul_146',0]],
    },
    {
        'name': 'MatMul_146',
        'op': 'MatMul',
        'attrs': [],
        'inport': [[0,'Softmax_145',0],[1,'Transpose_117',0]],
        'outport': [[0,'Transpose_147',0]],
    },
    {
        'name': 'Transpose_147',
        'op': 'Transpose',
        'attrs': [],
        'inport': [[0,'MatMul_146',0]],
        'outport': [],
    },
]

MHANodeNames=['Split_60','Transpose_147','MatMul_146','Softmax_145','Sub_144','Mul_143','Sub_141','Slice_138','Slice_135','Mul_139','Div_124','MatMul_122', 'Concat_121','Transpose_79','Transpose_98','Transpose_117','Transpose_118']

GeluNodes = ['Mul_213','Pow_215','Mul_217','Add_218','Mul_220','Tanh_221','Add_223','Mul_224']

def gpt2():
    file = 'data/public/gpt2-10.onnx'
    m = onnx.load_model(file)
    g = onnx_tool.Graph(m.graph, constant_folding=True, verbose=True)
    shapeengine = g.shape_regress(
        {
            'input1': [1, 1, 'seq']
        },
        {
            'seq': (1, 384),
        })
    serialize_shape_engine(shapeengine, 'gpt2.se')  # create shape engine before any fusion
    max_shape_key = {'batch': 4, 'seq': 384}
    max_shape = {'input1': numpy.zeros((max_shape_key['batch'], 1, max_shape_key['seq']))}
    g.shape_infer(max_shape)#update tensor shape with the max_shape

    #fusion without shape ops.
    pattern = FusionPattern(layernorm_pattern, inplace_fusion=False)
    nodes = pattern.search_pattern(g)
    for names in nodes:
        g.fuse_subgraph_node_names(names, 'Layernrom', names[0])
    RangeGatherPattern = createSerialPattern(RangeGatherOps)
    nodes = RangeGatherPattern.search_pattern(g)
    for names in nodes:
        g.fuse_subgraph_node_names(names, 'RangeGather', names[0])

    #fusion like MHA contains a lot of shape ops, remove these ops will simplify the fusion pattern
    cg = g.get_compute_graph()
    cg = removeShapeOps(cg)

    pattern = FusionPattern(MHAPattern)#define your pattern
    nodes = pattern.search_pattern(cg)
    for names in nodes:
        cg.fuse_subgraph_node_names(names, 'MHA', names[0])

    GeluDescs = create_descs_from_nodenames(cg, GeluNodes)#create a pattern from current graph
    pattern = FusionPattern(GeluDescs)
    nodes = pattern.search_pattern(cg)
    for names in nodes:
        cg.fuse_subgraph_node_names(names, 'Gelu', names[0])

    MadDescs =  create_descs_from_nodenames(cg, ['Mul_41','Add_42'])
    pattern = FusionPattern(MadDescs)
    nodes = pattern.search_pattern(cg)
    for names in nodes:
        cg.fuse_subgraph_node_names(names, 'Mad', names[0])

    cg.graph_reorder_nodes()#reorder to make sure the execution sequence is right
    compress_mem = cg.compress_memory()
    serialize_memory_compression(compress_mem, max_shape_key, 'gpt2.cm')

    serialize_graph(cg, 'gpt2.cg')
    cg.save_model('gpt2-fused-cg.onnx', rawmodel=m)

gpt2()