import onnx
import onnx_tool
import numpy
from onnx_tool.fusion import layernorm_pattern, FusionPattern, createSerialPattern,removeShapeOps
from onnx_tool.serialization import serialize_shape_engine, serialize_graph,serialize_memory_compression



RangeGatherOps=['ConstantOfShape','NonZero','Transpose','Squeeze','Mul','Add','Cast','Unsqueeze','Reshape','Gather']

ShapeOps = ['Any',['Flatten','Reshape']]

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

    g.shape_infer(max_shape)#ready for memory compression
    pattern = FusionPattern(layernorm_pattern, inplace_fusion=False)
    nodes = pattern.find_pattern(g)
    for names in nodes:
        g.fuse_subgraph_node_names(names, 'Layernrom', names[0])
    RangeGatherPattern = createSerialPattern(RangeGatherOps)
    nodes = RangeGatherPattern.find_pattern(g)
    for names in nodes:
        g.fuse_subgraph_node_names(names, 'RangeGather', names[0])

    g = removeShapeOps(g)
    g.graph_reorder()

    cg = g.get_compute_graph()
    compress_mem = cg.compress_memory()
    serialize_memory_compression(compress_mem, max_shape_key, 'gpt2.cm')



    serialize_graph(cg, 'gpt2.cg')
    cg.save_model('gpt2-fused-cg.onnx', rawmodel=m)

gpt2()