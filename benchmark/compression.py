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
    nodes = pattern.search_pattern(cg)
    for names in nodes:
        cg.fuse_postop_node_names(names, True)
    pattern = FusionPattern(Conv_Res)
    nodes = pattern.search_pattern(cg)
    for names in nodes:
        cg.fuse_postop_node_names(names, True)
    # remove flattern
    pattern = FusionPattern(remove_shapeop)
    nodes = pattern.search_pattern(cg)
    for names in nodes:
        cg.fuse_postop_node_names(names, False)
    cg.graph_reorder_nodes()
    compress_mem = cg.compress_memory()
    serialize_memory_compression(compress_mem, max_shape_key, 'resnet50_fused.cm')
    serialize_graph(cg, 'resnet50_fused.cg')
    cg.save_model('resnet50_fused.onnx')

def bevformer():
    from onnx_tool import NODE_REGISTRY
    from onnx_tool.node import PWNode,Node,_get_shape
    from onnx_tool.tensor import Tensor
    # this is the TensorRT version of BEVFormer
    # It fused some ops as two TRT plugins
    @NODE_REGISTRY.register()
    class RotateTRTNode(PWNode):
        def __init__(self, n):
            super().__init__(n)
            self.op_mac = 4 * 4  # assuming 4x4 transformation matrix

    @NODE_REGISTRY.register()
    class MultiScaleDeformableAttnTRTNode(Node):
        def __init__(self, n):
            super().__init__(n)

        def shape_infer(self, intensors: list[Tensor],outtensors: list[Tensor]):
            s0 = intensors[0].get_shape()
            s3 = intensors[3].get_shape()
            s0[1] = s3[1]
            outtensors[0].update_shape(s0)

        def profile(self, intensors: [], outtensors: []):
            macs = 8
            batch = intensors[0].get_shape()[0]
            num_heads = intensors[0].get_shape()[2]
            channels = intensors[0].get_shape()[3]
            num_levels = intensors[1].get_shape()[0]
            num_query = intensors[3].get_shape()[1]
            num_points = intensors[4].get_shape()[3]
            base_num = batch * num_query * num_heads * channels * num_levels * num_points
            return [base_num * macs,0]

    file = 'data/public/bevformer_tiny.onnx'
    m = onnx.load_model(file)
    g = Graph(m.graph,constant_folding=True, verbose=True)
    g.shape_infer()
    g.profile()
    g.print_node_map()
    g.save_model('bevformer_tiny_shapes.onnx',rawmodel=m)
    compress_mem = g.compress_memory()
    print('compressed memory allocation: ',compress_mem[1])
    cg=g.get_compute_graph()
    cg.graph_reorder_nodes()
    cg.compress_memory()
    cg.save_model('bevformer_tiny_cg.onnx',rawmodel=m)

def gpt2():
    file = 'data/public/gpt2-10.onnx'
    m = onnx.load_model(file)
    g = Graph(m.graph,constant_folding=True,verbose=True)
    shapeengine = g.shape_regress(
        {
            'input1': ['batch', 1, 'seq']
        },
        {
            'batch': (1, 4),
            'seq': (1, 384),
        })
    serialize_shape_engine(shapeengine, 'gpt2.se')  # create shape engine before any fusion
    max_shape_key = {'batch': 4, 'seq': 384}
    max_shape = {'input1': numpy.zeros((max_shape_key['batch'],1, max_shape_key['seq']))}
    g.shape_infer(max_shape)

    cg = g.get_compute_graph()
    compress_mem = cg.compress_memory()
    serialize_memory_compression(compress_mem, max_shape_key, 'gpt2.cm')
    serialize_graph(cg, 'gpt2.cg')
    cg.save_model('gpt2_cg.onnx')

# resnet_compress()
# resnet_fusion_compression()
bevformer()
# gpt2()