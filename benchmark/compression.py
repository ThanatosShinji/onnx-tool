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

def bevformer():
    from onnx_tool import NODE_REGISTRY
    from onnx_tool.node import PWNode,Node,_get_shape
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

        def shape_infer(self, intensors: []):
            s0 = _get_shape(intensors[0])
            s3 = _get_shape(intensors[3])
            s0[1] = s3[1]
            return [s0]

        def profile(self, intensors: [], outtensors: []):
            macs = 8
            batch = _get_shape(intensors[0])[0]
            num_heads = _get_shape(intensors[0])[2]
            channels = _get_shape(intensors[0])[3]
            num_levels = _get_shape(intensors[1])[0]
            num_query = _get_shape(intensors[3])[1]
            num_points = _get_shape(intensors[4])[3]
            base_num = batch * num_query * num_heads * channels * num_levels * num_points
            return base_num * macs

    file = 'data/public/bevformer_tiny.onnx'
    m = onnx.load_model(file)
    g = Graph(m.graph,verbose=True)
    g.shape_infer()
    g.profile()
    g.print_node_map()
    g.save_model('bevformer_tiny_shapes.onnx',rawmodel=m)
    compress_mem = g.compress_memory()
    print('compressed memory allocation: ',compress_mem[1])
    cg=g.get_compute_graph()
    cg.save_model('bevformer_tiny_cg.onnx',rawmodel=m)

# resnet_compress()
# resnet_fusion_compression()
bevformer()
