from typing import List

import numpy

from onnx_tool.tensor import Tensor


def parse_and_edit():
    import onnx_tool
    modelpath = 'data/public/resnet50-v1-7.onnx'
    m = onnx_tool.Model(modelpath)
    g = m.graph
    g.nodemap['resnetv17_batchnorm0_fwd'].set_attr('epsilon', 0.0001)  # update the epsilon attribute value
    g.nodemap['resnetv17_batchnorm0_fwd'].set_attr('lol', 'haha')  # add new attributes of OP
    raw = g.tensormap['resnetv17_batchnorm0_gamma'].numpy
    g.tensormap['resnetv17_batchnorm0_gamma'].numpy = raw.astype(numpy.float16)  # convert weight tensor to float16
    g.skip_node('flatten_473')  # remove_node will break the input and output tensor relation
    m.save_model('resnet50-v1-7-edited.onnx')


def profile_model():
    import onnx_tool
    modelpath = 'data/public/resnet50-v1-7.onnx'
    m = onnx_tool.Model(modelpath)
    m.graph.shape_infer({'data': numpy.zeros((1, 3, 224, 224))})  # update tensor shapes with new input tensor
    m.graph.profile()
    m.graph.print_node_map()  # console print
    m.graph.print_node_map('resnet50-224.txt')  # save file

    m.graph.shape_infer({'data': numpy.zeros((1, 3, 256, 256))})  # update new resolution
    m.graph.profile()
    m.graph.print_node_map(exclude_ops=['Flatten', 'Relu', 'BatchNormalization'])  # remove ops from the profile
    m.graph.print_node_map('resnet50-256.csv')  # csv file

    m.save_model('resnet50_shapes_only.onnx',
                 shape_only=True)  # only with weight tensor shapes and dynamic tensor shapes
    # remove static weights, minimize storage space. 46KB


def weight_compression():
    import onnx_tool
    modelpath = 'data/public/resnet50-v1-7.onnx'
    m = onnx_tool.Model(modelpath)
    g = m.graph

    def tofloat16():
        for key in g.initials:
            tensor = g.tensormap[key]
            raw = tensor.numpy
            tensor.numpy = raw.astype(numpy.float16)
        m.save_model(m.modelname + '-fp16.onnx')

    def quantize_sym():
        from onnx_tool.quantization import graph_quantize
        for key in g.initials:
            graph_quantize(g, key, block=-1, type='sym', bits=8)
        m.save_model(m.modelname + '-8bits-sym-default.onnx')

    def quantize_asym():
        from onnx_tool.quantization import graph_quantize
        for key in g.initials:
            graph_quantize(g, key, block=-1, type='asym', bits=8)
        m.save_model(m.modelname + '-8bits-asym-default.onnx')

    def quantize_sym_b32():
        from onnx_tool.quantization import graph_quantize
        for key in g.initials:
            graph_quantize(g, key, block=32, type='sym', bits=8)
        m.save_model(m.modelname + '-8bits-sym-b32.onnx')

    def quantize_4bits_sym_b32():
        from onnx_tool.quantization import graph_quantize
        for key in g.initials:
            graph_quantize(g, key, block=32, type='sym', bits=4)
        m.save_model(m.modelname + '-4bits-sym-b32.onnx')


def simple_inference():
    import onnx
    import onnx_tool
    modelpath = 'data/public/resnet50-v1-7.onnx'
    tmppath = 'tmp.onnx'
    m = onnx_tool.Model(modelpath)
    dumptensors = ['resnetv17_stage1_conv3_fwd', 'resnetv17_stage1_conv3_fwd']
    m.graph.add_dump_tensors(dumptensors)
    m.save_model(tmppath)

    # add two hidden tensors resnetv17_stage1_conv3_fwd resnetv17_stage1_conv3_fwd to 'resnet50_shapes.onnx' model's output tensors

    def infer_with_ort(onnxfile, dumptensors, inputm):
        import onnxruntime as ort
        sess = ort.InferenceSession(onnxfile)
        output = sess.run(dumptensors, inputm)
        return output

    inputm = {'data': numpy.ones((1, 3, 224, 224), dtype=numpy.float32)}
    # outputs = infer_with_ort(tmppath,dumptensors,inputm) #with onnxruntime
    # print(outputs[0])
    outputs = m.graph.value_infer(inputm)  # limited models, very slow, for debug purpose
    print(m.graph.tensormap['resnetv17_stage1_conv3_fwd'].numpy)


def dynamic_input_shapes():
    import numpy
    import onnx_tool
    from onnx_tool import create_ndarray_f32  # or use numpy.ones(shape,numpy.float32) is ok
    modelpath = 'data/public/rvm_mobilenetv3_fp32.onnx'
    m = onnx_tool.Model(modelpath)
    inputs = {'src': create_ndarray_f32((1, 3, 1080, 1920)), 'r1i': create_ndarray_f32((1, 16, 135, 240)),
              'r2i': create_ndarray_f32((1, 20, 68, 120)), 'r3i': create_ndarray_f32((1, 40, 34, 60)),
              'r4i': create_ndarray_f32((1, 64, 17, 30)), 'downsample_ratio': numpy.array((0.25,), dtype=numpy.float32)}
    m.graph.shape_infer(inputs)
    m.graph.profile()
    m.graph.print_node_map()
    m.save_model('rvm_mobilenetv3_fp32_shapes.onnx')


def custom_layer_register():
    import onnx_tool
    from onnx_tool.node import _get_shape
    from onnx_tool import create_ndarray_f32

    @onnx_tool.NODE_REGISTRY.register()
    class CropPluginNode(onnx_tool.Node):
        # you can implement either shape_infer(faster) or value_infer.
        # it's not necessary to implement both
        def shape_infer(self, intensors: []):
            # if you know how to calculate shapes of this op, you can implement shape_infer
            return [_get_shape(intensors[1])]

        # for upgrade of node_profilers.py, node_profilers.py's 'infer_shape' method should be placed
        # as 'value_infer' method here, and do not create this class' 'shape_infer' method.
        def value_infer(self, intensors: []):
            # if you don't know how to calculate the shapes of this op, you can implement value_infer.
            shape1 = intensors[1].shape
            outtensor = intensors[0][:, :, :shape1[2], :shape1[3]]
            return [outtensor]

        def profile(self, intensors: [], outtensors: []):
            macs = 0
            # accumulate macs here
            # this node has no calculation
            return macs

    onnx_tool.model_profile('./rrdb_new.onnx', {'input': create_ndarray_f32((1, 3, 335, 619))},
                            savenode='rrdb_new_nodemap.txt', saveshapesmodel='rrdb_new_shapes.onnx')


def bert_mha_fuse():
    import onnx_tool
    modelpath = 'data/public/bertsquad-12.onnx'
    m = onnx_tool.Model(modelpath, mcfg={})
    g = m.graph
    g.graph_reorder_nodes()

    in_tensor_names = ['bert/encoder/Reshape_1:0']
    out_tensor_names = ['bert/encoder/layer_0/attention/output/dense/BiasAdd:0']
    g.fuse_subgraph_iotensors(inputs=in_tensor_names, outputs=out_tensor_names, name_prefix='MHA',
                              nodeop='MHA', keep_attr=True)
    g.graph_reorder_nodes()
    m.save_model('bertsquad_mha.onnx')


def bert_mha_layernorm_fuse():
    import onnx_tool
    modelpath = 'data/public/bertsquad-12.onnx'
    m = onnx_tool.Model(modelpath, mcfg={})
    g = m.graph
    g.graph_reorder_nodes()

    in_tensor_names = ['bert/encoder/Reshape_1:0']
    out_tensor_names = ['bert/encoder/layer_0/attention/output/dense/BiasAdd:0']
    # automatically find all MHA nodes
    g.fuse_subgraph_iotensors(inputs=in_tensor_names, outputs=out_tensor_names, name_prefix='MHA_0',
                              nodeop='MHA',
                              keep_attr=True)

    in_tensor_names = ['bert/encoder/layer_0/attention/output/add:0']
    out_tensor_names = ['bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1:0']
    g.fuse_subgraph_iotensors(inputs=in_tensor_names, outputs=out_tensor_names, name_prefix='layernrom',
                              nodeop='LayerNormalization',
                              keep_attr=True)
    g.graph_reorder_nodes()
    m.save_model('bertsquad_mha_layernorm.onnx')


def computegraph_with_shapeengine():
    import onnx_tool
    import onnx
    model_config = {
        'name': 'data/public/BERT_quan95.onnx',
        'dynamic_input': None,
        'input_desc':
            {
                'input_ids': ('batch', 'seq'),
                'attention_mask': ('batch', 'seq'),
                'token_type_ids': ('batch', 'seq'),
            },
        'input_range':
            {
                'batch': (1, 4),
                'seq': (16, 384)
            }
    }

    m = onnx_tool.Model(model_config['name'])
    g = m.graph
    g.graph_reorder_nodes()

    shape_engine = g.shape_regress(model_config['input_desc'], model_config['input_range'])
    cg = g.get_compute_graph()
    cg.save_model('compute_graph.onnx', rawmodel=m.mproto)

    shape_engine.update_variable('batch', 3)  # update batch size
    shape_engine.update_variable('seq', 155)  # update batch size
    shape_engine.update_variables()  # all shapes updated

    print(shape_engine.get_tensorshape('1979'))  # query tensor shapes


def serialization():
    import onnx_tool
    resnetinfo = {
        'name': 'data/public/resnet18-v1-7.onnx',
        'input_desc':
            {
                'data': [1, 3, 'h', 'w']
            },
        'input_range':
            {
                'h': (224, 299),
                'w': (224, 299),
            }
    }
    shape_engie, compute_graph = onnx_tool.model_shape_regress(resnetinfo['name'], resnetinfo['input_desc'],
                                                               resnetinfo['input_range'])
    onnx_tool.serialize_graph(compute_graph, 'resnet18.cg')
    onnx_tool.serialize_shape_engine(shape_engie, 'resnet18.se')


def detic_profile():
    import onnx_tool
    minfo = {
        'name': 'data/public/model_custom_vocabulary.onnx',
        'dynamic_input': None,
        'mcfg': {
            'constant_folding': False,
            'verbose': True,
            'if_fixed_branch': 'else',
            'fixed_topk': 1000,
        }
    }
    m = onnx_tool.Model(minfo['name'], minfo['mcfg'])
    m.graph.graph_reorder_nodes()
    m.graph.shape_infer(minfo['dynamic_input'])
    m.graph.profile()
    m.graph.print_node_map()
    m.save_model('detic_shapes.onnx')


def ssd300_vgg16():
    import onnx_tool
    minfo = {
        'name': 'data/public/ssd300_vgg16.onnx',
        'dynamic_input': None,
        'mcfg': {
            'constant_folding': True,
            'verbose': True,
            'if_fixed_branch': 'else',
            'fixed_topk': 0
        }
    }
    m = onnx_tool.Model(minfo['name'], minfo['mcfg'])
    m.graph.graph_reorder_nodes()
    m.graph.shape_infer(minfo['dynamic_input'])
    m.graph.profile()
    m.graph.print_node_map()
    m.save_model('ssd300_shapes.onnx')


def paraformer_profile():
    import onnx_tool
    from onnx_tool.node import NODE_REGISTRY
    batch = 4
    len = 100
    hidden_size = 512
    minfo = {
        'name': './data/public/paraformer.onnx',
        'dynamic_input': {
            'speech': numpy.zeros((batch, len, 560), dtype=numpy.float32),
            'speech_lengths': numpy.array((len,) * batch, dtype=numpy.int32)
        },
        'mcfg': {
            'constant_folding': True,
            'verbose': True,
            'remove_dangling': True
        }
    }

    @NODE_REGISTRY.register()
    class SequenceMaskNode(onnx_tool.Node):
        def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
            shape = intensors[0].get_shape()
            outtensors[0].update_shape(shape + [len, ])

    @NODE_REGISTRY.register()
    class cif_searchNode(onnx_tool.Node):
        def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
            outtensors[0].update_shape([batch, hidden_size])

    m = onnx_tool.Model(minfo['name'], minfo['mcfg'])
    m.graph.remove_dangling_nodes()
    in_tensor_names = ['onnx::ReduceMax_7938']
    out_tensor_names = ['tgt_mask']

    m.graph.fuse_subgraph_iotensors(inputs=in_tensor_names, outputs=out_tensor_names, name_prefix='SequenceMask',
                                    nodeop='SequenceMask', keep_attr=False)
    m.graph.fuse_subgraph_iotensors(inputs=['onnx::Unsqueeze_7851'], outputs=['onnx::ReduceMean_7914'],
                                    name_prefix='cif_search',
                                    nodeop='cif_search', keep_attr=False)
    m.graph.update_graph()
    m.graph.graph_reorder_nodes()
    # m.save_model('para_new.onnx')

    m.graph.shape_infer(minfo['dynamic_input'])
    m.graph.profile()
    m.graph.print_node_map()
    m.save_model('paraformer_shapes.onnx', shape_only=True)


paraformer_profile()
