def get_profile_for_each_layer():
    import onnx_tool
    modelpath = 'resnet50-v1-12.onnx'
    onnx_tool.model_profile(modelpath)  # pass file name
    onnx_tool.model_profile(modelpath, savenode='node_table.txt')  # save profile table to txt file
    onnx_tool.model_profile(modelpath, savenode='node_table.csv')  # save profile table to csv file


def add_op_filter():
    import onnx_tool
    from torchvision.models import ResNet
    from torchvision.models.resnet import Bottleneck
    import torch
    from onnx_tool import NoMacsOps
    net = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000)  # build ResNet50 for ImageNet
    x = torch.randn(1, 3, 224, 224)
    torch.onnx.export(net, x, 'tmp.onnx')
    hops = list(NoMacsOps)  # import onnx_tool's suggested ops
    hops.extend(('Add', 'Flatten', 'Relu'))  # add Add and Flatten op to hidden list
    onnx_tool.model_profile('tmp.onnx', hidden_ops=hops)  # conv, Gemm, GlobalAveragePool, MaxPool left


def save_graph_structure_only():
    import onnx
    import onnx_tool
    modelpath = 'resnet50-v1-12.onnx'
    model = onnx.load_model(modelpath)
    onnx_tool.model_shape_infer(model, None, saveshapesmodel='resnet50_shapes.onnx', shapesonly=True)
    # pass ONNX.ModelProto and remove static weights, minimize storage space. 46KB


def add_dump_tensors():
    import onnx
    import onnx_tool
    modelpath = 'resnet50-v1-12.onnx'
    model = onnx.load_model(modelpath)
    onnx_tool.model_shape_infer(model, None, saveshapesmodel='resnet50_shapes.onnx', shapesonly=True,
                                dump_outputs=['resnetv17_stage1_conv3_fwd', 'resnetv17_stage1_conv3_fwd'])
    # add two hidden tensors resnetv17_stage1_conv3_fwd resnetv17_stage1_conv3_fwd to 'resnet50_shapes.onnx' model's output tensors


def dynamic_input_shapes():
    import numpy
    import onnx_tool
    from onnx_tool import create_ndarray_f32  # or use numpy.ones(shape,numpy.float32) is ok
    modelpath = 'data/public/rvm_mobilenetv3_fp32.onnx'
    inputs = {'src': create_ndarray_f32((1, 3, 1080, 1920)), 'r1i': create_ndarray_f32((1, 16, 135, 240)),
              'r2i': create_ndarray_f32((1, 20, 68, 120)), 'r3i': create_ndarray_f32((1, 40, 34, 60)),
              'r4i': create_ndarray_f32((1, 64, 17, 30)), 'downsample_ratio': numpy.array((0.25,), dtype=numpy.float32)}
    onnx_tool.model_profile(modelpath, inputs, None, saveshapesmodel='rvm_mobilenetv3_fp32_shapes.onnx')


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
            outtensor = intensors[:, :, :shape1[2], :shape1[3]]
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
    import onnx
    modelpath = 'data/public/bertsquad-12.onnx'
    mproto = onnx.load_model(modelpath)
    g = onnx_tool.Graph(mproto.graph)
    g.graph_reorder()

    # do some graph search here
    in_tensor_names = ['bert/encoder/Reshape_1:0']
    out_tensor_names = ['bert/encoder/layer_0/attention/output/dense/BiasAdd:0']
    fused_g = g.fuse_subgraph_iotensors(inputs=in_tensor_names, outputs=out_tensor_names, name='MHA_0',
                                        nodeop='MHA', keep_attr=True)
    fused_g.save_model('bertsquad_mha.onnx')


def bert_mha_layernorm_fuse():
    import onnx_tool
    import onnx
    modelpath = 'data/public/bertsquad-12.onnx'
    mproto = onnx.load_model(modelpath)
    g = onnx_tool.Graph(mproto.graph)
    g.graph_reorder()

    # do some graph search here
    in_tensor_names = ['bert/encoder/Reshape_1:0']
    out_tensor_names = ['bert/encoder/layer_0/attention/output/dense/BiasAdd:0']
    fused_g = g.fuse_subgraph_iotensors(inputs=in_tensor_names, outputs=out_tensor_names, name='MHA_0',
                                        nodeop='MHA',
                                        keep_attr=True)  # ignore new op warning 'node MHA is not registed for profiling xxxx'

    in_tensor_names = ['bert/encoder/layer_0/attention/output/add:0']
    out_tensor_names = ['bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1:0']
    fused_g = fused_g.fuse_subgraph_iotensors(inputs=in_tensor_names, outputs=out_tensor_names, name='layernrom_0',
                                              nodeop='layernorm',
                                              keep_attr=True)  # ignore new op warning 'node layernorm is not registed for profiling xxxx'
    fused_g.save_model('bertsquad_mha_layernorm.onnx')


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

    mproto = onnx.load_model(model_config['name'])
    g = onnx_tool.Graph(mproto.graph)
    g.graph_reorder()

    shape_engine = g.shape_regress(model_config['input_desc'], model_config['input_range'])
    cg = g.get_compute_graph()
    cg = onnx_tool.Graph(cg)
    cg.save_model('compute_graph.onnx')

    shape_engine.update_variable('batch', 3)  # update batch size
    shape_engine.update_variable('seq', 155)  # update batch size
    shape_engine.update_variables()  # all shapes updated

    print(shape_engine.get_tensorshape('1979'))  # query tensor shapes
