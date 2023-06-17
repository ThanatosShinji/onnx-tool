import os
import warnings

import numpy
import onnx

from .graph import Graph
from .node import NODE_REGISTRY, Node
from .tensor import create_ndarray_f32, create_ndarray_int64
from .utils import timer, VERSION
from .serialization import serialize_shape_engine, serialize_graph


def __remove_initilisers(graph: onnx.GraphProto):
    graph.ClearField('initializer')


def model_export_tensors_numpy(m, tensornames: [str] = None, savefolder: str = None, fp16: bool = False) -> None:
    if isinstance(m, str):
        m = onnx.load_model(m)

    def save_numpy(arr: numpy.ndarray, fp16: bool, filename):
        if fp16 and arr.dtype in [numpy.float32, numpy.float64]:
            arr = arr.astype(numpy.float16)
        numpy.save(filename, arr)

    if isinstance(m, onnx.ModelProto):
        g = Graph(m.graph)
        if savefolder is not None:
            os.makedirs(savefolder, exist_ok=True)
        else:
            savefolder = './'
        if tensornames is None:
            for key in g.initials:
                name = key
                if '/' in key:
                    name = key.replace('/', '_')
                if '\\' in key:
                    name = key.replace('\\', '_')
                narr = g.tensormap[key].numpy
                save_numpy(narr, fp16, os.path.join(savefolder, name + '.npy'))

        else:
            for name in tensornames:
                if name not in g.initials:
                    warnings.warn(f'tensor {name} not found ')
                    continue
                fname = name
                if '/' in name:
                    fname = name.replace('/', '_')
                if '\\' in name:
                    fname = name.replace('\\', '_')
                narr = g.tensormap[name].numpy
                save_numpy(narr, fp16, os.path.join(savefolder, fname + '.npy'))


# These ops are created by onnx exporter, they are out of programmer's sense
DefaultFilter = (
    'Identity', 'Constant',
)

# These ops have no computation
NoMacsOps = (
    'Identity', 'Constant', 'Shape', 'Squeeze', 'Unsqueeze', 'Reshape', 'ConstantOfShape', 'Cast', 'Pad', 'Concat',
    'Slice', 'Gather'
)


def model_profile(m, dynamic_shapes: {str: tuple} = None, savenode: str = None,
                  saveshapesmodel: str = None, shapesonly: bool = False, verbose: bool = False,
                  constant_folding: bool = False,
                  hidden_ops: [str] = NoMacsOps,
                  dump_outputs: [str] = None) -> None:
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        g = Graph(m.graph, constant_folding=constant_folding, verbose=verbose)
        gtmr = timer()
        g.graph_reorder_nodes()
        gtmr.start()
        g.shape_infer(dynamic_shapes)
        if verbose:
            print(f'infered all tensor shapes, time cost {gtmr.stop():.3f} s')
        gtmr.start()
        g.profile()
        if verbose:
            print(f'profile all nodes, time cost {gtmr.stop():.3f} s')
        g.print_node_map(savenode, exclude_nodes=hidden_ops)

        if saveshapesmodel is not None:
            if dump_outputs is not None:
                g.add_dump_tensors(dump_outputs)
            g.save_model(saveshapesmodel, shapesonly)


def model_shape_regress(m, input_desc: {}, input_range: {}):
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        G = Graph(m.graph)
        G.graph_reorder_nodes()
        shape_engine = G.shape_regress(input_desc, input_range)
        cg = G.get_compute_graph()
        return shape_engine, cg


def model_constant_folding(m, f: str):
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        g = Graph(m.graph, constant_folding=True, verbose=True)
        g.save_model(f, rawmodel=m)


def model_shape_infer(m, dynamic_shapes: {str: tuple} = None,
                      saveshapesmodel: str = None, shapesonly: bool = False, verbose: bool = False,
                      dump_outputs: [str] = None):
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        g = Graph(m.graph, verbose=verbose)
        g.shape_infer(dynamic_shapes)
        if saveshapesmodel is not None:
            if dump_outputs is not None:
                g.add_dump_tensors(dump_outputs)
            g.save_model(saveshapesmodel, shape_only=shapesonly)


def graph_simplify_names(graph, renametensor=True, renamelayer=True, custom_inputs=None, custom_outputs=None,
                         remove_unused_tensors=True):
    '''
        Args:
            graph: onnx.GraphProto
            renametensor: boolean  eg.: resnetblock1_conv0_weight => 123
            renamelayer: boolean eg.: resnetblock_conv0 => Conv_0
            custom_inputs: [str] | {str:str} eg.: ['input'] without shapes, {'input':'Nx3xwidthxheight'} with shapes
            custom_outputs: [str] | {str:str} eg.: ['output'] without shapes, {'output':'Nx1xwidthxheight'} with shapes
        Returns:

    '''
    if remove_unused_tensors:
        graph_remove_unused_tensors(graph)
    if renamelayer:
        count = 0
        for node in graph.node:
            node.name = node.op_type + '_' + str(count)
            count += 1
    if renametensor:
        total_t = {}
        for node in graph.node:
            for input in node.input:
                total_t[input] = 0
            for output in node.output:
                total_t[output] = 0
        count = 0
        for key in total_t.keys():
            total_t[key] = str(count)
            count += 1

        if custom_inputs is not None:
            if isinstance(custom_inputs, list):
                assert (len(custom_inputs) == len(graph.input))
                for i, input in enumerate(graph.input):
                    total_t[input.name] = custom_inputs[i]
            elif isinstance(custom_inputs, dict):
                keylist = list(custom_inputs.keys())
                assert (len(keylist) == len(graph.input))
                for i, input in enumerate(graph.input):
                    total_t[input.name] = keylist[i]

                    # maybe consider create a new valueinfoproto
                    shapes = custom_inputs[keylist[i]].split('x')
                    dim = input.type.tensor_type.shape.dim
                    assert (len(shapes) == len(dim))
                    for nb, shapeval in zip(dim, shapes):
                        if shapeval.isnumeric():
                            if nb.HasField('dim_param'):
                                nb.ClearField('dim_param')
                            nb.dim_value = int(shapeval)
                        else:
                            if nb.HasField('dim_value'):
                                nb.ClearField('dim_value')
                            nb.dim_param = shapeval
            else:
                raise NotImplementedError()

        if custom_outputs is not None:
            if isinstance(custom_outputs, list):
                assert (len(custom_outputs) == len(graph.output))
                for i, output in enumerate(graph.output):
                    total_t[output.name] = custom_outputs[i]
            elif isinstance(custom_outputs, dict):
                keylist = list(custom_outputs.keys())
                assert (len(keylist) == len(graph.output))
                for i, output in enumerate(graph.output):
                    total_t[output.name] = keylist[i]
                    shapes = custom_outputs[keylist[i]].split('x')
                    # maybe consider create a new valueinfoproto
                    dim = output.type.tensor_type.shape.dim
                    assert (len(shapes) == len(dim))
                    for nb, shapeval in zip(dim, shapes):
                        if shapeval.isnumeric():
                            if nb.HasField('dim_param'):
                                nb.ClearField('dim_param')
                            nb.dim_value = int(shapeval)
                        else:
                            if nb.HasField('dim_value'):
                                nb.ClearField('dim_value')
                            nb.dim_param = shapeval
            else:
                raise NotImplementedError()

        for initial in graph.initializer:
            initial.name = total_t[initial.name]
        for node in graph.node:
            for i, input in enumerate(node.input):
                node.input[i] = total_t[input]
            for i, output in enumerate(node.output):
                node.output[i] = total_t[output]

        for input in graph.input:
            input.name = total_t[input.name]

        for output in graph.output:
            output.name = total_t[output.name]


def graph_remove_unused_tensors(graph):
    producer = {}
    consumer = {}
    for initial in graph.initializer:
        producer[initial.name] = 0
    for node in graph.node:
        for input in node.input:
            consumer[input] = 0
        for output in node.output:
            producer[output] = 0
    inputs = []
    outputs = []
    for key in consumer.keys():
        if key not in producer:
            inputs.append(key)
    for key in producer.keys():
        if key not in consumer:
            outputs.append(key)
    valid_inputs = []
    valid_outputs = []
    for input in graph.input:
        if input.name in inputs:
            valid_inputs.append(input)
    for output in graph.output:
        if output.name in outputs:
            valid_outputs.append(output)
    graph.ClearField('input')
    for input in valid_inputs:
        graph.input.append(input)
    graph.ClearField('output')
    for output in valid_outputs:
        graph.output.append(output)


def model_simplify_names(m, savemodel: str, renametensor=True, renamelayer=True, custom_inputs=None,
                         custom_outputs=None, remove_unused_tensors=True, node_reorder=False):
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        graph_simplify_names(m.graph, renametensor, renamelayer, custom_inputs, custom_outputs, remove_unused_tensors)
        G = Graph(m.graph)
        if node_reorder:
            G = G.graph_reorder_nodes()
        G.save_model(savemodel)


def model_reorder_nodes(m, savemodel: str, ):
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        G = Graph(m.graph)
        G = G.graph_reorder_nodes()
        G.save_model(savemodel)


def model_io_modify(m, savemodel: str, custom_io):
    '''
        Args:
            m: onnx.ModelProto or file path
            custom_io: {str:str} e.g. {'input':'Nx3xwidthxheight'}
        Returns:

    '''
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        graph = m.graph
        if custom_io is not None:
            keylist = list(custom_io.keys())
            for i, input in enumerate(graph.input):
                if input.name in keylist:
                    shapes = custom_io[input.name].split('x')
                    # maybe consider create a new valueinfoproto
                    dim = input.type.tensor_type.shape.dim
                    assert (len(shapes) == len(dim))
                    for nb, shapeval in zip(dim, shapes):
                        if shapeval.isnumeric():
                            if nb.HasField('dim_param'):
                                nb.ClearField('dim_param')
                            nb.dim_value = int(shapeval)
                        else:
                            if nb.HasField('dim_value'):
                                nb.ClearField('dim_value')
                            nb.dim_param = shapeval

            for i, output in enumerate(graph.output):
                if output.name in keylist:
                    shapes = custom_io[output.name].split('x')
                    # maybe consider create a new valueinfoproto
                    dim = output.type.tensor_type.shape.dim
                    assert (len(shapes) == len(dim))
                    for nb, shapeval in zip(dim, shapes):
                        if shapeval.isnumeric():
                            if nb.HasField('dim_param'):
                                nb.ClearField('dim_param')
                            nb.dim_value = int(shapeval)
                        else:
                            if nb.HasField('dim_value'):
                                nb.ClearField('dim_value')
                            nb.dim_param = shapeval
        graph = Graph(graph)
        graph.save_model(savemodel, rawmodel=m)


def model_subgraph(m, in_tensor_names: [str] = None, out_tensor_names: [str] = None, nodenames: [str] = None,
                   savefolder='./'):
    if isinstance(m, str):
        mname = os.path.basename(m)
        mname = os.path.splitext(mname)[0]
        m = onnx.load_model(m)
    else:
        mname = ''
    if isinstance(m, onnx.ModelProto):
        graph = Graph(m.graph)
        if in_tensor_names is not None and out_tensor_names is not None:
            graph_lvl0, graph_lvl1, graph_lvl2 = graph.get_subgraph(inputs=in_tensor_names, outputs=out_tensor_names)
            graph_lvl0.save_model(os.path.join(savefolder, mname + '_level0.onnx'))
            graph_lvl1.save_model(os.path.join(savefolder, mname + '_level1.onnx'))
            graph_lvl2.save_model(os.path.join(savefolder, mname + '_level2.onnx'))
        if nodenames is not None:
            rawgraph = graph.get_onnxgraph_by_nodenames(nodenames)
            subgraph = Graph(rawgraph)
            subgraph.save_model(os.path.join(savefolder, mname + '_subgraph.onnx'))


def model_opfusion(m, op_type: str, op_name: str, savefile, in_tensor_names: [str] = None,
                   out_tensor_names: [str] = None, nodenames: [str] = None, keep_attr=True):
    if isinstance(m, str):
        m = onnx.load_model(m)

    if isinstance(m, onnx.ModelProto):
        graph = Graph(m.graph)
        if in_tensor_names is not None and out_tensor_names is not None:
            graph = graph.fuse_subgraph_iotensors(inputs=in_tensor_names, outputs=out_tensor_names, name=op_name,
                                                  nodeop=op_type, keep_attr=keep_attr)
            graph.save_model(savefile)
        if nodenames is not None:
            graph = graph.fuse_subgraph_node_names(nodenames, nodeop=op_type, name=op_name, keep_attr=keep_attr)
            graph.save_model(savefile)
