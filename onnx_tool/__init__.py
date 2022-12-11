import os
import warnings

import numpy
import onnx

from .graph import Graph
from .node_profilers import NodeBase, node_profile, node_infer_shape, Constant
from .tensor import graph_addoutputs, graph_set_inputs, shape_of_tensor, is_valid_ndarray, tensorproto2ndarray, volume, \
    create_ndarray_f32, create_ndarray_int64, update_static_tensors, volume_tensor
from .utils import NODEPROFILER_REGISTRY, timer, tuple2str, GLOBAL_VARS, VERSION


def __remove_initilisers(graph: onnx.GraphProto):
    graph.ClearField('initializer')


def __remove_constantnodes(graph: onnx.GraphProto):
    validnodes = []
    for node in graph.node:
        if node.op_type != 'Constant':
            validnodes.append(node)
    graph.ClearField('node')
    for node in validnodes:
        graph.node.append(node)


def model_export_tensors_numpy(m, tensornames: [str] = None, savefolder: str = None, fp16: bool = False) -> None:
    if isinstance(m, str):
        m = onnx.load_model(m)
    GLOBAL_VARS['tensor_map'] = {}
    GLOBAL_VARS['params_map'] = {}

    def save_numpy(arr: numpy.ndarray, fp16: bool, filename):
        if fp16 and arr.dtype in [numpy.float32, numpy.float64]:
            arr = arr.astype(numpy.float16)
        numpy.save(filename, arr)

    if isinstance(m, onnx.ModelProto):
        update_static_tensors(m.graph)
        if savefolder is not None:
            os.makedirs(savefolder, exist_ok=True)
        else:
            savefolder = './'
        tensor_map = GLOBAL_VARS['tensor_map']
        if tensornames is None:
            for key in tensor_map.keys():
                name = key
                if '/' in key:
                    name = key.replace('/', '_')
                if '\\' in key:
                    name = key.replace('\\', '_')
                save_numpy(tensor_map[key], fp16, os.path.join(savefolder, name + '.npy'))

        else:
            for name in tensornames:
                if name not in tensor_map.keys():
                    warnings.warn(f'tensor {name} not found ')
                    continue
                fname = name
                if '/' in name:
                    fname = name.replace('/', '_')
                if '\\' in name:
                    fname = name.replace('\\', '_')
                save_numpy(tensor_map[name], fp16, os.path.join(savefolder, fname + '.npy'))


def narray_calc_sparsity(arr):
    if len(arr.shape) != 2 and len(arr.shape) != 4:
        return None
    if arr.dtype == numpy.float32 or arr.dtype == numpy.float64 or arr.dtype == numpy.int32 or arr.dtype == numpy.int8:
        flag = arr == 0
        return flag.sum() / arr.size
    if arr.dtype == numpy.uint8:
        flag = arr == 128
        return flag.sum() / arr.size


def narray_zero_flag(arr):
    if arr.dtype == numpy.float32 or arr.dtype == numpy.float64 or arr.dtype == numpy.int32 or arr.dtype == numpy.int8:
        flag = arr == 0
    if arr.dtype == numpy.uint8:
        flag = arr == 128
    return flag


def search_sparse_blocksize(arr, ratio, deltar_thres=0.1):
    if len(arr.shape) == 2:  # gemm or matmul
        initsize = 2
        validsize = 1
        prevalid0 = True
        prevalid1 = True
        validratio = ratio
        while True:
            # try axis=1
            if prevalid1 and arr.shape[1] % initsize == 0:
                rearr = arr.reshape(arr.shape[0], -1, initsize)
                flag = narray_zero_flag(rearr)
                arrsum = numpy.sum(flag, -1)
                ratio1 = (arrsum == initsize).sum() / arrsum.size
                if ratio1 > ratio - deltar_thres:
                    valid1 = True
                    validratio = ratio1
                else:
                    valid1 = False
            else:
                valid1 = False

            # try axis=0
            if prevalid0 and arr.shape[0] % initsize == 0:
                rearr = arr.reshape(-1, initsize, arr.shape[1])
                flag = narray_zero_flag(rearr)
                arrsum = numpy.sum(flag, 1)
                ratio0 = (arrsum == initsize).sum() / arrsum.size
                if ratio0 > ratio - deltar_thres:
                    valid0 = True
                    validratio = ratio0
                else:
                    valid0 = False
            else:
                valid0 = False

            if not valid1 and not valid0:
                break
            validsize = initsize
            initsize *= 2
            prevalid0 = valid0
            prevalid1 = valid1

        # check square
        if prevalid1 and prevalid0:
            rearr = arr.reshape(arr.shape[0] // validsize, validsize, arr.shape[1] // validsize, validsize)
            flag = narray_zero_flag(rearr)
            arrsum = numpy.sum(flag, axis=(1, -1))
            ratios = (arrsum == (validsize * validsize)).sum() / arrsum.size
            if ratios > ratio - deltar_thres:
                return (validsize, validsize), ratios

        return (validsize if prevalid0 else 1, validsize if prevalid1 else 1), validratio

    if len(arr.shape) == 4:  # conv2d
        initsize = 2
        validsize = 1
        prevalid0 = True
        prevalid1 = True
        validratio0 = ratio
        validratio1 = ratio
        while True:
            # try axis=1
            if prevalid1 and arr.shape[1] % initsize == 0:
                rearr = arr.reshape(arr.shape[0], -1, initsize, *arr.shape[2:])
                flag = narray_zero_flag(rearr)
                arrsum = numpy.sum(flag, 2)
                ratio1 = (arrsum == initsize).sum() / arrsum.size
                if ratio1 > ratio - deltar_thres:
                    valid1 = True
                    validratio1 = ratio1
                else:
                    valid1 = False
            else:
                valid1 = False

            # try axis=0
            if prevalid0 and arr.shape[0] % initsize == 0:
                rearr = arr.reshape(-1, initsize, *arr.shape[1:])
                flag = narray_zero_flag(rearr)
                arrsum = numpy.sum(flag, 1)
                ratio0 = (arrsum == initsize).sum() / arrsum.size
                if ratio0 > ratio - deltar_thres:
                    valid0 = True
                    validratio0 = ratio0
                else:
                    valid0 = False
            else:
                valid0 = False

            if not valid1 and not valid0:
                break
            validsize = initsize
            initsize *= 2
            prevalid0 = valid0
            prevalid1 = valid1
        # check square
        if validsize > 1 and prevalid1 and prevalid0:
            rearr = arr.reshape(arr.shape[0] // validsize, validsize, arr.shape[1] // validsize, validsize,
                                *arr.shape[2:])
            flag = narray_zero_flag(rearr)
            arrsum = numpy.sum(flag, axis=(1, 3))
            ratios = (arrsum == (validsize * validsize)).sum() / arrsum.size
            if ratios > ratio - deltar_thres:
                return (validsize, validsize), ratios
        if validratio0 > validratio1:
            return (validsize, 1), validratio0
        return (1, validsize), validratio1

    return (1, 1), ratio


def sparsity_search(thres_size=128, thres_ratio=0.4):
    tensor_map = GLOBAL_VARS['tensor_map']
    GLOBAL_VARS['sparse_map'] = {}
    sparse_map = GLOBAL_VARS['sparse_map']
    for key in tensor_map.keys():
        arr = tensor_map[key]
        if (volume_tensor(arr) > thres_size):
            ratio = narray_calc_sparsity(arr)
            if ratio is not None and ratio > thres_ratio:
                blocksize, blockratio = search_sparse_blocksize(arr, ratio)
                sparse_map[key] = {'blocksize': blocksize, 'blockratio': blockratio, 'ratio': ratio}
    GLOBAL_VARS['sparse_map'] = sparse_map


def infer_shapes(graph: onnx.GraphProto, dynamic_tensors: {}, verbose: bool = False) -> [map, map]:
    """
        Returns: {TensorName:ndarray},{NodeName:int}
    """
    GLOBAL_VARS['tensor_map'] = {}
    GLOBAL_VARS['params_map'] = {}

    update_static_tensors(graph)

    sparsity_search()

    if dynamic_tensors is not None:
        graph_set_inputs(graph, dynamic_tensors)

    tensor_map = GLOBAL_VARS['tensor_map']
    params_map = GLOBAL_VARS['params_map']
    for input in graph.input:
        shape = shape_of_tensor(input)
        for d in shape:
            if d < 0:
                raise ValueError(f"Input {input.name}'s shape is dynamic, please set it a fixed input dimension")
        if input.name not in tensor_map:
            tensor_map.update({input.name: numpy.zeros(shape, dtype=numpy.float32)})

        if not is_valid_ndarray(tensor_map[input.name]):
            raise ValueError(f"Input {input.name}'s shape is dynamic, please set it a fixed input dimension")

    for node in graph.node:
        ins = []
        for input in node.input:
            if input == '':
                continue
            ins.append(tensor_map[input])
        outs = []
        for output in node.output:
            if output == '':
                continue
            outs.append(output)
        outtensors = node_infer_shape(node, ins)
        for tensor, name in zip(outtensors, outs):
            tensor_map[name] = tensor

    for key in tensor_map.keys():
        shape = tensor_map[key].shape
        if len(shape) == 0:
            shape = (0,)
        vinf = onnx.helper.make_tensor_value_info(key, onnx.TensorProto.FLOAT, shape)
        graph.value_info.append(vinf)

    for output in graph.output:
        dim = output.type.tensor_type.shape.dim
        for nb, dnb in zip(dim, tensor_map[output.name].shape):
            nb.dim_value = dnb
    GLOBAL_VARS['tensor_map'] = tensor_map
    GLOBAL_VARS['params_map'] = params_map
    return tensor_map, params_map


def graph_profile(graph: onnx.GraphProto, dynamic_shapes: {}, verbose=False, hidden_ops: [str] = None
                  ) -> [float, float, map]:
    """
        return MACs,Params,NodeMap
    """
    macs = 0.0
    params = 0
    memory = 0

    gtmr = timer()

    gtmr.start()
    tmap, pmap = infer_shapes(graph, dynamic_shapes, verbose=verbose)
    t_shapeinfer = gtmr.stop()
    if verbose:
        print(f'infered all tensor shapes, time cost {gtmr.stop():.3f} s')

    node_map = {}
    index = 0
    gtmr.start()
    params_map = GLOBAL_VARS['params_map']

    sparse_map = GLOBAL_VARS['sparse_map']
    sparse_model = len(sparse_map.keys()) > 0
    params_flag_map = {}
    for key in params_map.keys():
        params_flag_map[key] = 0
    params_shared_nodes = {}
    for input in graph.input:
        tensor = tmap[input.name]
        _memory = volume(tensor.shape) * 4
        nodedata = {'macs': 0, 'params': 0, 'memory': _memory, 'inshape': tensor.shape,
                    'outshape': tensor.shape}
        nodedata.update({'blocksize': (1, 1), 'ratio': 0, 'blockratio': 0})
        node_map.update({input.name: nodedata})
        memory += _memory

    for node in graph.node:
        ins = []
        _params = 0
        _memory = 0
        if hidden_ops is not None:
            if node.op_type in hidden_ops:
                continue
        for input in node.input:
            if input == '':
                continue
            ins.append(tmap[input])
            if input in pmap.keys():
                if params_flag_map[input] == 0:
                    _params += pmap[input]
                    _memory += pmap[input]

                params_flag_map[input] += 1

        outs = []
        for output in node.output:
            if tmap.keys().__contains__(output):
                outs.append(tmap[output])
                if node.op_type == 'Constant':
                    # Constant's output tensors are already counted as weight tensors
                    continue
                _memory += volume(tmap[output].shape)
        _macs, _params_c = node_profile(node, ins, outs)
        # @deprecated _params_c

        outshape = (0,)
        if len(outs) > 0:
            outshape = outs[0].shape
            outshape = (0,) if len(outshape) == 0 else outshape
        inshape = (0,)
        if len(ins) > 0:
            inshape = ins[0].shape
            inshape = (0,) if len(inshape) == 0 else inshape
        if len(node.name) == 0:
            node.name = node.op_type + '_{}'.format(index)
        index += 1
        _memory *= 4
        nodedata = {'macs': _macs, 'params': _params, 'memory': _memory, 'inshape': inshape,
                    'outshape': outshape}
        if sparse_model:
            is_sparse = False
            for input in node.input:
                if input == '':
                    continue
                if input in sparse_map.keys():
                    nodedata.update(sparse_map[input])
                    is_sparse = True
                    break
            if not is_sparse:
                nodedata.update({'blocksize': (1, 1), 'ratio': 0, 'blockratio': 0})
        node_map.update({node.name: nodedata})
        macs += _macs
        params += _params
        memory += _memory

    t_profile = gtmr.stop()
    if verbose:
        print(f'profile all nodes, time cost {t_profile:.3f} s')

    for node in graph.node:
        for input in node.input:
            if input == '':
                continue
            if input in pmap.keys():
                if params_flag_map[input] > 1 and volume(tmap[input].shape) > 128:  # set 128 as sharing threshold
                    if input in params_shared_nodes:
                        params_shared_nodes[input].append(node.name)
                    else:
                        params_shared_nodes[input] = [node.name]

    GLOBAL_VARS['macs'] = macs
    GLOBAL_VARS['params'] = params
    GLOBAL_VARS['memory'] = memory
    GLOBAL_VARS['node_map'] = node_map
    GLOBAL_VARS['params_shared_nodes'] = params_shared_nodes
    GLOBAL_VARS['t_shapeinfer'] = t_shapeinfer
    GLOBAL_VARS['t_profile'] = t_profile
    if verbose:
        tmem_count = 0
        for t in tmap:
            tmem_count += volume(tmap[t].shape)
        tmem_count *= 4
        diffratio = abs(memory - tmem_count) / tmem_count
        print(
            f'Memory sum from TensorMap:{tmem_count} Memory sum from NodeMap sum:{memory}, diff ratio:{diffratio:.3%}')
        assert (diffratio < 0.01)

    return macs, params, node_map


# These ops are created by onnx exporter, they are out of programmer's sense
DefaultFilter = (
    'Identity', 'Constant',
)

# These ops have no computation
NoMacsOps = (
    'Identity', 'Constant', 'Shape', 'Squeeze', 'Unsqueeze', 'Reshape', 'ConstantOfShape', 'Cast', 'Pad', 'Concat',
    'Slice', 'Gather'
)


def model_profile(m, dynamic_shapes: {str: tuple} = None, savenode: str = None, saveshapesmodel: str = None,
                  topsort: bool = True, shapesonly: bool = False, verbose: bool = False,
                  hidden_ops: [str] = DefaultFilter,
                  dump_outputs: [str] = None, remove_unused_tensors=True) -> None:
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        graph = m.graph
        if remove_unused_tensors:
            graph_remove_unused_tensors(graph)
        if topsort:
            G = Graph(graph)
            G.graph_reorder()
            graph = G.rawgraph
        graph_profile(graph, dynamic_shapes, verbose, hidden_ops=hidden_ops)
        print_node_map(savenode)
        if saveshapesmodel is not None:
            if shapesonly:
                __remove_initilisers(graph)
                __remove_constantnodes(graph)

            if dump_outputs is not None:
                graph_addoutputs(graph, dump_outputs)
            G = Graph(graph)
            G.save_model(saveshapesmodel)


def model_profile_v2(m, dynamic_shapes: {str: tuple} = None, savenode: str = None,
                     saveshapesmodel: str = None, shapesonly: bool = False, verbose: bool = False,
                     hidden_ops: [str] = NoMacsOps,
                     dump_outputs: [str] = None, remove_unused_tensors=True) -> None:
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        G = Graph(m.graph, verbose=verbose)
        gtmr = timer()
        gtmr.start()
        G.shape_infer(dynamic_shapes)
        if verbose:
            print(f'infered all tensor shapes, time cost {gtmr.stop():.3f} s')
        gtmr.start()
        G.profile()
        if verbose:
            print(f'profile all nodes, time cost {gtmr.stop():.3f} s')
        G.print_node_map(savenode, exclude_nodes=hidden_ops)

        if saveshapesmodel is not None:
            if dump_outputs is not None:
                G.add_dump_tensors(dump_outputs)
            G.save_model(saveshapesmodel, shapesonly)


def model_api_test(m, dynamic_shapes: {str: tuple} = None):
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        G = Graph(m.graph)
        G.graph_reorder()
        gtmr = timer()
        gtmr.start()
        G.shape_infer(dynamic_shapes)
        t_shapeinfer_new = gtmr.stop()
        print(f'infered all tensor shapes, time cost {t_shapeinfer_new:.3f} s')

        gtmr.start()
        G.profile()
        t_profile_new = gtmr.stop()
        print(f'profile all nodes, time cost {t_profile_new:.3f} s')
        graph_profile(G.rawgraph, dynamic_shapes, True)
        node_map = GLOBAL_VARS['node_map']
        node_map_v2 = G.nodemap
        macsdiffacc = 0
        for key in node_map.keys():
            if key in node_map_v2.keys():
                diff = node_map[key]['macs'] - node_map_v2[key].macs
                macsdiffacc += diff
                if diff != 0:
                    print(f"Error macs: {key} {node_map[key]['macs']} {node_map_v2[key].macs}")
                diff = node_map[key]['params'] - node_map_v2[key].params
                if diff != 0:
                    print(f"Error params: {key} {node_map[key]['params']} {node_map_v2[key].params}")
                diff = node_map[key]['memory'] - node_map_v2[key].memory
                if diff != 0:
                    print(f"Error memory: {key} {node_map[key]['memory']} {node_map_v2[key].memory}")
            else:
                print(f"Error Key not match {key}")
        # G.print_node_map(exclude_nodes=NoMacsOps)
        # G.save_model('shape.onnx')
        return {'t_shapeinfer_new': t_shapeinfer_new, 't_profile_new': t_profile_new,
                't_shapeinfer': GLOBAL_VARS['t_shapeinfer'], 't_profile': GLOBAL_VARS['t_profile'],
                'macsdiff': macsdiffacc}


def model_shape_regress(m, min_shape: {}, max_shape: {}):
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        G = Graph(m.graph)
        G.graph_reorder()
        G.shape_regress(min_shape, max_shape)


def model_remove_Identity(m, f: str):
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        graph = m.graph
        iden_map = {}
        iden_set = []
        for node in graph.node:
            if node.op_type == 'Identity':
                iden_map[node.output[0]] = node.input[0]
                iden_set.append(node.name)
        for node in graph.node:
            for i, input in enumerate(node.input):
                if input in iden_map.keys():
                    node.input[i] = iden_map[input]
        G = Graph(graph)
        nodes = list(G.nodemap.keys())
        for idenn in iden_set:
            nodes.remove(idenn)
        onnxg = G.get_onnxgraph_by_nodenames(nodes)
        G = Graph(onnxg)
        G.save_model(f)


def model_shape_infer(m, dynamic_shapes: {str: tuple} = None,
                      saveshapesmodel: str = None, shapesonly: bool = False, verbose: bool = False,
                      dump_outputs: [str] = None):
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        _, _ = infer_shapes(m.graph, dynamic_shapes, verbose)
        if saveshapesmodel is not None:
            if shapesonly:
                __remove_initilisers(m)
                __remove_constantnodes(m)

            if dump_outputs is not None:
                graph_addoutputs(m.graph, dump_outputs)
            G = Graph(m.graph)
            G.save_model(saveshapesmodel)


def print_node_map(f: str = None, metric='MACs'):
    from tabulate import tabulate
    assert (metric in ['MACs', 'FLOPs'])
    node_map = GLOBAL_VARS['node_map']
    sparse_map = GLOBAL_VARS['sparse_map']
    print_sparse_table = True
    if len(sparse_map.keys()) == 0:
        print_sparse_table = False
    saveformat = 'txt'
    splitch = 'x'

    if f is not None and '.csv' in f:
        saveformat = 'csv'
        csvformat = True
    else:
        csvformat = False

    ptable = []

    macs = int(round(GLOBAL_VARS['macs']))
    params = int(GLOBAL_VARS['params'])
    memory = int(GLOBAL_VARS['memory'])

    shared_params = GLOBAL_VARS['params_shared_nodes']
    if len(shared_params.keys()):
        print()
        print('*' * 64)
        print(f'Please note that Weight Tensors Sharing is detected:')
        for key in shared_params.keys():
            print(f'Tensor:{key} ')
            print('Shared by: ')
            for node in shared_params[key]:
                print('           ', node)
            print()
        print('*' * 64)

    factor = 1
    if metric == 'FLOPs':
        factor = 2

    def num2str(num, csv=False):
        if csv:
            return '{}'.format(num)
        else:
            return '{:,}'.format(num)

    params += 1e-18
    macs += 1e-18
    for key in node_map.keys():
        item = node_map[key]
        row = [key]
        if print_sparse_table:
            row.append(tuple2str(item['blocksize'], splitch))
            row.append('{:.2%}'.format(item['blockratio']))
            row.append('{:.2%}'.format(item['ratio']))
        row.append(num2str(int(item['macs']) * factor, csvformat))
        row.append('{:.2%}'.format(item['macs'] / macs))
        row.append(num2str(int(item['memory']), csvformat))
        row.append('{:.2%}'.format(item['memory'] / memory))
        row.append(num2str(int(item['params']), csvformat))
        row.append('{:.2%}'.format(item['params'] / params))
        row.append(tuple2str(item['inshape'], splitch))
        row.append(tuple2str(item['outshape'], splitch))

        ptable.append(row)
    row = ['Total']
    if print_sparse_table:
        row.append('_')
        row.append('_')
        row.append('_')
    row.append(num2str(int(macs * factor), csvformat))
    row.append('100%')
    row.append(num2str(int(memory), csvformat))
    row.append('100%')
    row.append(num2str(int(params), csvformat))
    row.append('100%')
    row.append('_')
    row.append('_')

    ptable.append(row)
    header = ['Name']
    if print_sparse_table:
        header.append('Sparse Pattern')
        header.append('Sparse Block Ratio')
        header.append('Sparse Ratio')
    header.extend([metric, 'CPercent', 'Memory', 'MPercent', 'Params', 'PPercent', 'InShape',
                   'OutShape'])

    if f is None:
        print(tabulate(ptable, headers=header))
    else:
        fp = open(f, 'w')
        if saveformat == 'csv':
            headerstr = ''
            for i, item in enumerate(header):
                headerstr += item
                if i < len(header) - 1:
                    headerstr += ','
            headerstr += '\n'
            fp.write(headerstr)
            for row in ptable:
                str = ''
                for i, ele in enumerate(row):
                    str += ele
                    if i != len(row) - 1:
                        str += ','
                str += '\n'
                fp.write(str)
        else:
            fp.write(tabulate(ptable, headers=header))
        fp.close()


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
            G = G.graph_reorder()
        G.save_model(savemodel)


def model_reorder_nodes(m, savemodel: str, ):
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        G = Graph(m.graph)
        G = G.graph_reorder()
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
        graph.save_model(savemodel)

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
