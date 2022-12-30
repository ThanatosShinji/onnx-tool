import warnings

import numpy
import onnx
from .node import create_node
from .tensor import get_attribute_data, Tensor, volume
from .utils import VERSION, tuple2str


def __shape_of_initializer__(initial):
    shape = []
    # for nb in tensor.shape.dim
    for nb in initial.dims:
        shape.append(nb)
    return shape





_SHAPE_TENSORS = {
    'Reshape': ('1of2',),
    'Resize': ('2of3', '3of4', '1of2'),
    'Upsample': ('2of3', '3of4', '1of2'),
    'Expand': ('1of2',),
    'Slice': ('1,2of3', '1,2,3of4', '1,2,3,4of5'),
    'ConstantOfShape': ('0of1',),
    'Tile': ('1of2',),
    'Range': ('0,1,2of3',),
    'OneHot': ('1of3',),
    'TopK': ('1of2',),
    'Pad': ('1of3',),
    'NonMaxSuppression': ('2of5',),
}

def _contains_shape_tensor(n):
    nodeset = _SHAPE_TENSORS.keys()
    shape_tensors = []
    if n.op_type in nodeset:
        tensor_descs = _SHAPE_TENSORS[n.op_type]
        for desc in tensor_descs:
            strs = desc.split('of')
            indice = strs[0]
            count = int(strs[1])
            if len(n.input) == count:
                indistr = indice.split(',')
                for istr in indistr:
                    shape_tensors.append(n.input[int(istr)])
    return shape_tensors


class Graph():
    def __init__(self, g: onnx.GraphProto, noderename: bool = False, verbose=False):
        self.verbose = verbose
        self.nodemap = {}
        self.tensormap = {}
        self.producedby = {}
        self.consumedby = {}
        self.rawgraph = g
        self.initials = []
        self.dynamics = []
        self.input = []
        self.output = []
        self.__init_graph_from_onnxproto__(g, noderename)
        self.__find_shape_tensors__()

    def log(self, str):
        if self.verbose:
            print(str)

    def __init_graph_from_onnxproto__(self, g, noderename, remove_dummytensors=True):
        if g is None:
            return
        ncount = 0
        from .utils import timer

        tm = timer()
        tm.start()
        for node in g.node:
            newnode = create_node(node)
            if noderename or len(newnode.name) == 0:
                newnode.name = newnode.op_type + '_' + str(ncount)
            ncount += 1
            for tensor in node.input:
                if tensor in self.producedby:
                    for producer in self.producedby[tensor]:
                        newnode.prevnodes.append(self.nodemap[producer])
                if tensor not in self.consumedby:
                    self.consumedby[tensor] = []
                self.consumedby[tensor].append(newnode.name)
                newnode.input.append(tensor)
            for tensor in node.output:
                if tensor not in self.producedby:
                    self.producedby[tensor] = []
                self.producedby[tensor].append(newnode.name)
                newnode.output.append(tensor)

            self.nodemap[newnode.name] = newnode

        self.log(f'Node Init Time Elapsed {tm.stop()}')

        tm.start()
        for input in g.input:
            tensor = Tensor(input)
            self.tensormap[input.name] = tensor
            self.input.append(input.name)
            self.dynamics.append(input.name)

        for output in g.output:
            tensor = Tensor(output)
            self.tensormap[output.name] = tensor
            self.output.append(output.name)
            self.dynamics.append(output.name)

        for key in self.nodemap.keys():
            node = self.nodemap[key]
            dummy_lists = []
            for tensor in node.output:
                if tensor in self.consumedby:
                    for consumer in self.consumedby[tensor]:
                        self.nodemap[node.name].nextnodes.append(self.nodemap[consumer])
                else:
                    if tensor not in self.output:
                        self.log(f'Dummy tensors detected: {tensor}')
                        if remove_dummytensors:
                            dummy_lists.append(tensor)
            for tensor in dummy_lists:
                node.output.remove(tensor)

        for valinfo in g.value_info:
            tensor = Tensor(valinfo)
            self.tensormap[valinfo.name] = tensor

        self.log(f'IO Tensor Init Time Elapsed {tm.stop()}')

        tm.start()
        for initial in g.initializer:
            self.initials.append(initial.name)
            tensor = Tensor(initial)
            self.tensormap[initial.name] = tensor

        for key in self.nodemap.keys():
            node = self.nodemap[key]
            if node.op_type == 'Constant':
                tensor = Tensor(node)
                self.tensormap[node.output[0]] = tensor
                self.initials.append(node.output[0])
        self.initials = set(self.initials)

        self.log(f'Static Tensor Init Time Elapsed {tm.stop()}')

        rmlist = []
        for input in self.input:
            if input in self.initials:
                rmlist.append(input)
        for key in rmlist:
            self.dynamics.remove(key)
            self.input.remove(key)

        rmlist = []
        for output in self.output:
            if output in self.consumedby.keys():
                rmlist.append(output)
        for key in rmlist:
            self.dynamics.remove(key)
            self.output.remove(key)

        tm.start()
        for key in self.nodemap.keys():
            node = self.nodemap[key]
            for input in node.input:
                if input not in self.initials and input not in self.tensormap.keys():
                    self.dynamics.append(input)
                    self.tensormap[input] = Tensor(input)

        self.dynamics = set(self.dynamics)
        self.sparse_model = False
        for key in self.tensormap.keys():
            tensor = self.tensormap[key]
            if tensor.sparsity is not None and tensor.sparsity['ratio'] > 0.4:
                self.sparse_model = True
                break

        self.log(f'Misc Tensor Init Time Elapsed {tm.stop()}')

    def __find_shape_tensors__(self):
        self.shape_tensors = []
        for n in self.nodemap.keys():
            shape_tensors = _contains_shape_tensor(self.nodemap[n])
            for st in shape_tensors:
                self.shape_tensors.append(st)
        self.shape_tensors = set(self.shape_tensors)
        # print(self.shape_tensors)
        for tensor in self.shape_tensors:
            if tensor not in self.initials and tensor in self.producedby.keys():
                searchnodes = self.producedby[tensor]
                while len(searchnodes) > 0:
                    nextnodes = []
                    for nname in searchnodes:
                        node = self.nodemap[nname]
                        node.shape_calc = True
                        # if node.op_type == 'Shape':
                        #     continue
                        for input in node.input:
                            if input not in self.initials and input in self.producedby.keys():
                                producers = self.producedby[input]
                                nextnodes.extend([p for p in producers if self.nodemap[p].shape_calc == False])
                    searchnodes = nextnodes

    def __get_subnodes_byio__(self, inputs: [], outputs: []):
        graph_level0 = []
        graph_level1 = []
        graph_level2 = []
        searchlist = outputs
        while len(searchlist):
            newlist = []
            for name in searchlist:
                if name in self.consumedby:
                    for consumer in self.consumedby[name]:
                        if consumer in graph_level2:
                            continue
                        graph_level2.append(consumer)
                        for tensor in self.nodemap[consumer].output:
                            newlist.append(tensor)
            searchlist = newlist

        searchlist = inputs
        while len(searchlist):
            newlist = []
            for name in searchlist:
                if name in self.consumedby:
                    for consumer in self.consumedby[name]:
                        if consumer in graph_level2 or consumer in graph_level1:
                            continue
                        graph_level1.append(consumer)
                        for tensor in self.nodemap[consumer].output:
                            newlist.append(tensor)
            searchlist = newlist

        for node in self.nodemap.keys():
            if node not in graph_level1 and node not in graph_level2:
                graph_level0.append(node)

        return graph_level0, graph_level1, graph_level2

    def get_subgraph(self, inputs: [], outputs: []):
        graph_level0, graph_level1, graph_level2 = self.__get_subnodes_byio__(inputs, outputs)

        graph_level0 = Graph(self.get_onnxgraph_by_nodenames(graph_level0))
        graph_level1 = Graph(self.get_onnxgraph_by_nodenames(graph_level1))
        graph_level2 = Graph(self.get_onnxgraph_by_nodenames(graph_level2))

        group_outputs = [graph_level0.output, graph_level1.output, graph_level2.output]
        group_inputs = [graph_level0.input, graph_level1.input, graph_level2.input]

        extern_outputs = []
        extern_inputs = []
        for ele in group_outputs:
            extern_outputs.extend(ele)
        extern_outputs = set(extern_outputs)

        for ele in group_inputs:
            extern_inputs.extend(ele)
        extern_inputs = set(extern_inputs)

        for inputs in group_inputs:
            extern_outputs = extern_outputs - set(inputs)

        for outputs in group_outputs:
            extern_inputs = extern_inputs - set(outputs)

        if len(extern_inputs) != len(self.input):
            warnings.warn("subgraph input and output tensors can not reverse to raw graph.")

        if len(extern_outputs) != len(self.output):
            warnings.warn("subgraph input and output tensors can not reverse to raw graph.")

        return graph_level0, graph_level1, graph_level2

    def fuse_subgraph_node_names(self, nodes: [str], nodeop: str, name: str, keep_attr=True):
        _inputs, _outputs = self.get_iotensors(nodes, remove_initials=False)
        newnode = onnx.helper.make_node(nodeop, _inputs, _outputs, name=name)
        count = 0
        if keep_attr:
            for node in nodes:
                for attribute in self.nodemap[node].proto.attribute:
                    attr = onnx.helper.make_attribute(self.nodemap[node].proto.name + '_' + attribute.name,
                                                      get_attribute_data(attribute))
                    newnode.attribute.append(attr)
                count += 1

        allnodes = set(self.nodemap.keys())
        remainnodes = allnodes - set(nodes)
        nodes = []
        for name in remainnodes:
            nodes.append(self.nodemap[name].proto)
        nodes.append(newnode)
        inputs = []
        outputs = []
        for name in self.input:
            if name in self.tensormap:
                inputs.append(self.tensormap[name].proto)
            else:
                inputs.append(onnx.helper.make_tensor_value_info(name, 1, None))
        for name in self.output:
            if name in self.tensormap:
                outputs.append(self.tensormap[name].proto)
            else:
                outputs.append(onnx.helper.make_tensor_value_info(name, 1, None))
        graph = onnx.helper.make_graph(nodes=nodes, name='fused_graph', inputs=inputs, outputs=outputs,
                                       initializer=self.rawgraph.initializer)
        newgraph = Graph(graph)
        return newgraph

    def fuse_subgraph_iotensors(self, inputs: [], outputs: [], nodeop: str, name: str, keep_attr=True):
        _, nodes, _ = self.__get_subnodes_byio__(inputs, outputs)
        _inputs, _outputs = self.get_iotensors(nodes, remove_initials=True)
        nodes = self.reorder_nodes(nodes, _inputs)
        return self.fuse_subgraph_node_names(nodes, nodeop, name, keep_attr)

    def get_onnxgraph_by_nodenames(self, nodenames):
        if len(nodenames):
            _inputs0, _outputs0 = self.get_iotensors(nodenames)
            graph_level0 = self.reorder_nodes(nodenames, _inputs0)
            subgraph = self.make_graph(graph_level0, 'subgraph', _inputs0, _outputs0)
            return subgraph
        return None

    def save_model(self, f: str, shape_only: bool = False):
        graph = self.make_graph(self.nodemap.keys(), 'graph', self.input, self.output, not shape_only)
        if graph is not None and f is not None:
            model = onnx.helper.make_model(graph, producer_name='onnx-tool', producer_version='v' + VERSION)
            onnx.save_model(model, f)

    def make_graph(self, nodenames, gname, inputnames, outputnames, with_initializer=True):
        nodes = []
        for name in nodenames:
            nodes.append(self.nodemap[name].make_nodeproto())

        initializer = None
        if with_initializer:
            names = []
            for name in nodenames:
                for input in self.nodemap[name].input:
                    if input in self.initials:
                        names.append(input)
            initializer = []
            for initial in self.rawgraph.initializer:
                if initial.name in names:
                    initializer.append(initial)

        inputs = []
        outputs = []
        for name in inputnames:
            if name in self.tensormap:
                inputs.append(self.tensormap[name].make_value_proto())
            else:
                inputs.append(onnx.helper.make_tensor_value_info(name, 1, None))
        for name in outputnames:
            if name in self.tensormap:
                outputs.append(self.tensormap[name].make_value_proto())
            else:
                outputs.append(onnx.helper.make_tensor_value_info(name, 1, None))
        value_infos = []
        for key in self.tensormap.keys():
            tensor = self.tensormap[key]
            vinfo = tensor.make_value_proto()
            value_infos.append(vinfo)
        graph = onnx.helper.make_graph(nodes=nodes, name=gname, inputs=inputs, outputs=outputs, initializer=initializer,
                                       value_info=value_infos)
        return graph

    def is_node_constant(self, node):
        for input in self.nodemap[node].input:
            if input not in self.initials:
                return False
        return True

    def graph_reorder(self):
        old_order = self.nodemap.keys()
        ordered_nodes = self.reorder_nodes(old_order, self.input)
        new_map = {}
        for nname in ordered_nodes:
            new_map[nname] = self.nodemap[nname]
        self.nodemap = new_map
        self.rawgraph = self.make_graph(self.nodemap.keys(), 'reordered', self.input, self.output)

    def reorder_nodes(self, nodenames, itnames):
        tensor_consumed = []
        tensor_produced = []
        nextnodes = []
        reorderednode = []
        search_flag = {}
        for name in itnames:
            for consumer in self.consumedby[name]:
                if consumer in nodenames:
                    if consumer not in nextnodes:
                        search_flag[consumer] = True
                        nextnodes.append(consumer)
            tensor_produced.append(name)

        for node in nodenames:
            if self.is_node_constant(node):
                dummy_node = True
                for output in self.nodemap[node].output:
                    if output in self.consumedby.keys():
                        for consumer in self.consumedby[output]:
                            if consumer in nodenames:
                                if consumer not in nextnodes:
                                    search_flag[consumer] = True
                                    nextnodes.append(consumer)
                        dummy_node = False
                    else:
                        if dummy_node:
                            if output in self.output:
                                dummy_node = False
                    tensor_produced.append(output)
                if not dummy_node:
                    reorderednode.append(node)

        while len(nextnodes):
            execnodes = []
            for node in nextnodes:
                produced = True
                for input in self.nodemap[node].input:
                    if len(input) == 0:
                        continue
                    if input in self.initials:
                        continue
                    if input not in tensor_produced:
                        produced = False
                        break
                if produced:
                    execnodes.append(node)

            newnodes = []
            for node in nextnodes:
                if node not in execnodes:
                    newnodes.append(node)

            reorderednode.extend(execnodes)
            for node in execnodes:
                for input in self.nodemap[node].input:
                    if input in self.initials:
                        continue
                    tensor_consumed.append(input)
                for output in self.nodemap[node].output:
                    tensor_produced.append(output)
                    if output in self.consumedby:
                        for consumer in self.consumedby[output]:
                            if consumer in nodenames:
                                if consumer in search_flag:
                                    continue
                                newnodes.append(consumer)
                                search_flag[consumer] = True
            nextnodes = set(newnodes)

        return reorderednode

    def get_iotensors(self, nodenames, remove_initials=True):
        intensors = []
        outtensors = []
        for name in nodenames:
            for input in self.nodemap[name].input:
                if len(input) == 0:
                    continue
                if remove_initials and input in self.initials:
                    continue
                if input in self.producedby:
                    producers = self.producedby[input]
                    inner = True
                    for producer in producers:
                        if producer not in nodenames:
                            inner = False
                            break
                    if inner:
                        continue
                if input not in intensors:

                    intensors.append(input)

            for output in self.nodemap[name].output:
                if remove_initials and output in self.initials:
                    continue
                if output in self.consumedby:
                    consumers = self.consumedby[output]
                    inner = True
                    for consumer in consumers:
                        if consumer not in nodenames:
                            inner = False
                            break
                    if inner:
                        continue
                if output not in outtensors:
                    outtensors.append(output)

        return intensors, outtensors

    def update_input_by_map(self, inputs: {}):
        for key in inputs.keys():
            if key in self.tensormap.keys():
                self.tensormap[key].update_tensor(inputs[key])

    def get_dynamic_tensors(self):
        dtensors = {}
        import copy
        for key in self.dynamics:
            dtensors[key] = copy.deepcopy(self.tensormap[key])
        return dtensors

    def shape_infer(self, inputs: {} = None):
        if inputs is not None:
            self.update_input_by_map(inputs)
        self.shapeinfer_optime_map = {}
        from .utils import timer
        tm = timer()
        for key in self.nodemap.keys():
            tm.start()
            node = self.nodemap[key]
            if node.shape_calc:
                itensors = []
                for input in node.input:
                    if self.tensormap[input].numpy is None:
                        assert (node.op_type in ('Shape', 'Slice'))
                        itensors.append(self.tensormap[input].get_shape())
                    else:
                        itensors.append(self.tensormap[input].numpy)
                otensors = node.value_infer(itensors)
                if len(otensors) > 0:
                    for i, output in enumerate(node.output):
                        self.tensormap[output].update_tensor(otensors[i])
            else:
                itensors = []
                for input in node.input:
                    if self.tensormap[input].numpy is not None:
                        itensors.append(self.tensormap[input].numpy)
                    else:
                        itensors.append(self.tensormap[input].get_shape())
                oshapes = node.shape_infer(itensors)
                if len(oshapes) > 0:
                    for i, output in enumerate(node.output):
                        self.tensormap[output].update_shape(oshapes[i])
            if node.op_type in self.shapeinfer_optime_map.keys():
                self.shapeinfer_optime_map[node.op_type] += tm.stop()
            else:
                self.shapeinfer_optime_map[node.op_type] = tm.stop()
        self.log(self.shapeinfer_optime_map)

    def value_infer(self, inputs: {}):
        self.update_input_by_map(inputs)
        for key in self.nodemap.keys():
            node = self.nodemap[key]
            itensors = []
            for input in node.input:
                itensors.append(self.tensormap[input].numpy)
            otensors = node.value_infer(itensors)
            for i, output in enumerate(node.output):
                self.tensormap[output].update_tensor(otensors[i])
        outputs = []
        for output in self.output:
            outputs.append(self.tensormap[output].numpy)
        return outputs

    def add_dump_tensors(self, dump_tensor_names: []):
        for name in dump_tensor_names:
            if name in self.tensormap.keys():
                self.output.append(name)

    def shape_regress(self, min_inputshape: {}, max_inputshape: {}):

        self.shape_infer(min_inputshape)
        mintensormap = self.get_dynamic_tensors()
        self.shape_infer(max_inputshape)
        maxtensormap = self.get_dynamic_tensors()
        input_dynamic_axis = {}
        variable_count = 0
        for key in min_inputshape.keys():
            min_shape = min_inputshape[key].shape
            max_shape = max_inputshape[key].shape
            flag = []
            for mins, maxs in zip(min_shape, max_shape):
                if mins == maxs:
                    flag.append(False)
                else:
                    flag.append(True)
                    variable_count += 1
            input_dynamic_axis[key] = flag
        print(input_dynamic_axis)

    def profile(self):
        params_flag_map = {}
        for key in self.initials:
            params_flag_map[key] = 0

        self.macs = 0.0
        self.params = 0
        self.memory = 0
        for key in self.nodemap.keys():
            node = self.nodemap[key]
            itensors = []
            _params = 0
            _memory = 0
            max_sparsity = 0
            block_sparsity = {'blocksize': (1, 1), 'blockratio': 0, 'ratio': 0}
            for input in node.input:
                tensor = self.tensormap[input]
                itensors.append(tensor.get_valueorshape())
                if input in self.initials:
                    if params_flag_map[input] == 0:
                        elesize = volume(self.tensormap[input].get_shape())
                        _params += elesize
                        _memory += elesize * self.tensormap[input].get_elementsize()
                    params_flag_map[input] += 1
                if tensor.sparsity is not None and tensor.sparsity['ratio'] > max_sparsity:
                    max_sparsity = tensor.sparsity['ratio']
                    block_sparsity = tensor.sparsity
            otensors = []
            for output in node.output:
                if self.tensormap[output].numpy is not None:
                    otensors.append(self.tensormap[output].numpy)
                else:
                    otensors.append(self.tensormap[output].get_shape())
                if node.op_type == 'Constant':
                    # Constant's output tensors are already counted as weight tensors
                    continue
                _memory += self.tensormap[output].get_memsize()
            macs = node.profile(itensors, otensors)
            outshape = (0,)
            if len(node.output) > 0:
                outshape = self.tensormap[node.output[0]].get_shape()
                outshape = (0,) if len(outshape) == 0 else outshape
            inshape = (0,)
            if len(node.input) > 0:
                inshape = self.tensormap[node.input[0]].get_shape()
                inshape = (0,) if len(inshape) == 0 else inshape

            node.macs = macs
            node.inshape = inshape
            node.outshape = outshape
            node.params = _params
            node.memory = _memory
            node.sparsity = block_sparsity
            self.macs += macs
            self.params += _params
            self.memory += _memory

    def print_node_map(self, f: str = None, metric='MACs', exclude_nodes=None):
        from tabulate import tabulate
        assert (metric in ['MACs', 'FLOPs'])
        print_sparse_table = self.sparse_model
        saveformat = 'txt'
        splitch = 'x'

        if f is not None and '.csv' in f:
            saveformat = 'csv'
            csvformat = True
        else:
            csvformat = False

        ptable = []

        macs = int(round(self.macs))
        params = int(self.params)
        memory = int(self.memory)

        shared_size = 0
        for key in self.tensormap.keys():
            if key in self.initials:
                if key in self.consumedby.keys() and len(self.consumedby[key]) > 1:
                    tensor = self.tensormap[key]
                    shared_size += volume(tensor.get_shape())

        if shared_size > 1024:
            print()
            print('*' * 64)
            print(f'Please note that Weight Tensors Sharing is detected:')
            for key in self.tensormap.keys():
                if key in self.initials:
                    if key in self.consumedby.keys() and len(self.consumedby[key]) > 1:
                        print(f'Tensor:{key} ')
                        print('Shared by: ')
                        for node in self.consumedby[key]:
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
        for key in self.nodemap.keys():
            node = self.nodemap[key]
            if exclude_nodes is not None and node.op_type in exclude_nodes:
                continue
            row = [key]
            if print_sparse_table:
                sparsity = node.sparsity
                row.append(tuple2str(sparsity['blocksize'], splitch))
                row.append('{:.2%}'.format(sparsity['blockratio']))
                row.append('{:.2%}'.format(sparsity['ratio']))
            row.append(num2str(int(node.macs) * factor, csvformat))
            row.append('{:.2%}'.format(node.macs / macs))
            row.append(num2str(int(node.memory), csvformat))
            row.append('{:.2%}'.format(node.memory / memory))
            row.append(num2str(int(node.params), csvformat))
            row.append('{:.2%}'.format(node.params / params))
            row.append(tuple2str(node.inshape, splitch))
            row.append(tuple2str(node.outshape, splitch))

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

if __name__ == '__main__':
    f = 'data/public/bertsquad-12.onnx'
    f = 'data/public/resnet18-v1-7_shapes.onnx'
    # f='data/public/rvm_mobilenetv3_fp32.onnx'
    m = onnx.load(f)
    graph = Graph(m.graph)
    graph.get_subgraph(['resnetv15_stage4_conv0_fwd'], ['resnetv15_stage4_batchnorm1_fwd'])
    graph.fuse_subgraph_iotensors(['resnetv15_stage3_activation1'], ['resnetv15_stage4__plus0'], 'fused', 'fused_0')

    # graph.get_subgraph(['393'],['601'])
    # graph.get_subgraph(['bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add_1:0'],['bert/encoder/layer_2/output/add:0'])
    print(graph)
