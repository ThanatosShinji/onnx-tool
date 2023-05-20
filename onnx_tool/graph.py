import copy
import math
import warnings

import numpy
import onnx

import onnx_tool
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


class ValueExpr():
    def __init__(self, srcrange: [], dstrange: []):
        self.alpha = 1
        self.beta = 0
        self.factor = 0
        self.truncmode = 0
        self.build_expr(srcrange, dstrange)

    def error(self, srcrange, dstrange):
        err = 0
        for sval, dval in zip(srcrange, dstrange):
            err += abs(self.__call__(sval) - dval)
        return err

    def build_expr(self, srcrange: [], dstrange: []):
        lastidx = len(srcrange) - 1
        srclen = srcrange[lastidx] - srcrange[0]
        dstlen = dstrange[lastidx] - dstrange[0]
        if srclen <= dstlen:
            self.factor = 0
            self.alpha = round(dstlen / srclen)
            self.beta = round(dstrange[lastidx] - srcrange[lastidx] * self.alpha)
        else:
            self.factor = 1
            self.alpha = round(srclen / dstlen)
            self.beta = round(dstrange[lastidx] - srcrange[lastidx] / self.alpha)

        test = [self.__call__(x) for x in srcrange]
        diff = 0
        for x0, x1 in zip(test, dstrange):
            diff += x0 - x1
        if diff != 0:
            self.truncmode = 1

    def __call__(self, x):
        if self.factor == 0:
            y = self.alpha * x + self.beta
        else:
            y = x / self.alpha + self.beta
        return self.truncate(y)

    def truncate(self, x):
        if self.truncmode == 0:
            return math.ceil(x)
        else:
            return math.floor(x)


class ShapeEngine():
    def __init__(self, input_desc):
        self.input_desc = input_desc
        self.variables = {}
        self.tensor_desc = {}
        self.tensor_epxr = {}
        for key in input_desc.keys():
            self.add_tensor_desc(key, input_desc[key])

    def update_variable(self, key, val):
        self.variables[key] = val

    def __get_shape_from_desc__(self, desc):
        shape = []
        for val in desc:
            if isinstance(val, int):
                shape.append(val)
            else:
                shape.append(self.variables[val])
        return shape

    def add_tensor_desc(self, tensor, desc):
        self.tensor_desc[tensor] = desc

    def update_tensor_desc(self, tensor, axis, vkey):
        desc = self.tensor_desc[tensor]
        assert (not isinstance(desc[axis], int))
        desc[axis] = vkey

    def get_tensor_desc(self, tensor):
        if tensor not in self.tensor_desc.keys():
            return None
        return self.tensor_desc[tensor]

    def add_expr(self, var_name, src_name, expr):
        self.tensor_epxr[var_name] = [src_name, expr]
        self.variables[var_name] = 0

    def update_variables(self):
        for key in self.tensor_epxr.keys():
            expr = self.tensor_epxr[key]
            self.variables[key] = expr[1](self.variables[expr[0]])

    def get_tensorshape(self, tname):
        desc = self.tensor_desc[tname]
        shape = self.__get_shape_from_desc__(desc)
        return shape

    def generate_input(self):
        tmp_input = {}
        for key in self.input_desc:
            shape = self.get_tensorshape(key)
            tmp_input[key] = numpy.zeros(shape)
        return tmp_input


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
        self.valid_shape = False
        self.valid_profile = False

    def log(self, str):
        if self.verbose:
            print(str)

    def remove_constant(self):
        rmlist = []
        for key in self.nodemap:
            if self.nodemap[key].op_type == 'Constant':
                rmlist.append(key)

        for key in rmlist:
            self.nodemap.pop(key)

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
                if input not in self.initials:
                    if input not in self.dynamics:
                        self.dynamics.append(input)
                    if input not in self.tensormap.keys():
                        self.tensormap[input] = Tensor(input)

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
                        if node.op_type == 'Shape':
                            continue
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

    def add_initial(self, name, data):
        from .tensor import create_initial_Tensor
        self.initials.add(name)
        if isinstance(data, numpy.ndarray):
            self.tensormap[name] = create_initial_Tensor(name, data)
        else:
            raise NotImplementedError('unsupported data type')

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

    def get_initials_from_nodenames(self, nodenames):
        initializer = []
        for name in nodenames:
            for input in self.nodemap[name].input:
                if input in self.initials:
                    initializer.append(self.tensormap[input].proto)
        return initializer

    def remove_node(self, nodename):
        node = self.nodemap[nodename]
        # update producer
        for o in node.output:
            self.producedby[o].remove(nodename)
            if len(self.producedby[o]) == 0:
                self.producedby.pop(o)
        # update consumer
        for i in node.input:
            if i in self.consumedby.keys():
                self.consumedby[i].remove(nodename)
        self.nodemap.pop(nodename)

    def skip_node(self, nodename):
        node = self.nodemap[nodename]
        count = 0
        for input in node.input:
            if input in self.dynamics:
                count += 1
                indtensor = input
        if count == 1 and len(node.output) == 1:
            self.consumedby[indtensor].remove(nodename)
            for con in self.consumedby[node.output[0]]:
                con_node = self.nodemap[con]
                for i in range(len(con_node.input)):
                    if con_node.input[i] == node.output[0]:
                        con_node.input[i] = indtensor
                        self.consumedby[indtensor].append(con)
                        break

    def fuse_subgraph_node_names(self, nodes: [str], nodeop: str, nodename: str, keep_attr=True):
        _inputs, _outputs = self.get_iotensors(nodes, remove_initials=False)
        newnode = onnx.helper.make_node(nodeop, _inputs, _outputs, name=nodename)
        count = 0
        if keep_attr:
            for node in nodes:
                for attribute in self.nodemap[node].proto.attribute:
                    attr = onnx.helper.make_attribute(
                        self.nodemap[node].proto.op_type + str(count) + '_' + attribute.name,
                        get_attribute_data(attribute))
                    newnode.attribute.append(attr)
                count += 1
        from .node import Node
        for name in nodes:
            self.remove_node(name)
        newnode = Node(newnode)
        newnode.input = _inputs
        newnode.output = _outputs
        for i in _inputs:
            self.consumedby[i].append(nodename)
            if i in self.producedby.keys():
                newnode.prevnodes.append(self.producedby[i])
        for o in _outputs:
            self.producedby[o] = [nodename]
            if o in self.consumedby.keys():
                newnode.nextnodes.append(self.consumedby[o])
        self.nodemap[nodename] = newnode

    def fuse_postop_node_names(self, nodes: [str], append_attr: bool, **extrakeys):
        _inputs, _outputs = self.get_iotensors(nodes, remove_initials=False)
        mainnode = self.nodemap[nodes[0]]
        newnode = onnx.helper.make_node(mainnode.op_type, _inputs, _outputs, name=mainnode.name)
        count = 0
        for attr in mainnode.proto.attribute:
            if attr.name == 'postop_count':
                count = onnx.helper.get_attribute_value(attr)
            else:
                newnode.attribute.append(attr)

        for key in extrakeys:
            attr = onnx.helper.make_attribute(key, extrakeys[key])
            newnode.attribute.append(attr)

        if append_attr:
            for nname in nodes:
                if nname == mainnode.name:
                    continue
                pnode = self.nodemap[nname]
                attr = onnx.helper.make_attribute(
                    'postop_' + str(count), pnode.op_type)
                newnode.attribute.append(attr)
                for attribute in pnode.proto.attribute:
                    attr = onnx.helper.make_attribute(
                        'postop_' + str(count) + '_' + attribute.name,
                        get_attribute_data(attribute))
                    newnode.attribute.append(attr)
                count += 1

        attr = onnx.helper.make_attribute('postop_count', count)
        newnode.attribute.append(attr)

        for name in nodes:
            self.remove_node(name)
        newnode = create_node(newnode)
        newnode.input = _inputs
        newnode.output = _outputs
        for i in _inputs:
            if i in self.consumedby.keys():
                self.consumedby[i].append(mainnode.name)
            if i in self.producedby.keys():
                newnode.prevnodes.append(self.producedby[i])
        for o in _outputs:
            self.producedby[o] = [mainnode.name]
            if o in self.consumedby.keys():
                newnode.nextnodes.append(self.consumedby[o])
        self.nodemap[mainnode.name] = newnode

    def fuse_subgraph_iotensors(self, inputs: [], outputs: [], nodeop: str, name: str, keep_attr=True):
        _, nodes, _ = self.__get_subnodes_byio__(inputs, outputs)
        _inputs, _outputs = self.get_iotensors(nodes, remove_initials=True)
        nodes = self.reorder_nodes(nodes, _inputs)
        return self.fuse_subgraph_node_names(nodes, nodeop, name, keep_attr)

    def get_onnxgraph_by_nodenames(self, nodenames):
        if len(nodenames):
            _inputs0, _outputs0 = self.get_iotensors(nodenames)
            graph_level0 = self.reorder_nodes(nodenames, _inputs0)
            subgraph = self.make_graph_onnx(graph_level0, 'subgraph', _inputs0, _outputs0)
            return subgraph
        return None

    def save_model(self, f: str, shape_only: bool = False, rawmodel: onnx.ModelProto = None):
        graph = self.make_graph_onnx(self.nodemap.keys(), 'graph', self.input, self.output, not shape_only)
        if graph is not None and f is not None:
            attr = {}
            attr['producer_name'] = 'onnx_tool'
            attr['producer_version'] = 'v' + VERSION
            model = onnx.helper.make_model(graph, **attr)
            if rawmodel is not None:
                model.ir_version = rawmodel.ir_version
                model.opset_import[0].version = rawmodel.opset_import[0].version
            onnx.save_model(model, f)

    def make_graph_onnx(self, nodenames, gname, inputnames, outputnames, with_initializer=True):
        nodes = []
        for name in nodenames:
            nodes.append(self.nodemap[name].make_nodeproto())

        initializer = None
        if with_initializer:
            initializer = self.get_initials_from_nodenames(nodenames)

        inputs = []
        outputs = []
        for name in inputnames:
            if name in self.tensormap:
                inputs.append(self.tensormap[name].make_value_proto())
            else:
                inputs.append(onnx.helper.make_tensor_value_info(name, 1, None))
        for name in outputnames:
            if name in self.tensormap:
                proto = self.tensormap[name].make_value_proto(make_dummy=True)
                outputs.append(proto)
        value_infos = []
        for key in self.dynamics:
            if key in self.input or key in self.output:
                continue
            tensor = self.tensormap[key]
            vinfo = tensor.make_value_proto()
            if vinfo is None:
                continue
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
        self.rawgraph = self.make_graph_onnx(self.nodemap.keys(), 'reordered', self.input, self.output)

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
                    if input not in self.producedby:
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

    def check_inputs(self):
        for name in self.input:
            shape = self.tensormap[name].shape
            for val in shape:
                if isinstance(val, str):
                    return False, name
                if val < 0:
                    return False, name
        return True, None

    def shape_infer(self, inputs: {} = None):
        self.valid_shape = False
        if inputs is not None:
            self.update_input_by_map(inputs)
        in_valid, tname = self.check_inputs()
        if not in_valid:
            raise ValueError(
                f"The input tensor {tname}'s shape {self.tensormap[tname].shape2str()} is not valid, Please set it to a valid shape.")
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
        self.valid_shape = True

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
                if name not in self.output:
                    self.output.append(name)

    def shape_regress(self, input_desc: {}, input_range: {}):
        shapeengine = ShapeEngine(input_desc)

        for key in input_range.keys():
            shapeengine.update_variable(key, input_range[key][1])
        tmp_input = shapeengine.generate_input()
        self.shape_infer(tmp_input)
        maxtensormap = self.get_dynamic_tensors()

        for key in input_range.keys():
            shapeengine.update_variable(key, input_range[key][0])
        tmp_input = shapeengine.generate_input()
        self.shape_infer(tmp_input)
        mintensormap = self.get_dynamic_tensors()

        for key in mintensormap.keys():
            if key in input_desc.keys():
                continue
            shape_desc = []
            minshape = mintensormap[key].shape
            maxshape = maxtensormap[key].shape
            for i, a in zip(minshape, maxshape):
                if i == a:
                    shape_desc.append(i)
                else:
                    shape_desc.append('')
            shapeengine.add_tensor_desc(key, shape_desc)

        vcount = 0
        for key in input_range.keys():
            minv = input_range[key][0]
            maxv = input_range[key][1]
            vranges = [(minv + maxv) // 2, maxv]
            shapes_range = []

            for val in vranges:
                shapeengine.update_variable(key, val)
                tmpinputs = shapeengine.generate_input()
                self.shape_infer(tmpinputs)
                shapes_range.append(self.get_dynamic_tensors())

            shapeengine.update_variable(key, input_range[key][0])
            srcrange = [minv, ] + vranges
            dstranges = [srcrange]
            tensor_range_map = {}

            def range_in_rangelist(range, rangelist):
                for i, r in enumerate(rangelist):
                    flag = True
                    for sv, dv in zip(range, r):
                        if sv != dv:
                            flag = False
                            break
                    if flag:
                        return i
                return -1

            for vkey in mintensormap.keys():
                if vkey in input_desc.keys():
                    continue
                shapes = [mintensormap[vkey].shape, shapes_range[0][vkey].shape, shapes_range[1][vkey].shape]
                for i in range(len(shapes[0])):
                    if shapes[0][i] != shapes[1][i] or shapes[0][i] != shapes[2][i]:
                        newrange = [val[i] for val in shapes]
                        idx = range_in_rangelist(newrange, dstranges)
                        if idx == -1:
                            nodename = self.producedby[vkey][0]
                            srcvalkey = []
                            for iname in self.nodemap[nodename].input:
                                desc = shapeengine.get_tensor_desc(iname)
                                if desc is not None:
                                    for ele in desc:
                                        if not isinstance(ele, int) and ele.isnumeric():
                                            if int(ele) >= vcount:
                                                srcvalkey.append(ele)
                            dstranges.append(newrange)
                            vidx = vcount + len(dstranges) - 1
                            if len(srcvalkey) == 0:
                                expr = ValueExpr(dstranges[0], newrange)
                                err = expr.error(dstranges[0], newrange)
                                srckey = key
                            else:
                                # TODO multiple src variables comparison
                                srckey = srcvalkey[0]
                                expr = ValueExpr(tensor_range_map[srckey], newrange)
                                err = expr.error(tensor_range_map[srckey], newrange)
                            if err > 0:
                                warnings.warn(f'src key {srckey} dst key {vidx} error:{err}')

                            shapeengine.add_expr(str(vidx), srckey, expr)
                            tensor_range_map[str(vidx)] = newrange
                            valkey = str(vidx)
                        else:
                            if idx == 0:
                                valkey = key
                            else:
                                vidx = vcount + idx
                                valkey = str(vidx)
                        shapeengine.update_tensor_desc(vkey, i, valkey)
            vcount += len(dstranges)
        return shapeengine

    def compress_memory(self, size_padding=64):
        self.remove_constant()
        tensor_in_mem = copy.deepcopy(self.input)
        tensor_mem_per_node = []
        tensor_consumed = {}
        for node in self.nodemap.keys():
            node_ = self.nodemap[node]
            for name in node_.output:
                tensor_in_mem.append(name)
            new_list = copy.deepcopy(tensor_in_mem)
            tensor_mem_per_node.append(new_list)
            for name in node_.input:
                if name in self.consumedby:
                    if name in tensor_consumed:
                        tensor_consumed[name].append(node)
                    else:
                        tensor_consumed[name] = [node]
                    consumers = self.consumedby[name]
                    if len(tensor_consumed[name]) == len(consumers):
                        if name in tensor_in_mem:
                            tensor_in_mem.remove(name)
        for output in self.output:
            if output not in tensor_in_mem:
                warnings.warn(f'tensor list is wrong, {output} is missing!')
        # print(tensor_mem_per_node)
        compress_mem = {}
        mem_tags = []
        mem_block = []
        for nodetensors in tensor_mem_per_node:
            # clear mem tags
            for i, tag in enumerate(mem_tags):
                if tag == "":
                    continue
                if tag not in nodetensors:
                    premerge = False
                    if i >= 1 and mem_tags[i - 1] == "":
                        premerge = True
                    nextmerge = False
                    if i < len(mem_tags) - 1 and mem_tags[i + 1] == "":
                        nextmerge = True
                    if premerge and nextmerge:
                        newblock = [mem_block[i - 1][0], mem_block[i + 1][0] + mem_block[i + 1][1]
                                    - mem_block[i - 1][0]]
                        mem_tags.pop(i)
                        mem_tags.pop(i)
                        mem_block.pop(i)
                        mem_block.pop(i)
                        mem_tags[i - 1] = ""
                        mem_block[i - 1] = newblock
                    elif premerge:
                        newblock = [mem_block[i - 1][0], mem_block[i][0] + mem_block[i][1]
                                    - mem_block[i - 1][0]]
                        mem_tags.pop(i)
                        mem_block.pop(i)
                        mem_tags[i - 1] = ""
                        mem_block[i - 1] = newblock
                    elif nextmerge:
                        newblock = [mem_block[i][0], mem_block[i + 1][0] + mem_block[i + 1][1]
                                    - mem_block[i][0]]
                        mem_tags.pop(i)
                        mem_block.pop(i)
                        mem_tags[i] = ""
                        mem_block[i] = newblock
                    else:
                        mem_tags[i] = ""

            # push tensor mem to block
            for tname in nodetensors:
                t_ = self.tensormap[tname]
                size_ = t_.get_memsize()
                size_ = int((size_ + size_padding - 1) // size_padding * size_padding)
                if tname in mem_tags:
                    continue
                block_found = False
                for i, tag in enumerate(mem_tags):
                    if tag == "":
                        if mem_block[i][1] >= size_:
                            remain_size = mem_block[i][1] - size_
                            if remain_size > 1024 * 1024:
                                remain_block = [mem_block[i][0] + size_, remain_size]
                                mem_tags[i] = tname
                                mem_block[i][1] = size_
                                mem_tags.insert(i + 1, "")
                                mem_block.insert(i + 1, remain_block)
                            else:
                                mem_tags[i] = tname
                            block_found = True
                            break
                if not block_found:
                    lastidx = len(mem_tags) - 1
                    if lastidx >= 0 and mem_tags[lastidx] == "":
                        mem_block[lastidx][1] = size_
                        mem_tags[lastidx] = tname
                    else:
                        mem_tags.append(tname)
                        if lastidx >= 0:
                            mem_block.append([mem_block[lastidx][0] + mem_block[lastidx][1], size_])
                        else:
                            mem_block.append([0, size_])

                idx = mem_tags.index(tname)
                compress_mem[tname] = mem_block[idx]

        lastblock = mem_block[-1]
        compress_size = lastblock[0] + lastblock[1]
        # print(compress_mem)
        # validate memory
        for nodetensors in tensor_mem_per_node:
            maxmem = 0
            overlap = False
            for tname in nodetensors:
                block = compress_mem[tname]
                if block[0] + block[1] > maxmem:
                    maxmem = block[0] + block[1]
            for tname in nodetensors:
                block0 = compress_mem[tname]
                for tname1 in nodetensors:
                    if tname1 == tname:
                        continue
                    block1 = compress_mem[tname1]
                    if block0[0] >= block1[0] and block0[0] < block1[0] + block1[1]:
                        overlap = True
                        laptensor = [tname, tname1]
                        break
                    if block1[0] >= block0[0] and block1[0] < block0[0] + block0[1]:
                        overlap = True
                        laptensor = [tname, tname1]
                        break
            if maxmem > compress_size:
                warnings.warn(f'Wrong compress total memory size:{compress_size}. larger size detected:{maxmem}')
            if overlap:
                warnings.warn(f'Memory overlap detected!{laptensor[0]} v.s. {laptensor[1]}')

        raw_memsize = 0
        for tname in self.dynamics:
            if tname in self.input or tname in self.output:
                continue
            raw_memsize += self.tensormap[tname].get_memsize()

        print(f"Raw memory size (without parameter memory): {raw_memsize:,} bytes")
        print(f"Compressed memory size: {compress_size:,} bytes")
        print(f'Comression ratio: {compress_size / raw_memsize * 100:.3f}%')
        return compress_mem, compress_size

    def get_compute_graph_onnx(self):
        nodes = []
        for key in self.nodemap.keys():
            node = self.nodemap[key]
            if node.shape_calc:
                continue
            dummy_node = True
            for output in node.output:
                dummy = True
                if output in self.consumedby.keys():
                    for consumer in self.consumedby[output]:
                        if not self.nodemap[consumer].shape_calc:
                            dummy = False
                            break
                else:
                    dummy = False
                dummy_node = dummy_node and dummy
            if dummy_node:
                continue
            nodes.append(node.name)

        _inputs0, _outputs0 = self.get_iotensors(nodes)
        graph_level0 = self.reorder_nodes(nodes, _inputs0)
        subgraph = self.make_graph_onnx(graph_level0, 'compute_graph', self.input, self.output)
        return subgraph

    def get_compute_graph(self):
        cg = copy.copy(self)
        nodes = []
        rmnodes = []
        for key in cg.nodemap.keys():
            node = cg.nodemap[key]
            if node.shape_calc:
                rmnodes.append(key)
                continue
            dummy_node = True
            for output in node.output:
                dummy = True
                if output in cg.consumedby.keys():
                    for consumer in cg.consumedby[output]:
                        if not cg.nodemap[consumer].shape_calc:
                            dummy = False
                            break
                else:
                    dummy = False
                dummy_node = dummy_node and dummy
            if dummy_node:
                rmnodes.append(key)
                continue
            nodes.append(node.name)
        for key in rmnodes:
            cg.remove_node(key)
        cg.graph_reorder()
        cg.dynamics = []
        cg.dynamics.extend(cg.input)
        cg.dynamics.extend(cg.output)
        for name in cg.nodemap.keys():
            for output in cg.nodemap[name].output:
                if name not in cg.dynamics:
                    cg.dynamics.append(output)
        return cg

    def profile(self):
        self.valid_profile = False
        if not self.valid_shape:
            warnings.warn('Please perform a valid shape_infer() before profile().')
            return
        params_flag_map = {}
        for key in self.initials:
            params_flag_map[key] = 0

        self.macs = 0.0
        self.params = 0
        self.memory = 0
        tensors=[]
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
                tensors.append(output)
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

        self.valid_profile = True

    def print_node_map(self, f: str = None, metric='MACs', exclude_nodes=None):
        if not self.valid_profile:
            warnings.warn('Please perform a valid profile() before print_node_map().')
            return
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
            row = [key, self.nodemap[key].op_type]
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
        row = ['Total', '_']
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
        header = ['Name', 'Type']
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
