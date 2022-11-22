import warnings

import numpy
import onnx
from .node import create_node
from .tensor import get_attribute_data, tensorproto2ndarray, shape_of_tensor, is_valid_ndarray
from .utils import VERSION


def __shape_of_initializer__(initial):
    shape = []
    # for nb in tensor.shape.dim
    for nb in initial.dims:
        shape.append(nb)
    return shape


class Tensor():
    def __init__(self, t):
        if isinstance(t, str):
            self.name = t
            self.proto = None
            self.shape = []
            self.numpy = None
        elif isinstance(t, onnx.ValueInfoProto):
            self.name = t.name
            self.proto = t
            self.shape = shape_of_tensor(t)
            self.numpy = None
        elif isinstance(t, onnx.TensorProto):
            self.name = t.name
            self.proto = t
            self.numpy = tensorproto2ndarray(t)
            self.shape = self.numpy.shape
        elif isinstance(t, onnx.NodeProto):
            if t.op_type == 'Constant':
                self.name = t.output[0]
                for att in t.attribute:
                    if att.name == 'value':
                        self.numpy = get_attribute_data(att)
                        if not is_valid_ndarray(self.numpy):
                            self.numpy = None
                        if self.numpy is not None:
                            self.shape = self.numpy.shape
                        else:
                            self.shape = []

    def update_tensor(self, data: numpy.ndarray):
        self.numpy = data
        self.shape = data.shape

    def update_shape(self, shape: numpy.ndarray):
        if isinstance(shape, numpy.ndarray):
            print("111")
        self.shape = shape

    def get_shape(self):
        return self.shape


_SHAPE_TENSORS = {
    'Reshape': ('1of2',),
    'Resize': ('2of3', '3of4'),
    'Slice': ('1,2of3', '1,2,3of4', '1,2,3,4of5')
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


#
# class Node():
#     def __init__(self, n: onnx.NodeProto):
#         self.name = n.name
#         self.op_type = n.op_type
#         self.nextnodes = []
#         self.prevnodes = []
#         self.output = []
#         self.input = []
#         self.proto = n
#         self.shape_calc = False


class Graph():
    def __init__(self, g: onnx.GraphProto):
        self.nodemap = {}
        self.tensormap = {}
        self.producedby = {}
        self.consumedby = {}
        self.rawgraph = g
        self.initials = []
        self.dynamics = []
        self.input = []
        self.output = []
        self.__init_graph_from_onnxproto__(g)
        self.__find_shape_tensors__()

    def __init_graph_from_onnxproto__(self, g):
        if g is None:
            return

        for node in g.node:
            newnode = create_node(node)

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

        for node in g.node:
            for tensor in node.output:
                if tensor in self.consumedby:
                    for consumer in self.consumedby[tensor]:
                        self.nodemap[node.name].nextnodes.append(self.nodemap[consumer])

        for input in g.input:
            tensor = Tensor(input)
            self.tensormap[input.name] = tensor

        for output in g.output:
            tensor = Tensor(output)
            self.tensormap[output.name] = tensor

        for valinfo in g.value_info:
            tensor = Tensor(valinfo)
            self.tensormap[valinfo.name] = tensor

        for initial in g.initializer:
            self.initials.append(initial.name)
            tensor = Tensor(initial)
            self.tensormap[initial.name] = tensor

        for node in g.node:
            if node.op_type == 'Constant':
                tensor = Tensor(node)
                self.tensormap[node.output[0]] = tensor
                self.initials.append(node.output[0])
        self.initials = set(self.initials)

        for key in self.nodemap.keys():
            node = self.nodemap[key]
            for input in node.input:
                if input not in self.initials:
                    self.dynamics.append(input)
                    self.tensormap[input] = Tensor(input)

        for t in g.input:
            self.input.append(t.name)
            self.dynamics.append(t.name)

        for t in g.output:
            self.output.append(t.name)
            self.dynamics.append(t.name)

        self.dynamics = set(self.dynamics)

    def __find_shape_tensors__(self):
        self.shape_tensors = []
        shape_calc_nodes = []
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
                        shape_calc_nodes.append(nname)
                        if node.op_type == 'Shape':
                            continue
                        for input in node.input:
                            if input not in self.initials and input in self.producedby.keys():
                                nextnodes.extend(self.producedby[input])
                    searchnodes = nextnodes
        # print(shape_calc_nodes)
        shape_calc_nodes = set(shape_calc_nodes)

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

        # print(len(self.nodemap.keys()))
        # print(len(graph_level0))
        # print(len(graph_level1))
        # print(len(graph_level2))
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
        # graph_level0.save_model('graph_level0.onnx')
        # graph_level1.save_model('graph_level1.onnx')
        # graph_level2.save_model('graph_level2.onnx')

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

    def save_model(self, f: str):
        if self.rawgraph is not None and f is not None:
            model = onnx.helper.make_model(self.rawgraph, producer_name='onnx-tool', producer_version='v' + VERSION)
            onnx.save_model(model, f)

    def make_graph(self, nodenames, gname, inputnames, outputnames, with_initializer=True):
        nodes = []
        for name in nodenames:
            nodes.append(self.nodemap[name].proto)

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
                inputs.append(self.tensormap[name].proto)
            else:
                inputs.append(onnx.helper.make_tensor_value_info(name, 1, None))
        for name in outputnames:
            if name in self.tensormap:
                outputs.append(self.tensormap[name].proto)
            else:
                outputs.append(onnx.helper.make_tensor_value_info(name, 1, None))
        graph = onnx.helper.make_graph(nodes=nodes, name=gname, inputs=inputs, outputs=outputs, initializer=initializer)
        return graph

    def is_node_constant(self, node):
        for input in self.nodemap[node].input:
            if input not in self.initials:
                return False
        return True

    def graph_reorder(self):
        return Graph(self.get_onnxgraph_by_nodenames(self.nodemap.keys()))

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

    def shape_infer(self, inputs: {} = None):
        if inputs is not None:
            self.update_input_by_map(inputs)

        for key in self.nodemap.keys():
            node = self.nodemap[key]
            if node.name == 'Add_348':
                print("111")
            if node.shape_calc:
                itensors = []
                for input in node.input:
                    if self.tensormap[input].numpy is None:
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
                    if input in self.shape_tensors:
                        itensors.append(self.tensormap[input].numpy)
                    else:
                        itensors.append(self.tensormap[input].get_shape())
                oshapes = node.shape_infer(itensors)
                if len(oshapes) > 0:
                    for i, output in enumerate(node.output):
                        self.tensormap[output].update_shape(oshapes[i])
        print('done!')

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
