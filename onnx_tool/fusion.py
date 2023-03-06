import numpy
import onnx.helper

import onnx_tool
from .graph import Graph
from .node import Node

ConvBN = [
    {
        'name': 'conv_0',
        'op': 'Conv',
        'attrs': [
            ['kernel_shape', '<=', 0, 7],
            ['kernel_shape', '>=', 0, 1],
            ['kernel_shape', '<=', 1, 7],
            ['kernel_shape', '>=', 1, 1],
        ],
        'inport': [],
        'outport': [[0, 'bn_0', 0]],
    },
    {
        'name': 'bn_0',
        'op': 'BatchNormalization',
        'attrs': [
        ],
        'inport': [[0, 'conv_0', 0]],
        'outport': [],
    },
]

Conv_Act = [
    {
        'name': 'conv_0',
        'op': 'Conv',
        'attrs': [],
        'inport': [],
        'outport': [[0, 'act_0', 0]],
    },
    {
        'name': 'act_0',
        'op': ['Relu', 'LeakyRelu'],
        'attrs': [
        ],
        'inport': [[0, 'conv_0', 0]],
        'outport': [],
    },
]

Conv_Res = [
    {
        'name': 'conv_0',
        'op': 'Conv',
        'attrs': [],
        'inport': [],
        'outport': [[0, 'add_0', 1]],
    },
    {
        'name': 'add_0',
        'op': ['Add', ],
        'attrs': [],
        'inport': [[1, 'conv_0', 0]],
        'outport': [[0, 'act_0', 0]],
    },
    {
        'name': 'act_0',
        'op': ['Relu', 'LeakyRelu'],
        'attrs': [],
        'inport': [[0, 'add_0', 0]],
        'outport': [],
    },
]

MHAint8_Pattern = [
    {
        'name': 'MatMul_0',
        'op': 'QLinearMatMul',
        'attrs': [
        ],
        'inport': [],
        'outport': [[0, 'DequantizeLinea_0', 0]],
    },
    {
        'name': 'DequantizeLinea_0',
        'op': 'DequantizeLinear',
        'attrs': [
        ],
        'inport': [[0, 'MatMul_0', 0]],
        'outport': [[0, 'div_0', 0]],
    },
    {
        'name': 'div_0',
        'op': 'Div',
        'attrs': [
        ],
        'inport': [[0, 'DequantizeLinea_0', 0]],
        'outport': [[0, 'Add_0', 0]],
    },
    {
        'name': 'Add_0',
        'op': 'Add',
        'attrs': [
        ],
        'inport': [[0, 'div_0', 0]],
        'outport': [[0, 'Softmax_0', 0]],
    },
    {
        'name': 'Softmax_0',
        'op': 'Softmax',
        'attrs': [
        ],
        'inport': [[0, 'Add_0', 0]],
        'outport': [[0, 'QuantizeLinear_1', 0]],
    },
    {
        'name': 'QuantizeLinear_1',
        'op': 'QuantizeLinear',
        'attrs': [
        ],
        'inport': [[0, 'Add_0', 0]],
        'outport': [[0, 'MatMul_1', 0]],
    },
    {
        'name': 'MatMul_1',
        'op': 'QLinearMatMul',
        'attrs': [
        ],
        'inport': [[0, 'QuantizeLinear_1', 0]],
        'outport': [],
    },
]

layernorm_pattern = [
    {
        'name': 'ReduceMean_196',
        'op': 'ReduceMean',
        'attrs': [],
        'inport': [],
        'outport': [[0, 'Sub_197', 1]]
    },
    {
        'name': 'Sub_197',
        'op': 'Sub',
        'attrs': [],
        'inport': [[1, 'ReduceMean_196', 0]],
        'outport': [[0, 'Pow_0', 0]]
    },
    {
        'name': 'Pow_0',
        'op': 'Pow',
        'attrs': [],
        'inport': [[0, 'Sub_197', 0]],
        'outport': [[0, 'ReduceMean_0', 0]]
    },
    {
        'name': 'ReduceMean_0',
        'op': 'ReduceMean',
        'attrs': [],
        'inport': [[0, 'Pow_0', 0]],
        'outport': [[0, 'Add_0', 0]]
    },
    {
        'name': 'Add_0',
        'op': 'Add',
        'attrs': [],
        'inport': [[0, 'ReduceMean_0', 0]],
        'outport': [[0, 'Sqrt_0', 0]]
    },
    {
        'name': 'Sqrt_0',
        'op': 'Sqrt',
        'attrs': [],
        'inport': [[0, 'Add_0', 0]],
        'outport': [[0, 'Div_0', 1]]
    },
    {
        'name': 'Div_0',
        'op': 'Div',
        'attrs': [],
        'inport': [[1, 'Sqrt_0', 0]],
        'outport': []
    },
]


class AttrExpr():
    def __init__(self, raw: []):
        self.attrname = raw[0]
        self.expr = raw[1]
        if len(raw) == 4:
            self.idx = raw[2]
            self.num = raw[3]
        else:
            self.num = raw[2]
            self.idx = -1

    def __call__(self, x):
        if hasattr(x, self.attrname):
            if self.idx == -1:
                return self.logical(x.__getattribute__(self.attrname))
            else:
                return self.logical(x.__getattribute__(self.attrname)[self.idx])
        return False

    def logical(self, attr):
        if self.expr == '<=':
            return attr <= self.num
        if self.expr == '<':
            return attr < self.num
        if self.expr == '>=':
            return attr >= self.num
        if self.expr == '>':
            return attr > self.num
        if self.expr == '==':
            return attr == self.num


class NodeCondition():
    def __init__(self, attrmap: {}):
        self.op = attrmap['op']
        self.name = attrmap['name']
        self.attexprs = []
        for att in attrmap['attrs']:
            self.attexprs.append(AttrExpr(att))
        self.outport = attrmap['outport']
        self.inport = attrmap['inport']

    def is_node(self, node: Node):
        flag = True

        if isinstance(self.op, list):
            flag &= node.op_type in self.op
        else:
            if self.op != 'Any':
                flag &= node.op_type == self.op
        for attrexpr in self.attexprs:
            flag &= attrexpr(node)
        return flag


class FusionPattern():
    def __init__(self, nodedescs: {}):
        self.nodedesc = {}
        self.first_key = nodedescs[0]['name']
        for desc in nodedescs:
            self.nodedesc[desc['name']] = NodeCondition(desc)

    def expand_from_descandnode(self, desc, node, graph):
        nodename_to_search = []
        searched_desc = []
        for outset in desc.outport:
            outidx = outset[0]
            outdescky = outset[1]
            next_inidx = outset[2]
            nextdesc = self.nodedesc[outdescky]
            if outidx < len(node.output):
                tname = node.output[outidx]
                if tname in graph.consumedby:
                    consumed_nodes = graph.consumedby[tname]
                    for nodename in consumed_nodes:
                        if nodename in self.found_node_names:
                            continue
                        if nodename in self.tmp_names:
                            continue
                        nodeobject = graph.nodemap[nodename]
                        if nodeobject.input.index(tname) == next_inidx:
                            if nextdesc.is_node(nodeobject):
                                nodename_to_search.append(nodename)
                                searched_desc.append(outdescky)
                                break

        # expand in nodes
        for inset in desc.inport:
            inidx = inset[0]
            indesckey = inset[1]
            prev_outidx = inset[2]
            nextdesc = self.nodedesc[indesckey]
            if inidx < len(node.input):
                tname = node.input[inidx]
                producer_node = graph.producedby[tname]
                for nodename in producer_node:
                    if nodename in self.found_node_names:
                        continue
                    if nodename in self.tmp_names:
                        continue
                    nodeobject = graph.nodemap[nodename]
                    if nodeobject.output.index(tname) == prev_outidx:
                        if nextdesc.is_node(nodeobject):
                            nodename_to_search.append(nodename)
                            searched_desc.append(indesckey)
                            break
        return nodename_to_search, searched_desc

    def find_pattern(self, graph: Graph):
        ls_nodes = []
        first_desc = self.nodedesc[self.first_key]
        self.found_node_names = []
        self.tmp_names = []
        for name in graph.nodemap.keys():
            if name in self.found_node_names:
                continue
            node = graph.nodemap[name]
            if first_desc.is_node(node):
                searched_desc_keys = [self.first_key]
                found_nodes = [name]
                nodename_to_search, descs_to_search = self.expand_from_descandnode(first_desc, node, graph)
                searched_desc_keys.extend(descs_to_search)
                found_nodes.extend(nodename_to_search)
                self.tmp_names = [name]
                self.tmp_names.extend(nodename_to_search)
                while len(nodename_to_search) > 0:
                    next_desc_search = []
                    next_nodes_search = []
                    for desc, nodename in zip(descs_to_search, nodename_to_search):
                        _nodenames, _desc = self.expand_from_descandnode(self.nodedesc[desc], graph.nodemap[nodename],
                                                                         graph)
                        searched_desc_keys.extend(_desc)
                        for name in _nodenames:
                            self.tmp_names.append(name)
                        next_desc_search.extend(_desc)
                        next_nodes_search.extend(_nodenames)
                    descs_to_search = next_desc_search
                    nodename_to_search = next_nodes_search
                    found_nodes.extend(next_nodes_search)

                if len(searched_desc_keys) == len(self.nodedesc.keys()):
                    ls_nodes.append(found_nodes)
                    self.found_node_names.extend(self.tmp_names)
        return ls_nodes


def ConvBNFusion(graph: Graph):
    import math
    pattern = FusionPattern(ConvBN)
    node_sets = pattern.find_pattern(graph)
    for nodes in node_sets:
        conv_node = graph.nodemap[nodes[0]]
        bn_node = graph.nodemap[nodes[1]]
        weight = graph.tensormap[conv_node.input[1]]
        gamma = graph.tensormap[bn_node.input[1]]
        beta = graph.tensormap[bn_node.input[2]]
        mean = graph.tensormap[bn_node.input[3]]
        var = graph.tensormap[bn_node.input[4]]
        wshape = weight.shape
        if len(conv_node.input) == 3:
            hasbias = True
            newbias = graph.nodemap[conv_node.input[2]].numpy
        else:
            hasbias = False
            newbias = numpy.zeros((wshape[0],), dtype=numpy.float32)
        for i in range(wshape[0]):
            sm = gamma.numpy[i] / math.sqrt(var.numpy[i] + bn_node.epsilon)
            sv = beta.numpy[i]
            for j in range(wshape[1]):
                for k in range(wshape[2]):
                    for l in range(wshape[3]):
                        weight.numpy[i, j, k, l] = sm * weight.numpy[i, j, k, l]
            newbias[i] = sm * (newbias[i] - mean.numpy[i]) + sv
        weight.update_proto(weight.numpy)
        newbiasname = conv_node.name + "_bias_"
        graph.add_initial(newbiasname, newbias)
        graph.remove_node(nodes[1])
        if hasbias:
            conv_node.input[2] = newbiasname
        else:
            conv_node.input.append(newbiasname)
        graph.producedby[bn_node.output[0]] = [conv_node.name]
        conv_node.output[0] = bn_node.output[0]
