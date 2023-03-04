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

ResBlock = [
    {
        'name': 'conv_0',
        'op': 'Conv',
        'attrs': [
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
        'outport': [[0, 'relu_0', 0]],
    },
    {
        'name': 'relu_0',
        'op': 'Relu',
        'attrs': [
        ],
        'inport': [[0, 'bn_0', 0]],
        'outport': [[0, 'conv_1', 0]],
    },
    {
        'name': 'conv_1',
        'op': 'Conv',
        'attrs': [
        ],
        'inport': [[0, 'relu_0', 0]],
        'outport': [[0, 'bn_1', 0]],
    },
    {
        'name': 'bn_1',
        'op': 'BatchNormalization',
        'attrs': [
        ],
        'inport': [[0, 'conv_1', 0]],
        'outport': [[0, 'add_0', 1]],
    },
    {
        'name': 'add_0',
        'op': 'Add',
        'attrs': [
        ],
        'inport': [[1, 'bn_1', 0]],
        'outport': [[0, 'relu_1', 0]],
    },
    {
        'name': 'relu_1',
        'op': 'Relu',
        'attrs': [
        ],
        'inport': [[0, 'add_0', 0]],
        'outport': [],
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
