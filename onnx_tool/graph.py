import warnings
import onnx
from .tensors import get_attribute_data
from .utils import VERSION


def __shape_of_initializer__(initial):
    shape = []
    # for nb in tensor.shape.dim
    for nb in initial.dims:
        shape.append(nb)
    return shape

class Tensor():
    def __init__(self,t):
        self.name=t.name
        self.proto=t
        if isinstance(t,onnx.ValueInfoProto):
            self.shape=[2,]
        elif isinstance(t,onnx.TensorProto):
            self.shape = [2,]

class Node():
    def __init__(self,n:onnx.NodeProto):
        self.name=n.name
        self.nextnodes=[]
        self.prevnodes=[]
        self.output=[]
        self.input=[]
        self.proto=n

class Graph():
    def __init__(self,g:onnx.GraphProto):
        self.nodemap= {}
        self.tensormap={}
        self.producedby={}
        self.consumedby={}
        self.rawgraph=g
        self.initials=[]
        self.input=[]
        self.output=[]

        if g is None:
            return

        for node in g.node:
            newnode = Node(node)
            for tensor in node.input:
                if tensor in self.producedby:
                    for producer in self.producedby[tensor]:
                        newnode.prevnodes.append(self.nodemap[producer])
                if tensor not in self.consumedby:
                    self.consumedby[tensor]=[]
                self.consumedby[tensor].append(newnode.name)
                newnode.input.append(tensor)
            for tensor in node.output:
                if tensor not in self.producedby:
                    self.producedby[tensor]=[]
                self.producedby[tensor].append(newnode.name)
                newnode.output.append(tensor)

            self.nodemap[newnode.name]=newnode

        for node in g.node:
            for tensor in node.output:
                if tensor in self.consumedby:
                    for consumer in self.consumedby[tensor]:
                        self.nodemap[node.name].nextnodes.append(self.nodemap[consumer])

        for input in g.input:
            tensor=Tensor(input)
            self.tensormap[input.name]=tensor

        for output in g.output:
            tensor=Tensor(output)
            self.tensormap[output.name]=tensor

        for valinfo in g.value_info:
            tensor = Tensor(valinfo)
            self.tensormap[valinfo.name] = tensor

        for initial in g.initializer:
            self.initials.append(initial.name)

        for node in g.node:
            if node.op_type == 'Constant':
                self.initials.append(node.output[0])

        for t in g.input:
            self.input.append(t.name)

        for t in g.output:
            self.output.append(t.name)

    def __get_subnodes_byio__(self,inputs:[],outputs:[]):
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
        return graph_level0,graph_level1,graph_level2

    def get_subgraph(self,inputs:[],outputs:[]):
        graph_level0, graph_level1, graph_level2=self.__get_subnodes_byio__(inputs,outputs)

        graph_level0=Graph(self.get_onnxgraph_by_nodenames(graph_level0))
        graph_level1=Graph(self.get_onnxgraph_by_nodenames(graph_level1))
        graph_level2=Graph(self.get_onnxgraph_by_nodenames(graph_level2))

        group_outputs=[graph_level0.output,graph_level1.output,graph_level2.output]
        group_inputs=[graph_level0.input,graph_level1.input,graph_level2.input]

        extern_outputs=[]
        extern_inputs=[]
        for ele in group_outputs:
            extern_outputs.extend(ele)
        extern_outputs=set(extern_outputs)

        for ele in group_inputs:
            extern_inputs.extend(ele)
        extern_inputs=set(extern_inputs)

        for inputs in group_inputs:
            extern_outputs=extern_outputs - set(inputs)

        for outputs in group_outputs:
            extern_inputs = extern_inputs - set(outputs)

        if len(extern_inputs)!=len(self.input):
            warnings.warn("subgraph input and output tensors can not reverse to raw graph.")

        if len(extern_outputs)!=len(self.output):
            warnings.warn("subgraph input and output tensors can not reverse to raw graph.")

        return graph_level0,graph_level1,graph_level2
        # graph_level0.save_model('graph_level0.onnx')
        # graph_level1.save_model('graph_level1.onnx')
        # graph_level2.save_model('graph_level2.onnx')

    def fuse_subgraph_node_names(self,nodes:[str],nodeop:str,name:str,keep_attr=True):
        _inputs, _outputs = self.get_iotensors(nodes,remove_initials=False)
        newnode=onnx.helper.make_node(nodeop,_inputs,_outputs,name=name)
        count=0
        if keep_attr:
            for node in nodes:
                for attribute in self.nodemap[node].proto.attribute:
                    attr=onnx.helper.make_attribute(self.nodemap[node].proto.name+'_'+attribute.name,get_attribute_data(attribute))
                    newnode.attribute.append(attr)
                count+=1

        allnodes=set(self.nodemap.keys())
        remainnodes=allnodes-set(nodes)
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
        graph=onnx.helper.make_graph(nodes=nodes,name='fused_graph',inputs=inputs,outputs=outputs,initializer=self.rawgraph.initializer)
        newgraph=Graph(graph)
        return newgraph

    def fuse_subgraph_iotensors(self,inputs:[],outputs:[],nodeop:str,name:str,keep_attr=True):
        _,nodes,_=self.__get_subnodes_byio__(inputs,outputs)
        _inputs, _outputs = self.get_iotensors(nodes,remove_initials=True)
        nodes=self.reorder_nodes(nodes,_inputs)
        return self.fuse_subgraph_node_names(nodes,nodeop,name,keep_attr)

    def get_onnxgraph_by_nodenames(self,nodenames):
        if len(nodenames):
            _inputs0, _outputs0 = self.get_iotensors(nodenames)
            graph_level0 = self.reorder_nodes(nodenames, _inputs0)
            subgraph = self.make_graph(graph_level0, 'subgraph', _inputs0, _outputs0)
            return subgraph
        return None

    def save_model(self,f:str):
        if self.rawgraph is not None and f is not None:
            model=onnx.helper.make_model(self.rawgraph, producer_name='onnx-tool',producer_version=VERSION)
            onnx.save_model(model,f)

    def make_graph(self,nodenames,gname,inputnames,outputnames,with_initializer=True):
        nodes=[]
        for name in nodenames:
            nodes.append(self.nodemap[name].proto)

        initializer=None
        if with_initializer:
            names=[]
            for name in nodenames:
                for input in self.nodemap[name].input:
                    if input in self.initials:
                        names.append(input)
            initializer=[]
            for initial in self.rawgraph.initializer:
                if initial.name in names:
                    initializer.append(initial)

        inputs=[]
        outputs=[]
        for name in inputnames:
            if name in self.tensormap:
                inputs.append(self.tensormap[name].proto)
            else:
                inputs.append(onnx.helper.make_tensor_value_info(name,1,None))
        for name in outputnames:
            if name in self.tensormap:
                outputs.append(self.tensormap[name].proto)
            else:
                outputs.append(onnx.helper.make_tensor_value_info(name, 1, None))
        graph=onnx.helper.make_graph(nodes=nodes,name=gname,inputs=inputs,outputs=outputs,initializer=initializer)
        return graph

    def reorder_nodes(self,nodenames,itnames):
        tensor_consumed=[]
        tensor_produced=[]
        nextnodes=[]
        reorderednode=[]
        for name in itnames:
            for consumer in self.consumedby[name]:
                if consumer in nodenames:
                    if consumer not in nextnodes:
                        nextnodes.append(consumer)
            tensor_produced.append(name)

        while len(nextnodes):
            execnodes=[]
            for node in nextnodes:
                produced=True
                for input in self.nodemap[node].input:
                    if input in self.initials:
                        continue
                    if input not in tensor_produced:
                        produced=False
                        break
                if produced:
                    execnodes.append(node)

            newnodes=[]
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
                                newnodes.append(consumer)
            nextnodes=set(newnodes)

        return reorderednode

    def get_iotensors(self,nodenames,remove_initials=True):
        intensors=[]
        outtensors=[]
        for name in nodenames:
            for input in self.nodemap[name].input:
                if remove_initials and input in self.initials:
                    continue
                if input in self.producedby:
                    producers=self.producedby[input]
                    inner=True
                    for producer in producers:
                        if producer not in nodenames:
                            inner=False
                            break
                    if inner:
                        continue
                if input not in intensors:
                    intensors.append(input)

            for output in self.nodemap[name].output:
                if remove_initials and output in self.initials:
                    continue
                if output in self.consumedby:
                    consumers=self.consumedby[output]
                    inner = True
                    for consumer in consumers:
                        if consumer not in nodenames:
                            inner = False
                            break
                    if inner:
                        continue
                if output not in outtensors:
                    outtensors.append(output)

        return intensors,outtensors

if __name__ == '__main__':
    f='data/public/bertsquad-12.onnx'
    f='data/public/resnet18-v1-7_shapes.onnx'
    # f='data/public/rvm_mobilenetv3_fp32.onnx'
    m=onnx.load(f)
    graph=Graph(m.graph)
    graph.get_subgraph(['resnetv15_stage4_conv0_fwd'],['resnetv15_stage4_batchnorm1_fwd'])
    graph.fuse_subgraph_iotensors(['resnetv15_stage3_activation1'],['resnetv15_stage4__plus0'],'fused','fused_0')

    # graph.get_subgraph(['393'],['601'])
    # graph.get_subgraph(['bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add_1:0'],['bert/encoder/layer_2/output/add:0'])
    print(graph)