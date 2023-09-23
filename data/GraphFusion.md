# Motivation
Help pre-process and simplify the ONNX graph structure.  
Here are some cases:
1. Multi-head attention: some models use matmul, some models use einsum. You fuse them to one MHA node.
2. Useless ops for inference: quantized models with ['Quantize','Dequantize'], just remove these two nodes. 'Flatten' Op
has no meaning for inference, just remove it.
3. Get subgraph from a whole ONNX model: if you want to split one ONNX model by model parallelism. You can save subgraphs to 
new ONNX models then you can load subgraph models on different inference device.

# Pattern Fusion
To simplify the fusion process, we introduce a pattern-base method.
## What is pattern?
onnx_tool.fusion.FusionPattern is a set of linked node descriptions.   
E.g.:  
```python
from onnx_tool.fusion import FusionPattern
Fused_Element = [# the node linkage relations
    {
        'name': 'any', # id
        'op': 'Any', # op match pattern, Any match every op
        'attrs': [], # attributes conditions 
        'inport': [], 
        'outport': [[0, 'act_0', 0]], # output tensor linkage: this node's 0th tensor to node 'act_0' 0th input tenosr
    },
    {
        'name': 'act_0',
        'op': ['Relu', 'LeakyRelu', 'Add'],# a set of ops
        'attrs': [
        ],# you can add something like  ['alpha', '<=', 7] which creates a condition that node.alpha<=7
        'inport': [[0, 'any', 0]],# input tensor linkage: node 'any' 0th output tensor to this node 0th input tensor 
        'outport': [],
    },
]

pattern = FusionPattern(Fused_Element)
nodes = pattern.search_pattern(graph)# nodes:list[names] like[['Conv0','Relu1'],['Linear2','Add3'],...]
```
A more easy way to create node descriptions:
```python
from onnx_tool.fusion import create_descs_from_nodenames, FusionPattern
GeluNodes = ['Mul_213','Pow_215','Mul_217','Add_218','Mul_220','Tanh_221','Add_223','Mul_224']
MadDescs =  create_descs_from_nodenames(graph, GeluNodes)#find these nodes in the graph, build a nodedescs for them
pattern = FusionPattern(MadDescs)
nodes = pattern.search_pattern(graph)# find all Gelu nodes in the graph
```

## Do Fusion
Fuse some nodes as one node(simple fusion, keep all attributes and static tensors):
```python
for names in nodes:
    graph.fuse_subgraph_node_names(names, 'Mad', names[0])
```
Fuse with post op way:
```python
for names in nodes:
    graph.fuse_postop_node_names(names, True)
#Conv+Relu will be fused as Conv(append new attribute 'postop_0':'Relu')
```
Even easier with two tensors:
```python
from onnx_tool.graph import Graph
g = Graph(...)
in_tensor_names = ['bert/encoder/Reshape_1:0'] # whole MHA subgraph's input tensor name
out_tensor_names = ['bert/encoder/layer_0/attention/output/dense/BiasAdd:0'] # whole MHA subgraph's output tensor name
g.fuse_subgraph_iotensors(inputs=in_tensor_names, outputs=out_tensor_names, name_prefix='MHA',
                          nodeop='MHA', keep_attr=True) #all MHA will be fused
```


# Extraction of SubGraph
In this case, the onnx model may be split into three onnx models: level 0 model, level 1 model, and level 2 model.  
Level 0 model: need to be executed before the subgraph.  
Level 1 model: the subgraph model.  
Level 2 model: need to be executed after the subgraph.  
You may see this image:
<p align="center">
  <img src="./resnet18_subgraph.png">
</p>

Step1: execute level 0 model to get resnetv15_stage4_batchnorm2_fwd and resnetv15_stage4_conv0_fwd tensors.  
Step2: feed resnetv15_stage4_conv0_fwd to level 1 model and execute it.  
Step3: feed resnetv15_stage4_batchnorm2_fwd and resnetv15_stage4_batchnorm1_fwd from level 0 and level 1 models. Then 
get the final output tensor.    

## Save Subgraph Models
```python
import onnx_tool
modelpath = 'resnet18-v1-7_shapes.onnx'
onnx_tool.model_subgraph(modelpath,['resnetv15_stage4_conv0_fwd'],['resnetv15_stage4_batchnorm1_fwd']) 
#get the subgraph by selecting its input and output tensors, view the source code for more usages.

onnx_tool.model_subgraph(modelpath,nodenames=['resnetv15_stage1_conv0_fwd','resnetv15_stage1_batchnorm0_fwd', \ 
                                                                                     'resnetv15_stage1_relu0_fwd'])
#get the subgraph by selecting node names

```    