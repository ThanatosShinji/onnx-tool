# Architecture Update

## Key Features

### Huge shape inference speedup

Model | v0.4.0 Time(s) | v0.5.0 Time(s) | speedup
---|---|---|---
VAE_Encoder|1.598|0.003|532
VAE_Decoder|3.425|0.001|1712
Text_Encoder|1.089|0.004|272
Bert_Base|0.710|0.006|109

Also, memory space is decreased by a lot.

### From function-call to class-call

There are three new python classes: Graph, Tensor, and Node.

~~~python
import onnx_tool
import onnx
#previous usage
onnx_tool.model_profile('raw.onnx',saveshapesmodel='shapes.onnx')
#new usage
m = onnx.load('raw.onnx')
graph = onnx_tool.Graph(m.graph)
graph.shape_infer()
graph.profile()
graph.print_node_map()
graph.save_model('shapes.onnx')
~~~

You can get all tensors from graph.tensormap and nodes from graph.nodemap.

### Split tensor value inference and shape inference

~~~python

class Node():
    def shape_infer(self, intensors: []):
        #compute output tensor's shape

    def value_infer(self, intensors: []):
        #compute output tensor's value

    def profile(self, intensors: [], outtensors: []):
        #get MACs of these IO tensors
~~~

## Deprecate

* node_profilers.py will be deprecated in the future. 
