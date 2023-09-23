<a href="ConstantFolding_CN.md">简体中文</a>
## Motivation
Sometimes, ONNX exporter creates a constant subgraph to get a constant tensor, like this:
<p align="left">
  <img src="ComputeGraphIssue.png">
</p>

Actually, the shape tensor of the four Add nodes matter. If we pre-compute this subgraph, then we only need 
to keep one shape tensor instead of a constant subgraph.
The constant folding of onnx_tool will replace these constant subgraphs to constant tensors. 
## Usage
python API
```python
import onnx_tool
rawonnx='test.onnx'
m = onnx_tool.Model(rawonnx,constant_folding=True)
m.save_model('folded.onnx')
```
cli
```commandline
python -m onnx_tool -i test.onnx -m constant_folding -o folded.onnx
```

## Typical models
### BevFormer
GraphProto Nodes Count:7689  
Contant folding ['/Constant', '/Constant_1', '/Constant_2']... 3343 Nodes  
Removed 3343 layers without any compute fusion.  
<a href="../benchmark/compression.py">ReproduceCode</a> 
### gpt-j
GraphProto Nodes Count:1139  
Contant folding ['Identity_0', 'Identity_1', 'Identity_2']... 1029 Nodes  
Only 110 layers remained. A successful case.   
<a href="../benchmark/transfomer_models.py">ReproduceCode</a>


## Known Issue
Constant folding depends on onnx_tool.Node's value_infer implementation.  
You may see this exception if your model is not supported for constant folding:  
```python
raise NotImplementedError(f'this Node {self.op_type}-{self.name} has no value_infer')
```
You need to disable the constant folding flag, or add this op's value_infer implementation.