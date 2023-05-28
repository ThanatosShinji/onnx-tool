<a href="ConstantFolding_CN.md">简体中文</a>
## The definition
Placeholder
## Usage
python API
```python
import onnx_tool
import onnx
rawonnx='test.onnx'
m = onnx.load_model(rawonnx)
g = onnx_tool.Graph(m.graph,constant_folding=True)
g.save_model('folded.onnx',rawmodel=m)
```
cli
```commandline
python -m onnx_tool -i test.onnx -m constant_folding -o folded.onnx
```

## Typical models
### BevFormer
GraphProto Nodes Count:7689  
Contant folding ['/Constant', '/Constant_1', '/Constant_2']... 3343 Nodes  
<a href="../benchmark/compression.py">ReproduceCode</a> 
### gpt-j
GraphProto Nodes Count:1139  
Contant folding ['Identity_0', 'Identity_1', 'Identity_2']... 1029 Nodes  
<a href="../benchmark/transfomer_models.py">ReproduceCode</a>

## Known Issue
placeholder