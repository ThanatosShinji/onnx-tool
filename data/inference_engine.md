# Compute Graph & Shape Engine cpp Integration

This md will show you how to integrate compute graph & shape engine to a cpp based inference engine.  
The integration will reduce the engine's work by a lot while supporting dynamic runtime input shapes.

## Python Serialization of Compute Graph & Shape Engine

~~~python
    import onnx_tool

resnetinfo = {
    'name': 'data/public/resnet18-v1-7.onnx',
    'input_desc':
        {
            'data': [1, 3, 'h', 'w']
        },
    'input_range':
        {
            'h': (224, 299),
            'w': (224, 299),
        }
}
shape_engie, compute_graph = onnx_tool.model_shape_regress(resnetinfo['name'], resnetinfo['input_desc'],
                                                           resnetinfo['input_range'])
onnx_tool.serialize_graph(compute_graph, 'resnet18.cg')
onnx_tool.serialize_shape_engine(shape_engie, 'resnet18.se')
~~~

The file 'resnet18.cg' contains compute graph information.  
The file 'resnet18.se' contains shape engine structure. They will be used by the cpp graph loader and shape engine
loader.

## Integration cpp Example

To keep this repo simple and clean, please refer to my new
repo: [shape-engine-cpp](https://github.com/ThanatosShinji/shape-engine-cpp.git)