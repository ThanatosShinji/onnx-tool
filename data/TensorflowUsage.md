## This md will show you how to apply onnx-tool to a tensorflow-keras model

* tensorflow export to ONNX, then use onnx_tool
    ```python
    import tensorflow
    from tensorflow import keras
    import onnx
    import tf2onnx
    import onnx_tool
    from onnx_tool import create_ndarray_f32
  
    temp_model_file = 'tmp.onnx'
    
    model = tensorflow.keras.applications.InceptionV3(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )
    onnx_model = tf2onnx.convert.from_keras(model,
                    input_signature=None, opset=None, custom_ops=None,
                    custom_op_handlers=None, custom_rewriter=None,
                    inputs_as_nchw=None, outputs_as_nchw=None, extra_opset=None,
                    shape_override=None, target=None, large_model=False, output_path=None)
    if isinstance(onnx_model,(list,tuple)):
        onnxproto=onnx_model[0]
    onnx.save_model(onnxproto, temp_model_file)
    inputshape=(1, 299,299,3)
    dynamics_input={
        'input_1':create_ndarray_f32(inputshape)
    }
    onnx_tool.model_profile(temp_model_file,dynamic_shapes=dynamics_input)
    ```    

* sample code [tfkeras_sample.py](https://github.com/ThanatosShinji/onnx-tool/blob/main/benchmark/tfkeras_sample.py)