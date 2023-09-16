import tensorflow
from tensorflow import keras
import onnx
import tf2onnx
import onnx_tool
from onnx_tool import create_ndarray_f32

temp_model_file = 'tmp.onnx'


def InceptionV3():
    inputshape = (1, 299, 299, 3)
    model = tensorflow.keras.applications.InceptionV3(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=inputshape[1:],
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )
    onnx_model = tf2onnx.convert.from_keras(model,
                                            input_signature=None, opset=None, custom_ops=None,
                                            custom_op_handlers=None, custom_rewriter=None,
                                            inputs_as_nchw=None, outputs_as_nchw=None, extra_opset=None,
                                            shape_override=None, target=None, large_model=False, output_path=None)
    if isinstance(onnx_model, (list, tuple)):
        onnxproto = onnx_model[0]
    onnx.save_model(onnxproto, temp_model_file)
    dynamics_input = {
        'input_1': create_ndarray_f32(inputshape)
    }
    onnx_tool.model_profile_v2(temp_model_file, dynamic_shapes=dynamics_input)


def MobileNetV3Large():
    inputshape = (1, 299, 299, 3)
    model = tensorflow.keras.applications.MobileNetV3Large(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=inputshape[1:],
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )
    onnx_model = tf2onnx.convert.from_keras(model,
                                            input_signature=None, opset=None, custom_ops=None,
                                            custom_op_handlers=None, custom_rewriter=None,
                                            inputs_as_nchw=None, outputs_as_nchw=None, extra_opset=None,
                                            shape_override=None, target=None, large_model=False, output_path=None)
    if isinstance(onnx_model, (list, tuple)):
        onnxproto = onnx_model[0]
    onnx.save_model(onnxproto, temp_model_file)
    dynamics_input = {
        'input_2': create_ndarray_f32(inputshape)
    }
    onnx_tool.model_profile_v2(temp_model_file, saveshapesmodel='shapes.onnx', dynamic_shapes=dynamics_input)


InceptionV3()
MobileNetV3Large()
