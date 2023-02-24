import onnxruntime as ort
import numpy as np


def resnet18():
    onnxfile = 'debug.onnx'
    sess = ort.InferenceSession(onnxfile)
    input = np.ones((1, 3, 224, 224), dtype=np.float32)
    input = input / 2
    output = sess.run(['resnetv15_dense0_fwd', 'resnetv15_conv0_fwd'], {'data': input})
    print(output)


resnet18()
