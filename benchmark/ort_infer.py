import onnxruntime as ort
import numpy as np


def resnet18():
    onnxfile = 'debug.onnx'
    sess = ort.InferenceSession(onnxfile)
    input = np.ones((1, 3, 224, 224), dtype=np.float32)
    input = input / 2
    output = sess.run(['resnetv15_dense0_fwd', 'resnetv15_conv0_fwd'], {'data': input})
    print(output)

def resnet50():
    onnxfile = 'data/public/resnet50-v2-7.onnx'
    sess = ort.InferenceSession(onnxfile)
    input = np.ones((1, 3, 224, 224), dtype=np.float32)
    input = input / 2
    output = sess.run(['resnetv24_dense0_fwd'], {'data': input})
    print(output)


def asr():
    onnxfile = 'data/private/asr_500G.onnx'
    import onnx_tool
    import onnx
    m = onnx.load_model(onnxfile)
    g = onnx_tool.Graph(m.graph)
    debugtensor = 'output'
    g.add_dump_tensors([debugtensor])
    g.save_model('asr_dump.onnx', rawmodel=m)
    sess = ort.InferenceSession('asr_dump.onnx')
    input = np.ones((1, 3, 16, 16), dtype=np.float32)
    input = input / 2
    output = sess.run([debugtensor], {'input': input})
    # print(output[0])
    print(output[0][0, 0, :, :])


# resnet50()
asr()
