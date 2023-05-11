import onnxruntime as ort
import numpy as np

def debug_with_onnxrt(onnxfile, dumpnames: [str], inputm):
    import onnx_tool
    import onnx
    m = onnx.load_model(onnxfile)
    g = onnx_tool.Graph(m.graph)
    g.add_dump_tensors(dumpnames)
    g.save_model('tmp.onnx', rawmodel=m)
    sess = ort.InferenceSession('tmp.onnx')
    output = sess.run(dumpnames, inputm)
    return output

def resnet18():
    onnxfile = 'data/public/resnet18-v1-7.onnx'
    input = np.ones((1, 3, 224, 224), dtype=np.float32)
    input = input / 2
    output = debug_with_onnxrt(onnxfile, ['resnetv15_dense0_fwd', 'resnetv15_conv0_fwd'], {'data': input})
    print(output)


def resnet50():
    onnxfile = 'data/public/resnet50-v2-7.onnx'
    input = np.ones((1, 3, 224, 224), dtype=np.float32)
    input = input / 2
    output = debug_with_onnxrt(onnxfile, ['resnetv24_pool0_fwd'], {'data': input})
    print(output)

if __name__ == '__main__':
    resnet50()
    resnet18()
