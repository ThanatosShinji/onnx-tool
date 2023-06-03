import onnxruntime as ort
import numpy as np
import onnx
import onnx_tool


def debug_with_onnxrt(onnxfile, dumpnames: [str], inputm):
    if len(dumpnames) > 0:
        m = onnx.load_model(onnxfile)
        g = onnx_tool.Graph(m.graph)
        g.add_dump_tensors(dumpnames)
        g.save_model('tmp.onnx', rawmodel=m)
        sess = ort.InferenceSession('tmp.onnx')
    else:
        sess = ort.InferenceSession(onnxfile)
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
    onnxfile = 'data/public/resnet50.onnx'
    # onnx_tool.model_io_modify(onnxfile,'resnet50.onnx',{'data':'1x3xhxw'})
    input = np.ones((1, 3, 32, 32), dtype=np.float32)
    input = input / 2
    inm = {'data': input}
    output = debug_with_onnxrt(onnxfile, [], inm)
    print(output)

    m = onnx.load_model(onnxfile)
    g = onnx_tool.Graph(m.graph)
    outm = g.value_infer(inm)
    print(outm)


def EdgeNeXt_small():
    onnxfile = 'data/public/convnext_large.onnx'
    input = np.ones((1, 3, 224, 224), dtype=np.float32)
    input = input / 2
    sess = ort.InferenceSession(onnxfile)
    output = sess.run(['1509'], {'input.1': input})
    print(output)
    sess = ort.InferenceSession('shape.onnx')
    output = sess.run(['1509'], {'input.1': input})
    print(output)


def text_encoder():
    onnxfile = 'data/public/text_encoder.onnx'
    input = np.ones((1, 77), dtype=np.int64)
    sess = ort.InferenceSession(onnxfile)
    output = sess.run(['emb'], {'tokens': input})
    print(output)
    sess = ort.InferenceSession('shape.onnx')
    output = sess.run(['emb'], {'tokens': input})
    print(output)

def gpt2():
    onnxfile = 'data/public/gpt2-10.onnx'
    input = np.ones((1, 1, 8), dtype=np.int64)
    sess = ort.InferenceSession(onnxfile)
    output = sess.run(['output1'], {'input1': input})
    print(output)
    m = onnx.load_model(onnxfile)
    g = onnx_tool.Graph(m.graph,constant_folding=True,verbose=True)
    _=g.value_infer({'input1':input})
    print(g.tensormap['output1'].numpy)


def gpt2():
    import numpy
    onnxfile = 'data/public/gpt2-10.onnx'
    input = np.ones((1, 1, 8), dtype=np.int64)
    sess = ort.InferenceSession(onnxfile)
    output = sess.run(['output1', "output13"], {'input1': input})
    ort_ret=output[0]
    print(ort_ret)
    m = onnx.load_model(onnxfile)
    g = onnx_tool.Graph(m.graph, verbose=True)
    output=g.value_infer({'input1':input})
    idx=g.output.index('output1')
    ot_ret=output[idx]
    print(ot_ret)
    diff=abs(ort_ret-ot_ret)
    print(numpy.max(diff))
    # g.shape_infer({
    #     'input1': onnx_tool.create_ndarray_int64((1, 1, 8)),
    # })
    # g.save_model('shape.onnx',rawmodel=m)
    # sess = ort.InferenceSession('shape.onnx')
    # output = sess.run(['output1', "output13"], {'input1': input})
    # print(output[0])

if __name__ == '__main__':
    # resnet50()
    gpt2()
    # resnet18()
    # EdgeNeXt_small()
    # text_encoder()
