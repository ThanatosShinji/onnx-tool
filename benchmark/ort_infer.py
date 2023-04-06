import time

import numpy
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
    debugtensor = 'input.4'
    g.add_dump_tensors([debugtensor])
    g.save_model('asr_dump.onnx', rawmodel=m)
    sess = ort.InferenceSession('asr_dump.onnx')
    input = np.ones((1, 3, 16, 16), dtype=np.float32)
    input = input / 2
    ts = time.time()
    output = sess.run([debugtensor], {'input': input})
    print(time.time() - ts)
    # print(output[0])
    # print(output[0][0, 0, :, :])
    t = np.transpose(output[0][0], axes=(1, 2, 0))
    print(t.shape)
    print(t)
    onnxshape = output[0][0].shape
    # file=open('test.output','rb')
    # buf=file.read()
    # ntensor = numpy.frombuffer(buf, dtype=numpy.float32).reshape((onnxshape[1], onnxshape[2], onnxshape[0]))
    # diff = numpy.abs(t - ntensor)
    # for i in range(3):
    #     print(diff[i, :, :].min())
    #     print(diff[i, :, :].max())
    # print(diff.min())
    # print(diff.max())


def asr_pipeline():
    onnxfile = 'data/private/asr_500G.onnx'
    import onnx_tool
    import onnx
    m = onnx.load_model(onnxfile)
    g = onnx_tool.Graph(m.graph)
    debugtensor = 'output'
    g.add_dump_tensors([debugtensor])
    g.save_model('asr_dump.onnx', rawmodel=m)
    sess = ort.InferenceSession('asr_dump.onnx')
    import cv2
    img = cv2.imread('data/0030.jpg')
    cv2.imshow('raw', img)
    # rx=int(img.shape[1]*0.5//8*8)
    # ry=int(img.shape[0]*0.5//8*8)
    # img=cv2.resize(img,dsize=(rx,ry))
    input = np.transpose(img, (2, 0, 1))
    input = input.astype(np.float32).reshape((1,) + input.shape)
    input = input / 255.0
    file = open('test.input', 'wb')
    tin = np.transpose(input[0], axes=(1, 2, 0))

    file.write(tin.tobytes())
    file.close()
    ts = time.time()
    output = sess.run(['output', debugtensor], {'input': input})
    print(time.time() - ts)
    t = np.transpose(output[0][0], axes=(1, 2, 0))

    def output2img(output):
        tensor = np.transpose(output, (1, 2, 0))
        tensor = np.clip(tensor, 0, 1.0)
        tensor = (tensor) * 255.0
        tensor = np.abs(tensor)
        tensor = tensor.astype(np.uint8)
        return tensor

    outimg = output2img(output[0][0, :, :, :])
    cv2.imshow('out', outimg)
    cv2.imwrite('test.png', outimg)
    cv2.waitKey()

    file = open('test.output', 'rb')
    buf = file.read()
    onnxshape = output[1][0].shape
    ntensor = numpy.frombuffer(buf, dtype=numpy.float32).reshape((onnxshape[1], onnxshape[2], onnxshape[0]))
    diff = numpy.abs(t - ntensor)
    for i in range(3):
        print(diff[i, :, :].min())
        print(diff[i, :, :].max())
    print(diff.min())
    print(diff.max())
    ntensor = np.transpose(ntensor, (2, 0, 1))
    cv2.imshow('out1', output2img(ntensor))

    # print(output[0])
    cv2.waitKey()


# resnet50()
asr_pipeline()
# asr()
