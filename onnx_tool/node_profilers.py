import math
import warnings

import numpy
import numpy as np

from .tensors import volume, create_ndarray_f32, create_ndarray_int64, get_attribute_data, is_valid_ndarray
from .utils import NODEPROFILER_REGISTRY, tuple2str

ADD_MACS = 1
EXP_MACS = 16
LOG_MACS = 16
SQRT_MACS = 4
POW_MACS = 32
MUL_MACS = 1
DIV_MACS = 2
CMP_MACS = 1
SIN_MACS = 14
COS_MACS = 14

RESIZE_LINEAR_MACS = 4
RESIZE_CUBIC_MACS = 8


def max_shape(shapes: []):
    maxvol = volume(shapes[0])
    maxshape = ()
    for shape in shapes:
        vol = volume(shape)
        if vol > maxvol:
            maxshape = shape
            maxvol = vol
    return maxshape


def max_shape_ndarray(lndarr: [numpy.ndarray]):
    maxshape = lndarr[0].shape
    maxvol = volume(maxshape)
    for ndarr in lndarr:
        vol = volume(ndarr.shape)
        if vol > maxvol:
            maxshape = ndarr.shape
            maxvol = vol
        elif vol == maxvol:
            if len(ndarr.shape) > len(maxshape):
                maxshape = ndarr.shape
    return maxshape


def conv_output_shape(xin, pad, ksize, stride, dilation):
    return int((xin + pad - dilation * (ksize - 1) - 1) / stride + 1)


def convtranspose_output_shape(xin, output_padding, pad, ksize, stride, dilation):
    return stride * (xin - 1) + output_padding + ((ksize - 1) * dilation + 1) - pad


def pooling_shape_calc(inshape, pad, kshape, dilation, stride, ceilmode):
    outshape = (inshape + pad - ((kshape - 1) * dilation + 1)) / stride + 1
    if ceilmode:
        return math.ceil(outshape)
    return math.floor(outshape)


class NodeBase():
    def __init__(self, nodeproto):
        self.nbinput = len(nodeproto.input)
        self.nboutput = len(nodeproto.output)
        self.name = nodeproto.name

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        return 0, 0

    def infer_shape(self, intensors: [numpy.ndarray]):
        return [(0), ]


@NODEPROFILER_REGISTRY.register()
class Conv(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        validAttr = ['pads', 'strides', 'dilations', 'auto_pad']
        self.auto_pad = None
        self.pads = (0, 0, 0, 0)
        self.strides = (1, 1)
        self.dilations = (1, 1)
        auto_add_attributes(nodeproto.attribute, validAttr, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        outtensors = []
        xtensor = intensors[0]
        wtensor = intensors[1]
        xshape = xtensor.shape
        wshape = wtensor.shape
        shape = []
        if self.auto_pad is not None and self.auto_pad != b'NOTSET':
            if self.auto_pad in [b'SAME_LOWER', b'SAME_UPPER']:
                shape = (xshape[0], wshape[0], math.ceil(xshape[2] / self.strides[0]))
                if len(xshape) == 4:
                    shape += (math.ceil(xshape[3] / self.strides[1]),)
        else:
            if len(xshape) == 4:
                oh = conv_output_shape(xshape[2], self.pads[0] + self.pads[2], wshape[2], self.strides[0],
                                       self.dilations[0])
                ow = conv_output_shape(xshape[3], self.pads[1] + self.pads[3], wshape[3], self.strides[1],
                                       self.dilations[1])
                shape = (xshape[0], wshape[0], oh, ow)
            elif len(xshape) == 3:
                oh = conv_output_shape(xshape[2], self.pads[0] + self.pads[1], wshape[2], self.strides[0],
                                       self.dilations[0])
                shape = (xshape[0], wshape[0], oh)
        outtensors.append(create_ndarray_f32(shape))
        return outtensors

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        macs = 0
        params = 0
        if self.nboutput == 1:
            if self.nbinput == 3 or self.nbinput == 2:
                kernel_shape = intensors[1].shape
                params += volume(kernel_shape)
                if self.nbinput == 3:
                    params += kernel_shape[0]

                if len(kernel_shape) > 3:
                    outvol = volume(outtensors[0].shape)
                    macs += outvol * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]
                    macs += outvol * ADD_MACS  # treat bias add as 0.5 MACs
                elif len(kernel_shape) == 3:
                    outvol = volume(outtensors[0].shape)
                    macs += outvol * kernel_shape[1] * kernel_shape[2]
                    macs += (outvol * ADD_MACS)
        return macs, params


@NODEPROFILER_REGISTRY.register()
class ConvTranspose(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        validAttr = ['pads', 'strides', 'dilations', 'output_padding', 'output_shape', 'group']
        self.pads = (0, 0, 0, 0)
        self.strides = (1, 1)
        self.dilations = (1, 1)
        self.output_padding = (0, 0, 0, 0)
        self.output_shape = (0, 0)
        self.group = 1
        auto_add_attributes(nodeproto.attribute, validAttr, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        xtensor = intensors[0]
        wtensor = intensors[1]
        xshape = xtensor.shape
        wshape = wtensor.shape
        shape = []
        outc = self.group * wshape[1]
        if len(xshape) == 4:
            ow = convtranspose_output_shape(xshape[2], self.output_padding[0], self.pads[0] + self.pads[2], wshape[2],
                                            self.strides[0],
                                            self.dilations[0])
            oh = convtranspose_output_shape(xshape[3], self.output_padding[1], self.pads[1] + self.pads[3], wshape[3],
                                            self.strides[1],
                                            self.dilations[1])
            shape = [xshape[0], outc, ow, oh]
            if volume(self.output_shape) != 0:
                shape[2:] = self.output_shape
        elif len(xshape) == 3:
            ow = convtranspose_output_shape(xshape[2], self.output_padding[0], self.pads[0] + self.pads[1], wshape[2],
                                            self.strides[0],
                                            self.dilations[0])
            shape = [xshape[0], outc, ow]
            if volume(self.output_shape) != 0:
                shape[2] = self.output_shape[0]
        return [create_ndarray_f32(shape)]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        macs = 0
        params = 0
        if self.nboutput == 1:
            if self.nbinput == 3 or self.nbinput == 2:
                kernel_shape = intensors[1].shape
                params += volume(kernel_shape)
                if self.nbinput == 3:
                    params += kernel_shape[0]

                if len(kernel_shape) > 3:
                    outvol = volume(outtensors[0].shape)
                    macs += outvol * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]
                    macs += outvol * ADD_MACS  # treat bias add as 0.5 MACs
                elif len(kernel_shape) == 3:
                    outvol = volume(outtensors[0].shape)
                    macs += outvol * kernel_shape[1] * kernel_shape[2]
                    macs += (outvol * ADD_MACS)
        return macs, params


class PWNBase(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = 0
        self.ratio = max(1, self.nbinput - 1)

    def infer_shape(self, intensors: [numpy.ndarray]):
        outtensors = []
        outtensors.append(create_ndarray_f32(max_shape_ndarray(intensors)))
        return outtensors

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        macs = 0
        params = 0

        macs += volume(outtensors[0].shape) * self.ratio * self.op_mac
        return macs, params


@NODEPROFILER_REGISTRY.register()
class Add(PWNBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = ADD_MACS

    def infer_shape(self, intensors: [numpy.ndarray]):
        return [intensors[0] + intensors[1]]


@NODEPROFILER_REGISTRY.register()
class Sum(PWNBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = ADD_MACS

    def infer_shape(self, intensors: [numpy.ndarray]):
        y = intensors[0]
        for i in range(1, len(intensors)):
            y = y + intensors[i]
        return [y]


@NODEPROFILER_REGISTRY.register()
class Abs(PWNBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = CMP_MACS

    def infer_shape(self, intensors: [numpy.ndarray]):
        return [numpy.abs(intensors[0])]


@NODEPROFILER_REGISTRY.register()
class Neg(PWNBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = CMP_MACS

    def infer_shape(self, intensors: [numpy.ndarray]):
        return [-intensors[0]]


@NODEPROFILER_REGISTRY.register()
class Sub(PWNBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = ADD_MACS

    def infer_shape(self, intensors: [numpy.ndarray]):
        return [intensors[0] - intensors[1]]


@NODEPROFILER_REGISTRY.register()
class Resize(NodeBase):
    def __init__(self, nodeproto):
        attnames = ['mode']
        auto_add_attributes(nodeproto.attribute, attnames, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        x = intensors[0]
        roi = []
        sizes = []
        if len(intensors) == 2:
            scales = intensors[1]
        elif len(intensors) >= 3:
            roi = intensors[1]
            scales = intensors[2]
            if len(intensors) >= 4:
                sizes = intensors[3]

        newshape = []
        if is_valid_ndarray(sizes):
            if len(sizes) == 4:
                newshape = sizes
            if len(sizes) == 2:
                newshape = x.shape[:2] + sizes
        else:
            if is_valid_ndarray(scales):
                newshape = []
                for src, scale in zip(x.shape, scales):
                    newshape.append(math.floor(src * scale))

        if is_valid_ndarray(newshape):
            if newshape.dtype != numpy.int64:
                newshape = newshape.astype(dtype=numpy.int64)
        return [create_ndarray_f32(newshape)]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        macs = 0
        outvol = volume(outtensors[0].shape)
        if self.mode == b'nearest':
            outvol *= 0
        elif self.mode == b'linear':
            outvol *= RESIZE_LINEAR_MACS
        elif self.mode == b'cubic':
            outvol *= RESIZE_CUBIC_MACS
        macs += outvol
        return macs, 0


# copy from https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/scatternd.py
def scatter_nd_impl(data, indices, updates, reduction='none'):  # type: ignore

    # Check tensor shapes
    assert indices.shape[-1] <= len(data.shape)
    assert updates.shape == indices.shape[:-1] + data.shape[indices.shape[-1]:]

    # Compute output
    output = numpy.copy(data)
    for i in numpy.ndindex(indices.shape[:-1]):
        # NOTE: The order of iteration in this loop is not specified.
        if reduction == 'add':
            output[indices[i]] += updates[i]
        elif reduction == 'mul':
            output[indices[i]] *= updates[i]
        else:
            output[indices[i]] = updates[i]
    return output


@NODEPROFILER_REGISTRY.register()
class ScatterND(NodeBase):
    def infer_shape(self, intensors: [numpy.ndarray]):
        data = intensors[0]
        indices = intensors[1]
        updates = intensors[2]
        return [scatter_nd_impl(data, indices, updates)]


# copy from https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/argmax.py
def argmax_use_numpy(data: numpy.ndarray, axis: int = 0, keepdims: int = 1) -> (numpy.ndarray):
    result = numpy.argmax(data, axis=axis)
    if (keepdims == 1):
        result = numpy.expand_dims(result, axis)
    return result.astype(numpy.int64)


@NODEPROFILER_REGISTRY.register()
class ArgMax(NodeBase):
    def __init__(self, n):
        super().__init__(n)
        attnames = ['axis', 'keepdims']
        self.keepdims = 1
        auto_add_attributes(n.attribute, attnames, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        data = intensors[0]
        return [argmax_use_numpy(data, self.axis, self.keepdims)]


@NODEPROFILER_REGISTRY.register()
class Upsample(Resize):
    pass


@NODEPROFILER_REGISTRY.register()
class Expand(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)

    def infer_shape(self, intensors: [numpy.ndarray]):
        input = intensors[0]
        shape = intensors[1]
        output = input * numpy.ones(shape.astype(numpy.int64), dtype=numpy.float32)
        return [output]


@NODEPROFILER_REGISTRY.register()
class Tile(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)

    def infer_shape(self, intensors: [numpy.ndarray]):
        input = intensors[0]
        repeats = intensors[1]
        output = numpy.tile(input, repeats)
        return [output]


@NODEPROFILER_REGISTRY.register()
class GRU(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        attnames = ['hidden_size']
        auto_add_attributes(nodeproto.attribute, attnames, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        x = intensors[0]
        w = intensors[1]
        seq_len = x.shape[0]
        batch = x.shape[1]
        num_dir = w.shape[0]
        h_len = w.shape[1] // 3
        return [create_ndarray_f32((seq_len, num_dir, batch, h_len)), create_ndarray_f32((num_dir, batch, h_len))]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        w = intensors[1]
        r = intensors[2]
        b = intensors[3]
        params = 0
        params += volume(w.shape) + volume(r.shape) + volume(b.shape)
        batch = intensors[0].shape[1]
        macs = volume(w.shape) + volume(r.shape) + volume(b.shape) * ADD_MACS
        macs *= batch
        return macs, params


@NODEPROFILER_REGISTRY.register()
class LSTM(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        attnames = ['direction', 'hidden_size']
        self.direction = None
        self.hidden_size = None
        auto_add_attributes(nodeproto.attribute, attnames, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        x = intensors[0]
        w = intensors[1]
        seq_len = x.shape[0]
        batch = x.shape[1]
        num_dir = w.shape[0]
        h_len = w.shape[1] // 4
        return [create_ndarray_f32((seq_len, num_dir, batch, h_len)), create_ndarray_f32((num_dir, batch, h_len))]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        w = intensors[1]
        r = intensors[2]
        b = intensors[3]
        params = 0
        params += volume(w.shape) + volume(r.shape) + volume(b.shape)
        batch = intensors[0].shape[1]
        macs = volume(w.shape) + volume(r.shape) + volume(b.shape) * ADD_MACS
        macs *= batch
        return macs, params


def auto_add_attributes(atts, attnames, obj):
    for att in atts:
        if att.name in attnames:
            obj.__setattr__(att.name, get_attribute_data(att))


@NODEPROFILER_REGISTRY.register()
class PoolBase(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        validAttr = ['kernel_shape', 'pads', 'strides', 'ceil_mode']
        self.kernel_shape = (3, 3)
        self.ceil_mode = 0
        self.pads = (0, 0, 0, 0)
        self.strides = (1, 1)
        self.dilations = (1, 1)
        auto_add_attributes(nodeproto.attribute, validAttr, self)
        self.op_mac = CMP_MACS

    def infer_shape(self, intensors: [numpy.ndarray]):
        if len(self.kernel_shape) == 1:
            inshape = intensors[0].shape
            outshape = inshape[:2] + (
                pooling_shape_calc(inshape[2], self.pads[0] + self.pads[1], self.kernel_shape[0], self.dilations[0],
                                   self.strides[0], self.ceil_mode),)
            return [create_ndarray_f32(outshape)]
        if len(self.kernel_shape) == 2:
            inshape = intensors[0].shape
            outshape = inshape[:2] + (
                pooling_shape_calc(inshape[2], self.pads[0] + self.pads[2], self.kernel_shape[0], self.dilations[0],
                                   self.strides[0],
                                   self.ceil_mode),
                pooling_shape_calc(inshape[3], self.pads[1] + self.pads[3], self.kernel_shape[1], self.dilations[1],
                                   self.strides[1],
                                   self.ceil_mode),
            )
            return [create_ndarray_f32(outshape)]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        outvol = volume(outtensors[0].shape)
        macs = outvol * CMP_MACS * self.kernel_shape[0]
        if len(self.kernel_shape) == 2:
            macs *= self.kernel_shape[1]
        return macs, 0


@NODEPROFILER_REGISTRY.register()
class MaxPool(PoolBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = CMP_MACS


@NODEPROFILER_REGISTRY.register()
class AveragePool(PoolBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = ADD_MACS


def axes_neg2pos(len, axes):
    newaxes = []
    for axis in axes:
        if axis < 0:
            newaxes.append(len + axis)
        else:
            newaxes.append(axis)
    return newaxes


@NODEPROFILER_REGISTRY.register()
class ReduceMean(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.axes = None
        self.keepdims = 1
        attnames = ['axes', 'keepdims']
        auto_add_attributes(nodeproto.attribute, attnames, self)
        self.axes = tuple(self.axes) if self.axes is not None else None

    def infer_shape(self, intensors: [numpy.ndarray]):
        reduced = numpy.mean(intensors[0], axis=self.axes, keepdims=self.keepdims == 1)
        return [reduced]
        # inshape=intensors[0].shape
        # shape=[]
        # if self.axes is None:
        #     if self.keepdims:
        #         for i in range(len(inshape)):
        #             shape.append(1)
        #     else:
        #         shape.append(1)
        # else:
        #     self.axes=axes_neg2pos(len(inshape),self.axes)
        #     for i in range(len(inshape)):
        #         if i in self.axes:
        #             if not self.keepdims:
        #                 continue
        #             else:
        #                 shape.append(1)
        #         else:
        #             shape.append(inshape[i])
        # return [create_ndarray_f32(shape)]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        data = intensors[0]
        vol = volume(data.shape)
        return vol * ADD_MACS, 0


@NODEPROFILER_REGISTRY.register()
class ReduceProd(ReduceMean):
    def infer_shape(self, intensors: [numpy.ndarray]):
        reduced = numpy.prod(intensors[0], axis=self.axes, keepdims=self.keepdims == 1)
        return [reduced]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        data = intensors[0]
        vol = volume(data.shape)
        return vol * MUL_MACS, 0


@NODEPROFILER_REGISTRY.register()
class ReduceL2(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.axes = None
        self.keepdims = 1
        attnames = ['axes', 'keepdims']
        auto_add_attributes(nodeproto.attribute, attnames, self)
        self.axes = tuple(self.axes) if self.axes is not None else None

    def infer_shape(self, intensors: [numpy.ndarray]):
        reduced = np.sqrt(numpy.sum(intensors[0], axis=self.axes, keepdims=self.keepdims == 1))
        return [reduced]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        data = intensors[0]
        vol = volume(data.shape)
        return vol * (ADD_MACS + SQRT_MACS), 0


@NODEPROFILER_REGISTRY.register()
class ReduceSum(ReduceMean):
    def infer_shape(self, intensors: [numpy.ndarray]):
        reduced = numpy.sum(intensors[0], axis=self.axes, keepdims=self.keepdims == 1)
        return [reduced]


@NODEPROFILER_REGISTRY.register()
class ReduceMin(ReduceMean):
    def infer_shape(self, intensors: [numpy.ndarray]):
        data = intensors[0]
        reduced = numpy.minimum.reduce(data, axis=self.axes, keepdims=self.keepdims == 1)
        return [reduced]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        data = intensors[0]
        vol = volume(data.shape)
        return vol * CMP_MACS, 0


@NODEPROFILER_REGISTRY.register()
class ReduceMax(ReduceMin):
    def infer_shape(self, intensors: [numpy.ndarray]):
        data = intensors[0]
        reduced = numpy.maximum.reduce(data, axis=self.axes, keepdims=self.keepdims == 1)
        return [reduced]


#
# @NODEPROFILER_REGISTRY.register()
# class ArgMax(NodeBase):
#     def __init__(self, nodeproto):
#         super().__init__(nodeproto)
#         self.axis = None
#         self.keepdims = 1
#         attnames = ['axis', 'keepdims']
#         auto_add_attributes(nodeproto.attribute, attnames, self)
#
#     def infer_shape(self,intensors:[numpy.ndarray]):
#         newshape=[]
#         ndim=len(intensors[0].shape)
#         self.axis=axes_neg2pos(ndim,[self.axis])[0]
#         for i in range(ndim):
#             if i == self.axis:
#                 if self.keepdims==1:
#                     newshape.append(1)
#             else:
#                 newshape.append(intensors[0].shape[i])
#         return [create_ndarray_f32(newshape)]

@NODEPROFILER_REGISTRY.register()
class Scan(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.num_scan_inputs = None
        self.scan_input_directions = None
        attnames = ['num_scan_inputs', 'scan_input_directions']
        auto_add_attributes(nodeproto.attribute, attnames, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        # TODO
        return [create_ndarray_f32((1, 1)), create_ndarray_f32((1, 1)), create_ndarray_f32((1,)),
                intensors[3], intensors[3], ]


@NODEPROFILER_REGISTRY.register()
class Compress(NodeBase):
    def __init__(self, node):
        super().__init__(node)
        self.axis = None
        attnames = ['axis']
        auto_add_attributes(node.attribute, attnames, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        return [numpy.compress(intensors[1], intensors[0], self.axis)]


@NODEPROFILER_REGISTRY.register()
class NonZero(NodeBase):
    def infer_shape(self, intensors: [numpy.ndarray]):
        condi = intensors[0]
        result = numpy.array(numpy.nonzero(condi), dtype=numpy.int64)
        if volume(result.shape) == 0:
            condi = numpy.ones_like(intensors[0])
            result = numpy.array(numpy.nonzero(condi), dtype=numpy.int64)
        return [result]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        return volume(outtensors[0].shape) * CMP_MACS, 0


@NODEPROFILER_REGISTRY.register()
class Less(NodeBase):
    def infer_shape(self, intensors: [numpy.ndarray]):
        result = numpy.less(intensors[0], intensors[1])
        return [result]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        return volume(outtensors[0].shape) * CMP_MACS, 0


@NODEPROFILER_REGISTRY.register()
class LessOrEqual(Less):
    def infer_shape(self, intensors: [numpy.ndarray]):
        result = numpy.less_equal(intensors[0], intensors[1])
        return [result]


@NODEPROFILER_REGISTRY.register()
class Not(NodeBase):
    def infer_shape(self, intensors: [numpy.ndarray]):
        result = numpy.logical_not(intensors[0].astype(numpy.bool))
        return [result]


@NODEPROFILER_REGISTRY.register()
class And(NodeBase):
    def infer_shape(self, intensors: [numpy.ndarray]):
        result = numpy.logical_and(intensors[0].astype(numpy.bool), intensors[1].astype(numpy.bool))
        return [result]


@NODEPROFILER_REGISTRY.register()
class Min(PWNBase):
    def __init__(self, node):
        super().__init__(node)
        self.op_mac = CMP_MACS
        self.ratio = self.nbinput - 1

    def infer_shape(self, intensors: [numpy.ndarray]):
        result = intensors[0]
        for i in range(1, self.nbinput):
            result = numpy.minimum(result, intensors[i])
        return [result]


@NODEPROFILER_REGISTRY.register()
class Where(NodeBase):
    def infer_shape(self, intensors: [numpy.ndarray]):
        result = numpy.where(intensors[0], intensors[1], intensors[2])
        return [result]


@NODEPROFILER_REGISTRY.register()
class Max(Min):
    def infer_shape(self, intensors: [numpy.ndarray]):
        result = intensors[0]
        for i in range(1, self.nbinput):
            result = numpy.maximum(result, intensors[i])
        return [result]


@NODEPROFILER_REGISTRY.register()
class Equal(NodeBase):
    def infer_shape(self, intensors: [numpy.ndarray]):
        result = numpy.equal(intensors[0], intensors[1])
        return [result]


@NODEPROFILER_REGISTRY.register()
class Greater(NodeBase):
    def infer_shape(self, intensors: [numpy.ndarray]):
        result = numpy.greater(intensors[0], intensors[1])
        return [result]


@NODEPROFILER_REGISTRY.register()
class RoiAlign(NodeBase):
    def __init__(self, node):
        super().__init__(node)
        attnames = ['mode', 'output_height', 'output_width', 'sampling_ratio', 'spatial_scale']
        auto_add_attributes(node.attribute, attnames, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        x = intensors[0]
        if len(x.shape) == 4 and self.output_height is not None and self.output_width is not None:
            newshape = x.shape[:2] + (self.output_height, self.output_width)
        else:
            raise NotImplementedError()
        return [create_ndarray_f32(newshape)]


# copy from https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/scatterelements.py
def scatter_elements(data, indices, updates, axis=0, reduction='none'):  # type: ignore
    if axis < 0:
        axis = data.ndim + axis

    idx_xsection_shape = indices.shape[:axis] + indices.shape[axis + 1:]

    def make_slice(arr, axis, i):  # type: ignore
        slc = [slice(None)] * arr.ndim
        slc[axis] = i
        return slc

    def unpack(packed):  # type: ignore
        unpacked = packed[0]
        for i in range(1, len(packed)):
            unpacked = unpacked, packed[i]
        return unpacked

    def make_indices_for_duplicate(idx):  # type: ignore
        final_idx = ()
        for i in range(len(idx[0])):
            final_idx.append(tuple(idx_element[i] for idx_element in idx))
        return (final_idx)

    # We use indices and axis parameters to create idx
    # idx is in a form that can be used as a NumPy advanced indices for scattering of updates param. in data
    idx = [[unpack(np.indices(idx_xsection_shape).reshape(indices.ndim - 1, -1)),
            indices[tuple(make_slice(indices, axis, i))].reshape(1, -1)[0]] for i in range(indices.shape[axis])]
    idx = (np.concatenate(idx, axis=1))
    idx.insert(axis, idx.pop())

    # updates_idx is a NumPy advanced indices for indexing of elements in the updates
    updates_idx = (idx)
    updates_idx.pop(axis)
    updates_idx.insert(axis, np.repeat(np.arange(indices.shape[axis]), np.prod(idx_xsection_shape)))

    scattered = np.copy(data)
    if reduction == 'none':
        scattered[tuple(idx)] = updates[tuple(updates_idx)]
    else:
        idx, updates_idx = make_indices_for_duplicate(idx), make_indices_for_duplicate(updates_idx)
        for iter, idx_set in enumerate(idx):
            if reduction == 'add':
                scattered[idx_set] += updates[updates_idx[iter]]
            elif reduction == 'mul':
                scattered[idx_set] *= updates[updates_idx[iter]]
    return scattered


@NODEPROFILER_REGISTRY.register()
class ScatterElements(NodeBase):
    def __init__(self, node):
        super().__init__(node)
        attnames = ['axis']
        auto_add_attributes(node.attribute, attnames, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        # TODO
        # y = scatter_elements(intensors[0], intensors[1], intensors[2], self.axis)
        # return [create_ndarray_f32(y)]
        return [intensors[0]]


@NODEPROFILER_REGISTRY.register()
class Hardmax(PWNBase):
    pass


@NODEPROFILER_REGISTRY.register()
class GlobalAveragePool(NodeBase):
    def infer_shape(self, intensors: [numpy.ndarray]):
        outtensors = []
        inshape = intensors[0].shape
        shape = inshape[0:2]
        for i in range(2, len(inshape)):
            shape += (1,)

        outtensors.append(create_ndarray_f32(shape))
        return outtensors

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        macs = 0
        params = 0
        macs += volume(intensors[0].shape) * ADD_MACS + volume(outtensors[0].shape) * DIV_MACS
        return macs, params


@NODEPROFILER_REGISTRY.register()
class CategoryMapper(PWNBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = 0


@NODEPROFILER_REGISTRY.register()
class TopK(NodeBase):
    def __init__(self, node):
        super().__init__(node)
        names = ['axis']
        self.axis = None
        auto_add_attributes(node.attribute, names, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        x = intensors[0]
        k = intensors[1][0]
        newshape = []
        for i in range(len(x.shape)):
            if i == self.axis:
                newshape.append(k)
            else:
                newshape.append(x.shape[i])
        return [create_ndarray_f32(newshape), create_ndarray_int64(newshape)]


@NODEPROFILER_REGISTRY.register()
class Relu(PWNBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = CMP_MACS
        self.ratio = 1


@NODEPROFILER_REGISTRY.register()
class PRelu(PWNBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = CMP_MACS + MUL_MACS
        self.ratio = 1


@NODEPROFILER_REGISTRY.register()
class Clip(PWNBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = CMP_MACS * 2
        self.ratio = 1


@NODEPROFILER_REGISTRY.register()
class Relu6(Clip):
    pass


@NODEPROFILER_REGISTRY.register()
class LRN(PWNBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        for att in nodeproto.attribute:
            if att.name == 'size':
                self.size = att.i

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        macs = 0
        params = 0
        outvol = volume(outtensors[0].shape)
        outvol *= (DIV_MACS + EXP_MACS + ADD_MACS + self.size * MUL_MACS)
        macs += outvol
        return macs, params


@NODEPROFILER_REGISTRY.register()
class Gemm(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        attnames = ['transA', 'transB']
        self.transA = None
        self.transB = None
        auto_add_attributes(nodeproto.attribute, attnames, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        x = intensors[0]
        w = intensors[1]

        if self.__class__ == Gemm:
            if self.transA is not None and self.transA > 0:
                xshape = x.shape[::-1]
            else:
                xshape = x.shape
            if self.transB is not None and self.transB > 0:
                yshape = xshape[:-1] + (w.shape[-2],)
            else:
                yshape = xshape[:-1] + (w.shape[-1],)
        else:
            yshape = x.shape[:-1] + (w.shape[-1],)

        return [create_ndarray_f32(yshape)]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        macs = 0
        params = 0
        xshape = intensors[0].shape
        if self.nbinput >= 2:
            weight_shape = intensors[1].shape
            params += volume(weight_shape)
            if self.nbinput == 3:
                params += volume(intensors[2].shape)

            macs = volume(xshape)
            if self.__class__ == Gemm:
                macs *= weight_shape[0]
            else:
                macs *= weight_shape[-1]

            if self.nbinput == 3:
                macs += volume(outtensors[0].shape) * ADD_MACS
        else:
            raise NotImplementedError()
        return macs, params


@NODEPROFILER_REGISTRY.register()
class MatMul(Gemm):
    pass


@NODEPROFILER_REGISTRY.register()
class Shape(NodeBase):
    def infer_shape(self, intensors: [numpy.ndarray]):
        outtensors = []
        outtensors.append(numpy.array(intensors[0].shape, numpy.int32))
        return outtensors


@NODEPROFILER_REGISTRY.register()
class Gather(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.axis = 0
        for att in nodeproto.attribute:
            if att.name == 'axis':
                self.axis = att.i

    def infer_shape(self, intensors: [numpy.ndarray]):
        outtensors = []
        out = numpy.take(intensors[0], intensors[1].astype(dtype=numpy.int), axis=self.axis)
        outtensors.append(out)
        return outtensors


@NODEPROFILER_REGISTRY.register()
class Constant(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.value = 0
        for att in nodeproto.attribute:
            if att.name == 'value':
                self.value = get_attribute_data(att)

    def infer_shape(self, intensors: [numpy.ndarray]):
        return [self.value]


@NODEPROFILER_REGISTRY.register()
class Unsqueeze(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.axes = [0]
        for att in nodeproto.attribute:
            if att.name == 'axes':
                self.axes = get_attribute_data(att)

    def infer_shape(self, intensors: [numpy.ndarray]):
        outtensor = intensors[0]
        for axis in self.axes:
            outtensor = numpy.expand_dims(outtensor, axis=axis)
        return [outtensor]


@NODEPROFILER_REGISTRY.register()
class Squeeze(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.axes = [0]
        for att in nodeproto.attribute:
            if att.name == 'axes':
                self.axes = get_attribute_data(att)

    def infer_shape(self, intensors: [numpy.ndarray]):
        outtensor = intensors[0]
        idx = 0
        if self.nbinput == 2:
            self.axes = intensors[1]
        for axis in self.axes:
            outtensor = numpy.squeeze(outtensor, axis=axis - idx)
            idx += 1
        return [outtensor]


@NODEPROFILER_REGISTRY.register()
class Concat(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.axis = 0
        for att in nodeproto.attribute:
            if att.name == 'axis':
                self.axis = get_attribute_data(att)

    def infer_shape(self, intensors: [numpy.ndarray]):
        outtensor = numpy.concatenate(intensors, self.axis)
        return [outtensor]


@NODEPROFILER_REGISTRY.register()
class Reshape(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)

    def infer_shape(self, intensors: [numpy.ndarray]):
        srcshape = intensors[0].shape
        shape = intensors[1]
        newshape = []
        for i in range(len(shape)):
            if shape[i] == 0:
                newshape.append(int(srcshape[i]))
            else:
                newshape.append(int(shape[i]))
        try:
            outtensor = (intensors[0].reshape(newshape))
        except:
            warnings.warn(
                f'node {self.name} cannot reshape array of size {tuple2str(srcshape)} into shape {tuple2str(shape)} ')
            outtensor = numpy.zeros(shape.astype(numpy.int64), intensors[0].dtype)
        return [outtensor]


# copy from https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/onehot.py
def one_hot(indices, depth, axis=-1, dtype=numpy.float32):  # type: ignore
    ''' Compute one hot from indices at a specific axis '''
    values = numpy.asarray(indices)
    rank = len(values.shape)
    depth_range = numpy.arange(depth)
    if axis < 0:
        axis += (rank + 1)
    ls = values.shape[0:axis]
    rs = values.shape[axis:rank]
    targets = numpy.reshape(depth_range, (1,) * len(ls) + depth_range.shape + (1,) * len(rs))
    values = numpy.reshape(numpy.mod(values, depth), ls + (1,) + rs)
    return numpy.asarray(targets == values, dtype=dtype)


@NODEPROFILER_REGISTRY.register()
class OneHot(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        attnames = ['axis']
        self.axis = -1
        auto_add_attributes(nodeproto.attribute, attnames, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        indices = intensors[0]
        depth = intensors[1]
        values = intensors[2]
        y = one_hot(indices, depth, self.axis)
        return [y]


@NODEPROFILER_REGISTRY.register()
class NonMaxSuppression(NodeBase):
    # TODO
    def infer_shape(self, intensors: [numpy.ndarray]):
        if self.nbinput >= 3:
            max_output_boxes_per_class = int(intensors[2][0])
            return [numpy.zeros((max_output_boxes_per_class, 3), dtype=numpy.int)]
        return [numpy.zeros((200, 3), dtype=numpy.int)]


class FusedNode(NodeBase):
    def __init__(self, node_proto):
        super().__init__(node_proto)

    def infer_shape(self, intensors: [numpy.ndarray]):
        return [intensors[0]]


@NODEPROFILER_REGISTRY.register()
class Identity(FusedNode):
    pass


@NODEPROFILER_REGISTRY.register()
class Erf(FusedNode):
    pass


@NODEPROFILER_REGISTRY.register()
class Dropout(FusedNode):
    pass


@NODEPROFILER_REGISTRY.register()
class Pad(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        attnames = ['pads', 'value']
        self.pads = None
        self.value = 0
        auto_add_attributes(nodeproto.attribute, attnames, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        data = intensors[0]
        newshape = []
        if self.pads is None:
            if self.nbinput > 1:
                pads = intensors[1]
                for i, v in enumerate(data.shape):
                    newshape.append(v + pads[i] + pads[i + len(data.shape)])
        else:
            for i, v in enumerate(data.shape):
                newshape.append(v + self.pads[i] + self.pads[i + len(data.shape)])
        newshape = [int(val) for val in newshape]
        return [create_ndarray_f32(newshape)]


@NODEPROFILER_REGISTRY.register()
class Split(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        attnames = ['axis', 'split']
        self.axis = None
        self.split = None
        auto_add_attributes(nodeproto.attribute, attnames, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        split = []
        end = 0
        if self.split is None:
            self.split = [intensors[0].shape[self.axis] // 2]
        self.axis = axes_neg2pos(len(intensors[0].shape), [self.axis])[0]
        for v in self.split:
            if end + v >= intensors[0].shape[self.axis]:
                break
            split.append(end + v)
            end += v
        return numpy.split(intensors[0], split, self.axis)


@NODEPROFILER_REGISTRY.register()
class Transpose(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        attnames = ['perm']
        self.perm = None
        auto_add_attributes(nodeproto.attribute, attnames, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        return [numpy.transpose(intensors[0], self.perm)]


@NODEPROFILER_REGISTRY.register()
class ConstantOfShape(NodeBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        attnames = ['value']
        self.value = None
        auto_add_attributes(nodeproto.attribute, attnames, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        arr = numpy.zeros(intensors[0].astype(numpy.int64), dtype=numpy.float32)
        if self.value is not None and len(self.value) == 1:
            arr.fill(self.value[0])
        return [arr]


@NODEPROFILER_REGISTRY.register()
class BatchNormalization(FusedNode):
    pass


@NODEPROFILER_REGISTRY.register()
class Slice(FusedNode):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        attnames = ['axes', 'ends', 'starts']
        auto_add_attributes(nodeproto.attribute, attnames, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        if len(intensors) == 3:
            data = intensors[0]
            starts = intensors[1]
            ends = intensors[2]
            return [data[starts[0]:ends[0]]]
        if len(intensors) == 4:
            data = intensors[0]
            starts = intensors[1]
            ends = intensors[2]
            axes = intensors[3]
            index = 0
            x = data
            for i in range(len(data.shape)):
                if i in axes:
                    if i == 0:
                        x = x[starts[index]:ends[index], ...]
                    if i == 1:
                        x = x[:, starts[index]:ends[index], ...]
                    if i == 2:
                        x = x[:, :, starts[index]:ends[index], ...]
                    if i == 3:
                        x = x[:, :, :, starts[index]:ends[index], ...]
                    if i == 4:
                        x = x[:, :, :, :, starts[index]:ends[index], ...]
                    index += 1
            return [x]
        if len(intensors) == 5:
            data = intensors[0]
            starts = intensors[1]
            ends = intensors[2]
            axes = intensors[3]
            steps = intensors[4]
            index = 0
            x = data
            for i in range(len(data.shape)):
                if i in axes:
                    if i == 0:
                        x = x[starts[index]:ends[index]:steps[index], ...]
                    if i == 1:
                        x = x[:, starts[index]:ends[index]:steps[index], ...]
                    if i == 2:
                        x = x[:, :, starts[index]:ends[index]:steps[index], ...]
                    if i == 3:
                        x = x[:, :, :, starts[index]:ends[index]:steps[index], ...]
                    if i == 4:
                        x = x[:, :, :, :, starts[index]:ends[index]:steps[index], ...]
                    index += 1
            return [x]
        if len(intensors) == 1:
            data = intensors[0]
            index = 0
            x = data
            for i in range(len(data.shape)):
                if i in self.axes:
                    if i == 0:
                        x = x[self.starts[index]:self.ends[index], ...]
                    if i == 1:
                        x = x[:, self.starts[index]:self.ends[index], ...]
                    if i == 2:
                        x = x[:, :, self.starts[index]:self.ends[index], ...]
                    if i == 3:
                        x = x[:, :, :, self.starts[index]:self.ends[index], ...]
                    if i == 4:
                        x = x[:, :, :, :, self.starts[index]:self.ends[index], ...]
                    index += 1
            return [x]


@NODEPROFILER_REGISTRY.register()
class Cast(FusedNode):
    def infer_shape(self, intensors: [numpy.ndarray]):
        return [intensors[0]]


@NODEPROFILER_REGISTRY.register()
class Flatten(NodeBase):
    def __init__(self, node):
        super().__init__(node)
        attnames = ['axis']
        self.axis = None
        auto_add_attributes(node.attribute, attnames, self)

    def infer_shape(self, intensors: [numpy.ndarray]):
        if self.axis is None:
            return [intensors[0].reshape((intensors[0].shape[0], -1))]
        else:
            vol = 1
            for i in range(self.axis):
                vol *= intensors[0].shape[i]
            return [intensors[0].reshape((vol, -1))]


@NODEPROFILER_REGISTRY.register()
class Exp(PWNBase):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = EXP_MACS
        self.ratio = 1


@NODEPROFILER_REGISTRY.register()
class Log(PWNBase):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = LOG_MACS
        self.ratio = 1


@NODEPROFILER_REGISTRY.register()
class CumSum(PWNBase):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = ADD_MACS
        self.ratio = 1


@NODEPROFILER_REGISTRY.register()
class Softmax(Exp):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = EXP_MACS + DIV_MACS
        self.ratio = 1


@NODEPROFILER_REGISTRY.register()
class Sigmoid(Exp):
    pass


@NODEPROFILER_REGISTRY.register()
class Tanh(PWNBase):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = EXP_MACS
        self.ratio = 2


@NODEPROFILER_REGISTRY.register()
class Mul(PWNBase):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = MUL_MACS

    def infer_shape(self, intensors: [numpy.ndarray]):
        return [intensors[0] * intensors[1]]


@NODEPROFILER_REGISTRY.register()
class InstanceNormalization(PWNBase):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = ADD_MACS + MUL_MACS + ADD_MACS + DIV_MACS


@NODEPROFILER_REGISTRY.register()
class Sqrt(PWNBase):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = SQRT_MACS


@NODEPROFILER_REGISTRY.register()
class Pow(PWNBase):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = POW_MACS


@NODEPROFILER_REGISTRY.register()
class Sin(PWNBase):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = SIN_MACS


@NODEPROFILER_REGISTRY.register()
class Cos(PWNBase):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = COS_MACS


@NODEPROFILER_REGISTRY.register()
class Div(PWNBase):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = DIV_MACS

    def infer_shape(self, intensors: [numpy.ndarray]):
        return [intensors[0] / (intensors[1])]


@NODEPROFILER_REGISTRY.register()
class Range(PWNBase):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = 1

    def infer_shape(self, intensors: [numpy.ndarray]):
        start = intensors[0]
        limit = intensors[1]
        delta = intensors[2]
        return [numpy.arange(start, limit, delta, dtype=numpy.float32)]


@NODEPROFILER_REGISTRY.register()
class Floor(FusedNode):
    def infer_shape(self, intensors: [numpy.ndarray]):
        return [numpy.floor(intensors[0])]


@NODEPROFILER_REGISTRY.register()
class Ceil(FusedNode):
    def infer_shape(self, intensors: [numpy.ndarray]):
        return [numpy.ceil(intensors[0])]


@NODEPROFILER_REGISTRY.register()
class Reciprocal(PWNBase):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = DIV_MACS


@NODEPROFILER_REGISTRY.register()
class HardSigmoid(PWNBase):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = MUL_MACS + ADD_MACS + CMP_MACS * 2


@NODEPROFILER_REGISTRY.register()
class LeakyRelu(PWNBase):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = MUL_MACS + CMP_MACS


def node_profile(node_proto: str, ins: [], outs: []):
    node_class = NODEPROFILER_REGISTRY.get(node_proto.op_type)
    if node_class != None:
        profler = node_class(node_proto)
        return profler.profile(ins, outs)
    warnings.warn(f'node {node_proto.op_type} is not registed for profiling, return 0 Macs and 0 params as default. '
                  f'Use NODEPROFILER_REGISTRY to register your profiler for this node.')
    return 0, 0


def node_infer_shape(node_proto: str, ins: []):
    node_class = NODEPROFILER_REGISTRY.get(node_proto.op_type)
    if node_class != None:
        profler = node_class(node_proto)
        return profler.infer_shape(ins)
    raise NotImplementedError(
        f'node {node_proto.op_type} is not registed for profiling!!! Use NODEPROFILER_REGISTRY to register your profiler for this node.')
