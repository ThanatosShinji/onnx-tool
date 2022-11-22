import numpy
import onnx
import math
import warnings
from .tensor import get_attribute_data, volume, is_valid_ndarray, create_ndarray_f32
from .utils import NODE_REGISTRY

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

_SHAPE_TENSORS = {
    'Reshape': ('1of2',),
    'Resize': ('2of3', '3of4'),
    'Slice': ('1,2of3', '1,2,3of4', '1,2,3,4of5'),
    'Expand': ('1of2',),
}


def _conv_output_shape(xin, pad, ksize, stride, dilation):
    return int((xin + pad - dilation * (ksize - 1) - 1) / stride + 1)


def _pooling_shape_calc(inshape, pad, kshape, dilation, stride, ceilmode):
    outshape = (inshape + pad - ((kshape - 1) * dilation + 1)) / stride + 1
    if ceilmode:
        return math.ceil(outshape)
    return math.floor(outshape)


def _axes_neg2pos(len, axes):
    newaxes = []
    for axis in axes:
        if axis < 0:
            newaxes.append(len + axis)
        else:
            newaxes.append(axis)
    return newaxes


def _max_shape(shapes: [numpy.ndarray]):
    maxshape = shapes[0]
    maxvol = volume(maxshape)
    for shape in shapes:
        vol = volume(shape)
        if vol > maxvol:
            maxshape = shape
            maxvol = vol
        elif vol == maxvol:
            if len(shape) > len(maxshape):
                maxshape = shape
    return maxshape


def _contains_shape_tensor(n):
    nodeset = _SHAPE_TENSORS.keys()
    shape_tensors = []
    if n.op_type in nodeset:
        tensor_descs = _SHAPE_TENSORS[n.op_type]
        for desc in tensor_descs:
            strs = desc.split('of')
            indice = strs[0]
            count = int(strs[1])
            if len(n.input) == count:
                indistr = indice.split(',')
                for istr in indistr:
                    shape_tensors.append(n.input[int(istr)])
    return shape_tensors


def _get_shape(item):
    if isinstance(item, numpy.ndarray):
        return item.shape
    elif isinstance(item, (list, tuple)):
        return list(item)


def _get_tensor(item):
    if isinstance(item, numpy.ndarray):
        return item
    elif isinstance(item, (list, tuple)):
        return create_ndarray_f32(item)


class Node():
    def __init__(self, n: onnx.NodeProto):
        self.name = n.name
        self.op_type = n.op_type
        self.nextnodes = []
        self.prevnodes = []
        self.output = []
        self.input = []
        self.proto = n
        self.shape_calc = False

        for att in n.attribute:
            self.__setattr__(att.name, get_attribute_data(att))

    def shape_infer(self, intensors: []):
        return []

    def value_infer(self, intensors: []):
        return []

    def profile(self, intensors: [], outtensors: []):
        return 0


class PWNode(Node):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = ADD_MACS
        self.ratio = max(1, len(self.input) - 1)

    def shape_infer(self, intensors: []):
        outshapes = []
        inshapes = []
        for item in intensors:
            inshapes.append(_get_shape(item))
        outshapes.append(_max_shape(intensors))
        return outshapes

    def profile(self, intensors: [], outtensors: []):
        outshape = _get_shape(outtensors[0])
        macs = volume(outshape) * self.ratio * self.op_mac
        return macs


@NODE_REGISTRY.register()
class SubNode(PWNode):
    pass


@NODE_REGISTRY.register()
class AddNode(PWNode):
    pass


@NODE_REGISTRY.register()
class DivNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = DIV_MACS


@NODE_REGISTRY.register()
class MulNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = MUL_MACS


@NODE_REGISTRY.register()
class ExpNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = EXP_MACS
        self.ratio = 1


@NODE_REGISTRY.register()
class LogNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = LOG_MACS
        self.ratio = 1


@NODE_REGISTRY.register()
class SigmoidNode(ExpNode):
    pass


@NODE_REGISTRY.register()
class TanhNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = EXP_MACS
        self.ratio = 2


@NODE_REGISTRY.register()
class HardSigmoidNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = MUL_MACS + ADD_MACS + CMP_MACS * 2


@NODE_REGISTRY.register()
class ReluNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = CMP_MACS


@NODE_REGISTRY.register()
class ClipNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = CMP_MACS * 2
        self.ratio = 1


@NODE_REGISTRY.register()
class Relu6Node(PWNode):
    pass


@NODE_REGISTRY.register()
class ConstantNode(Node):
    pass


@NODE_REGISTRY.register()
class ConcatNode(Node):
    def shape_infer(self, intensors: [numpy.ndarray]):
        inshapes = []
        for tensor in intensors:
            inshapes.append(_get_shape(tensor))
        faketensors = []
        for shape in inshapes:
            faketensors.append(create_ndarray_f32(shape))
        outtensor = numpy.concatenate(faketensors, self.axis)
        return [outtensor.shape]

    def value_infer(self, intensors: [numpy.ndarray]):
        outtensor = numpy.concatenate(intensors, self.axis)
        return [outtensor]


@NODE_REGISTRY.register()
class ShapeNode(Node):
    def shape_infer(self, intensors: [numpy.ndarray]):
        return [_get_shape(intensors[0])]

    def value_infer(self, intensors: [numpy.ndarray]):
        return [numpy.array(_get_shape(intensors[0]), dtype=numpy.int)]


@NODE_REGISTRY.register()
class ResizeNode(Node):
    def shape_infer(self, intensors: [numpy.ndarray]):
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
                newshape = x[:2] + sizes
        else:
            if is_valid_ndarray(scales):
                newshape = []
                for src, scale in zip(x, scales):
                    newshape.append(math.floor(src * scale))

        if is_valid_ndarray(newshape):
            if newshape.dtype != numpy.int64:
                newshape = newshape.astype(dtype=numpy.int64)
        return [list(newshape)]


@NODE_REGISTRY.register()
class PoolBase(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = CMP_MACS

        if not hasattr(self, 'kernel_shape'):
            self.kernel_shape = (3, 3)
        if not hasattr(self, 'ceil_mode'):
            self.ceil_mode = 0
        if not hasattr(self, 'pads'):
            self.pads = (0, 0, 0, 0)
        if not hasattr(self, 'strides'):
            self.strides = (1, 1)
        if not hasattr(self, 'dilations'):
            self.dilations = (1, 1)

    def shape_infer(self, intensors: []):

        if len(self.kernel_shape) == 1:
            inshape = _get_shape(intensors[0])
            outshape = inshape[:2] + [
                _pooling_shape_calc(inshape[2], self.pads[0] + self.pads[1], self.kernel_shape[0], self.dilations[0],
                                    self.strides[0], self.ceil_mode), ]
            return [outshape]
        if len(self.kernel_shape) == 2:
            inshape = _get_shape(intensors[0])
            outshape = inshape[:2] + [
                _pooling_shape_calc(inshape[2], self.pads[0] + self.pads[2], self.kernel_shape[0], self.dilations[0],
                                    self.strides[0],
                                    self.ceil_mode),
                _pooling_shape_calc(inshape[3], self.pads[1] + self.pads[3], self.kernel_shape[1], self.dilations[1],
                                    self.strides[1],
                                    self.ceil_mode),
            ]
            return [outshape]

    def profile(self, intensors: [], outtensors: []):
        outshape = _get_shape(outtensors[0])
        outvol = volume(outshape)
        macs = outvol * CMP_MACS * self.kernel_shape[0]
        if len(self.kernel_shape) == 2:
            macs *= self.kernel_shape[1]
        return macs


@NODE_REGISTRY.register()
class AveragePoolNode(PoolBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = ADD_MACS


@NODE_REGISTRY.register()
class GlobalAveragePoolNode(Node):
    def shape_infer(self, intensors: []):
        inshape = _get_shape(intensors[0])
        shape = inshape[0:2]
        for i in range(2, len(inshape)):
            shape += (1,)
        return [shape]

    def profile(self, intensors: [], outtensors: []):
        inshape = _get_shape(intensors[0])
        outshape = _get_shape(outtensors[0])
        macs = volume(inshape) * ADD_MACS + volume(outshape) * DIV_MACS
        return macs


@NODE_REGISTRY.register()
class ExpandNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)

    def shape_infer(self, intensors: []):
        input = _get_shape(intensors[0])
        shape = _get_shape(intensors[1])
        output = numpy.ones(input, dtype=numpy.float32) * numpy.ones(shape, dtype=numpy.float32)
        return [output.shape]

    def value_infer(self, intensors: []):
        output = intensors[0] * numpy.ones(intensors[1].astype(numpy.int64), dtype=numpy.float32)
        return [output]


@NODE_REGISTRY.register()
class SliceNode(Node):
    def shape_infer(self, intensors: []):
        data = _get_tensor(intensors[0])
        datashape = _get_shape(intensors[0])
        if len(intensors) == 3:
            starts = intensors[1]
            ends = intensors[2]
            x = data[starts[0]:ends[0]]
        if len(intensors) == 4:
            starts = intensors[1]
            ends = intensors[2]
            axes = intensors[3]
            index = 0
            x = data
            for i in range(len(datashape)):
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
        if len(intensors) == 5:
            starts = intensors[1]
            ends = intensors[2]
            axes = intensors[3]
            steps = intensors[4]
            index = 0
            x = data
            for i in range(len(datashape)):
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
        if len(intensors) == 1:
            index = 0
            x = data
            for i in range(len(datashape)):
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
        return [x.shape]

    def value_infer(self, intensors: []):
        data = intensors[0]
        datashape = _get_shape(intensors[0])
        if len(intensors) == 3:
            starts = intensors[1]
            ends = intensors[2]
            x = data[starts[0]:ends[0]]
        if len(intensors) == 4:
            starts = intensors[1]
            ends = intensors[2]
            axes = intensors[3]
            index = 0
            x = data
            for i in range(len(datashape)):
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
        if len(intensors) == 5:
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
        if len(intensors) == 1:
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


@NODE_REGISTRY.register()
class ReduceMeanNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)

    def shape_infer(self, intensors: [numpy.ndarray]):
        inshape = _get_shape(intensors[0])
        reduced = numpy.mean(numpy.ones(inshape, dtype=numpy.float32), axis=tuple(self.axes),
                             keepdims=self.keepdims == 1)
        return [reduced.shape]

    def value_infer(self, intensors: [numpy.ndarray]):
        reduced = numpy.mean(intensors[0], axis=self.axes, keepdims=self.keepdims == 1)
        return [reduced]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        data = intensors[0]
        vol = volume(data.shape)
        return vol * ADD_MACS, 0


@NODE_REGISTRY.register()
class ConvNode(Node):
    def shape_infer(self, intensors: []):
        outshapes = []
        xshape = _get_shape(intensors[0])
        wshape = _get_shape(intensors[1])
        shape = []
        if hasattr(self, 'auto_pad') and self.auto_pad != b'NOTSET':
            if self.auto_pad in [b'SAME_LOWER', b'SAME_UPPER']:
                shape = (xshape[0], wshape[0], math.ceil(xshape[2] / self.strides[0]))
                if len(xshape) == 4:
                    shape += (math.ceil(xshape[3] / self.strides[1]),)
        else:
            if len(xshape) == 4:
                oh = _conv_output_shape(xshape[2], self.pads[0] + self.pads[2], wshape[2], self.strides[0],
                                        self.dilations[0])
                ow = _conv_output_shape(xshape[3], self.pads[1] + self.pads[3], wshape[3], self.strides[1],
                                        self.dilations[1])
                shape = (xshape[0], wshape[0], oh, ow)
            elif len(xshape) == 3:
                oh = _conv_output_shape(xshape[2], self.pads[0] + self.pads[1], wshape[2], self.strides[0],
                                        self.dilations[0])
                shape = (xshape[0], wshape[0], oh)
        outshapes.append(shape)
        return outshapes


@NODE_REGISTRY.register()
class SplitNode(Node):
    def shape_infer(self, intensors: []):
        split = []
        end = 0
        inshape = _get_shape(intensors[0])
        fakeintensor = create_ndarray_f32(inshape)
        if self.split is None:
            if len(intensors) == 2:
                self.split = intensors[1]
            else:
                self.split = [inshape[self.axis] // 2]

        self.axis = _axes_neg2pos(len(inshape), [self.axis])[0]
        for v in self.split:
            if end + v >= inshape[self.axis]:
                break
            split.append(end + v)
            end += v
        outtensor = numpy.split(fakeintensor, split, self.axis)
        outshape = []
        for t in outtensor:
            outshape.append(t.shape)
        return outshape

    def value_infer(self, intensors: []):
        split = []
        end = 0
        if self.split is None:
            if len(intensors) == 2:
                self.split = intensors[1]
            else:
                self.split = [intensors[0].shape[self.axis] // 2]

        self.axis = _axes_neg2pos(len(intensors[0].shape), [self.axis])[0]
        for v in self.split:
            if end + v >= intensors[0].shape[self.axis]:
                break
            split.append(end + v)
            end += v
        return numpy.split(intensors[0], split, self.axis)


def create_node(n: onnx.NodeProto):
    node_class = NODE_REGISTRY.get(n.op_type + 'Node')
    if node_class != None:
        instance = node_class(n)
        return instance
    warnings.warn(f'node {n.op_type} is not registed for profiling, return 0 Macs and 0 params as default. '
                  f'Use NODEPROFILER_REGISTRY to register your profiler for this node.')
    return Node(n)
