import numpy
import onnx
import math
import warnings
from .tensor import get_attribute_data, volume, is_valid_ndarray, create_ndarray_f32, onnxdtype2npdtype
from .utils import NODE_REGISTRY, tuple2str

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


def _convtranspose_output_shape(xin, output_padding, pad, ksize, stride, dilation):
    return stride * (xin - 1) + output_padding + ((ksize - 1) * dilation + 1) - pad


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
        return list(item.shape)
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
        self.attr = {}
        if n.output[0] == 'onnx::Reshape_251':
            print(111)
        for att in n.attribute:
            self.attr[att.name] = onnx.helper.get_attribute_value(att)
            self.__setattr__(att.name, get_attribute_data(att))
            if att.name == 'axes':
                if isinstance(self.axes, list):
                    self.axes = tuple(self.axes)

    def add_default_value(self, attname, defaultvalue):
        if not hasattr(self, attname):
            setattr(self, attname, defaultvalue)

    def make_nodeproto(self):
        return onnx.helper.make_node(self.op_type, self.input, self.output, self.name, **self.attr)
        self.proto.name = self.name
        self.proto.op_type = self.op_type
        return self.proto

    def shape_infer(self, intensors: []):
        faketensors = [_get_tensor(tensor) for tensor in intensors]
        outtensors = self.value_infer(faketensors)
        outshapes = [_get_shape(tensor) for tensor in outtensors]
        return outshapes

    def value_infer(self, intensors: []):
        return []

    def profile(self, intensors: [], outtensors: []):
        return 0


class FusedBase(Node):
    def shape_infer(self, intensors: []):
        outshapes = []
        outshapes.append(_get_shape(intensors[0]))
        return outshapes

    def value_infer(self, intensors: []):
        return [intensors[0]]

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
        outshapes.append(_max_shape(inshapes))
        return outshapes

    def profile(self, intensors: [], outtensors: []):
        outshape = _get_shape(outtensors[0])
        macs = volume(outshape) * self.ratio * self.op_mac
        return macs


class NpMathBase(Node):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = ADD_MACS
        self.ratio = max(1, len(self.input) - 1)

    def shape_infer(self, intensors: []):
        maxlen = 0
        for tensor in intensors:
            shape = _get_shape(tensor)
            if len(shape) > maxlen:
                maxlen = len(shape)
        inshapes = []
        for tensor in intensors:
            shape = _get_shape(tensor)
            for i in range(0, maxlen - len(shape)):
                shape = [1, ] + shape
            inshapes.append(shape)
        outshape = []
        for i in range(maxlen):
            maxdim = 0
            for shape in inshapes:
                if shape[i] > maxdim:
                    maxdim = shape[i]
            outshape.append(maxdim)
        return [outshape]

    def profile(self, intensors: [], outtensors: []):
        outshape = _get_shape(outtensors[0])
        macs = volume(outshape) * self.ratio * self.op_mac
        return macs


@NODE_REGISTRY.register()
class SubNode(NpMathBase):
    def value_infer(self, intensors: []):
        return [intensors[0] - intensors[1]]


@NODE_REGISTRY.register()
class AddNode(NpMathBase):
    def value_infer(self, intensors: []):
        return [intensors[0] + intensors[1]]


@NODE_REGISTRY.register()
class MinNode(NpMathBase):
    def __init__(self, node):
        super().__init__(node)
        self.op_mac = CMP_MACS

    def value_infer(self, intensors: []):
        result = intensors[0]
        for i in range(1, len(intensors)):
            result = numpy.minimum(result, intensors[i])
        return [result]


@NODE_REGISTRY.register()
class MaxNode(NpMathBase):
    def value_infer(self, intensors: []):
        result = intensors[0]
        for i in range(1, len(intensors)):
            result = numpy.maximum(result, intensors[i])
        return [result]


@NODE_REGISTRY.register()
class NegNode(NpMathBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = CMP_MACS

    def value_infer(self, intensors: []):
        return [-intensors[0]]


@NODE_REGISTRY.register()
class DivNode(NpMathBase):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = DIV_MACS

    def value_infer(self, intensors: []):
        return [intensors[0] / intensors[1]]


@NODE_REGISTRY.register()
class MulNode(NpMathBase):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = MUL_MACS

    def value_infer(self, intensors: []):
        return [intensors[0] * intensors[1]]


@NODE_REGISTRY.register()
class AbsNode(NpMathBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = CMP_MACS

    def value_infer(self, intensors: []):
        return [numpy.abs(intensors[0])]


@NODE_REGISTRY.register()
class CeilNode(NpMathBase):
    def value_infer(self, intensors: []):
        return [numpy.ceil(intensors[0])]


@NODE_REGISTRY.register()
class ExpNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = EXP_MACS
        self.ratio = 1


@NODE_REGISTRY.register()
class SoftmaxNode(ExpNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = EXP_MACS + DIV_MACS
        self.ratio = 1


@NODE_REGISTRY.register()
class LogNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = LOG_MACS
        self.ratio = 1


@NODE_REGISTRY.register()
class ImageScalerNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = ADD_MACS + MUL_MACS
        self.ratio = 1


@NODE_REGISTRY.register()
class InstanceNormalizationNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = ADD_MACS + MUL_MACS + ADD_MACS + DIV_MACS


@NODE_REGISTRY.register()
class SqrtNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = SQRT_MACS


@NODE_REGISTRY.register()
class PowNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = POW_MACS


@NODE_REGISTRY.register()
class SinNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = SIN_MACS


@NODE_REGISTRY.register()
class CosNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = COS_MACS


@NODE_REGISTRY.register()
class RangeNode(Node):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = 1

    def value_infer(self, intensors: []):
        start = intensors[0]
        limit = intensors[1]
        delta = intensors[2]
        return [numpy.arange(start, limit, delta, dtype=numpy.float32)]


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
class PReluNode(PWNode):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = CMP_MACS + MUL_MACS
        self.ratio = 1


@NODE_REGISTRY.register()
class LeakyReluNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = self.op_mac = MUL_MACS + CMP_MACS


@NODE_REGISTRY.register()
class SumNode(PWNode):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = ADD_MACS

    def value_infer(self, intensors: []):
        y = intensors[0]
        for i in range(1, len(intensors)):
            y = y + intensors[i]
        return [y]


@NODE_REGISTRY.register()
class NonMaxSuppressionNode(Node):
    # TODO
    def value_infer(self, intensors: []):
        if self.nbinput >= 3:
            max_output_boxes_per_class = int(intensors[2][0])
            return [numpy.zeros((max_output_boxes_per_class, 3), dtype=numpy.int)]
        return [numpy.zeros((200, 3), dtype=numpy.int)]


@NODE_REGISTRY.register()
class LRNNode(PWNode):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)

    def profile(self, intensors: [], outtensors: []):
        macs = 0
        outvol = volume(_get_shape(outtensors[0]))
        outvol *= (DIV_MACS + EXP_MACS + ADD_MACS + self.size * MUL_MACS)
        macs += outvol
        return macs


@NODE_REGISTRY.register()
class LessNode(Node):
    def value_infer(self, intensors: []):
        result = numpy.less(intensors[0], intensors[1])
        return [result]

    def profile(self, intensors: [], outtensors: []):
        return volume(_get_shape(outtensors[0])) * CMP_MACS


@NODE_REGISTRY.register()
class LessOrEqualNode(LessNode):
    def value_infer(self, intensors: [numpy.ndarray]):
        result = numpy.less_equal(intensors[0], intensors[1])
        return [result]


@NODE_REGISTRY.register()
class NotNode(Node):
    def value_infer(self, intensors: []):
        result = numpy.logical_not(intensors[0].astype(numpy.bool))
        return [result]


@NODE_REGISTRY.register()
class AndNode(Node):
    def value_infer(self, intensors: []):
        result = numpy.logical_and(intensors[0].astype(numpy.bool), intensors[1].astype(numpy.bool))
        return [result]


@NODE_REGISTRY.register()
class WhereNode(Node):
    def value_infer(self, intensors: []):
        result = numpy.where(intensors[0], intensors[1], intensors[2])
        return [result]


@NODE_REGISTRY.register()
class TransposeNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value('perm', None)

    def shape_infer(self, intensors: []):
        xshape = _get_shape(intensors[0])
        yshape = []
        if self.perm is None:
            return [xshape]
        for axis in self.perm:
            yshape.append(xshape[axis])
        return [yshape]

    def value_infer(self, intensors: []):
        return [numpy.transpose(intensors[0], self.perm)]


@NODE_REGISTRY.register()
class GemmNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value('transA', None)
        self.add_default_value('transB', None)

    def shape_infer(self, intensors: []):
        xshape = _get_shape(intensors[0])
        wshape = _get_shape(intensors[1])
        if self.__class__ == GemmNode:
            if self.transA is not None and self.transA > 0:
                xshape = xshape[::-1]
            else:
                xshape = xshape
            if self.transB is not None and self.transB > 0:
                yshape = xshape[:-1] + [wshape[-2], ]
            else:
                yshape = xshape[:-1] + [wshape[-1], ]
        else:
            yshape = xshape[:-1] + [wshape[-1], ]

        return [yshape]

    def value_infer(self, intensors: []):
        return [intensors[0] * intensors[1]]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        xshape = _get_shape(intensors[0])
        if len(intensors) >= 2:
            weight_shape = _get_shape(intensors[1])
            macs = volume(xshape)
            if self.__class__ == GemmNode:
                macs *= weight_shape[0]
            else:
                macs *= weight_shape[-1]

            if len(intensors) == 3:
                macs += volume(_get_shape(outtensors[0])) * ADD_MACS
        else:
            raise NotImplementedError()
        return macs


@NODE_REGISTRY.register()
class MatMulNode(GemmNode):
    pass


@NODE_REGISTRY.register()
class TileNode(Node):
    def value_infer(self, intensors: []):
        input = intensors[0]
        repeats = intensors[1]
        output = numpy.tile(input, repeats)
        return [output]


@NODE_REGISTRY.register()
class GatherNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value('axis', 0)

    def shape_infer(self, intensors: []):
        xshape = _get_shape(intensors[0])
        idxshape = _get_shape(intensors[1])
        axis = _axes_neg2pos(len(xshape), [self.axis])[0]
        yshape = []
        for i in range(len(xshape)):
            if i == axis:
                yshape.extend(idxshape)
            else:
                yshape.append(xshape[i])
        return [yshape]

    def value_infer(self, intensors: []):
        outtensors = []
        out = numpy.take(intensors[0], intensors[1].astype(dtype=numpy.int), axis=self.axis)
        outtensors.append(out)
        return outtensors


@NODE_REGISTRY.register()
class ClipNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = CMP_MACS * 2
        self.ratio = 1


@NODE_REGISTRY.register()
class ReciprocalNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = DIV_MACS


@NODE_REGISTRY.register()
class Relu6Node(PWNode):
    pass


@NODE_REGISTRY.register()
class ConstantNode(Node):
    pass


@NODE_REGISTRY.register()
class ConcatNode(Node):
    def shape_infer(self, intensors: [numpy.ndarray]):
        outshape = _get_shape(intensors[0])
        for i in range(len(intensors) - 1):
            shape = _get_shape(intensors[i + 1])
            outshape[self.axis] += shape[self.axis]
        return [outshape]

    def value_infer(self, intensors: [numpy.ndarray]):
        outtensor = numpy.concatenate(intensors, self.axis)
        return [outtensor]


from .node_profilers import one_hot


@NODE_REGISTRY.register()
class OneHotNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value('axis', -1)

    def value_infer(self, intensors: []):
        indices = intensors[0]
        depth = intensors[1]
        values = intensors[2]
        y = one_hot(indices, depth, self.axis)
        return [y]


@NODE_REGISTRY.register()
class EinsumNode(Node):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        strs = self.equation.split(b',')
        self.ashape = strs[0].replace(b' ', b'')
        strs = strs[1].split(b'->')
        self.bshape = strs[0].replace(b' ', b'')
        self.cshape = strs[1].replace(b' ', b'')

    def shape_infer(self, intensors: []):
        shape = []
        map = {}
        shape0 = _get_shape(intensors[0])
        shape1 = _get_shape(intensors[1])
        for i, v in enumerate(shape0):
            map[self.ashape[i]] = v
        for i, v in enumerate(shape1):
            map[self.bshape[i]] = v
        for k in self.cshape:
            shape.append(map[k])
        return [shape]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        macs = 1
        map = {}
        shape0 = _get_shape(intensors[0])
        shape1 = _get_shape(intensors[1])
        for i, v in enumerate(shape0):
            map[self.ashape[i]] = v
        for i, v in enumerate(shape1):
            map[self.bshape[i]] = v
        for key in map.keys():
            macs *= map[key]
        return macs


@NODE_REGISTRY.register()
class UnsqueezeNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value('axes', None)

    def value_infer(self, intensors: []):
        outtensor = intensors[0]
        if self.axes is None:
            axes = intensors[1]
        else:
            axes = self.axes
        for axis in axes:
            outtensor = numpy.expand_dims(outtensor, axis=axis)
        return [outtensor]


@NODE_REGISTRY.register()
class SqueezeNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value('axes', [0])

    def value_infer(self, intensors: [numpy.ndarray]):
        outtensor = intensors[0]
        idx = 0
        if len(intensors) == 2:
            self.axes = intensors[1]
        for axis in self.axes:
            outtensor = numpy.squeeze(outtensor, axis=axis - idx)
            idx += 1
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
        xshape = _get_shape(intensors[0])
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
                newshape = xshape[:2] + sizes
        else:
            if is_valid_ndarray(scales):
                newshape = []
                for src, scale in zip(xshape, scales):
                    newshape.append(math.floor(src * scale))

        if is_valid_ndarray(newshape):
            if newshape.dtype != numpy.int64:
                newshape = newshape.astype(dtype=numpy.int64)
        return [list(newshape)]

    def profile(self, intensors: [], outtensors: []):
        macs = 0
        outvol = volume(_get_shape(outtensors[0]))
        if self.mode == b'nearest':
            outvol *= 0
        elif self.mode == b'linear':
            outvol *= RESIZE_LINEAR_MACS
        elif self.mode == b'cubic':
            outvol *= RESIZE_CUBIC_MACS
        macs += outvol
        return macs


@NODE_REGISTRY.register()
class UpsampleNode(ResizeNode):
    pass


from .node_profilers import auto_pad_valid_shape_calc, auto_pad_same_shape_calc, pooling_shape_calc


@NODE_REGISTRY.register()
class PoolBase(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = CMP_MACS
        self.add_default_value('kernel_shape', (3, 3))
        self.add_default_value('ceil_mode', 0)
        self.add_default_value('pads', (0, 0, 0, 0))
        self.add_default_value('strides', (1, 1))
        self.add_default_value('dilations', (1, 1))
        self.add_default_value('auto_pad', None)

    def shape_infer(self, intensors: []):
        inshape = _get_shape(intensors[0])
        if self.auto_pad is not None and self.auto_pad != b'NOTSET':
            outshape = inshape[:2]
            if self.auto_pad in [b'SAME_LOWER', b'SAME_UPPER']:
                outshape += (auto_pad_same_shape_calc(inshape[2], self.strides[0]),)
                if len(self.strides) == 2:
                    outshape += [auto_pad_same_shape_calc(inshape[3], self.strides[1]), ]
            elif self.auto_pad == b'VALID':
                outshape += [auto_pad_valid_shape_calc(inshape[2], self.kernel_shape[0], self.strides[0]), ]
                if len(self.strides) == 2:
                    outshape += [auto_pad_valid_shape_calc(inshape[3], self.kernel_shape[1], self.strides[1]), ]
        else:
            if len(self.kernel_shape) == 1:
                outshape = inshape[:2] + [
                    pooling_shape_calc(inshape[2], self.pads[0] + self.pads[1], self.kernel_shape[0], self.dilations[0],
                                       self.strides[0], self.ceil_mode), ]
                return [create_ndarray_f32(outshape)]
            if len(self.kernel_shape) == 2:
                outshape = inshape[:2] + [
                    pooling_shape_calc(inshape[2], self.pads[0] + self.pads[2], self.kernel_shape[0], self.dilations[0],
                                       self.strides[0],
                                       self.ceil_mode),
                    pooling_shape_calc(inshape[3], self.pads[1] + self.pads[3], self.kernel_shape[1], self.dilations[1],
                                       self.strides[1],
                                       self.ceil_mode),
                ]
        return [outshape, ]

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
class MaxPoolNode(PoolBase):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = CMP_MACS


@NODE_REGISTRY.register()
class DropoutNode(FusedBase):
    def shape_infer(self, intensors: []):
        if len(self.input) == 1:
            return [_get_shape(intensors[0])]
        return [_get_shape(intensors[0]), _get_shape(intensors[0])]


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

    def value_infer(self, intensors: []):
        output = intensors[0] * numpy.ones(intensors[1].astype(numpy.int64), dtype=numpy.float32)
        return [output]


@NODE_REGISTRY.register()
class PadNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value('pads', None)
        self.add_default_value('value', 0)

    def shape_infer(self, intensors: []):
        inshape = _get_shape(intensors[0])
        newshape = []
        if self.pads is None:
            if len(intensors) > 1:
                pads = intensors[1]
                for i, v in enumerate(inshape):
                    newshape.append(v + pads[i] + pads[i + len(inshape)])
        else:
            for i, v in enumerate(inshape):
                newshape.append(v + self.pads[i] + self.pads[i + len(inshape)])
        newshape = [int(val) for val in newshape]
        return [newshape, ]


@NODE_REGISTRY.register()
class IdentityNode(FusedBase):
    pass


@NODE_REGISTRY.register()
class ErfNode(FusedBase):
    pass


@NODE_REGISTRY.register()
class BatchNormalizationNode(FusedBase):
    pass


@NODE_REGISTRY.register()
class FlattenNode(Node):
    def __init__(self, node):
        super().__init__(node)
        self.add_default_value('axis', None)

    def value_infer(self, intensors: []):
        if self.axis is None:
            return [intensors[0].reshape((intensors[0].shape[0], -1))]
        else:
            vol = 1
            for i in range(self.axis):
                vol *= intensors[0].shape[i]
            return [intensors[0].reshape((vol, -1))]


from .node_profilers import argmax_use_numpy


@NODE_REGISTRY.register()
class ArgMaxNode(Node):
    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('axis', 0)
        self.add_default_value('keepdims', 1)

    def value_infer(self, intensors: []):
        data = intensors[0]
        return [argmax_use_numpy(data, self.axis, self.keepdims)]


@NODE_REGISTRY.register()
class ArrayFeatureExtractorNode(Node):
    def shape_infer(self, intensors: []):
        return [_get_shape(intensors[1])]


@NODE_REGISTRY.register()
class ZipMapNode(Node):
    def shape_infer(self, intensors: []):
        return [(_get_shape(intensors[0])[0],)]


@NODE_REGISTRY.register()
class SliceNode(Node):
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
        self.add_default_value('keepdims', 1)

    def shape_infer(self, intensors: []):
        xshape = _get_shape(intensors[0])
        yshape = []
        self.axes = _axes_neg2pos(len(xshape), self.axes)

        for i in range(len(xshape)):
            if i in self.axes:
                if self.keepdims:
                    yshape.append(1)
            else:
                yshape.append(xshape[i])
        return [yshape]

    def value_infer(self, intensors: [numpy.ndarray]):
        reduced = numpy.mean(intensors[0], axis=self.axes, keepdims=self.keepdims == 1)
        return [reduced]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        vol = volume(_get_shape(intensors[0]))
        return vol * ADD_MACS


@NODE_REGISTRY.register()
class ReduceProdNode(ReduceMeanNode):
    def value_infer(self, intensors: [numpy.ndarray]):
        reduced = numpy.prod(intensors[0], axis=self.axes, keepdims=self.keepdims == 1)
        return [reduced]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        datashape = _get_shape(intensors[0])
        vol = volume(datashape)
        return vol * MUL_MACS


@NODE_REGISTRY.register()
class ReduceSumNode(ReduceMeanNode):
    def value_infer(self, intensors: [numpy.ndarray]):
        reduced = numpy.sum(intensors[0], axis=self.axes, keepdims=self.keepdims == 1)
        return [reduced]


@NODE_REGISTRY.register()
class ReduceMinNode(ReduceMeanNode):
    def value_infer(self, intensors: [numpy.ndarray]):
        data = intensors[0]
        reduced = numpy.minimum.reduce(data, axis=self.axes, keepdims=self.keepdims == 1)
        return [reduced]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        datashape = _get_shape(intensors[0])
        vol = volume(datashape)
        return vol * CMP_MACS


@NODE_REGISTRY.register()
class ReduceMaxNode(ReduceMinNode):
    def value_infer(self, intensors: [numpy.ndarray]):
        data = intensors[0]
        reduced = numpy.maximum.reduce(data, axis=self.axes, keepdims=self.keepdims == 1)
        return [reduced]


@NODE_REGISTRY.register()
class TopKNode(Node):
    def __init__(self, node):
        super().__init__(node)
        self.add_default_value('axis', None)

    def shape_infer(self, intensors: []):
        xshape = _get_shape(intensors[0])
        k = intensors[1][0]
        newshape = []
        for i in range(len(xshape)):
            if i == self.axis:
                newshape.append(k)
            else:
                newshape.append(xshape[i])
        return [newshape, newshape]


@NODE_REGISTRY.register()
class ScanNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value('num_scan_inputs', None)
        self.add_default_value('scan_input_directions', None)

    def shape_infer(self, intensors: []):
        # TODO
        return [create_ndarray_f32((1, 1)), create_ndarray_f32((1, 1)), create_ndarray_f32((1,)),
                intensors[3], intensors[3], ]


@NODE_REGISTRY.register()
class CompressNode(Node):
    def __init__(self, node):
        super().__init__(node)
        self.add_default_value('axis', None)

    def value_infer(self, intensors: []):
        return [numpy.compress(intensors[1], intensors[0], self.axis)]


@NODE_REGISTRY.register()
class HardmaxNode(PWNode):
    pass


@NODE_REGISTRY.register()
class CategoryMapperNode(PWNode):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = 0


@NODE_REGISTRY.register()
class LSTMNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value('direction', None)
        self.add_default_value('hidden_size', None)

    def shape_infer(self, intensors: []):
        xshape = _get_shape(intensors[0])
        wshape = _get_shape(intensors[1])
        seq_len = xshape[0]
        batch = xshape[1]
        num_dir = wshape[0]
        h_len = wshape[1] // 4
        return [(seq_len, num_dir, batch, h_len), (num_dir, batch, h_len)]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        wshape = _get_shape(intensors[1])
        rshape = _get_shape(intensors[2])
        bshape = _get_shape(intensors[3])
        batch = intensors[0].shape[1]
        macs = volume(wshape) + volume(rshape) + volume(bshape) * ADD_MACS
        macs *= batch
        return macs


@NODE_REGISTRY.register()
class ConvNode(Node):
    def __init__(self, n):
        super(ConvNode, self).__init__(n)
        self.add_default_value('auto_pad', None)
        self.add_default_value('pads', (0, 0, 0, 0))
        self.add_default_value('strides', (1, 1))
        self.add_default_value('dilations', (1, 1))
        self.add_default_value('group', 1)

    def shape_infer(self, intensors: []):
        outshapes = []
        xshape = _get_shape(intensors[0])
        wshape = _get_shape(intensors[1])
        shape = []
        if self.auto_pad is not None and self.auto_pad != b'NOTSET':
            if self.auto_pad in [b'SAME_LOWER', b'SAME_UPPER']:
                shape = (xshape[0], wshape[0], math.ceil(xshape[2] / self.strides[0]))
                if len(xshape) == 4:
                    shape += (math.ceil(xshape[3] / self.strides[1]),)
            elif self.auto_pad == b'VALID':
                oh = math.ceil((xshape[2] - wshape[2] + 1) / self.strides[0])
                shape = (xshape[0], wshape[0], oh)
                if len(xshape) == 4:
                    ow = math.ceil((xshape[3] - wshape[3] + 1) / self.strides[1])
                    shape += (ow,)
            else:
                assert 0

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

    def profile(self, intensors: [], outtensors: []):
        macs = 0
        if len(outtensors) == 1:
            if len(intensors) == 3 or len(intensors) == 2:
                kernel_shape = _get_shape(intensors[1])
                outvol = volume(_get_shape(outtensors[0]))
                if len(kernel_shape) > 3:
                    macs += outvol * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]
                elif len(kernel_shape) == 3:
                    macs += outvol * kernel_shape[1] * kernel_shape[2]
                macs += (outvol * ADD_MACS)
        return macs


@NODE_REGISTRY.register()
class ReduceL2Node(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value('axes', None)
        self.add_default_value('keepdims', 1)
        self.axes = tuple(self.axes) if self.axes is not None else None

    def value_infer(self, intensors: []):
        reduced = numpy.sqrt(numpy.sum(intensors[0], axis=self.axes, keepdims=self.keepdims == 1))
        return [reduced]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        vol = volume(_get_shape(intensors[0]))
        return vol * (ADD_MACS + SQRT_MACS)


@NODE_REGISTRY.register()
class CumSumNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = ADD_MACS
        self.ratio = 1


@NODE_REGISTRY.register()
class NonZeroNode(Node):
    def value_infer(self, intensors: []):
        condi = intensors[0]
        result = numpy.array(numpy.nonzero(condi), dtype=numpy.int64)
        if volume(result.shape) == 0:
            condi = numpy.ones_like(intensors[0])
            result = numpy.array(numpy.nonzero(condi), dtype=numpy.int64)
        return [result]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        return volume(outtensors[0].shape) * CMP_MACS


@NODE_REGISTRY.register()
class EqualNode(Node):
    def value_infer(self, intensors: []):
        result = numpy.equal(intensors[0], intensors[1])
        return [result]

    def profile(self, intensors: [], outtensors: []):
        return volume(_get_shape(outtensors[0])) * CMP_MACS


@NODE_REGISTRY.register()
class FloorNode(FusedBase):
    def value_infer(self, intensors: []):
        return [numpy.floor(intensors[0])]

    def profile(self, intensors: [], outtensors: []):
        return volume(_get_shape(outtensors[0])) * CMP_MACS


@NODE_REGISTRY.register()
class RoiAlignNode(Node):
    def __init__(self, node):
        super().__init__(node)

    def shape_infer(self, intensors: []):
        xshape = _get_shape(intensors[0])
        if len(xshape) == 4 and self.output_height is not None and self.output_width is not None:
            newshape = xshape[:2] + (self.output_height, self.output_width)
        else:
            raise NotImplementedError()
        return [newshape]


@NODE_REGISTRY.register()
class ScatterElementsNode(Node):
    def __init__(self, node):
        super().__init__(node)

    def shape_infer(self, intensors: []):
        # TODO
        # y = scatter_elements(intensors[0], intensors[1], intensors[2], self.axis)
        # return [create_ndarray_f32(y)]
        return [_get_shape(intensors[0])]


@NODE_REGISTRY.register()
class ScatterNDNode(Node):
    def value_infer(self, intensors: []):
        from .node_profilers import scatter_nd_impl
        data = intensors[0]
        indices = intensors[1]
        updates = intensors[2]
        return [scatter_nd_impl(data, indices, updates)]


@NODE_REGISTRY.register()
class GreaterNode(Node):
    def value_infer(self, intensors: []):
        result = numpy.greater(intensors[0], intensors[1])
        return [result]


@NODE_REGISTRY.register()
class DequantizeLinearNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = MUL_MACS


@NODE_REGISTRY.register()
class QuantizeLinearNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = MUL_MACS


@NODE_REGISTRY.register()
class MatMulIntegerNode(GemmNode):
    pass


@NODE_REGISTRY.register()
class QLinearMatMulNode(GemmNode):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value('transA', None)
        self.add_default_value('transB', None)

    def shape_infer(self, intensors: []):
        xshape = _get_shape(intensors[0])
        wshape = _get_shape(intensors[3])

        if self.__class__ == GemmNode:
            if self.transA is not None and self.transA > 0:
                xshape = xshape[::-1]
            else:
                xshape = xshape
            if self.transB is not None and self.transB > 0:
                yshape = xshape[:-1] + (wshape[-2],)
            else:
                yshape = xshape[:-1] + (wshape[-1],)
        else:
            yshape = xshape[:-1] + (wshape[-1],)

        return [yshape]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        xshape = _get_shape(intensors[0])
        weight_shape = _get_shape(intensors[3])
        macs = volume(xshape)
        if self.__class__ == GemmNode:
            macs *= weight_shape[0]
        else:
            macs *= weight_shape[-1]
        return macs


@NODE_REGISTRY.register()
class QLinearConvNode(ConvNode):
    def shape_infer(self, intensors: []):
        outtensors = []
        xshape = _get_shape(intensors[0])
        wshape = _get_shape(intensors[3])
        shape = []
        if self.auto_pad is not None and self.auto_pad != b'NOTSET':
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
        outtensors.append(shape)
        return outtensors

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        macs = 0
        if len(outtensors) == 1:
            kernel_shape = intensors[3].shape
            if len(kernel_shape) > 3:
                outvol = volume(outtensors[0].shape)
                macs += outvol * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]
            elif len(kernel_shape) == 3:
                outvol = volume(outtensors[0].shape)
                macs += outvol * kernel_shape[1] * kernel_shape[2]
            if len(intensors) == 9:
                macs += (outvol * ADD_MACS)
        return macs


@NODE_REGISTRY.register()
class ConvTransposeNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value('pads', (0, 0, 0, 0))
        self.add_default_value('output_padding', (0, 0, 0, 0))
        self.add_default_value('strides', (1, 1))
        self.add_default_value('dilations', (1, 1))
        self.add_default_value('output_shape', (0, 0))
        self.add_default_value('group', 1)

    def shape_infer(self, intensors: []):
        xshape = _get_shape(intensors[0])
        wshape = _get_shape(intensors[1])
        shape = []
        outc = self.group * wshape[1]
        if len(xshape) == 4:
            ow = _convtranspose_output_shape(xshape[2], self.output_padding[0], self.pads[0] + self.pads[2], wshape[2],
                                             self.strides[0],
                                             self.dilations[0])
            oh = _convtranspose_output_shape(xshape[3], self.output_padding[1], self.pads[1] + self.pads[3], wshape[3],
                                             self.strides[1],
                                             self.dilations[1])
            shape = [xshape[0], outc, ow, oh]
            if volume(self.output_shape) != 0:
                shape[2:] = self.output_shape
        elif len(xshape) == 3:
            ow = _convtranspose_output_shape(xshape[2], self.output_padding[0], self.pads[0] + self.pads[1], wshape[2],
                                             self.strides[0],
                                             self.dilations[0])
            shape = [xshape[0], outc, ow]
            if volume(self.output_shape) != 0:
                shape[2] = self.output_shape[0]
        return [shape, ]

    def profile(self, intensors: [], outtensors: []):
        macs = 0
        if len(outtensors) == 1:
            if len(intensors) == 3 or len(intensors) == 2:
                kernel_shape = intensors[1].shape
                if len(kernel_shape) > 3:
                    outvol = volume(outtensors[0].shape)
                    macs += outvol * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]
                    macs += outvol * ADD_MACS  # treat bias add as 0.5 MACs
                elif len(kernel_shape) == 3:
                    outvol = volume(outtensors[0].shape)
                    macs += outvol * kernel_shape[1] * kernel_shape[2]
                    macs += (outvol * ADD_MACS)
        return macs


@NODE_REGISTRY.register()
class ReshapeNode(Node):
    def shape_infer(self, intensors: []):
        srcshape = _get_shape(intensors[0])
        shape = intensors[1]
        newshape = []
        for i in range(len(shape)):
            if shape[i] == 0:
                newshape.append(int(srcshape[i]))
            else:
                newshape.append(int(shape[i]))
        sum = volume(newshape)
        raw = volume(srcshape)
        if sum < 0:
            remain = raw // -sum
            for i, val in enumerate(newshape):
                if val == -1:
                    newshape[i] = remain
                    break
        assert raw == volume(newshape)
        return [newshape]

    def value_infer(self, intensors: []):
        return [intensors[0].reshape(intensors[1])]


@NODE_REGISTRY.register()
class GRUNode(Node):
    def shape_infer(self, intensors: []):
        xshape = _get_shape(intensors[0])
        wshape = _get_shape(intensors[1])
        seq_len = xshape[0]
        batch = xshape[1]
        num_dir = wshape[0]
        h_len = wshape[1] // 3
        return [(seq_len, num_dir, batch, h_len), (num_dir, batch, h_len)]

    def profile(self, intensors: [], outtensors: []):
        w = intensors[1]
        r = intensors[2]
        b = intensors[3]
        params = 0
        params += volume(w.shape) + volume(r.shape) + volume(b.shape)
        batch = intensors[0].shape[1]
        macs = volume(w.shape) + volume(r.shape) + volume(b.shape) * ADD_MACS
        macs *= batch
        return macs, params


@NODE_REGISTRY.register()
class ConstantOfShapeNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value('value', None)

    def value_infer(self, intensors: []):
        arr = numpy.zeros(intensors[0].astype(numpy.int64), dtype=numpy.float32)
        if self.value is not None and len(self.value) == 1:
            arr.fill(self.value[0])
        return [arr]


@NODE_REGISTRY.register()
class CastNode(Node):
    def shape_infer(self, intensors: []):
        return [_get_shape(intensors[0])]

    def value_infer(self, intensors: []):
        return [intensors[0].astype(onnxdtype2npdtype(self.to))]


@NODE_REGISTRY.register()
class SplitNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value('axis', None)
        self.add_default_value('split', None)

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
