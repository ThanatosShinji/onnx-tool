import numpy
import onnx
import math
import warnings
from .tensor import get_attribute_data, volume, is_valid_ndarray, create_ndarray_f32, onnxdtype2npdtype
from .utils import NODE_REGISTRY, tuple2str

'''
Real MACs: the number of x86 instructions to finish numeric compute.
From a low-level view, float-point multiple and add is much expensive than
bit movement. But current hardware has added more and more float-point
units to accelerate this. So, we should treat every instruction equally to
measure the complexity of the model.


'''

MUL_MACS = 1
ADD_MACS = 1
CMP_MACS = 1
DIV_MACS = 4  # refers to vrcp14ps
# following refers to https://github.com/reyoung/avx_mathfun
EXP_MACS = 32
POW_MACS = EXP_MACS  # similar with exp
LOG_MACS = 43
SIN_MACS = 39
COS_MACS = 39

TANH_MACS = EXP_MACS + 2 * ADD_MACS + DIV_MACS  # (e^2x-1)/(e^2x+1)
ATANH_MACS = LOG_MACS + 2 * ADD_MACS + DIV_MACS + MUL_MACS  # 1/2*ln((1+x)/(1-x))

SQRT_MACS = 24  # refers to vsqrtps
ATAN_MACS = SQRT_MACS + 3 * MUL_MACS + MUL_MACS + DIV_MACS  # refers to "Approximations to inverse tangent function"

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


def _auto_pad_valid_shape_calc(x, ksize, stride):
    return math.ceil((x - ksize + 1) / stride)


def _auto_pad_same_shape_calc(x, stride):
    return math.ceil((x) / stride)


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
    return [1]


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

    def shape_infer(self, intensors: []):
        faketensors = [_get_tensor(tensor) for tensor in intensors]
        outtensors = self.value_infer(faketensors)
        outshapes = [_get_shape(tensor) for tensor in outtensors]
        return outshapes

    def value_infer(self, intensors: []):
        raise NotImplementedError(f'this Node {self.op_type}-{self.name} has no value_infer')

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
        if intensors[0].dtype == intensors[1].dtype:
            if intensors[0].dtype in [numpy.int64]:
                return [intensors[0] // intensors[1]]
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

    def value_infer(self, intensors: []):
        return [numpy.exp(intensors[0])]


@NODE_REGISTRY.register()
class SoftmaxNode(ExpNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = EXP_MACS + DIV_MACS
        self.ratio = 1

    def value_infer(self, intensors: []):
        xexp = numpy.exp(intensors[0])
        return [xexp / numpy.sum(xexp,axis=self.axis,keepdims=True)]


@NODE_REGISTRY.register()
class LogNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = LOG_MACS
        self.ratio = 1

    def value_infer(self, intensors: []):
        return [numpy.log(intensors[0])]


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

    def value_infer(self, intensors: []):
        return [numpy.sqrt(intensors[0])]


@NODE_REGISTRY.register()
class PowNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = POW_MACS

    def value_infer(self, intensors: []):
        return [numpy.power(intensors[0], intensors[1])]


@NODE_REGISTRY.register()
class SinNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = SIN_MACS

    def value_infer(self, intensors: []):
        return [numpy.sin(intensors[0])]


@NODE_REGISTRY.register()
class CosNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = COS_MACS

    def value_infer(self, intensors: []):
        return [numpy.cos(intensors[0])]


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
    def value_infer(self, intensors: []):
        y = 1 / (1 + numpy.exp(-intensors[0]))
        return [y]


@NODE_REGISTRY.register()
class TanhNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = EXP_MACS
        self.ratio = 2

    def value_infer(self, intensors: []):
        return [numpy.tanh(intensors[0])]


@NODE_REGISTRY.register()
class AtanNode(TanhNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = ATAN_MACS

    def value_infer(self, intensors: []):
        return [numpy.arctan(intensors[0])]


@NODE_REGISTRY.register()
class SignNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = CMP_MACS

    def value_infer(self, intensors: []):
        return [numpy.sign(intensors[0])]


@NODE_REGISTRY.register()
class HardSigmoidNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = MUL_MACS + ADD_MACS + CMP_MACS * 2
        self.add_default_value('alpha',0.2)
        self.add_default_value('beta',0.5)

    def value_infer(self, intensors: []):
        y = max(0, min(1, self.alpha * intensors[0] + self.beta))
        return [y]


@NODE_REGISTRY.register()
class ReluNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = CMP_MACS

    def value_infer(self, intensors: []):
        return [numpy.clip(intensors[0], 0, None)]


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
        self.add_default_value('alpha', 0.01)

    def value_infer(self, intensors: []):
        outtensor = numpy.zeros_like(intensors[0])
        for i in numpy.ndindex(intensors[0].shape):
            x = intensors[0][i]
            outtensor[i] = x if x >= 0 else x * self.alpha
        return [outtensor]


@NODE_REGISTRY.register()
class SumNode(PWNode):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.op_mac = ADD_MACS
        self.ratio = len(nodeproto.input) - 1

    def value_infer(self, intensors: []):
        y = intensors[0]
        for i in range(1, len(intensors)):
            y = y + intensors[i]
        return [y]


@NODE_REGISTRY.register()
class NonMaxSuppressionNode(Node):
    def shape_infer(self, intensors: []):
        if len(intensors) >= 3:
            max_output_boxes_per_class = int(intensors[2][0])
            return [(max_output_boxes_per_class, 3)]
        raise NotImplementedError()


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
    def shape_infer(self, intensors: []):
        return [_get_shape(intensors[0])]

    def value_infer(self, intensors: [numpy.ndarray]):
        result = numpy.less_equal(intensors[0], intensors[1])
        return [result]


@NODE_REGISTRY.register()
class NotNode(Node):
    def value_infer(self, intensors: []):
        result = numpy.logical_not(intensors[0].astype(numpy.bool_))
        return [result]


@NODE_REGISTRY.register()
class AndNode(Node):
    def value_infer(self, intensors: []):
        result = numpy.logical_and(intensors[0].astype(numpy.bool_), intensors[1].astype(numpy.bool_))
        return [result]


@NODE_REGISTRY.register()
class WhereNode(Node):
    def shape_infer(self, intensors: []):
        cond_shape = _get_shape(intensors[0])
        x_shape = _get_shape(intensors[1])
        y_shape = _get_shape(intensors[2])
        return [_max_shape((cond_shape, x_shape, y_shape))]

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
            return [xshape[::-1]]
        for axis in self.perm:
            yshape.append(xshape[axis])
        return [yshape]

    def value_infer(self, intensors: []):
        return [numpy.transpose(intensors[0], self.perm)]


@NODE_REGISTRY.register()
class GemmNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value('transA', 0)
        self.add_default_value('transB', 0)

    def shape_infer(self, intensors: []):
        xshape = _get_shape(intensors[0])
        wshape = _get_shape(intensors[1])
        if self.__class__ == GemmNode:
            if self.transA > 0:
                xshape = xshape[::-1]
            else:
                xshape = xshape
            if self.transB > 0:
                yshape = xshape[:-1] + [wshape[-2], ]
            else:
                yshape = xshape[:-1] + [wshape[-1], ]
        else:
            # broadcast support
            batchshape=xshape[:-2] if len(xshape)> len(wshape) else wshape[:-2]
            yshape = batchshape+[xshape[-2],wshape[-1]]

        return [yshape]

    def value_infer(self, intensors: []):
        if self.__class__ == MatMulNode:
            ashape = _get_shape(intensors[0])
            bshape = _get_shape(intensors[1])
            assert (ashape[-1] == bshape[-2])
            return [numpy.matmul(intensors[0], intensors[1])]
        if self.transA > 0:
            A = numpy.transpose(intensors[0])
        else:
            A = intensors[0]
        if self.transB > 0:
            B = numpy.transpose(intensors[1])
        else:
            B = intensors[1]
        C = numpy.matmul(A, B)
        if len(intensors) > 2:
            C = numpy.add(C, intensors[2])
        return [C]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        yshape = _get_shape(outtensors[0])
        if len(intensors) >= 2:
            weight_shape = _get_shape(intensors[1])
            macs = volume(yshape)
            if self.__class__ == GemmNode:
                if self.transB > 0:
                    macs *= weight_shape[-1]
                else:
                    macs *= weight_shape[-2]
            else:
                macs *= weight_shape[-2]
            if len(intensors) == 3:
                macs += volume(yshape) * ADD_MACS
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
        out = numpy.take(intensors[0], intensors[1].astype(dtype=numpy.int64), axis=self.axis)
        outtensors.append(out)
        return outtensors


@NODE_REGISTRY.register()
class ClipNode(PWNode):
    def __init__(self, n):
        super().__init__(n)
        self.op_mac = CMP_MACS * 2
        self.ratio = 1

    def value_infer(self, intensors: []):
        y = numpy.clip(intensors[0], intensors[1], intensors[2])
        return [y]


@NODE_REGISTRY.register()
class ReciprocalNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = DIV_MACS

    def value_infer(self, intensors: []):
        return [numpy.reciprocal(intensors[0])]


@NODE_REGISTRY.register()
class Relu6Node(PWNode):
    def value_infer(self, intensors: []):
        return [numpy.clip(intensors[0],0,6)]


@NODE_REGISTRY.register()
class ConstantNode(Node):
    def __init__(self, n):
        super().__init__(n)

    def shape_infer(self, intensors: []):
        return [self.value.shape]

    def value_infer(self, intensors: []):
        return [self.value]


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
class TriluNode(Node):
    def __init__(self, n):
        super().__init__(n)
        self.add_default_value('upper',1)


    def shape_infer(self, intensors: []):
        return [_get_shape(intensors[0])]

    def value_infer(self, intensors: []):
        if len(intensors)==2:
            k=intensors[1]
        else:
            k=numpy.array(0).astype(numpy.int64)
        if self.upper==0:
            return [numpy.tril(intensors[0],k)]
        else:
            return [numpy.triu(intensors[0],k)]

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
        self.add_default_value('axes', [0])

    def shape_infer(self, intensors: []):
        inshape = _get_shape(intensors[0])
        if len(intensors) == 2:
            axes = intensors[1]
        else:
            axes = self.axes
        newaxis_len = len(inshape) + len(axes)
        axes = _axes_neg2pos(newaxis_len, axes)
        newshape = []
        idx = 0
        for i in range(newaxis_len):
            if i in axes:
                newshape.append(1)
            else:
                newshape.append(inshape[idx])
                idx += 1
        return [newshape]

    def value_infer(self, intensors: []):
        outtensor = intensors[0]
        if len(intensors) == 2:
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

    def shape_infer(self, intensors: []):
        inshape = _get_shape(intensors[0])
        outshape = []
        if len(intensors) == 2:
            self.axes = intensors[1]
        axes = _axes_neg2pos(len(inshape), self.axes)
        for i in range(len(inshape)):
            if i in axes:
                continue
            else:
                outshape.append(inshape[i])
        return [outshape]

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
        newshape = [len(_get_shape(intensors[0]))]
        return [newshape]

    def value_infer(self, intensors: [numpy.ndarray]):
        return [numpy.array(_get_shape(intensors[0]), dtype=numpy.int64)]


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
            if len(sizes) == len(xshape):
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
                outshape += (_auto_pad_same_shape_calc(inshape[2], self.strides[0]),)
                if len(self.strides) == 2:
                    outshape += [_auto_pad_same_shape_calc(inshape[3], self.strides[1]), ]
            elif self.auto_pad == b'VALID':
                outshape += [_auto_pad_valid_shape_calc(inshape[2], self.kernel_shape[0], self.strides[0]), ]
                if len(self.strides) == 2:
                    outshape += [_auto_pad_valid_shape_calc(inshape[3], self.kernel_shape[1], self.strides[1]), ]
        else:
            if len(self.kernel_shape) == 1:
                outshape = inshape[:2] + [
                    _pooling_shape_calc(inshape[2], self.pads[0] + self.pads[1], self.kernel_shape[0],
                                        self.dilations[0],
                                        self.strides[0], self.ceil_mode), ]
            if len(self.kernel_shape) == 2:
                outshape = inshape[:2] + [
                    _pooling_shape_calc(inshape[2], self.pads[0] + self.pads[2], self.kernel_shape[0],
                                        self.dilations[0],
                                        self.strides[0],
                                        self.ceil_mode),
                    _pooling_shape_calc(inshape[3], self.pads[1] + self.pads[3], self.kernel_shape[1],
                                        self.dilations[1],
                                        self.strides[1],
                                        self.ceil_mode),
                ]
            if len(self.kernel_shape) == 3:
                outshape = inshape[:2] + [
                    _pooling_shape_calc(inshape[2], self.pads[0] + self.pads[0], self.kernel_shape[0],
                                        self.dilations[0],
                                        self.strides[0],
                                        self.ceil_mode),
                    _pooling_shape_calc(inshape[3], self.pads[1] + self.pads[1], self.kernel_shape[1],
                                        self.dilations[1],
                                        self.strides[1],
                                        self.ceil_mode),
                    _pooling_shape_calc(inshape[4], self.pads[2] + self.pads[2], self.kernel_shape[2],
                                        self.dilations[1],
                                        self.strides[2],
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

    def value_infer(self, intensors: []):
        xshape=intensors[0].shape
        oshape=self.shape_infer(intensors)[0]
        ot=numpy.zeros(oshape)
        for i in numpy.ndindex(ot.shape):
            batch = i[0]
            ocn = i[1]
            oh = i[2]
            ow = i[3]
            t = ot[i]
            ks=tuple(self.kernel_shape)
            for j in numpy.ndindex(ks):
                kh = j[0]
                kw = j[1]
                srch = oh * self.strides[0] + kh - self.pads[0]
                srcw = ow * self.strides[1] + kw - self.pads[1]
                if srch < 0 or srch >= xshape[2] or srcw < 0 or srcw >= xshape[3]:
                    continue
                else:
                    srcv = intensors[0][batch, ocn, srch, srcw]
                t = max(srcv,t)
            ot[i] = t
        return [ot]


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

    def value_infer(self, intensors: []):
        x=intensors[0]
        h=x.shape[2]
        w=x.shape[3]
        y=numpy.zeros(x.shape[:2],dtype=numpy.float32)
        for i in numpy.ndindex(y.shape):
            t=0
            for j in numpy.ndindex((h,w)):
                xi=i+j
                t+=x[xi]
            t/=(h*w)
            y[i]=t
        return [y]

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
        xshape = _get_shape(intensors[0])
        expandshape = intensors[1]
        yshape = []
        if len(xshape) < len(expandshape):
            for i in range(len(xshape), len(expandshape)):
                xshape = [1, ] + xshape
        for x, e in zip(xshape, expandshape):
            yshape.append(max(x, e))
        return [yshape]

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
    def value_infer(self, intensors: []):
        outtensor=numpy.zeros_like(intensors[0])
        for i in numpy.ndindex(intensors[0].shape):
            outtensor[i]=math.erf(intensors[0][i])
        return [outtensor]


@NODE_REGISTRY.register()
class BatchNormalizationNode(FusedBase):
    def __init__(self,n):
        super().__init__(n)
        self.add_default_value('epsilon',1e-05)
        self.add_default_value('momentum',0.9)
        self.add_default_value('training_mode',int(0))

    def value_infer(self, intensors: []):
        x=intensors[0]
        scale=intensors[1]
        b=intensors[2]
        mean=intensors[3]
        var=intensors[4]
        y=numpy.zeros_like(x)
        for i in numpy.ndindex(y.shape):
            cn=i[1]
            sqrt_var=math.sqrt(var[cn]+self.epsilon)
            sm=scale[cn]/sqrt_var
            sv=b[cn]
            m=mean[cn]
            y[i]=(x[i]-m)*sm+sv
        return [y]

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


# copy from https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/argmax.py
def argmax_use_numpy(data: numpy.ndarray, axis: int = 0, keepdims: int = 1) -> (numpy.ndarray):
    result = numpy.argmax(data, axis=axis)
    if (keepdims == 1):
        result = numpy.expand_dims(result, axis)
    return result.astype(numpy.int64)


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
    def __init__(self, n):
        super(SliceNode, self).__init__(n)
        self.add_default_value('steps', None)

    def shape_infer(self, intensors: []):
        inshape = _get_shape(intensors[0])
        if len(intensors) == 1:
            starts = self.starts
            ends = self.ends
            axes = self.axes
            if self.steps is None:
                steps = [1] * len(starts)
            else:
                steps = self.steps
        else:
            elesize = len(intensors[1])
            starts = intensors[1]
            ends = intensors[2]
            if len(intensors) == 3:
                # undef beheviour of bidaf-9.onnx
                axes = [0]
                steps = [1]
            else:
                axes = [0] * elesize
                steps = [1] * elesize
            if len(intensors) >= 4:
                axes = intensors[3]
            if len(intensors) >= 5:
                steps = intensors[4]

        axes = _axes_neg2pos(len(inshape), axes)
        newshape = inshape.copy()
        for a in axes:
            newshape[a] = 0
        for s, e, a, st in zip(starts, ends, axes, steps):
            if s < 0:
                s = max(0, inshape[a] + s)
            else:
                s = max(s, 0)
            if e < 0:
                e = max(0, inshape[a] + e)
            else:
                e = min(e, inshape[a])
            tmp = abs(e - s)
            newshape[a] += abs(math.ceil(tmp / st))
        return [newshape]

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
            for i in axes:
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
            for i in axes:
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
            for i in self.axes:
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
        self.add_default_value('axes', None)
        self.add_default_value('keepdims', 1)

    def shape_infer(self, intensors: []):
        xshape = _get_shape(intensors[0])
        yshape = []
        if len(intensors) == 2:
            axes = intensors[1]
        else:
            axes = self.axes
        if axes is None:
            return [[1]]
        else:
            axes = _axes_neg2pos(len(xshape), axes)

        for i in range(len(xshape)):
            if i in axes:
                if self.keepdims:
                    yshape.append(1)
            else:
                yshape.append(xshape[i])
        return [yshape]

    def value_infer(self, intensors: [numpy.ndarray]):
        if len(intensors) == 2:
            axes = intensors[1]
        else:
            axes = self.axes
        reduced = numpy.mean(intensors[0], axis=axes, keepdims=self.keepdims == 1)
        return [reduced]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        vol = volume(_get_shape(intensors[0]))
        return vol * ADD_MACS


@NODE_REGISTRY.register()
class ReduceProdNode(ReduceMeanNode):
    def value_infer(self, intensors: [numpy.ndarray]):
        if len(intensors) == 2:
            axes = intensors[1]
        else:
            axes = self.axes
        reduced = numpy.prod(intensors[0], axis=axes, keepdims=self.keepdims == 1)
        return [reduced]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        datashape = _get_shape(intensors[0])
        vol = volume(datashape)
        return vol * MUL_MACS


@NODE_REGISTRY.register()
class ReduceSumNode(ReduceMeanNode):
    def value_infer(self, intensors: [numpy.ndarray]):
        if len(intensors) == 2:
            axes = tuple(intensors[1].tolist())
        else:
            axes = self.axes
        reduced = numpy.sum(intensors[0], axis=axes, keepdims=self.keepdims == 1)
        return [reduced]


@NODE_REGISTRY.register()
class ReduceMinNode(ReduceMeanNode):
    def value_infer(self, intensors: [numpy.ndarray]):
        data = intensors[0]
        if len(intensors) == 2:
            axes = intensors[1]
        else:
            axes = self.axes
        reduced = numpy.minimum.reduce(data, axis=axes, keepdims=self.keepdims == 1)
        return [reduced]

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        datashape = _get_shape(intensors[0])
        vol = volume(datashape)
        return vol * CMP_MACS


@NODE_REGISTRY.register()
class ReduceMaxNode(ReduceMinNode):
    def value_infer(self, intensors: [numpy.ndarray]):
        data = intensors[0]
        if len(intensors) == 2:
            axes = intensors[1]
        else:
            axes = self.axes
        reduced = numpy.maximum.reduce(data, axis=axes, keepdims=self.keepdims == 1)
        return [reduced]


@NODE_REGISTRY.register()
class TopKNode(Node):
    def __init__(self, node):
        super().__init__(node)
        self.add_default_value('axis', None)

    def shape_infer(self, intensors: []):
        xshape = _get_shape(intensors[0])
        k = intensors[1][0]
        # when the input tensor only contain 1 dimension, the axis attribute (default: 0) may not appear in the node
        if len(xshape) == 1 and self.axis is None:
            self.axis = 0
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
        if len(self.output) == 2:
            # first 3 useless tensors are removed from the graph
            return [_get_shape(intensors[3]), _get_shape(intensors[3])]

        # TODO
        return [(1, 1), (1, 1), (1,),
                _get_shape(intensors[3]), _get_shape(intensors[3]), ]


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
        batch = _get_shape(intensors[0])[1]
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
            if len(xshape) == 5:
                od = _conv_output_shape(xshape[2], self.pads[0] + self.pads[3], wshape[2], self.strides[0],
                                        self.dilations[0])
                oh = _conv_output_shape(xshape[3], self.pads[1] + self.pads[4], wshape[3], self.strides[1],
                                        self.dilations[1])
                ow = _conv_output_shape(xshape[4], self.pads[2] + self.pads[5], wshape[4], self.strides[2],
                                        self.dilations[2])
                shape = (xshape[0], wshape[0], od, oh, ow)
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

    def value_infer(self, intensors: []):
        if self.group != 1:
            raise NotImplementedError()
        outshape = self.shape_infer(intensors)[0]
        outtensor = numpy.zeros(outshape, dtype=numpy.float32)
        has_bias = len(intensors) > 2
        xshape = _get_shape(intensors[0])
        wshape = _get_shape(intensors[1])
        if len(wshape) != 4:
            raise NotImplementedError()

        reduce_shape = tuple(wshape[1:])
        for i in numpy.ndindex(outshape):
            batch = i[0]
            ocn = i[1]
            oh = i[2]
            ow = i[3]
            t = outtensor[i]
            if has_bias:
                t = intensors[2][ocn]
            for j in numpy.ndindex(reduce_shape):
                icn = j[0]
                kh = j[1]
                kw = j[2]
                srch = oh * self.strides[0] + kh * self.dilations[0] - self.pads[0]
                srcw = ow * self.strides[1] + kw * self.dilations[1] - self.pads[1]
                if srch < 0 or srch >= xshape[2] or srcw < 0 or srcw >= xshape[3]:
                    srcv = 0
                else:
                    srcv = intensors[0][batch, icn, srch, srcw]
                wv = intensors[1][(ocn,) + j]
                t += srcv * wv
            outtensor[i] = t
        return [outtensor]

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
        reduced = numpy.sqrt(numpy.sum(intensors[0] * intensors[0], axis=self.axes, keepdims=self.keepdims == 1))
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
        self.add_default_value('exclusive', 0)
        self.add_default_value('reverse', 0)

    def value_infer(self, intensors: []):
        if self.exclusive == 0 and self.reverse == 0:
            return [numpy.cumsum(intensors[0], intensors[1])]
        raise NotImplementedError(f"CumSum doesnt support {self.exclusive} {self.reverse}")


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
        return volume(_get_shape(outtensors[0])) * CMP_MACS


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
            newshape = xshape[:2] + [self.output_height, self.output_width]
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


# copy from https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/scatternd.py
def scatter_nd_impl(data, indices, updates, reduction='none'):  # type: ignore

    # Check tensor shapes
    assert indices.shape[-1] <= len(data.shape)
    assert updates.shape == indices.shape[:-1] + data.shape[indices.shape[-1]:]

    # Compute output
    output = numpy.copy(data)
    for i in numpy.ndindex(indices.shape[:-1]):
        rawoutput = output[tuple(indices[i])]
        # NOTE: The order of iteration in this loop is not specified.
        if reduction == "add":
            rawoutput += updates[i]
        elif reduction == "mul":
            rawoutput *= updates[i]
        elif reduction == "max":
            rawoutput = numpy.maximum(rawoutput, updates[i])
        elif reduction == "min":
            rawoutput = numpy.minimum(rawoutput, updates[i])
        else:
            rawoutput = updates[i]
        output[tuple(indices[i])] = rawoutput
    return output


# copy from https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/gathernd.py
def gather_nd_impl(
        data: numpy.ndarray, indices: numpy.ndarray, batch_dims: int
) -> numpy.ndarray:
    # Note the data rank - will be reused multiple times later
    data_rank = len(data.shape)

    # Check input tensors' shape/rank condition
    assert indices.shape[-1] <= data_rank

    # The list of data/indice shape of batch_dims
    batch_dims_shape = []

    # The number of elements in the batch_dims for data/indice array
    batch_dims_size = 1

    # Check the shape of indice and data are identicial for batch dims.
    for i in range(batch_dims):
        batch_dims_shape.append(indices.shape[i])
        batch_dims_size *= indices.shape[i]

    # Compute output of the op as below

    # Compute shape of output array
    output_shape = (
        batch_dims_shape + list(indices.shape)[batch_dims:-1]
        if (indices.shape[-1] == data_rank - batch_dims)
        else batch_dims_shape
             + list(indices.shape)[batch_dims:-1]
             + list(data.shape)[batch_dims + indices.shape[-1]:]
    )

    # Placeholder for output data
    output_data_buffer = []

    # Flatten 'indices' to 2D array
    reshaped_indices = indices.reshape(batch_dims_size, -1, indices.shape[-1])

    # Flatten 'data' to array of shape (batch_dim_size, data.shape[batch_dimes:])
    reshaped_data = data.reshape((batch_dims_size,) + data.shape[batch_dims:])

    # gather each scalar value from 'data'
    for batch_dim in range(reshaped_indices.shape[0]):
        for outer_dim in range(reshaped_indices.shape[1]):
            gather_index = tuple(reshaped_indices[batch_dim][outer_dim])
            output_data_buffer.append(reshaped_data[(batch_dim, *gather_index)])
    return numpy.asarray(output_data_buffer, dtype=data.dtype).reshape(output_shape)


@NODE_REGISTRY.register()
class ScatterNDNode(Node):
    def shape_infer(self, intensors: []):
        return [_get_shape(intensors[0])]  # output=copy(data)

    def value_infer(self, intensors: []):
        data = intensors[0]
        indices = intensors[1].astype(numpy.int64)
        updates = intensors[2]
        return [scatter_nd_impl(data, indices, updates)]  # TODO this impl may fail some cases


@NODE_REGISTRY.register()
class GatherNDNode(Node):
    def shape_infer(self, intensors: []):
        data_shape = _get_shape(intensors[0])
        indice_shape = _get_shape(intensors[1])
        batch_dims = 0
        # Note the data rank - will be reused multiple times later
        data_rank = len(data_shape)

        # Check input tensors' shape/rank condition
        assert indice_shape[-1] <= data_rank

        # The list of data/indice shape of batch_dims
        batch_dims_shape = []

        # The number of elements in the batch_dims for data/indice array
        batch_dims_size = 1

        # Check the shape of indice and data are identicial for batch dims.
        for i in range(batch_dims):
            batch_dims_shape.append(indice_shape[i])
            batch_dims_size *= indice_shape[i]

        # Compute output of the op as below

        # Compute shape of output array
        output_shape = (
            batch_dims_shape + list(indice_shape)[batch_dims:-1]
            if (indice_shape[-1] == data_rank - batch_dims)
            else batch_dims_shape
                 + list(indice_shape)[batch_dims:-1]
                 + list(data_shape)[batch_dims + indice_shape[-1]:]
        )
        return [output_shape]

    def value_infer(self, intensors: []):
        data = intensors[0]
        indices = intensors[1].astype(numpy.int64)
        return [gather_nd_impl(data, indices, 0)]


@NODE_REGISTRY.register()
class RandomUniformLikeNode(Node):
    def shape_infer(self, intensors: []):
        return [_get_shape(intensors[0])]


@NODE_REGISTRY.register()
class RandomNormalLikeNode(RandomUniformLikeNode):
    pass


@NODE_REGISTRY.register()
class GreaterNode(Node):
    def value_infer(self, intensors: []):
        result = numpy.greater(intensors[0], intensors[1])
        return [result]

    def profile(self, intensors: [], outtensors: []):
        outshape = _get_shape(outtensors[0])
        return volume(outshape) * CMP_MACS


@NODE_REGISTRY.register()
class DequantizeLinearNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = MUL_MACS + ADD_MACS
        self.ratio = 1


@NODE_REGISTRY.register()
class QuantizeLinearNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)
        self.op_mac = MUL_MACS + ADD_MACS
        self.ratio = 1


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
                yshape = xshape[:-1] + [wshape[-2], ]
            else:
                yshape = xshape[:-1] + [wshape[-1], ]
        else:
            yshape = xshape[:-1] + [wshape[-1], ]

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
                shape = [xshape[0], wshape[0], math.ceil(xshape[2] / self.strides[0])]
                if len(xshape) == 4:
                    shape += [math.ceil(xshape[3] / self.strides[1]), ]
        else:
            if len(xshape) == 4:
                oh = _conv_output_shape(xshape[2], self.pads[0] + self.pads[2], wshape[2], self.strides[0],
                                        self.dilations[0])
                ow = _conv_output_shape(xshape[3], self.pads[1] + self.pads[3], wshape[3], self.strides[1],
                                        self.dilations[1])
                shape = [xshape[0], wshape[0], oh, ow]
            elif len(xshape) == 3:
                oh = _conv_output_shape(xshape[2], self.pads[0] + self.pads[1], wshape[2], self.strides[0],
                                        self.dilations[0])
                shape = [xshape[0], wshape[0], oh]
        outtensors.append(shape)
        return outtensors

    def profile(self, intensors: [numpy.ndarray], outtensors: [numpy.ndarray]):
        macs = 0
        outshape = _get_shape(outtensors[0])
        if len(outtensors) == 1:
            kernel_shape = intensors[3].shape
            if len(kernel_shape) > 3:
                outvol = volume(outshape)
                macs += outvol * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]
            elif len(kernel_shape) == 3:
                outvol = volume(outshape)
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
        if len(xshape) == 5:
            od = _convtranspose_output_shape(xshape[2], self.output_padding[0], self.pads[0] + self.pads[3], wshape[2],
                                             self.strides[0],
                                             self.dilations[0])
            ow = _convtranspose_output_shape(xshape[3], self.output_padding[1], self.pads[1] + self.pads[4], wshape[3],
                                             self.strides[1],
                                             self.dilations[1])
            oh = _convtranspose_output_shape(xshape[4], self.output_padding[2], self.pads[2] + self.pads[5], wshape[4],
                                             self.strides[2],
                                             self.dilations[2])
            shape = [xshape[0], outc, od, ow, oh]
            if volume(self.output_shape) != 0:
                shape[2:] = self.output_shape
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
                kernel_shape = _get_shape(intensors[1])
                if len(kernel_shape) > 3:
                    outvol = volume(_get_shape(outtensors[0]))
                    macs += outvol * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]
                    macs += outvol * ADD_MACS  # treat bias add as 0.5 MACs
                elif len(kernel_shape) == 3:
                    outvol = volume(_get_shape(outtensors[0]))
                    macs += outvol * kernel_shape[1] * kernel_shape[2]
                    macs += (outvol * ADD_MACS)
        return macs


@NODE_REGISTRY.register()
class ReshapeNode(Node):
    def shape_infer(self, intensors: []):
        srcshape = _get_shape(intensors[0])
        if not is_valid_ndarray(intensors[1]):
            return [[1, ]]
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
class GatherElementsNode(Node):
    def shape_infer(self, intensors: []):
        return [_get_shape(intensors[1])]

    def value_infer(self, intensors: []):
        x=intensors[0]
        indice=intensors[1].astype(numpy.int64)
        outtensor = numpy.zeros_like(indice)
        for i in numpy.ndindex(outtensor.shape):
            idx = list(i)
            idx[self.axis] = indice[i]
            outtensor[i] = x[tuple(idx)]
        return [outtensor]


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
        xshape = _get_shape(intensors[0])
        wshape = _get_shape(intensors[1])
        rshape = _get_shape(intensors[2])
        bshape = _get_shape(intensors[3])
        batch = xshape[1]
        macs = volume(wshape) + volume(rshape) + volume(bshape) * ADD_MACS
        macs *= batch
        return macs


@NODE_REGISTRY.register()
class ConstantOfShapeNode(Node):
    def __init__(self, nodeproto):
        super().__init__(nodeproto)
        self.add_default_value('value', None)

    def shape_infer(self, intensors: []):
        return [list(intensors[0].astype(numpy.int64))]

    def value_infer(self, intensors: []):
        arr = numpy.zeros(intensors[0].astype(numpy.int64), dtype=self.value.dtype)
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

    def shape_infer(self, intensors: []):
        end = 0
        inshape = _get_shape(intensors[0])
        if self.split is None:
            if len(intensors) == 2:
                split = intensors[1]
            else:
                split = [inshape[self.axis] // 2] * 2
        else:
            split = self.split
        axis = _axes_neg2pos(len(inshape), [self.axis])[0]
        shapes = []
        for v in split:
            shape = []
            for i in range(len(inshape)):
                if i == axis:
                    if end + v < inshape[axis]:
                        shape.append(v)
                    else:
                        shape.append(inshape[axis] - end)
                else:
                    shape.append(inshape[i])
            end += v
            shapes.append(shape)
        return shapes

    def value_infer(self, intensors: []):
        splitpos = []
        end = 0
        inshape = _get_shape(intensors[0])
        if self.split is None:
            if len(intensors) == 2:
                split = intensors[1]
            else:
                split = [inshape[self.axis] // 2]
        else:
            split = self.split

        axis = _axes_neg2pos(len(inshape), [self.axis])[0]
        for v in split:
            if end + v >= inshape[axis]:
                break
            splitpos.append(end + v)
            end += v
        return numpy.split(intensors[0], splitpos, axis)


def create_node(n: onnx.NodeProto):
    node_class = NODE_REGISTRY.get(n.op_type + 'Node')
    if node_class != None:
        instance = node_class(n)
        return instance
    warnings.warn(f'node {n.op_type} is not registed for profiling, return 0 Macs and 0 params as default. '
                  f'Use NODEPROFILER_REGISTRY to register your profiler for this node.')
    return Node(n)
