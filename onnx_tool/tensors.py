import numpy
import onnx

from .utils import GLOBAL_VARS


def create_ndarray_f32(shape):
    return numpy.ones(shape, dtype=numpy.float32)


def create_ndarray_int64(shape):
    return numpy.zeros(shape, dtype=numpy.int64)


def shape_of_tensor(tensor):
    shape = []
    # for nb in tensor.shape.dim
    for nb in tensor.type.tensor_type.shape.dim:
        assert (nb.dim_value != None)
        shape.append(nb.dim_value)
    return shape


def shape_of_initializer(initial):
    shape = []
    # for nb in tensor.shape.dim
    for nb in initial.dims:
        shape.append(nb)
    return shape


def onnxdtype2npdtype(initial):
    if initial.data_type == initial.FLOAT:
        return numpy.float32
    if initial.data_type == initial.FLOAT16:
        return numpy.float16
    if initial.data_type == initial.INT32:
        return numpy.int32
    if initial.data_type == initial.INT16:
        return numpy.int16
    if initial.data_type == initial.INT64:
        return numpy.int64
    if initial.data_type == initial.INT8:
        return numpy.int8
    if initial.data_type == initial.UINT8:
        return numpy.uint8
    if initial.data_type == initial.BOOL:
        return numpy.bool


def tensorproto2ndarray(initial):
    shape = shape_of_initializer(initial)
    ndtype = onnxdtype2npdtype(initial)
    if initial.raw_data == b'':
        arr = numpy.zeros(shape, ndtype).reshape((-1))
        if ndtype == numpy.float32:
            for i in range(len(initial.float_data)):
                arr[i] = initial.float_data[i]
        if ndtype == numpy.int32:
            for i in range(len(initial.int32_data)):
                arr[i] = initial.int32_data[i]
        if ndtype == numpy.float16:
            for i in range(len(initial.int32_data)):
                arr[i] = numpy.float16(initial.int32_data[i])  # TODO wrong conversion from int16 to float16
        if ndtype == numpy.int64:
            for i in range(len(initial.int64_data)):
                arr[i] = initial.int64_data[i]
        if ndtype == numpy.float64:
            for i in range(len(initial.int32_data)):
                arr[i] = initial.double_data[i]
        return arr.reshape(shape)
    else:
        return numpy.frombuffer(initial.raw_data, dtype=ndtype).reshape(shape)


def get_attribute_data(att):
    if att.type == att.INTS:
        val = []
        for ints in att.ints:
            val.append(ints)
        return val
    elif att.type == att.INT:
        return att.i
    elif att.type == att.FLOAT:
        return att.f
    elif att.type == att.STRING:
        return att.s
    elif att.type == att.FLOATS:
        val = []
        for f in att.floats:
            val.append(f)
        return val
    elif att.type == att.TENSOR:
        return tensorproto2ndarray(att.t)


def volume(shape: []):
    val = 1 if len(shape) > 0 else 0
    for v in shape:
        val *= v
    return val


def is_valid_ndarray(x):
    if x is None:
        return False
    if isinstance(x, (list, tuple)) and len(x) == 0:
        return False
    if isinstance(x, numpy.ndarray):
        if volume(x.shape) == 0:
            return True if x.size else False
        else:
            return True
    return False


def graph_addoutputs(graph: onnx.GraphProto, outnames: [str]) -> onnx.GraphProto:
    tensor_map = GLOBAL_VARS['tensor_map']
    for name in outnames:
        if tensor_map is not None and name in tensor_map.keys():
            newout = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, tensor_map[name].shape)
        else:
            newout = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, ())
        graph.output.append(newout)
    return graph


def graph_set_inputs(graph: onnx.GraphProto, dynamic_tensors: {}) -> onnx.GraphProto:
    tensor_map = GLOBAL_VARS['tensor_map']
    for input in graph.input:
        if dynamic_tensors.keys().__contains__(input.name):
            tensor_map[input.name] = dynamic_tensors[input.name]
            dim = input.type.tensor_type.shape.dim
            for nb, dnb in zip(dim, dynamic_tensors[input.name].shape):
                nb.dim_value = dnb
    return graph


def update_static_tensors(graph: onnx.GraphProto):
    tensor_map = GLOBAL_VARS['tensor_map']
    params_map = GLOBAL_VARS['params_map']
    for initial in graph.initializer:
        arr = tensorproto2ndarray(initial)
        tensor_map.update({initial.name: arr})

    for node in graph.node:
        if node.op_type == 'Constant':
            for att in node.attribute:
                if att.name == 'value':
                    tensor_map[node.output[0]] = get_attribute_data(att)

    totalparams = 0
    for key in tensor_map.keys():
        params_map[key] = volume(tensor_map[key].shape)
        totalparams += params_map[key]
    GLOBAL_VARS['totalparams'] = totalparams
