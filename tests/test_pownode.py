import numpy as np
import onnx
from onnx_tool.node import PowNode, TmpNodeProto
from onnx_tool.tensor import Tensor


def make_tensor_from_array(name, arr: np.ndarray):
    tproto = onnx.helper.make_tensor(name, onnx.TensorProto.FLOAT, arr.shape, arr.flatten().tolist())
    return Tensor(tproto)


def test_pownode_shape_and_value_broadcast():
    # base: shape (2,1), exp: shape (1,2) -> result (2,2)
    base = np.array([[2.0], [3.0]], dtype=np.float32)
    exp = np.array([[2.0, 3.0]], dtype=np.float32)

    t0 = make_tensor_from_array('base', base)
    t1 = make_tensor_from_array('exp', exp)
    out = Tensor('out')

    node = PowNode(TmpNodeProto('pow', 'Pow', {}))

    node.shape_infer([t0, t1], [out])
    assert out.get_shape() == [2, 2]

    node.value_infer([t0, t1], [out])
    expected = np.power(base, exp)
    np.testing.assert_allclose(out.get_numpy(), expected)


def test_pownode_scalar_exponent():
    # exponent is scalar
    base = np.array([[2.0, 3.0]], dtype=np.float32)
    exp = np.array(2.0, dtype=np.float32)  # scalar

    t0 = make_tensor_from_array('base2', base)
    t1 = make_tensor_from_array('exp2', exp)
    out = Tensor('out2')

    node = PowNode(TmpNodeProto('pow2', 'Pow', {}))
    node.shape_infer([t0, t1], [out])
    assert out.get_shape() == [1, 2]

    node.value_infer([t0, t1], [out])
    expected = np.power(base, exp)
    np.testing.assert_allclose(out.get_numpy(), expected)
