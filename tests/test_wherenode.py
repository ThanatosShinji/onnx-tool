import numpy as np
import onnx
from onnx_tool.node import WhereNode, TmpNodeProto
from onnx_tool.tensor import Tensor


def make_tensor_from_array(name, arr: np.ndarray):
    tproto = onnx.helper.make_tensor(name, onnx.TensorProto.FLOAT, arr.shape, arr.flatten().tolist())
    return Tensor(tproto)


def test_wherenode_shape_and_value_broadcast():
    # cond shape (2, 1), x shape (1, 3), y shape (1, 3) -> expected broadcast result (2, 3)
    cond = np.array([[True], [False]], dtype=np.bool_)
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    y = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)

    t_cond = make_tensor_from_array('cond', cond)
    t_x = make_tensor_from_array('x', x)
    t_y = make_tensor_from_array('y', y)
    out = Tensor('out')

    node = WhereNode(TmpNodeProto('where', 'Where', {}))

    # 1. Verify shape inference calculates the expanded dimensions correctly
    node.shape_infer([t_cond, t_x, t_y], [out])
    assert out.get_shape() == [2, 3]

    # 2. Verify value inference evaluates and broadcasts the arrays correctly
    node.value_infer([t_cond, t_x, t_y], [out])
    expected = np.where(cond, x, y)
    np.testing.assert_allclose(out.get_numpy(), expected)


def test_wherenode_bug_report_regression():
    # Direct regression test for the shapes mentioned in the bug report
    # cond: [1, 1, 9, 9], x: [1, 10, 1, 1], y: [1, 10, 1, 1] -> result [1, 10, 9, 9]
    cond = np.ones((1, 1, 9, 9), dtype=np.bool_)
    x = np.ones((1, 10, 1, 1), dtype=np.float32)
    y = np.zeros((1, 10, 1, 1), dtype=np.float32)

    t_cond = make_tensor_from_array('cond', cond)
    t_x = make_tensor_from_array('x', x)
    t_y = make_tensor_from_array('y', y)
    out = Tensor('out')

    node = WhereNode(TmpNodeProto('where', 'Where', {}))

    # Verify the profiler will no longer undercount the volume
    node.shape_infer([t_cond, t_x, t_y], [out])
    assert out.get_shape() == [1, 10, 9, 9]
