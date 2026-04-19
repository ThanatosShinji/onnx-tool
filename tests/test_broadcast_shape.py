import pytest
from onnx_tool.node import _broadcast_shape


def old_broadcast_shape(shapes):
    maxlen = 0
    for shape in shapes:
        maxlen = max(len(shape), maxlen)
    newshapes = []
    for shape in shapes:
        if len(shape) < maxlen:
            gap = maxlen - len(shape)
            newshape = [1] * gap + list(shape)
        else:
            newshape = list(shape)
        newshapes.append(newshape)
    outshape = newshapes[0]
    for i in range(len(newshapes) - 1):
        outshape = [max(a, b) for a, b in zip(newshapes[i + 1], outshape)]
    return outshape


@pytest.mark.parametrize("shapes, expect", [
    ([[2, 3, 4, 5], []], [2, 3, 4, 5]),
    ([[2, 3, 4, 5], [5]], [2, 3, 4, 5]),
    ([[4, 5], [2, 3, 4, 5]], [2, 3, 4, 5]),
    ([[1, 4, 5], [2, 3, 1, 1]], [2, 3, 4, 5]),
    ([[3, 4, 5], [2, 1, 1, 1]], [2, 3, 4, 5]),
])
def test_broadcast_same_as_old(shapes, expect):
    assert old_broadcast_shape(shapes) == expect
    assert _broadcast_shape(shapes) == expect


def test_incompatible_shapes():
    shapes = [[2, 3], [3, 4]]
    # old implementation would silently return [3,4]
    assert old_broadcast_shape(shapes) == [3, 4]
    # new implementation should raise ValueError for incompatible dims
    with pytest.raises(ValueError):
        _broadcast_shape(shapes)
