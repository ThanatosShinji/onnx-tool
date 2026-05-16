"""
ONNX Op Kernel Registry

注册式算子推理内核。每个算子注册为一个 Kernel 类，
实现 run(inputs, outputs, attrs) 方法，使用 PyTorch 进行计算。
"""

from typing import Dict, List, Optional, Tuple, Callable
import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Kernel Registry
# ---------------------------------------------------------------------------
class KernelRegistry:
    """全局算子内核注册表"""

    _kernels: Dict[str, type] = {}

    @classmethod
    def register(cls, op_type: str):
        """装饰器：注册一个算子内核类"""
        def wrapper(kernel_cls):
            cls._kernels[op_type] = kernel_cls
            return kernel_cls
        return wrapper

    @classmethod
    def get(cls, op_type: str):
        """获取算子内核类，未注册返回 None"""
        return cls._kernels.get(op_type)

    @classmethod
    def has(cls, op_type: str) -> bool:
        return op_type in cls._kernels

    @classmethod
    def registered_ops(cls) -> List[str]:
        return list(cls._kernels.keys())


# ---------------------------------------------------------------------------
# Base Kernel
# ---------------------------------------------------------------------------
class Kernel:
    """
    算子内核基类。
    子类需实现 run(inputs, outputs, attrs) 方法。

    可选实现 preprocess_weight，在 GraphInfer 初始化时对 weight 做前处理
    （如格式转换、常量折叠等），处理后的 tensor 会替换原始 weight 缓存。
    """
    op_type: str = ""

    @staticmethod
    def run(
        inputs: List[torch.Tensor],
        outputs: List[torch.Tensor],
        attrs: Dict,
    ):
        """
        执行算子。

        Args:
            inputs: 输入 tensor 列表（已正确 reshape）
            outputs: 输出 tensor 列表（已正确 reshape），结果写入其中
            attrs: 算子属性字典
        """
        raise NotImplementedError

    @staticmethod
    def preprocess_weight(src_name: str, tensor: torch.Tensor, attrs: Dict) -> torch.Tensor:
        """可选：对 weight 常量做前处理，返回处理后的 tensor。

        如果子类不重写此方法，默认返回原 tensor（不做处理）。

        Args:
            src_name: 该 tensor 在 ONNX 图中的名称（如 '/conv1/Conv_weight'）
            tensor: 已加载到目标设备的原始 weight tensor
            attrs: 使用该 weight 的节点的属性字典

        Returns:
            处理后的 weight tensor
        """
        return tensor


# ===========================================================================
# 常用辅助函数
# ===========================================================================

def _to_2d_padding(pads: tuple, auto_pad=None) -> int:
    """将 ONNX pad 格式转为 F.pad 需要的格式"""
    if auto_pad is not None and auto_pad != b'NOTSET' and auto_pad != b'':
        return None  # 由调用方处理 auto_pad
    if len(pads) == 2:
        return (pads[0], pads[0], pads[1], pads[1])
    elif len(pads) == 4:
        return (pads[0], pads[1], pads[2], pads[3])
    return 0


# ===========================================================================
# 注册算子内核
# ===========================================================================

@KernelRegistry.register("Conv")
class ConvKernel(Kernel):
    op_type = "Conv"

    @staticmethod
    def run(inputs, outputs, attrs):
        x = inputs[0]  # [N, C, H, W]
        w = inputs[1]  # [M, C, kH, kW]
        bias = inputs[2] if len(inputs) > 2 else None

        strides = attrs.get('strides', (1, 1))
        pads = attrs.get('pads', (0, 0, 0, 0))
        dilations = attrs.get('dilations', (1, 1))
        group = attrs.get('group', 1)

        # ONNX pads 格式: [x1_begin, x2_begin, x1_end, x2_end]
        if len(pads) == 2:
            pad_h, pad_w = pads[0], pads[1]
        elif len(pads) == 4:
            pad_h, pad_w = pads[0], pads[2]
        else:
            pad_h, pad_w = 0, 0

        result = F.conv2d(
            x, w, bias=bias,
            stride=strides,
            padding=(pad_h, pad_w),
            dilation=dilations,
            groups=group,
        )
        outputs[0].copy_(result)


@KernelRegistry.register("Add")
class AddKernel(Kernel):
    op_type = "Add"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(inputs[0] + inputs[1])


@KernelRegistry.register("Sub")
class SubKernel(Kernel):
    op_type = "Sub"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(inputs[0] - inputs[1])


@KernelRegistry.register("Mul")
class MulKernel(Kernel):
    op_type = "Mul"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(inputs[0] * inputs[1])


@KernelRegistry.register("Div")
class DivKernel(Kernel):
    op_type = "Div"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(inputs[0] / inputs[1])


@KernelRegistry.register("Relu")
class ReluKernel(Kernel):
    op_type = "Relu"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(F.relu(inputs[0]))


@KernelRegistry.register("Sigmoid")
class SigmoidKernel(Kernel):
    op_type = "Sigmoid"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(torch.sigmoid(inputs[0]))


@KernelRegistry.register("Tanh")
class TanhKernel(Kernel):
    op_type = "Tanh"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(torch.tanh(inputs[0]))


@KernelRegistry.register("LeakyRelu")
class LeakyReluKernel(Kernel):
    op_type = "LeakyRelu"

    @staticmethod
    def run(inputs, outputs, attrs):
        alpha = attrs.get('alpha', 0.01)
        outputs[0].copy_(F.leaky_relu(inputs[0], alpha))


@KernelRegistry.register("Clip")
class ClipKernel(Kernel):
    op_type = "Clip"

    @staticmethod
    def run(inputs, outputs, attrs):
        # ONNX Clip: min/max 可以是输入 tensor 或属性
        t = inputs[0]
        if len(inputs) > 1 and inputs[1] is not None:
            _min = inputs[1].item()
        else:
            _min = attrs.get('min', -3.4e38)
        if len(inputs) > 2 and inputs[2] is not None:
            _max = inputs[2].item()
        else:
            _max = attrs.get('max', 3.4e38)
        outputs[0].copy_(torch.clamp(t, _min, _max))


@KernelRegistry.register("Reshape")
class ReshapeKernel(Kernel):
    op_type = "Reshape"

    @staticmethod
    def run(inputs, outputs, attrs):
        # inputs[1] 是 shape tensor，但 memory pool 里可能没有它
        # 直接用 attrs 或 inputs[1] 的值
        if len(inputs) > 1 and inputs[1] is not None:
            shape = inputs[1].cpu().tolist()
        else:
            shape = attrs.get('shape', [])
        # 处理 0 维度: 保持原形状
        x = inputs[0]
        new_shape = []
        si = 0
        for s in shape:
            if s == 0:
                new_shape.append(x.shape[si])
            else:
                new_shape.append(s)
            si += 1
        outputs[0].copy_(x.reshape(new_shape))


@KernelRegistry.register("Transpose")
class TransposeKernel(Kernel):
    op_type = "Transpose"

    @staticmethod
    def run(inputs, outputs, attrs):
        perm = attrs.get('perm', None)
        if perm is not None:
            outputs[0].copy_(inputs[0].permute(*perm))
        else:
            # 默认反转所有维度
            outputs[0].copy_(inputs[0].T)


@KernelRegistry.register("Gemm")
class GemmKernel(Kernel):
    op_type = "Gemm"

    @staticmethod
    def run(inputs, outputs, attrs):
        alpha = attrs.get('alpha', 1.0)
        beta = attrs.get('beta', 1.0)
        transA = attrs.get('transA', 0)
        transB = attrs.get('transB', 0)

        A = inputs[0]
        B = inputs[1]
        C = inputs[2] if len(inputs) > 2 else None

        if transA:
            A = A.T
        # B 已在 preprocess_weight 中预转置，无需再次转置
        # if transB:
        #     B = B.T

        result = alpha * torch.mm(A, B)
        if C is not None:
            result += beta * C
        outputs[0].copy_(result)

    @staticmethod
    def preprocess_weight(src_name: str, tensor: torch.Tensor, attrs: Dict) -> torch.Tensor:
        """Gemm weight 前处理：如果 transB=1，预转置 weight 矩阵。"""
        transB = attrs.get('transB', 0)
        if transB:
            tensor = tensor.T.contiguous()
        return tensor


@KernelRegistry.register("MatMul")
class MatMulKernel(Kernel):
    op_type = "MatMul"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(torch.matmul(inputs[0], inputs[1]))


@KernelRegistry.register("BatchNormalization")
class BatchNormalizationKernel(Kernel):
    op_type = "BatchNormalization"

    @staticmethod
    def run(inputs, outputs, attrs):
        # inputs: [x, scale, bias, mean, var]
        x = inputs[0]
        scale = inputs[1]
        bias = inputs[2]
        mean = inputs[3]
        var = inputs[4]
        epsilon = attrs.get('epsilon', 1e-5)
        momentum = attrs.get('momentum', 0.9)

        # F.batch_norm 需要 1D 参数
        result = F.batch_norm(
            x, running_mean=mean, running_var=var,
            weight=scale, bias=bias,
            training=False, momentum=momentum, eps=epsilon,
        )
        outputs[0].copy_(result)


@KernelRegistry.register("AveragePool")
class AveragePoolKernel(Kernel):
    op_type = "AveragePool"

    @staticmethod
    def run(inputs, outputs, attrs):
        x = inputs[0]
        kernel_shape = attrs.get('kernel_shape', (1, 1))
        strides = attrs.get('strides', (1, 1))
        pads = attrs.get('pads', (0, 0, 0, 0))

        if len(pads) == 4:
            pad_h, pad_w = pads[0], pads[2]
        else:
            pad_h, pad_w = 0, 0

        outputs[0].copy_(
            F.avg_pool2d(x, kernel_shape, stride=strides, padding=(pad_h, pad_w))
        )


@KernelRegistry.register("MaxPool")
class MaxPoolKernel(Kernel):
    op_type = "MaxPool"

    @staticmethod
    def run(inputs, outputs, attrs):
        x = inputs[0]
        kernel_shape = attrs.get('kernel_shape', (1, 1))
        strides = attrs.get('strides', (1, 1))
        pads = attrs.get('pads', (0, 0, 0, 0))

        if len(pads) == 4:
            pad_h, pad_w = pads[0], pads[2]
        else:
            pad_h, pad_w = 0, 0

        outputs[0].copy_(
            F.max_pool2d(x, kernel_shape, stride=strides, padding=(pad_h, pad_w))
        )


@KernelRegistry.register("GlobalAveragePool")
class GlobalAveragePoolKernel(Kernel):
    op_type = "GlobalAveragePool"

    @staticmethod
    def run(inputs, outputs, attrs):
        x = inputs[0]
        # [N, C, H, W] -> [N, C, 1, 1]
        outputs[0].copy_(F.adaptive_avg_pool2d(x, (1, 1)))


@KernelRegistry.register("Flatten")
class FlattenKernel(Kernel):
    op_type = "Flatten"

    @staticmethod
    def run(inputs, outputs, attrs):
        axis = attrs.get('axis', 1)
        x = inputs[0]
        # 从 axis 处展平
        leading = x.shape[:axis]
        rest = x.shape[axis:]
        new_shape = list(leading) + [-1]
        outputs[0].copy_(x.reshape(new_shape))


@KernelRegistry.register("Concat")
class ConcatKernel(Kernel):
    op_type = "Concat"

    @staticmethod
    def run(inputs, outputs, attrs):
        axis = attrs.get('axis', 0)
        outputs[0].copy_(torch.cat(inputs, dim=axis))


@KernelRegistry.register("Softmax")
class SoftmaxKernel(Kernel):
    op_type = "Softmax"

    @staticmethod
    def run(inputs, outputs, attrs):
        axis = attrs.get('axis', -1)
        outputs[0].copy_(F.softmax(inputs[0], dim=axis))


@KernelRegistry.register("Squeeze")
class SqueezeKernel(Kernel):
    op_type = "Squeeze"

    @staticmethod
    def run(inputs, outputs, attrs):
        axes = attrs.get('axes', None)
        if axes is not None:
            outputs[0].copy_(inputs[0].squeeze(dim=tuple(axes)))
        else:
            outputs[0].copy_(inputs[0].squeeze())


@KernelRegistry.register("Unsqueeze")
class UnsqueezeKernel(Kernel):
    op_type = "Unsqueeze"

    @staticmethod
    def run(inputs, outputs, attrs):
        axes = attrs.get('axes', None)
        if axes is not None:
            for ax in sorted(axes):
                outputs[0].copy_(inputs[0].unsqueeze(ax))
        else:
            outputs[0].copy_(inputs[0].unsqueeze(0))


@KernelRegistry.register("Shape")
class ShapeKernel(Kernel):
    op_type = "Shape"

    @staticmethod
    def run(inputs, outputs, attrs):
        shape_tensor = torch.tensor(inputs[0].shape, dtype=torch.int64)
        outputs[0].copy_(shape_tensor)


@KernelRegistry.register("Gather")
class GatherKernel(Kernel):
    op_type = "Gather"

    @staticmethod
    def run(inputs, outputs, attrs):
        axis = attrs.get('axis', 0)
        data = inputs[0]
        indices = inputs[1].long()
        outputs[0].copy_(torch.gather(data, axis, indices))


@KernelRegistry.register("Cast")
class CastKernel(Kernel):
    op_type = "Cast"

    @staticmethod
    def run(inputs, outputs, attrs):
        # ONNX Cast: to 是整数表示的 dtype
        to = attrs.get('to', 1)  # 默认 float32
        # ONNX to tensor proto dtype mapping
        onnx_to_torch = {
            1: torch.float32,    # FLOAT
            2: torch.uint8,      # UINT8
            3: torch.int8,       # INT8
            5: torch.int16,      # INT16
            6: torch.int32,      # INT32
            7: torch.int64,      # INT64
            9: torch.bool,       # BOOL
            10: torch.float16,   # FLOAT16
            11: torch.double,    # DOUBLE
        }
        dtype = onnx_to_torch.get(to, torch.float32)
        outputs[0].copy_(inputs[0].to(dtype))


@KernelRegistry.register("Expand")
class ExpandKernel(Kernel):
    op_type = "Expand"

    @staticmethod
    def run(inputs, outputs, attrs):
        shape = inputs[1].cpu().tolist()
        outputs[0].copy_(inputs[0].expand(shape))


@KernelRegistry.register("Tile")
class TileKernel(Kernel):
    op_type = "Tile"

    @staticmethod
    def run(inputs, outputs, attrs):
        repeats = inputs[1].cpu().tolist()
        outputs[0].copy_(inputs[0].repeat(*repeats))


@KernelRegistry.register("Where")
class WhereKernel(Kernel):
    op_type = "Where"

    @staticmethod
    def run(inputs, outputs, attrs):
        condition = inputs[0].bool()
        outputs[0].copy_(torch.where(condition, inputs[1], inputs[2]))


@KernelRegistry.register("Equal")
class EqualKernel(Kernel):
    op_type = "Equal"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(inputs[0] == inputs[1])


@KernelRegistry.register("Greater")
class GreaterKernel(Kernel):
    op_type = "Greater"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(inputs[0] > inputs[1])


@KernelRegistry.register("Less")
class LessKernel(Kernel):
    op_type = "Less"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(inputs[0] < inputs[1])


@KernelRegistry.register("Neg")
class NegKernel(Kernel):
    op_type = "Neg"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(-inputs[0])


@KernelRegistry.register("Sqrt")
class SqrtKernel(Kernel):
    op_type = "Sqrt"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(torch.sqrt(inputs[0]))


@KernelRegistry.register("Pow")
class PowKernel(Kernel):
    op_type = "Pow"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(inputs[0] ** inputs[1])


@KernelRegistry.register("Erf")
class ErfKernel(Kernel):
    op_type = "Erf"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(torch.erf(inputs[0]))


@KernelRegistry.register("ReduceMean")
class ReduceMeanKernel(Kernel):
    op_type = "ReduceMean"

    @staticmethod
    def run(inputs, outputs, attrs):
        axes = attrs.get('axes', None)
        keepdims = attrs.get('keepdims', 1)
        if axes is not None:
            outputs[0].copy_(inputs[0].mean(dim=tuple(axes), keepdim=bool(keepdims)))
        else:
            outputs[0].copy_(inputs[0].mean(keepdim=bool(keepdims)))


@KernelRegistry.register("ReduceSum")
class ReduceSumKernel(Kernel):
    op_type = "ReduceSum"

    @staticmethod
    def run(inputs, outputs, attrs):
        axes = attrs.get('axes', None)
        keepdims = attrs.get('keepdims', 1)
        if axes is not None:
            outputs[0].copy_(inputs[0].sum(dim=tuple(axes), keepdim=bool(keepdims)))
        else:
            outputs[0].copy_(inputs[0].sum(keepdim=bool(keepdims)))


@KernelRegistry.register("Identity")
class IdentityKernel(Kernel):
    op_type = "Identity"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(inputs[0])


@KernelRegistry.register("Slice")
class SliceKernel(Kernel):
    op_type = "Slice"

    @staticmethod
    def run(inputs, outputs, attrs):
        x = inputs[0]
        # ONNX Slice: inputs[1]=starts, inputs[2]=ends, inputs[3]=axes, inputs[4]=steps
        starts = inputs[1].cpu().tolist() if len(inputs) > 1 else attrs.get('starts', [0])
        ends = inputs[2].cpu().tolist() if len(inputs) > 2 else attrs.get('ends', [x.shape[0]])
        axes = inputs[3].cpu().tolist() if len(inputs) > 3 else attrs.get('axes', list(range(len(starts))))
        steps = inputs[4].cpu().tolist() if len(inputs) > 4 else attrs.get('steps', [1] * len(starts))

        # 构建切片
        slices = [slice(None)] * x.ndim
        for i, ax in enumerate(axes):
            s = starts[i] if i < len(starts) else 0
            e = ends[i] if i < len(ends) else x.shape[ax]
            step = steps[i] if i < len(steps) else 1
            slices[ax] = slice(s, e, step)

        outputs[0].copy_(x[tuple(slices)])


@KernelRegistry.register("Pad")
class PadKernel(Kernel):
    op_type = "Pad"

    @staticmethod
    def run(inputs, outputs, attrs):
        x = inputs[0]
        pads = inputs[1].cpu().tolist() if len(inputs) > 1 else attrs.get('pads', [0, 0, 0, 0])
        mode = attrs.get('mode', 'constant')
        value = inputs[2].item() if len(inputs) > 2 else attrs.get('value', 0.0)

        # ONNX pads: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        # F.pad: (x1_begin, x1_end, x2_begin, x2_end, ...)
        ndim = len(pads) // 2
        pad_tuple = []
        for i in range(ndim):
            pad_tuple.append(pads[i])           # begin
            pad_tuple.append(pads[i + ndim])     # end
        # F.pad 格式从最后一个维度开始
        pad_tuple = tuple(pad_tuple[::-1])

        torch_mode = {'constant': 'constant', 'reflect': 'reflect', 'edge': 'replicate'}.get(mode, 'constant')
        outputs[0].copy_(F.pad(x, pad_tuple, mode=torch_mode, value=value))


@KernelRegistry.register("Resize")
class ResizeKernel(Kernel):
    op_type = "Resize"

    @staticmethod
    def run(inputs, outputs, attrs):
        x = inputs[0]
        # inputs[1] = roi (optional), inputs[2] = scales (optional), inputs[3] = sizes
        if len(inputs) > 3 and inputs[3] is not None:
            sizes = inputs[3].cpu().tolist()
            # sizes 是 [N, C, H, W] 格式
            mode = attrs.get('mode', 'nearest').decode() if isinstance(attrs.get('mode', ''), bytes) else attrs.get('mode', 'nearest')
            align_corners = attrs.get('coordinate_transformation_mode', 'half_pixel') == 'align_corners'
            torch_mode = {'nearest': 'nearest', 'linear': 'bilinear', 'cubic': 'bicubic'}.get(mode, 'nearest')
            result = F.interpolate(x, size=sizes[2:], mode=torch_mode, align_corners=align_corners if torch_mode != 'nearest' else None)
            outputs[0].copy_(result)
        elif len(inputs) > 2 and inputs[2] is not None:
            scales = inputs[2].cpu().tolist()
            sizes = [int(x.shape[i] * scales[i]) for i in range(len(scales))]
            mode = attrs.get('mode', 'nearest').decode() if isinstance(attrs.get('mode', ''), bytes) else attrs.get('mode', 'nearest')
            torch_mode = {'nearest': 'nearest', 'linear': 'bilinear', 'cubic': 'bicubic'}.get(mode, 'nearest')
            result = F.interpolate(x, size=sizes[2:], mode=torch_mode)
            outputs[0].copy_(result)
