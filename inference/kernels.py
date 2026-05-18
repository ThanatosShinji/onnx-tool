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
        torch.add(inputs[0], inputs[1], out=outputs[0])


@KernelRegistry.register("Sub")
class SubKernel(Kernel):
    op_type = "Sub"

    @staticmethod
    def run(inputs, outputs, attrs):
        torch.sub(inputs[0], inputs[1], out=outputs[0])


@KernelRegistry.register("Mul")
class MulKernel(Kernel):
    op_type = "Mul"

    @staticmethod
    def run(inputs, outputs, attrs):
        torch.mul(inputs[0], inputs[1], out=outputs[0])


@KernelRegistry.register("Div")
class DivKernel(Kernel):
    op_type = "Div"

    @staticmethod
    def run(inputs, outputs, attrs):
        torch.div(inputs[0], inputs[1], out=outputs[0])


@KernelRegistry.register("Relu")
class ReluKernel(Kernel):
    op_type = "Relu"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(torch.relu(inputs[0]))


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
        torch.clamp(t, _min, _max, out=outputs[0])


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
                new_shape.append(int(s))
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

        torch.mm(A, B, out=outputs[0])
        if alpha != 1.0:
            outputs[0].mul_(alpha)
        if C is not None:
            outputs[0].add_(beta * C)

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
        torch.matmul(inputs[0], inputs[1], out=outputs[0])
        if len(inputs) > 2 and inputs[2] is not None:
            outputs[0].add_(inputs[2])


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
        # ONNX Flatten: output shape = (d0, d1, ..., d_{axis-1}, d_{axis} * ... * d_{k-1})
        # 即保留前 axis 个维度，将剩余维度展平为一个维度
        if axis == 0:
            # 展平所有维度
            result = x.reshape(-1)
        else:
            leading = x.shape[:axis]
            rest_numel = 1
            for s in x.shape[axis:]:
                rest_numel *= s
            new_shape = tuple(leading) + (rest_numel,)
            result = x.reshape(new_shape)
        # 如果输出 tensor 的 shape 与计算结果不同（模型 shape inference 可能有误），
        # 尝试 reshape 到输出 tensor 的 shape
        if result.shape != outputs[0].shape:
            result = result.reshape(outputs[0].shape)
        outputs[0].copy_(result)


@KernelRegistry.register("Concat")
class ConcatKernel(Kernel):
    op_type = "Concat"

    @staticmethod
    def run(inputs, outputs, attrs):
        axis = attrs.get('axis', 0)
        torch.cat(inputs, dim=axis, out=outputs[0])


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
        # axes 来源: inputs[1] (ONNX 13+) 或 attrs['axes'] (ONNX 11-)
        if len(inputs) > 1 and inputs[1] is not None:
            axes = [int(x) for x in inputs[1].cpu().tolist()]
        else:
            axes = attrs.get('axes', None)

        if axes is not None:
            t = inputs[0]
            for ax in sorted(axes):
                t = t.unsqueeze(ax)
            outputs[0].copy_(t)
        else:
            outputs[0].copy_(inputs[0].unsqueeze(0))


@KernelRegistry.register("Shape")
class ShapeKernel(Kernel):
    op_type = "Shape"

    @staticmethod
    def run(inputs, outputs, attrs):
        shape_tensor = torch.tensor(inputs[0].shape, dtype=torch.int64, device=outputs[0].device)
        outputs[0].copy_(shape_tensor)


@KernelRegistry.register("Gather")
class GatherKernel(Kernel):
    op_type = "Gather"

    @staticmethod
    def run(inputs, outputs, attrs):
        axis = attrs.get('axis', 0)
        data = inputs[0]
        indices = inputs[1].long()
        # ONNX Gather: data[indices, ...] 沿 axis 维收集
        # 使用 torch.index_select 更高效且语义正确
        if indices.dim() > 1:
            # 多维 indices: 需要先展平再 reshape
            flat_idx = indices.reshape(-1)
            result = torch.index_select(data, axis, flat_idx)
            # 恢复 indices 的维度
            out_shape = list(data.shape)
            out_shape[axis] = indices.numel()
            outputs[0].copy_(result.reshape(out_shape))
        else:
            outputs[0].copy_(torch.index_select(data, axis, indices))


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
        # 确保 repeats 是扁平列表
        if isinstance(repeats, list) and len(repeats) == 1 and isinstance(repeats[0], list):
            repeats = repeats[0]
        repeats = [int(r) for r in repeats]
        outputs[0].copy_(inputs[0].repeat(*repeats))


@KernelRegistry.register("Where")
class WhereKernel(Kernel):
    op_type = "Where"

    @staticmethod
    def run(inputs, outputs, attrs):
        condition = inputs[0].bool()
        torch.where(condition, inputs[1], inputs[2], out=outputs[0])


@KernelRegistry.register("Equal")
class EqualKernel(Kernel):
    op_type = "Equal"

    @staticmethod
    def run(inputs, outputs, attrs):
        torch.eq(inputs[0], inputs[1], out=outputs[0])


@KernelRegistry.register("Greater")
class GreaterKernel(Kernel):
    op_type = "Greater"

    @staticmethod
    def run(inputs, outputs, attrs):
        torch.gt(inputs[0], inputs[1], out=outputs[0])


@KernelRegistry.register("Less")
class LessKernel(Kernel):
    op_type = "Less"

    @staticmethod
    def run(inputs, outputs, attrs):
        torch.lt(inputs[0], inputs[1], out=outputs[0])


@KernelRegistry.register("Neg")
class NegKernel(Kernel):
    op_type = "Neg"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(torch.neg(inputs[0]))


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
        torch.pow(inputs[0], inputs[1], out=outputs[0])


@KernelRegistry.register("Erf")
class ErfKernel(Kernel):
    op_type = "Erf"

    @staticmethod
    def run(inputs, outputs, attrs):
        torch.erf(inputs[0], out=outputs[0])


@KernelRegistry.register("ReduceMean")
class ReduceMeanKernel(Kernel):
    op_type = "ReduceMean"

    @staticmethod
    def run(inputs, outputs, attrs):
        # axes 来源: inputs[1] (ONNX 18+) 或 attrs['axes']
        if len(inputs) > 1 and inputs[1] is not None:
            axes = [int(x) for x in inputs[1].cpu().tolist()]
        else:
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
        # axes 来源: inputs[1] (ONNX 18+) 或 attrs['axes']
        if len(inputs) > 1 and inputs[1] is not None:
            axes = [int(x) for x in inputs[1].cpu().tolist()]
        else:
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
        starts = [int(v) for v in inputs[1].cpu().tolist()] if len(inputs) > 1 and inputs[1] is not None else attrs.get('starts', [0])
        ends = [int(v) for v in inputs[2].cpu().tolist()] if len(inputs) > 2 and inputs[2] is not None else attrs.get('ends', [x.shape[0]])
        axes = [int(v) for v in inputs[3].cpu().tolist()] if len(inputs) > 3 and inputs[3] is not None else attrs.get('axes', list(range(len(starts))))
        steps = [int(v) for v in inputs[4].cpu().tolist()] if len(inputs) > 4 and inputs[4] is not None else attrs.get('steps', [1] * len(starts))

        # 构建切片
        slices = [slice(None)] * x.ndim
        for i, ax in enumerate(axes):
            s = int(starts[i]) if i < len(starts) else 0
            e = int(ends[i]) if i < len(ends) else x.shape[ax]
            step = int(steps[i]) if i < len(steps) else 1
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


# ===========================================================================
# Fused Ops for LLM (GPT2)
# ===========================================================================

@KernelRegistry.register("Layernrom")
class LayerNormKernel(Kernel):
    """Fused Layer Normalization: y = (x - mean) / sqrt(var + eps) * weight + bias"""
    op_type = "Layernrom"

    @staticmethod
    def run(inputs, outputs, attrs):
        x = inputs[0]       # [B, N, C]
        weight = inputs[1]  # [C]
        bias = inputs[2]    # [C]
        axes = attrs.get('ReduceMean0_axes', [-1])
        eps = 1e-5

        mean = x.mean(dim=tuple(axes), keepdim=True)
        var = x.var(dim=tuple(axes), keepdim=True, unbiased=False)
        # y = (x - mean) / sqrt(var + eps) * weight + bias
        torch.sub(x, mean, out=outputs[0])
        inv_std = torch.rsqrt(var + eps)
        outputs[0].mul_(inv_std).mul_(weight).add_(bias)


@KernelRegistry.register("Mad")
class MadKernel(Kernel):
    """Fused Mul+Add: y = x * weight + bias"""
    op_type = "Mad"

    @staticmethod
    def run(inputs, outputs, attrs):
        x = inputs[0]
        weight = inputs[1]
        bias = inputs[2]
        torch.mul(x, weight, out=outputs[0])
        outputs[0].add_(bias)


@KernelRegistry.register("Gelu")
class GeluKernel(Kernel):
    """Fused GELU activation: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    op_type = "Gelu"

    @staticmethod
    def run(inputs, outputs, attrs):
        x = inputs[0]
        outputs[0].copy_(F.gelu(x))


@KernelRegistry.register("RangeGather")
class RangeGatherKernel(Kernel):
    """
    Fused Range + Gather for positional embedding lookup with KV cache offset.
    Generates position indices [n_past, n_past+1, ..., n_past+seq_len-1] and
    gathers from position embedding table.
    """
    op_type = "RangeGather"

    @staticmethod
    def run(inputs, outputs, attrs):
        # inputs: [seq_len_shape, const_of_shape_output, ..., pos_embed_weight, n_past]
        seq_len_tensor = inputs[0]  # scalar tensor, sequence length
        pos_weight = inputs[4]      # position embedding weight [max_len, C]
        n_past = inputs[5]          # scalar tensor, past length

        seq_len = int(seq_len_tensor.cpu().item())
        past = int(n_past.cpu().item())

        # Generate position indices [past, past+1, ..., past+seq_len-1]
        indices = torch.arange(past, past + seq_len, device=pos_weight.device, dtype=torch.int64)

        # Gather from position embedding
        outputs[0].copy_(torch.index_select(pos_weight, 0, indices))


# ===========================================================================
# LLM 标准 Ops (Qwen3 / LLaMA 等)
# ===========================================================================

@KernelRegistry.register("LayerNormalization")
class LayerNormKernel(Kernel):
    """
    Layer Normalization (支持 RMS Norm 和 per-head Q/K Norm).

    ONNX 标准 LayerNormalization 节点。
    attr['type'] == 'rms' 时为 RMS Norm（Qwen/LLaMA 使用）。

    支持 per-head norm：当 weight.shape[-1] != x.shape[-1] 时，
    自动将 x reshape 为 [..., N, head_dim] 沿最后一维做 norm。
    （Qwen3 的 q_norm/k_norm 使用此模式）
    """
    op_type = "LayerNormalization"

    @staticmethod
    def run(inputs, outputs, attrs):
        x = inputs[0]
        weight = inputs[1]
        bias = inputs[2] if len(inputs) > 2 else None
        eps = attrs.get('epsilon', 1e-5)
        norm_type = attrs.get('type', 'layer')

        # 检测 per-head norm：weight 维度小于 x 最后一维
        head_dim = weight.shape[-1]
        if head_dim < x.shape[-1]:
            # Per-head norm: reshape [B, S, N*head_dim] -> [B, S, N, head_dim]
            N = x.shape[-1] // head_dim
            x = x.reshape(*x.shape[:-1], N, head_dim)

        if norm_type == 'rms':
            # RMS Norm: y = x * rsqrt(mean(x^2) + eps) * weight
            variance = x.pow(2).mean(-1, keepdim=True)
            y = x * torch.rsqrt(variance + eps)
            y = y * weight
        else:
            # Layer Norm: y = (x - mean) / sqrt(var + eps) * weight + bias
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            y = (x - mean) / torch.sqrt(var + eps)
            y = y * weight
            if bias is not None:
                y = y + bias

        # 恢复原始 shape
        if head_dim < inputs[0].shape[-1]:
            y = y.reshape(inputs[0].shape)

        outputs[0].copy_(y)


@KernelRegistry.register("SDPA")
class SDPAKernel(Kernel):
    """
    Scaled Dot-Product Attention (with optional KV cache).

    支持两种输入格式：
      4D: Q/K/V [B, N, S, head_dim]  (Builder 的 MHA 模式)
      3D: Q/K/V [B, S, D]            (Builder 的 SDPA 模式, D = N * head_dim)

    当 inputs 包含 5 个元素时，后两个是 n_past 和 kv_cache：
      inputs[3]: n_past  (scalar tensor, current past length)
      inputs[4]: kv_cache [B, 2*num_layers, context_len, kv_hidden_size]
                 其中 kv_hidden_size = kv_head_num * head_dim
                 layer_i 的 K 在 [:, layer_i*2, :, :], V 在 [:, layer_i*2+1, :, :]
    """
    op_type = "SDPA"

    @staticmethod
    def run(inputs, outputs, attrs):
        q = inputs[0]
        k = inputs[1]
        v = inputs[2]

        num_heads = attrs.get('head_num', 16)
        kv_head_num = attrs.get('kv_head_num', num_heads)
        layer_i = attrs.get('layer_i', 0)

        # head_dim: 优先从 q 的实际 shape 推导（比 attrs['head_size'] 更可靠）
        if q.dim() == 3:
            head_dim = q.shape[-1] // num_heads
            kv_hidden_size = k.shape[-1]  # 保存原始 K 的最后一维
        else:
            head_dim = q.shape[-1]
            kv_hidden_size = k.shape[-1] * k.shape[-2] if k.dim() == 4 else k.shape[-1]

        # 统一转为 4D [B, N, S, head_dim]
        if q.dim() == 3:
            B, S, D = q.shape
            N = num_heads
            q = q.reshape(B, S, N, head_dim).transpose(1, 2)  # [B, N, S, head_dim]
            k = k.reshape(B, S, kv_head_num, head_dim).transpose(1, 2)
            v = v.reshape(B, S, kv_head_num, head_dim).transpose(1, 2)
        else:
            B, N, S, _ = q.shape

        # KV-cache: 拼接历史 K/V
        has_kv_cache = len(inputs) >= 5 and inputs[4] is not None
        if has_kv_cache:
            n_past_tensor = inputs[3]
            kv_cache = inputs[4]  # [B, 2*num_layers, context_len, kv_hidden_size]
            if n_past_tensor is None:
                n_past = 0
            elif n_past_tensor.numel() == 1:
                n_past = int(n_past_tensor.item())
            else:
                n_past = int(n_past_tensor[0, 0].item())

            # kv_cache 是 4D: [B, 2*num_layers, context_len, kv_hidden_size]
            # kv_hidden_size 已在上面从原始 K 的最后一维推导
            cache_k = kv_cache[:, layer_i * 2, :n_past, :kv_hidden_size]  # [B, n_past, kv_hidden]
            cache_v = kv_cache[:, layer_i * 2 + 1, :n_past, :kv_hidden_size]  # [B, n_past, kv_hidden]

            # Reshape cache K/V to [B, kv_head_num, n_past, head_dim]
            cache_k = cache_k.reshape(B, n_past, kv_head_num, head_dim).transpose(1, 2)
            cache_v = cache_v.reshape(B, n_past, kv_head_num, head_dim).transpose(1, 2)

            # 拼接历史和当前
            k = torch.cat([cache_k, k], dim=2)  # [B, kv_head_num, n_past+S, head_dim]
            v = torch.cat([cache_v, v], dim=2)

            # 把当前 K/V 写回 kv_cache（只写 kv_hidden_size 个元素）
            curr_k_flat = inputs[1].reshape(B, S, -1)  # [B, S, kv_hidden_size]
            curr_v_flat = inputs[2].reshape(B, S, -1)  # [B, S, kv_hidden_size]
            kv_cache[:, layer_i * 2, n_past:n_past + S, :kv_hidden_size] = curr_k_flat
            kv_cache[:, layer_i * 2 + 1, n_past:n_past + S, :kv_hidden_size] = curr_v_flat

            total_S = n_past + S
        else:
            total_S = S

        # GQA: expand KV heads to match Q heads
        if kv_head_num < num_heads:
            repeat = num_heads // kv_head_num
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Scaled dot-product attention
        scale = head_dim ** 0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask: Q 的每个 token 只能看到 K 中 <= 自己位置的 token
        # 对于 KV-cache: Q 位置 [n_past, n_past+S), K 位置 [0, n_past+S)
        if has_kv_cache:
            # 构建 causal mask: [S, total_S]
            mask = torch.full((S, total_S), float('-inf'), device=q.device)
            for i in range(S):
                q_pos = n_past + i
                mask[i, :q_pos + 1] = 0.0
            attn = attn + mask
        else:
            mask = torch.triu(torch.full((S, S), float('-inf'), device=q.device), diagonal=1)
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)

        # Weighted sum
        out = torch.matmul(attn, v)  # [B, N, S, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, S, -1)  # [B, S, D]

        outputs[0].copy_(out)


@KernelRegistry.register("Rope")
class RopeKernel(Kernel):
    """
    Rotary Position Embedding.

    支持两种输入格式：
      4D: x [B, N, S, head_dim]  (Builder 的 MHA 模式)
      3D: x [B, S, D]            (Builder 的 SDPA 模式, D = N * head_dim)

    RoPE 总是按 head_dim 维度应用，对于 3D 输入需要先 reshape。
    cos/sin table 的最后一维是 head_dim/2。

    inputs[1]: cos constant [1, 1, max_pos, head_dim/2] 或 [1, max_pos, head_dim/2]
    inputs[2]: sin constant [1, 1, max_pos, head_dim/2] 或 [1, max_pos, head_dim/2]
    inputs[3]: position [B, S] (每个 token 的位置索引)
    """
    op_type = "Rope"

    @staticmethod
    def run(inputs, outputs, attrs):
        x = inputs[0]
        cos_table = inputs[1]
        sin_table = inputs[2]
        position = inputs[3]   # [B, S]

        # 从 cos/sin table 推断 head_dim
        head_dim_half = cos_table.shape[-1]
        head_dim = head_dim_half * 2

        if x.dim() == 4:
            B, N, S, HD = x.shape
            # 已经是 [B, N, S, head_dim]
            pass
        else:
            B, S, D = x.shape
            N = D // head_dim  # 头数
            x = x.reshape(B, S, N, head_dim).transpose(1, 2)  # [B, N, S, head_dim]

        # 根据 position 索引从 cos/sin 表中 gather
        pos_idx = position.long()  # [B, S]

        # cos/sin table 可能是 [1, max_pos, half] 或 [1, 1, max_pos, half]
        if cos_table.dim() == 3:
            cos = cos_table.expand(B, -1, -1)  # [B, max_pos, half]
            sin = sin_table.expand(B, -1, -1)
            cos = torch.gather(cos, 1, pos_idx.unsqueeze(-1).expand(-1, -1, head_dim_half))
            sin = torch.gather(sin, 1, pos_idx.unsqueeze(-1).expand(-1, -1, head_dim_half))
            cos = cos.unsqueeze(1)  # [B, 1, S, half]
            sin = sin.unsqueeze(1)
        else:
            cos = cos_table.expand(B, -1, -1, -1)  # [B, 1, max_pos, half]
            sin = sin_table.expand(B, -1, -1, -1)
            cos = torch.gather(cos, 2, pos_idx.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, head_dim_half))
            sin = torch.gather(sin, 2, pos_idx.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, head_dim_half))

        # Apply RoPE on head_dim dimension
        x1 = x[..., :head_dim_half]
        x2 = x[..., head_dim_half:]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        y = torch.cat([y1, y2], dim=-1)  # [B, N, S, head_dim]

        # 如果输入是 3D，转回 [B, S, D]
        if inputs[0].dim() == 3:
            y = y.transpose(1, 2).reshape(B, S, -1)

        outputs[0].copy_(y)


@KernelRegistry.register("Silu")
class SiluKernel(Kernel):
    """SiLU (Swish) activation: y = x * sigmoid(x)"""
    op_type = "Silu"

    @staticmethod
    def run(inputs, outputs, attrs):
        outputs[0].copy_(F.silu(inputs[0]))


# ===========================================================================
# YOLO / CV detection 所需额外算子
# ===========================================================================

@KernelRegistry.register("GatherElements")
class GatherElementsKernel(Kernel):
    """ONNX GatherElements: 沿 axis 按 indices 收集元素，保持 indices 的维度结构"""
    op_type = "GatherElements"

    @staticmethod
    def run(inputs, outputs, attrs):
        axis = attrs.get('axis', 0)
        data = inputs[0]
        indices = inputs[1].long()
        outputs[0].copy_(torch.gather(data, axis, indices))


@KernelRegistry.register("Mod")
class ModKernel(Kernel):
    """ONNX Mod: 逐元素取模 (fmod=0 时行为同 Python % 即 truncation toward zero)"""
    op_type = "Mod"

    @staticmethod
    def run(inputs, outputs, attrs):
        fmod = attrs.get('fmod', 0)
        if fmod:
            outputs[0].copy_(torch.fmod(inputs[0], inputs[1]))
        else:
            outputs[0].copy_(torch.remainder(inputs[0], inputs[1]))


@KernelRegistry.register("ReduceMax")
class ReduceMaxKernel(Kernel):
    """ONNX ReduceMax: 沿指定轴取最大值"""
    op_type = "ReduceMax"

    @staticmethod
    def run(inputs, outputs, attrs):
        # axes 来源: inputs[1] (ONNX 18+) 或 attrs['axes']
        if len(inputs) > 1 and inputs[1] is not None:
            axes = [int(x) for x in inputs[1].cpu().tolist()]
        else:
            axes = attrs.get('axes', None)
        keepdims = attrs.get('keepdims', 1)
        if axes is not None:
            outputs[0].copy_(inputs[0].amax(dim=tuple(axes), keepdim=bool(keepdims)))
        else:
            outputs[0].copy_(inputs[0].amax(keepdim=bool(keepdims)))


@KernelRegistry.register("Split")
class SplitKernel(Kernel):
    """ONNX Split: 沿 axis 将 tensor 切分为多个子 tensor

    split 参数来源优先级:
      1. inputs[1] (动态 split sizes tensor)
      2. attrs['split'] (静态属性)
      3. 均匀分割 (num_outputs 等分)
    """
    op_type = "Split"

    @staticmethod
    def run(inputs, outputs, attrs):
        axis = attrs.get('axis', 0)

        # 获取 split sizes
        split_sizes = None
        if len(inputs) > 1 and inputs[1] is not None:
            split_sizes = [int(x) for x in inputs[1].cpu().tolist()]
        elif attrs.get('split') is not None:
            split_sizes = list(attrs['split'])

        if split_sizes is not None:
            chunks = torch.split(inputs[0], split_sizes, dim=axis)
        else:
            # 均匀分割
            num_outputs = len(outputs)
            chunks = torch.chunk(inputs[0], num_outputs, dim=axis)
        for i, chunk in enumerate(chunks):
            if i < len(outputs):
                outputs[i].copy_(chunk)


@KernelRegistry.register("TopK")
class TopKKernel(Kernel):
    """ONNX TopK: 沿 axis 取 top-k 值和索引 (largest=1 取最大, sorted=1 排序输出)"""
    op_type = "TopK"

    @staticmethod
    def run(inputs, outputs, attrs):
        axis = attrs.get('axis', -1)
        largest = attrs.get('largest', 1)
        sorted_result = attrs.get('sorted', 1)
        k = int(inputs[1].item()) if len(inputs) > 1 and inputs[1] is not None else attrs.get('k', 1)

        values, indices = torch.topk(
            inputs[0], k, dim=axis,
            largest=bool(largest),
            sorted=bool(sorted_result),
        )
        outputs[0].copy_(values)
        if len(outputs) > 1:
            outputs[1].copy_(indices)
