"""
Memory Pool for ONNX Compressed Memory

基于 cg.compress_memory() 的结果，使用 PyTorch 作为后端分配器。
compress_mem 格式: {tensor_name: [offset, size]}，每个张量在内存池中的偏移和字节大小。
compress_size: 内存池总字节数。

用法:
    from memory_pool import MemoryPool
    pool = MemoryPool(compress_mem, compress_size, dtype=torch.float32)
    tensor = pool.get('tensor_name')  # 返回该张量对应的 torch.Tensor（共享内存）
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np


class MemoryPool:
    """
    基于 compress_memory() 结果的 PyTorch 内存池。

    分配一块连续的 CPU/XPU 内存，compress_mem 中的每个张量都映射到
    该内存块的一个视图（view），实现零拷贝复用。

    compress_mem: Dict[str, List[int]] — {tensor_name: [offset_bytes, size_bytes]}
    compress_size: int — 内存池总大小（字节）
    dtype: torch.dtype — 张量元素类型（默认 float32）
    device: str — 'cpu' 或 'xpu'
    """

    def __init__(
        self,
        compress_mem: Dict[str, List[int]],
        compress_size: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        if compress_size <= 0:
            raise ValueError(f"compress_size must be > 0, got {compress_size}")

        self._compress_mem = compress_mem
        self._compress_size = compress_size
        self._dtype = dtype
        self._device = device
        self._element_size = torch.tensor([], dtype=dtype).element_size()

        # 分配一整块连续内存作为 memory pool
        self._pool = torch.empty(compress_size, dtype=torch.uint8, device=device)

        # 缓存已创建的视图张量
        self._tensors: Dict[str, torch.Tensor] = {}

        self._validate()

    def _validate(self):
        """验证所有张量不越界（compress_memory 自身已保证同生命周期无重叠，跨生命周期的复用是合法的）"""
        for name, (offset, size) in self._compress_mem.items():
            if offset < 0:
                raise ValueError(f"Tensor '{name}' has negative offset {offset}")
            if size <= 0:
                raise ValueError(f"Tensor '{name}' has non-positive size {size}")
            end = offset + size
            if end > self._compress_size:
                raise ValueError(
                    f"Tensor '{name}' ends at {end}, "
                    f"exceeds pool size {self._compress_size}"
                )

    @property
    def pool(self) -> torch.Tensor:
        """返回底层 uint8 内存池张量"""
        return self._pool

    @property
    def compress_size(self) -> int:
        """内存池总大小（字节）"""
        return self._compress_size

    @property
    def device(self) -> str:
        return self._device

    def get(self, key: str, shape: Optional[Union[List[int], Tuple[int, ...]]] = None) -> torch.Tensor:
        """
        获取指定张量名的 torch.Tensor（共享内存池视图）。

        Args:
            key: 张量名称（compress_mem 中的 key）
            shape: 可选，指定张量形状。如果为 None，返回 1D 展平张量。
                   形状的乘积必须 <= size / element_size。

        Returns:
            共享内存池的 torch.Tensor 视图。
        """
        if key not in self._compress_mem:
            raise KeyError(
                f"Tensor '{key}' not found in compress_mem. "
                f"Available keys: {list(self._compress_mem.keys())[:10]}..."
            )

        offset, size = self._compress_mem[key]
        num_elements = size // self._element_size

        if shape is not None:
            expected_elements = 1
            for d in shape:
                expected_elements *= d
            if expected_elements > num_elements:
                raise ValueError(
                    f"Shape {shape} requires {expected_elements} elements, "
                    f"but tensor '{key}' only has {num_elements} elements "
                    f"({size} bytes / {self._element_size} bytes per element)"
                )
            # 实际需要的字节数（可能小于分配的 size，支持动态 shape）
            used_bytes = expected_elements * self._element_size
        else:
            used_bytes = size

        # 从 uint8 pool 中取出视图并重新解释为目标 dtype
        # 只取实际需要的字节数（支持动态 shape：分配按最大 shape，使用按实际 shape）
        byte_view = self._pool[offset: offset + used_bytes]
        t = byte_view.view(dtype=self._dtype)

        if shape is not None:
            t = t.reshape(shape)

        # 缓存
        self._tensors[key] = t
        return t

    def get_batch(
        self,
        key: str,
        batch_size: int,
        element_shape: Optional[Union[List[int], Tuple[int, ...]]] = None,
    ) -> torch.Tensor:
        """
        获取支持 batch 维度的张量视图。

        Args:
            key: 张量名称
            batch_size: batch 维度大小
            element_shape: 每个元素的形状（不含 batch 维度）

        Returns:
            形状为 (batch_size, *element_shape) 的 torch.Tensor 视图
        """
        if element_shape is not None:
            shape = [batch_size] + list(element_shape)
        else:
            shape = [batch_size]
        return self.get(key, shape=shape)

    def __getitem__(self, key: str) -> torch.Tensor:
        """语法糖: pool['tensor_name']"""
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._compress_mem

    def __len__(self) -> int:
        return len(self._compress_mem)

    def keys(self):
        return self._compress_mem.keys()

    def items(self):
        return self._compress_mem.items()

    def to(self, device: str):
        """
        将整个内存池移动到指定设备（cpu/xpu）。
        注意：这会清除所有缓存的张量视图。
        """
        if device == self._device:
            return self
        self._pool = self._pool.to(device)
        self._device = device
        self._tensors.clear()
        return self

    def zero_(self):
        """将整个内存池置零"""
        self._pool.zero_()
        return self

    def fill_tensor(self, key: str, value: Union[torch.Tensor, np.ndarray]):
        """
        将数据填充到指定张量的内存区域。

        Args:
            key: 张量名称
            value: 要填充的数据（torch.Tensor 或 np.ndarray）
        """
        t = self.get(key)
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        t.copy_(value.flatten()[:t.numel()].to(device=self._device, dtype=self._dtype))

    def __repr__(self) -> str:
        return (
            f"MemoryPool(compress_size={self._compress_size:,} bytes, "
            f"num_tensors={len(self._compress_mem)}, "
            f"dtype={self._dtype}, device={self._device})"
        )


def create_memory_pool_from_model(
    model_path: str,
    input_desc: dict,
    input_range: dict,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Tuple["MemoryPool", dict]:
    """
    便捷函数：从 ONNX 模型直接创建 MemoryPool。

    Args:
        model_path: ONNX 模型路径
        input_desc: 输入描述，如 {'input': ('batch', 3, 'height', 'width')}
        input_range: 输入范围，如 {'batch': (1, 4), 'height': (224, 1080)}
        dtype: 张量数据类型
        device: 设备

    Returns:
        (MemoryPool, shape_engine) 元组
    """
    import onnx_tool

    model_config = {
        "name": model_path,
        "dynamic_input": None,
        "input_desc": input_desc,
        "input_range": input_range,
    }

    m = onnx_tool.Model(model_config["name"])
    g = m.graph
    g.graph_reorder_nodes()

    shape_engine = g.shape_regress(
        model_config["input_desc"], model_config["input_range"]
    )
    cg = g.get_compute_graph()
    compress_mem, compress_size = cg.compress_memory()

    pool = MemoryPool(compress_mem, compress_size, dtype=dtype, device=device)
    return pool, shape_engine
