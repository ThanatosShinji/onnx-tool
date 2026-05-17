"""
ONNX Compute Graph Inference Engine

基于 onnx_tool 的 compute graph 和 MemoryPool，实现图级别的推理框架。
"""

import time
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import onnx_tool
from memory_pool import MemoryPool
from kernels import KernelRegistry


class GraphInfer:
    """
    计算图推理引擎。

    在 __init__ 中完成：
      1. 加载 ONNX 模型
      2. 节点重排
      3. Shape regression（形状推导引擎）
      4. 提取 compute graph
      5. 内存压缩 + MemoryPool 创建

    forward 中遍历所有节点，检查每个 node 的输入输出 tensor 在 memory pool
    中的 buf 是否重叠（debug 模式）。
    """

    def __init__(
        self,
        onnx_path: str,
        input_desc: Dict[str, tuple],
        input_range: Dict[str, tuple],
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        safetensors_path: Optional[str] = None,
        weight_map: Optional[Dict[str, str]] = None,
    ):
        self.onnx_path = onnx_path
        self.input_desc = input_desc
        self.input_range = input_range
        self.dtype = dtype
        self.device = device
        self.safetensors_path = safetensors_path
        self.weight_map = weight_map or {}

        # 1. 加载模型（启用 constant folding，折叠常量子图）
        self._model = onnx_tool.Model(onnx_path, {'constant_folding': True})
        self._graph = self._model.graph

        # 2. 从 safetensors 补全 weight 的 shape 信息（ONNX 文件中可能缺失）
        if safetensors_path:
            self._update_weight_shapes_from_safetensors(safetensors_path)

        # 3. 节点重排
        self._graph.graph_reorder_nodes()

        # 4. Shape regression
        self.shape_engine = self._graph.shape_regress(input_desc, input_range)

        # 4. 用 input_range 的最大值更新 shape_engine 变量，
        #    并在原始 graph 上重新 shape_infer，确保 tensormap 中的 shape 是最大值
        for var_name, (lo, hi) in input_range.items():
            self.shape_engine.update_variable(var_name, hi)
        max_inputs = self.shape_engine.generate_input()
        # generate_input() 默认生成 float64，会导致 compress_memory 中
        # get_memsize() 按 8 bytes/elem 计算，使 compress_size 翻倍。
        # 转为 self.dtype 对应的 numpy 类型，确保 memsize 计算正确。
        np_dtype = self._get_np_dtype()
        for key in max_inputs:
            max_inputs[key] = max_inputs[key].astype(np_dtype)
        self._graph.shape_infer(max_inputs)

        # 5. 提取 compute graph（此时 tensormap 中的 shape 已是最大值）
        self._cg = self._graph.get_compute_graph()

        # 6. 内存压缩 + MemoryPool（compress_memory 基于最大 shape 分配）
        # 使用 graph.py 的 compress_memory（Best-Fit 策略，按大小降序重排以减少碎片）
        # 检测是否有 kv_cache tensor，有则使用 llm_test 的双池压缩
        has_kv_cache = 'kv_cache' in self._cg.tensormap
        if has_kv_cache:
            from benchmark.llm_test import compress_memory_with_kv_cache
            compress_mem, compress_size, kv_compress_mem, kv_compress_size = \
                compress_memory_with_kv_cache(self._cg)
        else:
            compress_mem, compress_size = self._cg.compress_memory()
            kv_compress_mem, kv_compress_size = {}, 0
        self.pool = MemoryPool(
            compress_mem, compress_size, dtype=dtype, device=device
        )
        self.kv_pool = MemoryPool(
            kv_compress_mem, kv_compress_size, dtype=dtype, device=device
        ) if kv_compress_size > 0 else None

        # 合并两个 pool 的 tensor blocks
        self._tensor_blocks: Dict[str, List[int]] = {}
        elem_size = torch.tensor([], dtype=dtype).element_size()
        for tname, (offset, _) in compress_mem.items():
            if tname in self._cg.tensormap:
                tm_shape = self._cg.tensormap[tname].get_shape()
                num_elems = 1
                for s in tm_shape:
                    num_elems *= s
                correct_size = num_elems * elem_size
            else:
                correct_size = 0
            self._tensor_blocks[tname] = [offset, correct_size]
        for tname, (offset, _) in kv_compress_mem.items():
            if tname in self._cg.tensormap:
                tm_shape = self._cg.tensormap[tname].get_shape()
                num_elems = 1
                for s in tm_shape:
                    num_elems *= s
                correct_size = num_elems * elem_size
            else:
                correct_size = 0
            self._tensor_blocks[tname] = [offset, correct_size]

        # 预创建所有 tensor 的 1D 视图（按最大 shape 分配），forward 时只 reshape
        self._tensor_views: Dict[str, torch.Tensor] = {}
        for tname, (offset, size) in self._tensor_blocks.items():
            if size > 0:
                if tname in kv_compress_mem and self.kv_pool is not None:
                    self._tensor_views[tname] = self.kv_pool._pool[offset:offset + size].view(dtype=dtype)
                else:
                    self._tensor_views[tname] = self.pool._pool[offset:offset + size].view(dtype=dtype)

        # 获取节点执行顺序（按 nodemap 的插入顺序，即拓扑序）
        self._node_names = list(self._cg.nodemap.keys())

        # 缓存常量 tensor（weight/bias 等），避免每次 forward 从 numpy 转换
        self._constant_tensors: Dict[str, torch.Tensor] = {}
        self._preload_constants()

        # 动态输入缓存：不在 memory pool 中的输入（如 n_past scalar），
        # forward 时由用户传入并暂存于此，供 _resolve_tensor 查找
        self._dynamic_inputs: Dict[str, torch.Tensor] = {}

        # 预缓存每个 node 的 input/output tensor 名称列表，避免 forward 中重复遍历
        self._node_input_names: List[List[str]] = []
        self._node_output_names: List[List[str]] = []
        for node_name in self._node_names:
            node = self._cg.nodemap[node_name]
            self._node_input_names.append(list(node.input))
            self._node_output_names.append(list(node.output))

        # shape 缓存：forward 中 _update_shape_from_input 之后批量刷新，
        # 避免每个 tensor 都调用 shape_engine.get_tensorshape()
        self._tensor_shape_cache: Dict[str, List[int]] = {}

        # 预构建每个 node 的 input/output tensor 视图缓存。
        # 每个元素是 (input_tensors, output_tensors)，其中 tensor 是 _tensor_views 的 1D 切片，
        # forward 中 shape 不变时直接 reshape 复用，跳过 _resolve_tensor 和 shape 查询。
        self._node_tensor_cache: List[tuple] = []
        for idx, node_name in enumerate(self._node_names):
            node = self._cg.nodemap[node_name]
            input_names = self._node_input_names[idx]
            output_names = self._node_output_names[idx]
            # input: 优先从常量取（weight/bias 等），否则从 pool 取 1D 视图
            in_tensors = []
            for tname in input_names:
                if tname in self._constant_tensors:
                    in_tensors.append(self._constant_tensors[tname])
                elif tname in self._tensor_views:
                    in_tensors.append(self._tensor_views[tname])
                else:
                    in_tensors.append(None)
            # output: 全部从 pool 取 1D 视图
            out_tensors = []
            for tname in output_names:
                if tname in self._tensor_views:
                    out_tensors.append(self._tensor_views[tname])
                else:
                    out_tensors.append(None)
            self._node_tensor_cache.append((in_tensors, out_tensors))

        print(f"GraphInfer initialized: {len(self._node_names)} nodes, "
              f"{len(self._tensor_blocks)} tensors, "
              f"{len(self._constant_tensors)} constants, "
              f"pool={compress_size:,} bytes")

    @property
    def compute_graph(self):
        return self._cg

    @property
    def node_names(self) -> List[str]:
        return list(self._node_names)

    def get_tensor_shape(self, tensor_name: str) -> List[int]:
        """获取 tensor 的形状：优先从缓存，fallback 到 shape_engine / tensormap"""
        if tensor_name in self._tensor_shape_cache:
            return self._tensor_shape_cache[tensor_name]
        try:
            shape = self.shape_engine.get_tensorshape(tensor_name)
            if shape is not None and all(isinstance(s, int) and s > 0 for s in shape):
                self._tensor_shape_cache[tensor_name] = shape
                return shape
        except Exception:
            pass
        if tensor_name in self._cg.tensormap:
            shape = self._cg.tensormap[tensor_name].get_shape()
            self._tensor_shape_cache[tensor_name] = shape
            return shape
        return []

    def get_tensor_block(self, tensor_name: str) -> Optional[List[int]]:
        """获取 tensor 在 memory pool 中的 [offset, size]"""
        return self._tensor_blocks.get(tensor_name)

    def get_tensor(self, tensor_name: str, shape: Optional[Union[List[int], Tuple[int, ...]]] = None) -> torch.Tensor:
        """从 memory pool 获取 tensor 视图"""
        return self.pool.get(tensor_name, shape=shape)

    def _reshape_view(self, tname: str, shape: List[int]) -> torch.Tensor:
        """对预创建的 1D 视图做 reshape，返回指定 shape 的视图。
        如果视图元素数大于目标 shape 所需，自动 slicing 取前 N 个元素。"""
        t = self._tensor_views[tname]
        needed = 1
        for s in shape:
            needed *= s
        if t.numel() > needed:
            return t[:needed].reshape(shape)
        if t.numel() == needed:
            return t.reshape(shape)
        # t.numel() < needed: 用 view 并允许广播
        return t.reshape(shape)

    def forward(self, inputs: Dict[str, torch.Tensor], debug: bool = False,
                profile: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        compress_memory 已按最大 shape 分配好所有 tensor 的 offset 和 size，
        forward 只需更新每个 tensor 的 shape（reshape 视图），指针不变。

        Args:
            inputs: 输入 dict {name: tensor}，key 对应 ONNX 模型的 input name
            debug: 是否打印详细的 debug 信息（tensor buf、重叠检查等）
            profile: 是否收集每个 op_type 和 node 的耗时统计

        Returns:
            outputs: 输出 dict {name: tensor}，key 对应 ONNX 模型的 output name
            当 profile=True 时，outputs 额外包含 '__profile__' key
        """
        # 根据输入 tensor 的实际形状更新 shape_engine 变量
        first_input = next(iter(inputs.values()))
        shape_changed = self._update_shape_from_input(first_input)

        # 批量刷新所有 tensor 的 shape 缓存（仅当 shape 实际变化时）
        if shape_changed or not self._tensor_shape_cache:
            self._tensor_shape_cache.clear()
            for tname in self._tensor_blocks:
                try:
                    shape = self.shape_engine.get_tensorshape(tname)
                    if shape is not None and all(isinstance(s, int) and s > 0 for s in shape):
                        self._tensor_shape_cache[tname] = shape
                except Exception:
                    pass
            # fallback: 不在 shape_engine 中的 tensor 从 tensormap 取
            for tname in self._tensor_blocks:
                if tname not in self._tensor_shape_cache and tname in self._cg.tensormap:
                    self._tensor_shape_cache[tname] = self._cg.tensormap[tname].get_shape()

        # 将输入 tensor 拷贝到 memory pool 中对应的位置
        for input_name, input_tensor in inputs.items():
            if input_name in self._tensor_views:
                flat = self._tensor_views[input_name]
                needed = input_tensor.numel()
                try:
                    t = flat[:needed].reshape(input_tensor.shape)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Failed to reshape input '{input_name}': "
                        f"flat.numel()={flat.numel()}, needed={needed}, "
                        f"input_tensor.shape={input_tensor.shape}, "
                        f"input_tensor.dtype={input_tensor.dtype}"
                    ) from e
                t.copy_(input_tensor.to(device=self.device, dtype=self.dtype))
            else:
                # 不在 pool 中的动态输入（如 n_past scalar），暂存供 kernel 使用
                self._dynamic_inputs[input_name] = input_tensor.to(device=self.device)

        if debug:
            print(f"\n{'='*80}")
            print(f"Forward Debug: {len(self._node_names)} nodes")
            print(f"{'='*80}")

        # profile 统计
        if profile:
            import collections
            op_time: Dict[str, float] = collections.defaultdict(float)
            op_count: Dict[str, int] = collections.defaultdict(int)
            node_times: List[tuple] = []  # (node_name, op_type, overhead_s, kernel_s)
            overhead_total = 0.0

        # shape 变化时重建缓存标记
        rebuild_cache = (shape_changed
                         or not hasattr(self, '_last_node_tensors')
                         or self._last_node_tensors is None)
        if rebuild_cache:
            self._last_node_tensors = []

        for idx, node_name in enumerate(self._node_names):
            node = self._cg.nodemap[node_name]
            input_names = self._node_input_names[idx]
            output_names = self._node_output_names[idx]
            cached_in, cached_out = self._node_tensor_cache[idx]

            if debug:
                print(f"\n[{idx+1}/{len(self._node_names)}] Node: {node.name} ({node.op_type})")

            # --- 准备输入 tensor ---
            if profile:
                t0 = time.perf_counter()
            if rebuild_cache:
                input_tensors = []
                for i, tname in enumerate(input_names):
                    t = cached_in[i]
                    if t is not None and tname in self._tensor_views:
                        shape = self._tensor_shape_cache.get(tname)
                        if shape is None:
                            shape = self.get_tensor_shape(tname)
                        needed = 1
                        for s in shape:
                            needed *= s
                        try:
                            if t.numel() > needed:
                                input_tensors.append(t[:needed].reshape(shape))
                            else:
                                input_tensors.append(t.reshape(shape))
                        except RuntimeError as e:
                            raise RuntimeError(
                                f"Failed to reshape input '{tname}' for node '{node_name}': "
                                f"t.numel()={t.numel()}, needed={needed}, shape={shape}"
                            ) from e
                    elif t is not None:
                        input_tensors.append(t)  # constant
                    else:
                        input_tensors.append(self._resolve_tensor(tname))
            else:
                input_tensors = self._last_node_tensors[idx][0]
                # 刷新动态输入（如 n_past），它们每步都可能变化
                for i, tname in enumerate(input_names):
                    if tname in self._dynamic_inputs:
                        input_tensors[i] = self._dynamic_inputs[tname]
            if profile:
                t_resolve = time.perf_counter()
            if debug:
                for tname in input_names:
                    self._print_tensor_info(tname, prefix="    In: ")

            # --- 准备输出 tensor ---
            if rebuild_cache:
                output_tensors = []
                for i, tname in enumerate(output_names):
                    t = cached_out[i]
                    shape = self._tensor_shape_cache.get(tname)
                    if shape is None:
                        shape = self.get_tensor_shape(tname)
                    needed = 1
                    for s in shape:
                        needed *= s
                    if t.numel() > needed:
                        output_tensors.append(t[:needed].reshape(shape))
                    else:
                        output_tensors.append(t.reshape(shape))
            else:
                output_tensors = self._last_node_tensors[idx][1]
            if profile:
                t_prep_output = time.perf_counter()

            # --- 查找 kernel ---
            kernel_cls = KernelRegistry.get(node.op_type)
            if profile:
                t_kernel_lookup = time.perf_counter()

            if kernel_cls is not None:
                if debug:
                    print(f"    >> Running kernel: {kernel_cls.__name__}")
                if profile:
                    t2 = time.perf_counter()
                kernel_cls.run(input_tensors, output_tensors, node.attr)
                if profile:
                    if self.device == 'xpu':
                        torch.xpu.synchronize()
                    t3 = time.perf_counter()
                    kernel_time = t3 - t2
                    resolve_input_time = t_resolve - t0
                    prep_output_time = t_prep_output - t_resolve
                    lookup_time = t_kernel_lookup - t_prep_output
                    overhead = t_kernel_lookup - t0
                    op_time[node.op_type] += kernel_time
                    op_count[node.op_type] += 1
                    node_times.append((node_name, node.op_type,
                                       resolve_input_time, prep_output_time,
                                       lookup_time, kernel_time))
                    overhead_total += overhead
            else:
                raise RuntimeError(
                    f"Unsupported op_type '{node.op_type}' in node '{node_name}'. "
                    f"No registered kernel found. "
                    f"Registered ops: {sorted(KernelRegistry.registered_ops())}"
                )

            # 缓存本次构建的 tensor 视图
            if rebuild_cache:
                self._last_node_tensors.append((input_tensors, output_tensors))

            # --- 检查输入输出重叠 ---
            if debug:
                self._check_overlap(node)

        # XPU 同步，确保所有 kernel 执行完毕
        if self.device == 'xpu' and not profile:
            torch.xpu.synchronize()

        # 收集输出
        outputs = {}
        for output_name in self._cg.output:
            if output_name in self._tensor_views:
                shape = self._tensor_shape_cache.get(output_name)
                if shape is None:
                    shape = self.get_tensor_shape(output_name)
                outputs[output_name] = self._reshape_view(output_name, shape)

        # 附加 profile 结果
        if profile:
            resolve_input_total = sum(nt[2] for nt in node_times)
            prep_output_total = sum(nt[3] for nt in node_times)
            lookup_total = sum(nt[4] for nt in node_times)
            outputs['__profile__'] = {
                'op_time': dict(op_time),
                'op_count': dict(op_count),
                'node_times': node_times,
                'overhead_total': overhead_total,
                'overhead_breakdown': {
                    'resolve_input': resolve_input_total,
                    'prepare_output': prep_output_total,
                    'kernel_lookup': lookup_total,
                },
            }

        return outputs

    def print_profile(self, profile_data: dict):
        """打印 profile 结果"""
        op_time = profile_data['op_time']
        op_count = profile_data['op_count']
        node_times = profile_data['node_times']
        overhead_total = profile_data['overhead_total']
        overhead_breakdown = profile_data.get('overhead_breakdown', {})
        kernel_total = sum(op_time.values())
        total = kernel_total + overhead_total

        print(f"\n{'='*70}")
        print(f"Profile Summary")
        print(f"{'='*70}")
        print(f"{'Op Type':<20} {'Count':>6} {'Total (s)':>12} {'Avg (ms)':>10} {'%':>8}")
        print(f"{'-'*56}")
        for op in sorted(op_time, key=lambda x: -op_time[x]):
            t = op_time[op]
            c = op_count[op]
            avg = t / c * 1000
            pct = t / total * 100
            print(f"{op:<20} {c:>6} {t:>12.6f} {avg:>10.4f} {pct:>7.1f}%")
        print(f"{'-'*56}")
        print(f"{'Kernel total':<20} {sum(op_count.values()):>6} {kernel_total:>12.6f} {'':>10} {kernel_total/total*100:>7.1f}%")
        print(f"{'Overhead (prepare)':<20} {'':>6} {overhead_total:>12.6f} {'':>10} {overhead_total/total*100:>7.1f}%")
        print(f"{'Total':<20} {'':>6} {total:>12.6f} {'':>10} {'100.0%':>8}")
        print(f"{'='*70}")

        # Overhead 细分
        if overhead_breakdown:
            print(f"\nOverhead Breakdown:")
            print(f"{'Category':<25} {'Total (s)':>12} {'Avg/Node (ms)':>16} {'% of Overhead':>16} {'% of Total':>12}")
            print(f"{'-'*81}")
            for cat in ['resolve_input', 'prepare_output', 'kernel_lookup']:
                t = overhead_breakdown.get(cat, 0)
                avg = t / len(node_times) * 1000 if node_times else 0
                pct_over = t / overhead_total * 100 if overhead_total > 0 else 0
                pct_total = t / total * 100 if total > 0 else 0
                label = {'resolve_input': 'Resolve Input',
                         'prepare_output': 'Prepare Output',
                         'kernel_lookup': 'Kernel Lookup'}.get(cat, cat)
                print(f"{label:<25} {t:>12.6f} {avg:>16.4f} {pct_over:>15.1f}% {pct_total:>11.1f}%")
            print(f"{'-'*81}")
            print(f"{'Overhead total':<25} {overhead_total:>12.6f} {'':>16} {'100.0%':>15} {overhead_total/total*100:>10.1f}%")

        # 最慢的 top-10 node
        print(f"\nTop-10 slowest nodes:")
        print(f"{'Node':<45} {'Op':>8} {'Res(ms)':>9} {'Prep(ms)':>9} {'Look(ms)':>9} {'Kernel(ms)':>11}")
        print(f"{'-'*91}")
        sorted_nodes = sorted(node_times, key=lambda x: x[5] + x[2] + x[3] + x[4], reverse=True)
        for name, op, rt, pt, lt, kt in sorted_nodes[:10]:
            short_name = name if len(name) < 43 else '...' + name[-40:]
            print(f"{short_name:<45} {op:>8} {rt*1000:>9.4f} {pt*1000:>9.4f} {lt*1000:>9.4f} {kt*1000:>11.4f}")

    def _safetensors_is_transposed_layout(self) -> bool:
        """safetensors 中的 2D weight 是否为 [N, K]（PyTorch transposed layout）。

        PyTorch 的 Linear weight 存储为 [out_features, in_features]，
        而 ONNX MatMul 期望 [in_features, out_features]。
        如果返回 True，表示 safetensors 的 2D weight 需要转置才能用于 ONNX MatMul。
        """
        return True

    def _needs_transpose_for_op(self, op_type: str) -> bool:
        """判断该 op_type 是否需要将 safetensors 的 [N, K] 转置为 [K, N]。

        MatMul: 需要，ONNX MatMul 期望 weight 为 [K, N]
        Gemm: 不需要，Gemm 有 transB 属性，由 preprocess_weight 处理
        Gather/其他: 不需要，embed_tokens 等保持原样
        """
        return op_type == 'MatMul'

    def _transpose_safetensors_shape(self, shape, op_type: str = ''):
        """将 safetensors 的 [N, K] shape 转置为 ONNX 的 [K, N]。
        仅对 MatMul 的 2D weight 生效。
        """
        if self._safetensors_is_transposed_layout() and self._needs_transpose_for_op(op_type) and len(shape) == 2:
            return [shape[1], shape[0]]
        return shape

    def _transpose_safetensors_tensor(self, t: torch.Tensor, op_type: str = '') -> torch.Tensor:
        """将 safetensors 的 [N, K] tensor 转置为 ONNX 的 [K, N]。
        仅对 MatMul 的 2D weight 生效。
        """
        if self._safetensors_is_transposed_layout() and self._needs_transpose_for_op(op_type) and t.dim() == 2:
            return t.T.contiguous()
        return t

    def _update_weight_shapes_from_safetensors(self, safetensors_path: str):
        """从 safetensors 文件读取 weight shape，更新到 graph 的 tensormap 中。

        onnx_tool.llm.Builder 导出的 ONNX 文件缺少 weight 的 shape 信息，
        需要从 safetensors 中补全，否则 shape_regress 无法推导 tensor shape。

        safetensors 中的 2D weight 为 [N, K]（PyTorch layout），
        对于 MatMul 的 weight 需要转置为 [K, N]（ONNX layout）以正确推导 shape。
        """
        import safetensors

        # 预计算每个 tensor 的消费者 op_type
        consumer_op: Dict[str, str] = {}
        for node in self._graph.nodemap.values():
            for inp in node.input:
                if inp not in consumer_op:
                    consumer_op[inp] = node.op_type

        with safetensors.safe_open(safetensors_path, framework='pt') as f:
            st_keys = set(f.keys())
            # Builder 导出的 ONNX 中 weight 被错误地放在 graph input 中。
            # 只排除真正的动态输入（由 input_desc 指定的）和 graph output
            dynamic_inputs = set(self.input_desc.keys()) if hasattr(self, 'input_desc') else set()
            output_set = set(self._graph.output)
            # 记录已更新的 tensor，供 _preload_constants 使用
            self._safetensors_updated: set = set()
            for tname in list(self._graph.tensormap.keys()):
                tm = self._graph.tensormap[tname]
                # 跳过已有 shape 的、动态输入的、输出的 tensor
                if tm.get_shape() or tname in dynamic_inputs or tname in output_set:
                    continue
                # 尝试直接匹配名称
                st_name = tname
                if st_name not in st_keys:
                    # 尝试通过 weight_map 匹配
                    st_name = self.weight_map.get(tname, tname)
                if st_name in st_keys:
                    shape = list(f.get_tensor(st_name).shape)
                    op_type = consumer_op.get(tname, '')
                    shape = self._transpose_safetensors_shape(shape, op_type)
                    tm.update_shape(shape)
                    self._safetensors_updated.add(tname)
                    print(f"  Updated shape from safetensors: {tname} -> {shape}")

    def _preload_constants(self):
        """预加载所有常量 tensor 到目标设备，并对 weight 执行 kernel 注册的前处理

        支持两种数据源：
          1. ONNX 文件内嵌的 weight（tensor_obj.numpy 不为 None）
          2. 外部 safetensors 文件（通过 safetensors_path + weight_map 匹配名称）

        注意：Builder 导出的 ONNX 中 weight 没有 numpy 数据且不在 initials 中，
        因此需要遍历所有 STATIC_TENSOR（type=1）来加载 safetensors 权重。
        """
        # 确定需要加载的 tensor 列表
        # 合并 initials（rope.cos/sin 等有 numpy 数据的常量）和
        # _safetensors_updated（从 safetensors 加载的 weight）
        load_names = set()
        if self._cg.initials:
            load_names.update(self._cg.initials)
        if hasattr(self, '_safetensors_updated') and self._safetensors_updated:
            load_names.update(self._safetensors_updated)
        if not load_names:
            dynamic_inputs = set(self.input_desc.keys())
            output_set = set(self._cg.output)
            load_names = set(n for n, tm in self._cg.tensormap.items()
                             if not tm.get_shape() and n not in dynamic_inputs
                             and n not in output_set)
        # 补充：对于有 shape 但 numpy 为 None 的 STATIC_TENSOR（weight/bias），
        # 也需要从 safetensors 加载（ONNX 中 weight 有 shape 但无数据）
        for n, tm in self._cg.tensormap.items():
            if tm.type == 1 and tm.numpy is None:  # STATIC_TENSOR without data
                load_names.add(n)
        load_names = list(load_names)

        # 预计算每个 constant 的消费者 op_type 和 attrs，用于 preprocess_weight
        const_consumer: Dict[str, tuple] = {}  # tname -> (op_type, attrs)
        for tname in load_names:
            if tname in self._cg.consumedby:
                consumers = self._cg.consumedby[tname]
                if consumers:
                    first_node = self._cg.nodemap[consumers[0]]
                    const_consumer[tname] = (first_node.op_type, first_node.attr)

        # 如果指定了 safetensors 文件，预加载所有 weight
        safetensors_data: Dict[str, torch.Tensor] = {}
        if self.safetensors_path:
            import safetensors
            with safetensors.safe_open(self.safetensors_path, framework='pt') as f:
                for key in f.keys():
                    safetensors_data[key] = f.get_tensor(key)

        for tname in load_names:
            if tname not in self._cg.tensormap:
                continue

            tensor_obj = self._cg.tensormap[tname]
            t = None

            # 1. 尝试从 safetensors 加载
            if self.safetensors_path:
                # 直接匹配名称
                if tname in safetensors_data:
                    t = safetensors_data[tname].to(device=self.device, dtype=self.dtype)
                # 通过 weight_map 匹配
                elif tname in self.weight_map and self.weight_map[tname] in safetensors_data:
                    st_name = self.weight_map[tname]
                    t = safetensors_data[st_name].to(device=self.device, dtype=self.dtype)

            # 2. fallback: 从 ONNX 内嵌数据加载
            if t is None and tensor_obj.numpy is not None:
                t = torch.from_numpy(tensor_obj.numpy.astype(
                    np.dtype(self._get_torch_dtype_name(self.dtype))
                )).to(device=self.device)

            if t is not None:
                # safetensors 的 2D weight 为 [N, K]，MatMul 的 weight 转置为 [K, N]
                op_type = const_consumer.get(tname, ('', {}))[0] if tname in const_consumer else ''
                t = self._transpose_safetensors_tensor(t, op_type)

                # 如果该 constant 有消费者 kernel 且实现了 preprocess_weight，则调用
                if tname in const_consumer:
                    op_type, attrs = const_consumer[tname]
                    kernel_cls = KernelRegistry.get(op_type)
                    if kernel_cls is not None and hasattr(kernel_cls, 'preprocess_weight'):
                        processed = kernel_cls.preprocess_weight(tname, t, attrs)
                        if processed is not None:
                            t = processed

                self._constant_tensors[tname] = t

    def _resolve_tensor(self, tname: str) -> Optional[torch.Tensor]:
        """
        解析 tensor：优先从 memory pool 取（reshape 预创建视图），
        如果不在 pool 中则从缓存中读取常量数据。
        """
        # 1. 尝试从 memory pool 获取（使用 shape 缓存，避免重复查询 shape_engine）
        if tname in self._tensor_views:
            shape = self._tensor_shape_cache.get(tname)
            if shape is None:
                shape = self.get_tensor_shape(tname)
            return self._reshape_view(tname, shape)

        # 2. 尝试从常量缓存获取
        if tname in self._constant_tensors:
            return self._constant_tensors[tname]

        # 3. 尝试从动态输入缓存获取（如 n_past scalar）
        if tname in self._dynamic_inputs:
            return self._dynamic_inputs[tname]

        # 4. 不在 pool 也不是常量/动态输入
        return None

    @staticmethod
    def _get_torch_dtype_name(dtype: torch.dtype) -> str:
        """将 torch.dtype 映射为 numpy dtype 名称"""
        mapping = {
            torch.float32: 'float32',
            torch.float64: 'float64',
            torch.float16: 'float16',
            torch.int32: 'int32',
            torch.int64: 'int64',
            torch.uint8: 'uint8',
            torch.int8: 'int8',
            torch.bool: 'bool',
        }
        return mapping.get(dtype, 'float32')

    def _get_np_dtype(self):
        """将 self.dtype (torch.dtype) 转为对应的 numpy dtype"""
        return np.dtype(self._get_torch_dtype_name(self.dtype))

    def _update_shape_from_input(self, input_tensor: torch.Tensor) -> bool:
        """根据实际输入 tensor 的形状，更新 shape_engine 中的变量。

        Returns:
            True 如果 shape 变量发生了实际变化，False 如果与上次相同。
        """
        if not hasattr(self, '_last_var_values'):
            self._last_var_values = {}
        changed = False
        for input_name, desc in self.input_desc.items():
            if len(desc) == len(input_tensor.shape):
                for i, dim_var in enumerate(desc):
                    if isinstance(dim_var, str) and dim_var in self.input_range:
                        actual_val = int(input_tensor.shape[i])
                        old_val = self._last_var_values.get(dim_var)
                        if old_val != actual_val:
                            self.shape_engine.update_variable(dim_var, actual_val)
                            self._last_var_values[dim_var] = actual_val
                            changed = True
        if changed:
            self.shape_engine.update_variables()
        return changed

    def _print_tensor_info(self, tname: str, prefix: str = ""):
        """打印单个 tensor 的详细信息"""
        block = self._tensor_blocks.get(tname)
        if block is None:
            # 可能是常量（initial）或不在 compress_mem 中
            is_init = tname in self._cg.initials if hasattr(self._cg, 'initials') else False
            print(f"{prefix}{tname}: NOT in memory pool{' (initial/constant)' if is_init else ''}")
            return

        offset, size = block
        try:
            shape = self.get_tensor_shape(tname)
        except Exception:
            shape = "?"

        # 计算实际元素数 vs 分配的元素数
        elem_size = torch.tensor([], dtype=self.dtype).element_size()
        allocated_elems = size // elem_size

        if isinstance(shape, list):
            shape_str = "x".join(str(s) for s in shape)
            total_elems = 1
            for s in shape:
                total_elems *= s
            waste = allocated_elems - total_elems
            waste_str = f" (浪费 {waste} 元素)" if waste > 0 else ""
        else:
            shape_str = str(shape)
            waste_str = ""

        print(f"{prefix}{tname}: offset={offset:>8,}, size={size:>8,} bytes, "
              f"shape=[{shape_str}]{waste_str}")

    def _check_overlap(self, node):
        """
        检查当前 node 的输入输出 tensor 在 memory pool 中是否有重叠。
        如果两个 tensor 的 [offset, offset+size) 区间相交，则打印警告。
        """
        all_tensors = list(node.input) + list(node.output)
        blocks = []
        for tname in all_tensors:
            block = self._tensor_blocks.get(tname)
            if block is not None:
                blocks.append((tname, block[0], block[0] + block[1]))

        # 两两检查重叠
        has_overlap = False
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                name_i, start_i, end_i = blocks[i]
                name_j, start_j, end_j = blocks[j]
                # 判断区间是否相交
                if start_i < end_j and start_j < end_i:
                    print(f"  ⚠ OVERLAP: '{name_i}' [{start_i}, {end_i}) 与 "
                          f"'{name_j}' [{start_j}, {end_j}) 重叠!")
                    has_overlap = True

        if not has_overlap:
            print(f"  [OK] 无重叠")

    def print_summary(self):
        """打印图结构摘要"""
        print(f"\n{'='*60}")
        print(f"GraphInfer Summary")
        print(f"{'='*60}")
        print(f"Model: {self.onnx_path}")
        print(f"Nodes: {len(self._node_names)}")
        print(f"Tensors in pool: {len(self._tensor_blocks)}")
        print(f"Pool size: {self.pool.compress_size:,} bytes ({self.pool.compress_size/1024/1024:.2f} MB)")
        print(f"Device: {self.device}")
        print(f"Dtype: {self.dtype}")

        # 按 op_type 统计
        op_counts = {}
        for name in self._node_names:
            op = self._cg.nodemap[name].op_type
            op_counts[op] = op_counts.get(op, 0) + 1
        print(f"\nOp breakdown:")
        for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1]):
            print(f"  {op}: {cnt}")

        # 最大 tensor
        max_tensor = max(self._tensor_blocks.items(), key=lambda x: x[1][1])
        print(f"\nLargest tensor: '{max_tensor[0]}' -> {max_tensor[1][1]:,} bytes")
        print(f"{'='*60}")


def create_graph_infer(
    onnx_path: str,
    input_desc: Dict[str, tuple],
    input_range: Dict[str, tuple],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> GraphInfer:
    """便捷函数：创建 GraphInfer 实例"""
    return GraphInfer(onnx_path, input_desc, input_range, dtype, device)
