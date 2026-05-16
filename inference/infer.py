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
    ):
        self.onnx_path = onnx_path
        self.input_desc = input_desc
        self.input_range = input_range
        self.dtype = dtype
        self.device = device

        # 1. 加载模型（启用 constant folding，折叠常量子图）
        self._model = onnx_tool.Model(onnx_path, {'constant_folding': True})
        self._graph = self._model.graph

        # 2. 节点重排
        self._graph.graph_reorder_nodes()

        # 3. Shape regression
        self.shape_engine = self._graph.shape_regress(input_desc, input_range)

        # 4. 用 input_range 的最大值更新 shape_engine 变量，
        #    并在原始 graph 上重新 shape_infer，确保 tensormap 中的 shape 是最大值
        for var_name, (lo, hi) in input_range.items():
            self.shape_engine.update_variable(var_name, hi)
        max_inputs = self.shape_engine.generate_input()
        self._graph.shape_infer(max_inputs)

        # 5. 提取 compute graph（此时 tensormap 中的 shape 已是最大值）
        self._cg = self._graph.get_compute_graph()

        # 6. 内存压缩 + MemoryPool（compress_memory 基于最大 shape 分配）
        # 使用两遍压缩：第一遍获取生命周期信息，第二遍按大小降序重排以减少碎片
        compress_mem, compress_size = self._compress_memory_optimized()
        self.pool = MemoryPool(
            compress_mem, compress_size, dtype=dtype, device=device
        )

        # 获取节点执行顺序（按 nodemap 的插入顺序，即拓扑序）
        self._node_names = list(self._cg.nodemap.keys())

        # 缓存每个 tensor 在 pool 中的 [offset, size]
        # compress_memory 返回的列表是 mem_block 的引用，且多个 tensor 可能共享
        # 同一个列表对象（offset=0 的 tensor 们），导致 size 相互覆盖。
        # 修复方案：用 tensormap 中的 shape 重新计算正确的 size，保留 offset。
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

        # 预创建所有 tensor 的 1D 视图（按最大 shape 分配），forward 时只 reshape
        self._tensor_views: Dict[str, torch.Tensor] = {}
        for tname, (offset, size) in self._tensor_blocks.items():
            if size > 0:
                self._tensor_views[tname] = self.pool._pool[offset:offset + size].view(dtype=dtype)

        # 缓存常量 tensor（weight/bias 等），避免每次 forward 从 numpy 转换
        self._constant_tensors: Dict[str, torch.Tensor] = {}
        self._preload_constants()

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
            # input: 优先从 pool 取 1D 视图，fallback 到常量
            in_tensors = []
            for tname in input_names:
                if tname in self._tensor_views:
                    in_tensors.append(self._tensor_views[tname])
                elif tname in self._constant_tensors:
                    in_tensors.append(self._constant_tensors[tname])
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
                t = flat[:needed].reshape(input_tensor.shape)
                t.copy_(input_tensor.to(device=self.device, dtype=self.dtype))

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
                        if t.numel() > needed:
                            input_tensors.append(t[:needed].reshape(shape))
                        else:
                            input_tensors.append(t.reshape(shape))
                    elif t is not None:
                        input_tensors.append(t)  # constant
                    else:
                        input_tensors.append(self._resolve_tensor(tname))
            else:
                input_tensors = self._last_node_tensors[idx][0]
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

    def _compress_memory_optimized(self, size_padding=64):
        """
        两遍内存压缩算法。

        第一遍：运行原始 compress_memory，获取每个 tensor 的生命周期信息。
        第二遍：按 node 顺序遍历，但在每个 node 内将新 tensor 按大小降序分配，
               大 tensor 优先选择最佳空闲块，减少碎片。

        Returns:
            (compress_mem, compress_size): 优化后的内存分配
        """
        import copy
        cg = self._cg

        # ---- Pass 1: 获取生命周期信息 ----
        tensor_in_mem = copy.deepcopy(cg.input)
        tensor_mem_per_node = []
        tensor_consumed = {}
        for node_name in cg.nodemap:
            node = cg.nodemap[node_name]
            for name in node.output:
                tensor_in_mem.append(name)
            new_list = copy.deepcopy(tensor_in_mem)
            tensor_mem_per_node.append(new_list)
            for name in node.input:
                if name in cg.consumedby:
                    if name in tensor_consumed:
                        tensor_consumed[name].append(node_name)
                    else:
                        tensor_consumed[name] = [node_name]
                    consumers = cg.consumedby[name]
                    if len(tensor_consumed[name]) == len(consumers):
                        if name in tensor_in_mem:
                            tensor_in_mem.remove(name)

        # 计算每个 tensor 的 padded size（只包含动态 tensor）
        elem_size = torch.tensor([], dtype=self.dtype).element_size()
        tensor_sizes = {}
        for tname in cg.dynamics:
            if tname in cg.tensormap:
                shape = cg.tensormap[tname].get_shape()
                num_elems = 1
                for s in shape:
                    num_elems *= s
                size_bytes = num_elems * elem_size
                size_bytes = int((size_bytes + size_padding - 1) // size_padding * size_padding)
                if size_bytes > 0:
                    tensor_sizes[tname] = size_bytes

        # ---- Pass 2: 按 node 顺序分配，每个 node 内新 tensor 按大小降序 ----
        compress_mem = {}
        mem_tags = []
        mem_block = []
        split_threshold = max(size_padding, 4096)

        for nodetensors in tensor_mem_per_node:
            # Phase 1: 释放不再存活的 tensor（与原始算法相同）
            for i, tag in enumerate(mem_tags):
                if tag == "":
                    continue
                if tag not in nodetensors:
                    premerge = False
                    if i >= 1 and mem_tags[i - 1] == "":
                        premerge = True
                    nextmerge = False
                    if i < len(mem_tags) - 1 and mem_tags[i + 1] == "":
                        nextmerge = True
                    if premerge and nextmerge:
                        newblock = [mem_block[i - 1][0], mem_block[i + 1][0] + mem_block[i + 1][1]
                                    - mem_block[i - 1][0]]
                        mem_tags.pop(i)
                        mem_tags.pop(i)
                        mem_block.pop(i)
                        mem_block.pop(i)
                        mem_tags[i - 1] = ""
                        mem_block[i - 1] = newblock
                    elif premerge:
                        newblock = [mem_block[i - 1][0], mem_block[i][0] + mem_block[i][1]
                                    - mem_block[i - 1][0]]
                        mem_tags.pop(i)
                        mem_block.pop(i)
                        mem_tags[i - 1] = ""
                        mem_block[i - 1] = newblock
                    elif nextmerge:
                        newblock = [mem_block[i][0], mem_block[i + 1][0] + mem_block[i + 1][1]
                                    - mem_block[i][0]]
                        mem_tags.pop(i)
                        mem_block.pop(i)
                        mem_tags[i] = ""
                        mem_block[i] = newblock
                    else:
                        mem_tags[i] = ""

            # Phase 2: 分配新 tensor，按大小降序排列
            new_tensors = [t for t in nodetensors if t not in mem_tags and t in tensor_sizes]
            new_tensors.sort(key=lambda t: -tensor_sizes[t])

            for tname in new_tensors:
                size_ = tensor_sizes[tname]

                # Best-Fit
                best_idx = -1
                best_remain = None
                for i, tag in enumerate(mem_tags):
                    if tag == "":
                        free_size = mem_block[i][1]
                        if free_size >= size_:
                            remain_size = free_size - size_
                            if best_idx == -1 or remain_size < best_remain[0]:
                                best_idx = i
                                best_remain = [remain_size, [mem_block[i][0] + size_, remain_size]]

                if best_idx >= 0:
                    i = best_idx
                    remain_size = best_remain[0]
                    if remain_size >= split_threshold:
                        remain_block = best_remain[1]
                        mem_tags[i] = tname
                        mem_block[i][1] = size_
                        mem_tags.insert(i + 1, "")
                        mem_block.insert(i + 1, remain_block)
                    else:
                        mem_tags[i] = tname
                else:
                    lastidx = len(mem_tags) - 1
                    if lastidx >= 0 and mem_tags[lastidx] == "":
                        mem_block[lastidx][1] = size_
                        mem_tags[lastidx] = tname
                    else:
                        mem_tags.append(tname)
                        if lastidx >= 0:
                            mem_block.append([mem_block[lastidx][0] + mem_block[lastidx][1], size_])
                        else:
                            mem_block.append([0, size_])

                idx = mem_tags.index(tname)
                compress_mem[tname] = [mem_block[idx][0], mem_block[idx][1]]

        # ---- 压缩尾部 ----
        last_used_end = max(b[0] + b[1] for b in mem_block) if mem_block else 0
        compress_size = last_used_end

        # ---- 验证重叠 ----
        for nodetensors in tensor_mem_per_node:
            for tname in nodetensors:
                if tname not in compress_mem:
                    continue
                block0 = compress_mem[tname]
                if block0[1] == 0:
                    continue
                for tname1 in nodetensors:
                    if tname1 == tname or tname1 not in compress_mem:
                        continue
                    block1 = compress_mem[tname1]
                    if block1[1] == 0:
                        continue
                    a_start, a_end = block0[0], block0[0] + block0[1]
                    b_start, b_end = block1[0], block1[0] + block1[1]
                    if a_start < b_end and b_start < a_end:
                        print(f"  WARNING: overlap detected between {tname} and {tname1}")

        raw_memsize = 0
        for tname in cg.dynamics:
            if tname in cg.input or tname in cg.output:
                continue
            if tname in cg.tensormap:
                raw_memsize += cg.tensormap[tname].get_memsize()

        print(f"Optimized compress: raw={raw_memsize:,} bytes, "
              f"pool={compress_size:,} bytes, "
              f"ratio={compress_size/raw_memsize*100:.2f}%")

        return compress_mem, compress_size

    def _preload_constants(self):
        """预加载所有常量 tensor 到目标设备，并对 weight 执行 kernel 注册的前处理"""
        # 预计算每个 constant 的消费者 op_type 和 attrs，用于 preprocess_weight
        const_consumer: Dict[str, tuple] = {}  # tname -> (op_type, attrs)
        for tname in self._cg.initials:
            if tname in self._cg.consumedby:
                consumers = self._cg.consumedby[tname]
                if consumers:
                    first_node = self._cg.nodemap[consumers[0]]
                    const_consumer[tname] = (first_node.op_type, first_node.attr)

        for tname in self._cg.initials:
            if tname in self._cg.tensormap:
                tensor_obj = self._cg.tensormap[tname]
                if tensor_obj.numpy is not None:
                    t = torch.from_numpy(tensor_obj.numpy.astype(
                        np.dtype(self._get_torch_dtype_name(self.dtype))
                    )).to(device=self.device)

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

        # 3. 不在 pool 也不是常量
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
