import importlib
import onnx_tool.llm as _llm
importlib.reload(_llm)
from onnx_tool.llm import *
import tabulate
import copy


def compress_memory_with_kv_cache(cg, size_padding=64):
    """LLM 专用内存压缩：将 kv_cache 与普通 activation 隔离为两个独立内存池。

    对 compute graph 执行两遍内存压缩：
      - activation pool: 所有动态 tensor（除 kv_cache 外），使用 graph.py 的 compress_memory
      - kv_cache pool: 仅 kv_cache tensor，独立分配

    Args:
        cg: compute graph（Graph 对象，需已调用 update_tensor_relations()）
        size_padding: 内存对齐粒度

    Returns:
        (compress_mem, compress_size, kv_compress_mem, kv_compress_size)
    """
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

    # 计算每个 tensor 的 padded size，分离 activation 和 kv_cache
    tensor_sizes = {}       # activation tensors
    kv_tensor_sizes = {}    # kv_cache only
    for tname in cg.dynamics:
        if tname not in cg.producedby and tname not in cg.input:
            continue
        if tname in cg.tensormap:
            size_ = cg.tensormap[tname].get_memsize()
            size_ = int((size_ + size_padding - 1) // size_padding * size_padding)
            if size_ > 0:
                if tname == 'kv_cache':
                    kv_tensor_sizes[tname] = size_
                else:
                    tensor_sizes[tname] = size_

    # ---- Pass 2: Best-Fit 分配 ----
    def _allocate_pool(tensor_sizes):
        """对一组 tensor 执行 Best-Fit 内存分配，返回 (compress_mem, compress_size)"""
        compress_mem = {}
        mem_tags = []
        mem_block = []
        split_threshold = max(size_padding, 4096)

        for nodetensors in tensor_mem_per_node:
            # Phase 1: 释放不再存活的 tensor
            for i, tag in enumerate(mem_tags):
                if tag == "":
                    continue
                if tag not in nodetensors:
                    premerge = i >= 1 and mem_tags[i - 1] == ""
                    nextmerge = i < len(mem_tags) - 1 and mem_tags[i + 1] == ""
                    if premerge and nextmerge:
                        newblock = [mem_block[i - 1][0], mem_block[i + 1][0] + mem_block[i + 1][1]
                                    - mem_block[i - 1][0]]
                        mem_tags.pop(i); mem_tags.pop(i)
                        mem_block.pop(i); mem_block.pop(i)
                        mem_tags[i - 1] = ""; mem_block[i - 1] = newblock
                    elif premerge:
                        newblock = [mem_block[i - 1][0], mem_block[i][0] + mem_block[i][1]
                                    - mem_block[i - 1][0]]
                        mem_tags.pop(i); mem_block.pop(i)
                        mem_tags[i - 1] = ""; mem_block[i - 1] = newblock
                    elif nextmerge:
                        newblock = [mem_block[i][0], mem_block[i + 1][0] + mem_block[i + 1][1]
                                    - mem_block[i][0]]
                        mem_tags.pop(i); mem_block.pop(i)
                        mem_tags[i] = ""; mem_block[i] = newblock
                    else:
                        mem_tags[i] = ""

            # Phase 2: 分配新 tensor，按大小降序
            new_tensors = [t for t in nodetensors if t not in mem_tags and t in tensor_sizes]
            new_tensors.sort(key=lambda t: -tensor_sizes[t])

            for tname in new_tensors:
                size_ = tensor_sizes[tname]
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
                        mem_tags[i] = tname; mem_block[i][1] = size_
                        mem_tags.insert(i + 1, ""); mem_block.insert(i + 1, remain_block)
                    else:
                        mem_tags[i] = tname
                else:
                    lastidx = len(mem_tags) - 1
                    if lastidx >= 0 and mem_tags[lastidx] == "":
                        mem_block[lastidx][1] = size_; mem_tags[lastidx] = tname
                    else:
                        mem_tags.append(tname)
                        if lastidx >= 0:
                            mem_block.append([mem_block[lastidx][0] + mem_block[lastidx][1], size_])
                        else:
                            mem_block.append([0, size_])

                idx = mem_tags.index(tname)
                compress_mem[tname] = [mem_block[idx][0], mem_block[idx][1]]

        last_used_end = max(b[0] + b[1] for b in mem_block) if mem_block else 0
        return compress_mem, last_used_end

    compress_mem, compress_size = _allocate_pool(tensor_sizes)
    kv_compress_mem, kv_compress_size = _allocate_pool(kv_tensor_sizes)

    # ---- 验证重叠（仅 activation pool）----
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

    print(f"LLM compress: raw={raw_memsize:,} bytes, "
          f"activation_pool={compress_size:,} bytes, "
          f"kv_cache_pool={kv_compress_size:,} bytes, "
          f"total_pool={compress_size + kv_compress_size:,} bytes, "
          f"ratio={(compress_size + kv_compress_size)/raw_memsize*100:.2f}%")

    return compress_mem, compress_size, kv_compress_mem, kv_compress_size


# ===========================================================================
# Qwen3-0.6B-Base 模型配置
# ===========================================================================
# 从 https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/main/config.json
# Qwen3 架构与 Qwen2 完全兼容，仅 model_type 名称不同
Qwen3_0_6B = {
    "name": "Qwen3-0.6B-Base",
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 32768,
    "max_window_layers": 28,
    "model_type": "qwen3",
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}


def export_qwen3_0_6b():
    """导出 Qwen3-0.6B-Base 到 ONNX 格式"""
    bs = 1
    seq_len = 128
    ids_shape = [bs, seq_len]

    print("Building Qwen3-0.6B-Base graph...")
    builder = Builder(**Qwen3_0_6B)
    builder.build_graph(ids_shape)

    onnx_path = 'qwen3_0_6b.onnx'
    print(f"Saving to {onnx_path}...")
    builder.save_graph(onnx_path)

    print("Profiling...")
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    macs = int(builder.graph.macs[0] / 1e9)
    params = builder.graph.params / 1e9
    print(f"\nQwen3-0.6B-Base: MACs={macs}G, Parameters={params:.3f}G")
    return onnx_path


def export_qwen3_0_6b_with_kv_cache():
    """导出 Qwen3-0.6B-Base 带 KV cache 的 ONNX 模型"""
    bs = 1
    seq_len = 128
    ids_shape = [bs, seq_len]
    past_sequence = 0
    context_length = 8192

    print("Building Qwen3-0.6B-Base with KV cache...")
    builder = Builder(**Qwen3_0_6B)
    builder.build_graph(ids_shape)
    builder.add_kv_cache(context_length, past_sequence)

    onnx_path = 'qwen3_0_6b_kvcache.onnx'
    print(f"Saving to {onnx_path}...")
    builder.save_graph(onnx_path)

    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()
    return onnx_path


# ===========================================================================
# Qwen3.5-4B 模型配置
# ===========================================================================
# Qwen3.5-4B 模型配置
# ===========================================================================
# 从 https://huggingface.co/Qwen/Qwen3.5-4B-Instruct/resolve/main/config.json
# Qwen3.5 使用混合架构：3×Gated DeltaNet + 1×Gated Attention，共 8 个 block（32 层）
# layer_types 模式: linear_attention × 3, full_attention, 重复 8 次
Qwen3_5_4B = {
    "name": "Qwen3.5-4B-Instruct",
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 248044,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 9216,
    "max_position_embeddings": 262144,
    "model_type": "qwen3_5",
    "num_attention_heads": 16,
    "num_hidden_layers": 32,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000000.0,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 248320,
    # Qwen3.5 DeltaNet 专用参数
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    # 混合层类型：3×linear_attention + 1×full_attention，重复 8 次 = 32 层
    "layer_types": [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ],
    # =========================================================================
    # Vision Encoder 配置（Qwen3.5-4B 多模态）
    # 基于 Qwen2.5-VL 架构：ViT + MLP Projector
    # =========================================================================
    "vision_hidden_size": 1024,       # ViT hidden dim
    "vision_num_layers": 24,          # ViT transformer 层数
    "vision_num_heads": 16,           # ViT attention heads
    "vision_intermediate_size": 4096, # ViT MLP intermediate dim (4× hidden)
    "vision_patch_size": 14,          # Patch size
    "vision_image_size": 448,         # 输入图像尺寸
    "vision_head_dim": 64,            # 每个 head 的维度 (1024/16)
    "projector_hidden_dim": 2560,     # Projector 中间维度 (= LLM hidden_size)
    "projector_type": "mlp",          # MLP projector (2-layer)
}


def export_qwen3_5_4b():
    """导出 Qwen3.5-4B-Instruct 到 ONNX 格式，输入序列长度 2048 (2K)"""
    bs = 1
    seq_len = 2048  # 2K input
    ids_shape = [bs, seq_len]

    print("Building Qwen3.5-4B-Instruct graph (2K input)...")
    builder = Builder(**Qwen3_5_4B)
    builder.build_graph(ids_shape)

    onnx_path = 'qwen3_5_4b.onnx'
    print(f"Saving to {onnx_path}...")
    builder.save_graph(onnx_path)

    print("Profiling...")
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    macs = int(builder.graph.macs[0] / 1e9)
    params = builder.graph.params / 1e9
    kv_params = builder.kv_params / 1e9
    print(f"\nQwen3.5-4B-Instruct (2K input):")
    print(f"  MACs={macs}G, Parameters={params:.3f}G, KV Cache={kv_params:.3f}G")
    return onnx_path


def export_qwen3_5_4b_with_kv_cache():
    """导出 Qwen3.5-4B-Instruct 带 KV cache 的 ONNX 模型，输入 2K"""
    bs = 1
    seq_len = 2048  # 2K input
    ids_shape = [bs, seq_len]
    past_sequence = 0
    context_length = 8192

    print("Building Qwen3.5-4B-Instruct with KV cache (2K input)...")
    builder = Builder(**Qwen3_5_4B)
    builder.build_graph(ids_shape)
    builder.add_kv_cache(context_length, past_sequence)

    onnx_path = 'qwen3_5_4b_kvcache.onnx'
    print(f"Saving to {onnx_path}...")
    builder.save_graph(onnx_path)

    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    macs = int(builder.graph.macs[0] / 1e9)
    params = builder.graph.params / 1e9
    kv_params = builder.kv_params / 1e9
    print(f"\nQwen3.5-4B-Instruct with KV cache (2K input):")
    print(f"  MACs={macs}G, Parameters={params:.3f}G, KV Cache={kv_params:.3f}G")
    return onnx_path


def profile_qwen3_5_4b():
    """对 Qwen3.5-4B 进行详细的性能分析（prefill + decode）"""
    from onnx_tool.device import Devices
    RuntimeCfg = {
        'Compute': {
            'MM': 'FP16',
            'MHA': 'FP16',
            'Others': 'FP16',
        },
        'Bits': {
            'MM': 16,
            'MHA': 16,
            'Others': 16,
        }
    }
    bs = 1
    prefill_length = 2048  # 2K prefill
    context_length = 8192
    ids_shape = [bs, prefill_length]

    print(f"\n{'='*60}")
    print("Qwen3.5-4B-Instruct Profile Analysis (2K input)")
    print(f"{'='*60}")

    builder = Builder(**Qwen3_5_4B)
    builder.build_graph(ids_shape)
    builder.add_kv_cache(context_length, 0)
    builder.graph.valid_shape = True

    model_name = builder.get_filename()
    device_names = ['Gaudi2H', 'H20']

    # Prefill profile
    print(f"\n--- Prefill (seq_len={prefill_length}) ---")
    for key in device_names:
        builder.profile(RuntimeCfg, Devices[key])
        print(f"\n  Device: {key}")
        print(f"  Latency: {builder.llm_profile[2]:.2f} ms")
        print(f"  Memory: {builder.context_mem[3]/1e9:.3f} GB")

    # Decode profile
    print(f"\n--- Decode (seq_len=1, past_kv={prefill_length}) ---")
    builder.set_past_kv_length(prefill_length)
    builder.graph.shape_infer(inputs={'ids': create_ndarray_int64([bs, 1])})
    builder.graph.profile()
    for key in device_names:
        builder.profile(RuntimeCfg, Devices[key])
        print(f"\n  Device: {key}")
        print(f"  Latency: {builder.llm_profile[2]:.2f} ms")

    # Summary
    macs = int(builder.graph.macs[0] / 1e9)
    params = builder.graph.params / 1e9
    kv_params = builder.kv_params / 1e9
    print(f"\n--- Summary ---")
    print(f"  MACs (decode): {macs}G")
    print(f"  Parameters: {params:.3f}G")
    print(f"  KV Cache: {kv_params:.3f}G")


# ===========================================================================
# Qwen3.5-4B Vision Model Profiling（多模态）
# ===========================================================================

def export_qwen3_5_4b_vision():
    """导出 Qwen3.5-4B 的 vision encoder 到 ONNX 格式"""
    image_shape = [3, 448, 448]

    print("Building Qwen3.5-4B Vision Encoder...")
    builder = Builder(**Qwen3_5_4B)
    builder.build_vision_graph(image_shape)

    onnx_path = 'qwen3_5_4b_vision.onnx'
    print(f"Saving to {onnx_path}...")
    builder.save_graph(onnx_path)

    print("Profiling...")
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    macs = int(builder.graph.macs[0] / 1e9)
    params = builder.graph.params / 1e9
    print(f"\nQwen3.5-4B Vision Encoder (448×448):")
    print(f"  MACs={macs}G, Parameters={params:.3f}G")
    return onnx_path


def profile_qwen3_5_4b_vision():
    """对 Qwen3.5-4B Vision Encoder 进行详细的性能分析"""
    from onnx_tool.device import Devices
    RuntimeCfg = {
        'Compute': {
            'MM': 'FP16',
            'MHA': 'FP16',
            'Others': 'FP16',
        },
        'Bits': {
            'MM': 16,
            'MHA': 16,
            'Others': 16,
        }
    }

    print(f"\n{'='*60}")
    print("Qwen3.5-4B Vision Encoder Profile Analysis")
    print(f"{'='*60}")

    image_shape = [3, 448, 448]
    builder = Builder(**Qwen3_5_4B)
    builder.build_vision_graph(image_shape)
    builder.graph.valid_shape = True
    builder.graph.profile()

    macs = int(builder.graph.macs[0] / 1e9)
    params = builder.graph.params / 1e9
    num_patches = (448 // 14) ** 2  # 1024 patches
    print(f"  Image: 448×448 → {num_patches} patches")
    print(f"  MACs: {macs}G")
    print(f"  Parameters: {params:.3f}G")

    for key in ['Gaudi2H', 'H20']:
        builder.profile(RuntimeCfg, Devices[key])
        print(f"  Device {key}: Latency={builder.llm_profile[2]:.2f} ms, "
              f"Memory={builder.context_mem[3]/1e9:.3f} GB")

    print(f"\n--- Architecture ---")
    print(f"  ViT: {Qwen3_5_4B['vision_num_layers']}L, "
          f"hidden={Qwen3_5_4B['vision_hidden_size']}, "
          f"heads={Qwen3_5_4B['vision_num_heads']}, "
          f"patch={Qwen3_5_4B['vision_patch_size']}")
    print(f"  Projector: {Qwen3_5_4B['projector_type']}, "
          f"hidden={Qwen3_5_4B['projector_hidden_dim']} → LLM dim={Qwen3_5_4B['hidden_size']}")


def profile_qwen3_5_4b_vision_multi_resolution():
    """对 Qwen3.5-4B Vision Encoder 进行多分辨率性能分析

    Qwen2.5-VL / Qwen3.5-VL 支持动态分辨率，图像会被切分为多个 tiles。
    测试不同分辨率下的 vision encoder 性能。
    """
    from onnx_tool.device import Devices
    RuntimeCfg = {
        'Compute': {
            'MM': 'FP16',
            'MHA': 'FP16',
            'Others': 'FP16',
        },
        'Bits': {
            'MM': 16,
            'MHA': 16,
            'Others': 16,
        }
    }

    print(f"\n{'='*60}")
    print("Qwen3.5-4B Vision Encoder Multi-Resolution Analysis")
    print(f"{'='*60}")

    resolutions = [
        ('224×224 (low)', [3, 224, 224]),
        ('336×336 (med)', [3, 336, 336]),
        ('448×448 (std)', [3, 448, 448]),
        ('672×672 (high)', [3, 672, 672]),
        ('896×896 (2K)', [3, 896, 896]),
    ]

    header = ['Resolution', 'Patches', 'MACs(G)', 'Params(G)',
              'Gaudi2H(ms)', 'H20(ms)', 'Memory(GB)']
    rows = []

    for label, image_shape in resolutions:
        builder = Builder(**Qwen3_5_4B)
        builder.build_vision_graph(image_shape)
        builder.graph.valid_shape = True
        builder.graph.profile()

        macs = int(builder.graph.macs[0] / 1e9)
        params = builder.graph.params / 1e9
        h, w = image_shape[1], image_shape[2]
        num_patches = (h // 14) * (w // 14)

        row = [label, num_patches, macs, f'{params:.3f}']

        for key in ['Gaudi2H', 'H20']:
            builder.profile(RuntimeCfg, Devices[key])
            row.append(f'{builder.llm_profile[2]:.2f}')

        row.append(f'{builder.context_mem[3]/1e9:.3f}')
        rows.append(row)

    print(tabulate.tabulate(rows, headers=header))


# ===========================================================================
# Qwen3.5-35B-A3B (Qwen3.5-MoE) 模型配置
# ===========================================================================
# 从 https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Instruct
# Sparse MoE: 256 experts, top-8 per token, + 1 shared expert
# 混合架构：3×Gated DeltaNet + 1×Gated Attention，共 10 个 block（40 层）
Qwen3_5_35B_A3B = {
    "architectures": ["Qwen3_5MoeForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 512,  # moe_intermediate_size (per-expert)
    "max_position_embeddings": 32768,
    "model_type": "qwen3_5_moe",
    "num_attention_heads": 16,
    "num_hidden_layers": 40,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000000.0,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 248320,
    # MoE 专用参数
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    # DeltaNet 专用参数
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    # 混合层类型：3×linear_attention + 1×full_attention，重复 10 次 = 40 层
    "layer_types": [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ],
    # =========================================================================
    # Vision Encoder 配置（Qwen3.5-35B-A3B 多模态）
    # 基于 Qwen2.5-VL 架构：ViT + MLP Projector
    # =========================================================================
    "vision_hidden_size": 1024,       # ViT hidden dim
    "vision_num_layers": 24,          # ViT transformer 层数
    "vision_num_heads": 16,           # ViT attention heads
    "vision_intermediate_size": 4096, # ViT MLP intermediate dim (4× hidden)
    "vision_patch_size": 14,          # Patch size
    "vision_image_size": 448,         # 输入图像尺寸
    "vision_head_dim": 64,            # 每个 head 的维度 (1024/16)
    "projector_hidden_dim": 2048,     # Projector 中间维度 (= LLM hidden_size)
    "projector_type": "mlp",          # MLP projector (2-layer)
}


def export_qwen3_5_35b_a3b():
    """导出 Qwen3.5-35B-A3B 到 ONNX 格式，输入序列长度 2048 (2K)"""
    bs = 1
    seq_len = 2048  # 2K input
    ids_shape = [bs, seq_len]

    print("Building Qwen3.5-35B-A3B-Instruct graph (2K input)...")
    builder = Builder(**Qwen3_5_35B_A3B)
    builder.build_graph(ids_shape)

    onnx_path = 'qwen3_5_35b_a3b.onnx'
    print(f"Saving to {onnx_path}...")
    builder.save_graph(onnx_path)

    print("Profiling...")
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    macs = int(builder.graph.macs[0] / 1e9)
    params = builder.graph.params / 1e9
    kv_params = builder.kv_params / 1e9
    print(f"\nQwen3.5-35B-A3B-Instruct (2K input):")
    print(f"  MACs={macs}G, Parameters={params:.3f}G, KV Cache={kv_params:.3f}G")
    return onnx_path


def profile_qwen3_5_35b_a3b():
    """对 Qwen3.5-35B-A3B 进行详细的性能分析（prefill + decode）"""
    from onnx_tool.device import Devices
    RuntimeCfg = {
        'Compute': {
            'MM': 'FP16',
            'MHA': 'FP16',
            'Others': 'FP16',
        },
        'Bits': {
            'MM': 16,
            'MHA': 16,
            'Others': 16,
        }
    }
    bs = 1
    prefill_length = 2048  # 2K prefill
    context_length = 8192
    ids_shape = [bs, prefill_length]

    print(f"\n{'='*60}")
    print("Qwen3.5-35B-A3B-Instruct Profile Analysis (2K input)")
    print(f"{'='*60}")

    builder = Builder(**Qwen3_5_35B_A3B)
    builder.build_graph(ids_shape)
    builder.add_kv_cache(context_length, 0)
    builder.graph.valid_shape = True

    device_names = ['Gaudi2H', 'H20']

    # Prefill profile
    print(f"\n--- Prefill (seq_len={prefill_length}) ---")
    for key in device_names:
        builder.profile(RuntimeCfg, Devices[key])
        print(f"\n  Device: {key}")
        print(f"  Latency: {builder.llm_profile[2]:.2f} ms")
        print(f"  Memory: {builder.context_mem[3]/1e9:.3f} GB")

    # Decode profile
    print(f"\n--- Decode (seq_len=1, past_kv={prefill_length}) ---")
    builder.set_past_kv_length(prefill_length)
    builder.graph.shape_infer(inputs={'ids': create_ndarray_int64([bs, 1])})
    builder.graph.profile()
    for key in device_names:
        builder.profile(RuntimeCfg, Devices[key])
        print(f"\n  Device: {key}")
        print(f"  Latency: {builder.llm_profile[2]:.2f} ms")

    # Summary
    macs = int(builder.graph.macs[0] / 1e9)
    params = builder.graph.params / 1e9
    kv_params = builder.kv_params / 1e9
    print(f"\n--- Summary ---")
    print(f"  MACs (decode): {macs}G")
    print(f"  Parameters: {params:.3f}G")
    print(f"  KV Cache: {kv_params:.3f}G")


# ===========================================================================
# DeepSeek-V4-Flash 模型配置
# ===========================================================================
# 从 https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash 的 config.json
# 架构: MLA (低秩注意力) + MoE (256 experts, top-6) + Hyper-Connections
# 总参数 284B，每 token 激活 13B，存储大小 158B (FP4/FP8 量化)
DeepSeek_V4_Flash = {
    "name": "DeepSeek-V4-Flash",
    "architectures": ["DeepSeekV4ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "head_dim": 512,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 2048,  # moe_intermediate_size (真实值)
    "max_position_embeddings": 4096,
    "model_type": "deepseek_v4",
    "num_attention_heads": 64,
    "num_hidden_layers": 43,  # 真实值 (非 model.py 默认的 7)
    "num_key_value_heads": 1,  # GQA: 1 KV head
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": 129280,
    # MoE 专用参数
    "num_experts": 256,
    "num_experts_per_tok": 6,  # 真实值 (非 model.py 默认的 8)
    "moe_intermediate_size": 2048,  # 真实值
    "shared_expert_intermediate_size": 2048,
    # MLA 参数
    "q_lora_rank": 1024,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "rope_head_dim": 64,
    "window_size": 128,
    # KV 压缩参数
    "compress_ratio": 4,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 512,
    # hc_mult=4 — Hyper-Connections 未建模
}


# ===========================================================================
# DeepSeek-V4-Pro 模型配置
# ===========================================================================
# 从 https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro 的 config.json
# 总参数 1.6T，每 token 激活 49B
DeepSeek_V4_Pro = {
    "name": "DeepSeek-V4-Pro",
    "architectures": ["DeepSeekV4ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "head_dim": 512,
    "hidden_act": "silu",
    "hidden_size": 7168,
    "initializer_range": 0.02,
    "intermediate_size": 3072,  # moe_intermediate_size
    "max_position_embeddings": 1048576,  # 1M
    "model_type": "deepseek_v4",
    "num_attention_heads": 128,
    "num_hidden_layers": 61,
    "num_key_value_heads": 1,  # GQA
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": 129280,
    # MoE 专用参数
    "num_experts": 384,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 3072,
    "shared_expert_intermediate_size": 3072,
    # MLA 参数
    "q_lora_rank": 1536,
    "o_lora_rank": 1024,
    "o_groups": 16,
    "rope_head_dim": 64,
    "window_size": 128,
    # KV 压缩参数
    "compress_ratio": 4,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 1024,
    # hc_mult=4 — Hyper-Connections 未建模
}


def export_deepseek_v4_pro():
    """导出 DeepSeek-V4-Pro 到 ONNX 格式，输入序列长度 2048 (2K)"""
    bs = 1
    seq_len = 2048
    ids_shape = [bs, seq_len]

    print("Building DeepSeek-V4-Pro graph (2K input)...")
    builder = Builder(**DeepSeek_V4_Pro)
    builder.build_graph(ids_shape)

    onnx_path = 'deepseek_v4_pro.onnx'
    print(f"Saving to {onnx_path}...")
    builder.save_graph(onnx_path)

    print("Profiling...")
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    macs = int(builder.graph.macs[0] / 1e9)
    params = builder.graph.params / 1e9
    kv_params = builder.kv_params / 1e9
    print(f"\nDeepSeek-V4-Pro (2K input):")
    print(f"  MACs={macs}G, Parameters={params:.3f}G, KV Cache={kv_params:.3f}G")
    return onnx_path


def profile_deepseek_v4_pro():
    """对 DeepSeek-V4-Pro 进行详细的性能分析（prefill + decode）"""
    from onnx_tool.device import Devices
    RuntimeCfg = {
        'Compute': {
            'MM': 'FP16',
            'MHA': 'FP16',
            'Others': 'FP16',
        },
        'Bits': {
            'MM': 16,
            'MHA': 16,
            'Others': 16,
        }
    }
    bs = 1
    prefill_length = 2048
    context_length = 4096
    ids_shape = [bs, prefill_length]

    print(f"\n{'='*60}")
    print("DeepSeek-V4-Pro Profile Analysis (2K input)")
    print(f"{'='*60}")

    builder = Builder(**DeepSeek_V4_Pro)
    builder.build_graph(ids_shape)
    builder.add_kv_cache(context_length, 0)
    builder.graph.valid_shape = True

    device_names = ['Gaudi2H', 'H20']

    print(f"\n--- Prefill (seq_len={prefill_length}) ---")
    for key in device_names:
        builder.profile(RuntimeCfg, Devices[key])
        print(f"\n  Device: {key}")
        print(f"  Latency: {builder.llm_profile[2]:.2f} ms")
        print(f"  Memory: {builder.context_mem[3]/1e9:.3f} GB")

    print(f"\n--- Decode (seq_len=1, past_kv={prefill_length}) ---")
    builder.set_past_kv_length(prefill_length)
    builder.graph.shape_infer(inputs={'ids': create_ndarray_int64([bs, 1])})
    builder.graph.profile()
    for key in device_names:
        builder.profile(RuntimeCfg, Devices[key])
        print(f"\n  Device: {key}")
        print(f"  Latency: {builder.llm_profile[2]:.2f} ms")

    macs = int(builder.graph.macs[0] / 1e9)
    params = builder.graph.params / 1e9
    kv_params = builder.kv_params / 1e9
    print(f"\n--- Summary ---")
    print(f"  MACs (decode): {macs}G")
    print(f"  Parameters: {params:.3f}G")
    print(f"  KV Cache: {kv_params:.3f}G")


def export_deepseek_v4_flash():
    """导出 DeepSeek-V4-Flash 到 ONNX 格式，输入序列长度 2048 (2K)"""
    bs = 1
    seq_len = 2048
    ids_shape = [bs, seq_len]

    print("Building DeepSeek-V4-Flash graph (2K input)...")
    builder = Builder(**DeepSeek_V4_Flash)
    builder.build_graph(ids_shape)

    onnx_path = 'deepseek_v4_flash.onnx'
    print(f"Saving to {onnx_path}...")
    builder.save_graph(onnx_path)

    print("Profiling...")
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    macs = int(builder.graph.macs[0] / 1e9)
    params = builder.graph.params / 1e9
    kv_params = builder.kv_params / 1e9
    print(f"\nDeepSeek-V4-Flash (2K input):")
    print(f"  MACs={macs}G, Parameters={params:.3f}G, KV Cache={kv_params:.3f}G")
    return onnx_path


def profile_deepseek_v4_flash():
    """对 DeepSeek-V4-Flash 进行详细的性能分析（prefill + decode）"""
    from onnx_tool.device import Devices
    RuntimeCfg = {
        'Compute': {
            'MM': 'FP16',
            'MHA': 'FP16',
            'Others': 'FP16',
        },
        'Bits': {
            'MM': 16,
            'MHA': 16,
            'Others': 16,
        }
    }
    bs = 1
    prefill_length = 2048
    context_length = 4096
    ids_shape = [bs, prefill_length]

    print(f"\n{'='*60}")
    print("DeepSeek-V4-Flash Profile Analysis (2K input)")
    print(f"{'='*60}")

    builder = Builder(**DeepSeek_V4_Flash)
    builder.build_graph(ids_shape)
    builder.add_kv_cache(context_length, 0)
    builder.graph.valid_shape = True

    device_names = ['Gaudi2H', 'H20']

    print(f"\n--- Prefill (seq_len={prefill_length}) ---")
    for key in device_names:
        builder.profile(RuntimeCfg, Devices[key])
        print(f"\n  Device: {key}")
        print(f"  Latency: {builder.llm_profile[2]:.2f} ms")
        print(f"  Memory: {builder.context_mem[3]/1e9:.3f} GB")

    print(f"\n--- Decode (seq_len=1, past_kv={prefill_length}) ---")
    builder.set_past_kv_length(prefill_length)
    builder.graph.shape_infer(inputs={'ids': create_ndarray_int64([bs, 1])})
    builder.graph.profile()
    for key in device_names:
        builder.profile(RuntimeCfg, Devices[key])
        print(f"\n  Device: {key}")
        print(f"  Latency: {builder.llm_profile[2]:.2f} ms")

    macs = int(builder.graph.macs[0] / 1e9)
    params = builder.graph.params / 1e9
    kv_params = builder.kv_params / 1e9
    print(f"\n--- Summary ---")
    print(f"  MACs (decode): {macs}G")
    print(f"  Parameters: {params:.3f}G")
    print(f"  KV Cache: {kv_params:.3f}G")


# Export the model with pytorch tensor names
# Not necessary to convert safetensors to ONNX format
def export_with_pytorch_weight_name():
    bs = 1
    seq_len = 1024
    ids_shape = [bs, seq_len]
    builder = Builder(**phi3_mini)
    builder.build_graph(ids_shape, WeightMap)
    for name in builder.graph.initials:
        print(name)
    builder.save_graph('phi3.onnx')
    # each name response the same tensor in this file:
    # https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/model.safetensors.index.json


# Add one new model from hugging face
def add_hugging_face_model():
    # get transformer config.json from hugging face
    # copy https://huggingface.co/google/gemma-2-2b-it/blob/main/config.json here
    gemma2b = {
        "architectures": [
            "Gemma2ForCausalLM"
        ],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attn_logit_softcapping": 50.0,
        "bos_token_id": 2,
        "cache_implementation": "hybrid",
        "eos_token_id": [
            1,
            107
        ],
        "final_logit_softcapping": 30.0,
        "head_dim": 256,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 2304,
        "initializer_range": 0.02,
        "intermediate_size": 9216,
        "max_position_embeddings": 8192,
        "model_type": "gemma2",
        "num_attention_heads": 8,
        "num_hidden_layers": 26,
        "num_key_value_heads": 4,
        "pad_token_id": 0,
        "query_pre_attn_scalar": 256,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000.0,
        "sliding_window": 4096,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.42.4",
        "use_cache": true,
        "vocab_size": 256000
    }

    # ref the modeling file, add model arch config
    # code: transformers/src/transformers/models/gemma2/modeling_gemma2.py
    ArchMap['Gemma2ForCausalLM'] = {
        "mlp_gate": True,
        "norm_scale": True,
        "norm_bias": False,
        "fuse_qkv": False,
        "qkv_bias": False,
        "o_bias": False,
        "mlp_bias": False,
        "lm_head_bias": False,
        'post_mlp_norm': True,
        'post_attn_norm': True,
    }
    ActMap['gelu_pytorch_tanh'] = 'Gelu'  # map new activation name to op_type

    # Gemma2ForCausalLM redefines this name
    WeightMap['mlp']['input_norm'] = 'pre_feedforward_layernorm'

    bs = 1
    seq_len = 2048
    ids_shape = [bs, seq_len]
    builder = Builder(**gemma2b)
    builder.build_graph(ids_shape, WeightMap)
    builder.save_graph('gemma2b.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()


# build these hugging face models to ONNX file, and do profiling.
def build_onnx_models():
    bs = 1
    seq_len = 128
    ids_shape = [bs, seq_len]
    builder = Builder(**gptj_6b)
    builder.build_graph(ids_shape)
    builder.save_graph('gptj_6b.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**QWen_7B)
    builder.build_graph(ids_shape)
    builder.save_graph('QWen_7B.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**Qwen2_72B_Instruct)
    builder.build_graph(ids_shape)
    builder.save_graph('Qwen2_72B_Instruct.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**Llama3_8B)
    builder.build_graph(ids_shape)
    builder.save_graph('Llama3_8B.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**llama_31_70B)
    builder.build_graph(ids_shape)
    builder.save_graph('llama_31_70B.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**phi3_mini)
    builder.build_graph(ids_shape)
    builder.save_graph('phi3_mini.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**Phi_3_medium_4k_instruct)
    builder.build_graph(ids_shape)
    builder.save_graph('Phi_3_medium_4k_instruct.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**Phi_3_small_8k_instruct)
    builder.build_graph(ids_shape)
    builder.save_graph('Phi_3_small_8k_instruct.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**phi2)
    builder.build_graph(ids_shape)
    builder.save_graph('phi2.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    builder = Builder(**yi_34B)
    builder.build_graph(ids_shape)
    builder.save_graph('yi_34B.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()


# generate summary table of these models
def profile_models():
    bs = 1
    seq_len = 1024
    ids_shape = [bs, seq_len]
    models = [gptj_6b, yi_34B, phi2, phi3_mini, Phi_3_small_8k_instruct, Phi_3_medium_4k_instruct, Llama3_8B,
              llama_31_70B, QWen_7B, Qwen2_72B_Instruct]

    # export model profile
    header = ['model_type', 'MACs(G)', 'Parameters(G)', 'KV Cache(G)']  # number not memory bytes
    rows = []
    for model in models:
        builder = Builder(**model)
        builder.build_graph(ids_shape)
        builder.graph.valid_shape = True
        builder.graph.profile()
        row = [builder.name, int(builder.graph.macs[0] / 1e9), builder.graph.params / 1e9, builder.kv_params / 1e9]
        rows.append(row)
    print(tabulate.tabulate(rows, headers=header))


def add_kv_cache():
    bs = 1
    seq_len = 128
    ids_shape = [bs, seq_len]
    past_sequence = 1024  # past length of KV cache
    context_length = 8096  # total length of KV cache
    builder = Builder(**Llama3_8B)
    builder.build_graph(ids_shape)
    builder.add_kv_cache(context_length, past_sequence)
    builder.save_graph('Llama3_8B.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()


def profile_model_with_devices():
    RuntimeCfg = {
        'Compute': {
            'MM': 'FP16',
            'MHA': 'FP16',
            'Others': 'FP16',
        },
        'Bits': {
            'MM': 16,
            'MHA': 16,
            'Others': 16,
        }
    }

    bs = 1
    prefill_length = 1024
    context_length = 4096

    models = [gptj_6b, yi_34B, phi2, phi3_mini, Phi_3_small_8k_instruct, Phi_3_medium_4k_instruct, llama2_7b, Llama3_8B,
              llama_31_70B, QWen_7B, Qwen2_72B_Instruct]

    # estimate latencies from hardware specs in onnx_tool.device
    from onnx_tool.device import Devices
    header = ['Model', 'Memory(G bytes)']
    rows = []
    device_names = ['Gaudi2H', 'H20']
    for key in device_names:
        header.append(key + '_prefill_latency')
    for key in device_names:
        header.append(key + '_decode_latency')
    for model in models:
        builder = Builder(**model)
        ids_shape = [bs, prefill_length]
        builder.build_graph(ids_shape)
        past_kv_length = 0
        builder.add_kv_cache(context_length, past_kv_length)
        builder.graph.valid_shape = True
        builder.profile(RuntimeCfg, None)
        row = [builder.name, builder.context_mem[3] / 1e9]
        for key in device_names:
            builder.profile(RuntimeCfg, Devices[key])
            row.append(builder.llm_profile[2])

        # change to decode shape
        builder.set_past_kv_length(prefill_length)
        builder.graph.shape_infer(inputs={'ids': create_ndarray_int64([bs, 1])})
        builder.graph.profile()
        for key in device_names:
            builder.profile(RuntimeCfg, Devices[key])
            row.append(builder.llm_profile[2])
        rows.append(row)

    print(tabulate.tabulate(rows, headers=header))


def gpt2_kv_cache():
    bs = 1
    seq_len = 128
    ids_shape = [bs, seq_len]
    past_sequence = 0  # past length of KV cache
    context_length = 8096  # total length of KV cache
    builder = Builder(**gpt2)
    WeightMap = {
        'embedding': {
            'embed': 'wte',
            'pos': 'wpe'
        },
        'layer_prefix': 'h.',
        'attention': {
            'input_norm': 'ln_1',
            'qkv': 'attn.c_attn',
            'q': 'attn.q_proj',
            'k': 'attn.k_proj',
            'v': 'attn.v_proj',
            'o': 'attn.c_proj',
            'output_norm': 'post_attention_layernorm'
        },
        'mlp': {
            'input_norm': 'ln_2',
            'gate': 'mlp.gate_proj',
            'up': 'mlp.c_fc',
            'down': 'mlp.c_proj',
            'gate_up': 'mlp.gate_up_proj',
            'output_norm': 'post_feedforward_layernorm',
        },
        'lm_head': {
            'input_norm': 'model.norm',
            'lm': 'lm_head'
        }
    }

    builder.build_graph(ids_shape, WeightMap)
    builder.add_kv_cache(context_length, past_sequence)
    builder.save_graph('gpt2.onnx')
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()


# generate summary table of these models
def profile_model():
    from onnx_tool.device import Devices
    RuntimeCfg = {
        'Compute': {
            'MM': 'FP16',
            'MHA': 'FP16',
            'Others': 'FP16',
        },
        'Bits': {
            'MM': 16,
            'MHA': 16,
            'Others': 16,
        }
    }
    bs = 1
    prefill_length = 1024
    context_length = 4096
    ids_shape = [bs, prefill_length]
    models = [llama2_7b, Llama3_8B]

    device_names = ['Gaudi2H']

    for model in models:
        builder = Builder(**model)
        # set prefill shape
        builder.build_graph(ids_shape)
        builder.add_kv_cache(context_length, 0)
        builder.graph.valid_shape = True
        model_name = builder.get_filename()
        for key in device_names:
            builder.profile(RuntimeCfg, Devices[key])
            file = None  # print
            # file = f'{model_name}_{key}_prefill.csv' # save file
            builder.print_profile(file)

        # change to decode shape
        builder.set_past_kv_length(prefill_length)
        builder.graph.shape_infer(inputs={'ids': create_ndarray_int64([bs, 1])})
        builder.graph.profile()
        for key in device_names:
            builder.profile(RuntimeCfg, Devices[key])
            file = None  # print
            # file = f'{model_name}_{key}_decode.csv' # save file
            builder.print_profile(file)


# generate summary table of these models
def profile_model_multicards():
    from onnx_tool.device import Devices
    RuntimeCfg = {
        'Compute': {
            'MM': 'FP16',
            'MHA': 'FP16',
            'Others': 'FP16',
        },
        'Bits': {
            'MM': 16,
            'MHA': 16,
            'Others': 16,
        }
    }
    bs = 1
    prefill_length = 1024
    context_length = 4096
    ids_shape = [bs, prefill_length]
    models = [Llama3_8B]

    device_name = 'Gaudi2H'
    device = {
        'FP32': 11000,
        'FP16': 428000, # benchmark number
        'INT8': 848000, # benchmark number
        'Bandwidth': 2230, # benchmark number
        'LinkBandwidth': 525,
        'Number': 4,
    }

    for model in models:
        builder = Builder(**model)
        # set prefill shape
        builder.build_graph(ids_shape)
        builder.add_kv_cache(context_length, 0)
        builder.graph.valid_shape = True
        model_name = builder.get_filename()
        builder.profile(RuntimeCfg, device)
        file = None  # print
        # file = f'{model_name}_{device_name}_prefill.csv' # save file
        builder.print_profile(file)

        # change to decode shape
        builder.set_past_kv_length(prefill_length)
        builder.graph.shape_infer(inputs={'ids': create_ndarray_int64([bs, 1])})
        builder.graph.profile()
        builder.profile(RuntimeCfg, device)
        file = None  # print
        # file = f'{model_name}_{device_name}_decode.csv' # save file
        builder.print_profile(file)


# ===========================================================================
# MiniMax-M2.7 模型配置
# ===========================================================================
# 从 https://huggingface.co/MiniMaxAI/MiniMax-M2.7/resolve/main/config.json
# MiniMax-M2.7: Sparse MoE (256 experts, top-8, sigmoid routing)
# Attention: GQA (48 heads, 8 KV heads, head_dim=128), per-layer QK norm
# Partial RoPE: rotary_dim=64 (仅前64维参与旋转)
# MoE: SwiGLU experts, NO shared expert (shared_intermediate_size=0)
# 总参数: ~230B, 激活参数: ~9.8B (per token, top-8 experts)
# 参考: arXiv 2605.26494 - The MiniMax-M2 Series
MiniMax_M2_7 = {
    "name": "MiniMax-M2.7",
    "architectures": ["MiniMaxM2ForCausalLM"],
    "attention_bias": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 3072,
    "intermediate_size": 1536,
    "max_position_embeddings": 204800,
    "model_type": "minimax_m2",
    "num_attention_heads": 48,
    "num_hidden_layers": 62,
    "num_key_value_heads": 8,
    "num_local_experts": 256,
    "num_experts_per_tok": 8,
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000,
    "rotary_dim": 64,
    "scoring_func": "sigmoid",
    "shared_intermediate_size": 0,
    "tie_word_embeddings": False,
    "use_cache": True,
    "use_qk_norm": True,
    "vocab_size": 200064,
}


def export_minimax_m2_7():
    """导出 MiniMax-M2.7 到 ONNX 格式"""
    bs = 1
    seq_len = 128
    ids_shape = [bs, seq_len]

    print("Building MiniMax-M2.7 graph...")
    builder = Builder(**MiniMax_M2_7)
    builder.build_graph(ids_shape)

    onnx_path = 'minimax_m2_7.onnx'
    print(f"Saving to {onnx_path}...")
    builder.save_graph(onnx_path)

    print("Profiling...")
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    macs = int(builder.graph.macs[0] / 1e9)
    params = builder.graph.params / 1e9
    print(f"\nMiniMax-M2.7: MACs={macs}G, Parameters={params:.3f}G")
    return onnx_path


def export_minimax_m2_7_with_kv_cache():
    """导出 MiniMax-M2.7 带 KV cache 的 ONNX 模型"""
    bs = 1
    seq_len = 128
    ids_shape = [bs, seq_len]
    past_sequence = 0
    context_length = 8192

    print("Building MiniMax-M2.7 with KV cache...")
    builder = Builder(**MiniMax_M2_7)
    builder.build_graph(ids_shape)
    builder.add_kv_cache(context_length, past_sequence)

    onnx_path = 'minimax_m2_7_kvcache.onnx'
    print(f"Saving to {onnx_path}...")
    builder.save_graph(onnx_path)

    print("Profiling...")
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    macs = int(builder.graph.macs[0] / 1e9)
    params = builder.graph.params / 1e9
    print(f"\nMiniMax-M2.7 (KV cache): MACs={macs}G, Parameters={params:.3f}G")
    return onnx_path


def profile_minimax_m2_7():
    """MiniMax-M2.7 prefill + decode 详细 profiling"""
    from onnx_tool.device import Devices

    RuntimeCfg = {
        'Compute': {
            'MM': 'FP16',
            'MHA': 'FP16',
            'Others': 'FP16',
        },
        'Bits': {
            'MM': 16,
            'MHA': 16,
            'Others': 16,
        }
    }

    bs = 1
    prefill_length = 2048
    context_length = 8192
    ids_shape = [bs, prefill_length]

    print("Building MiniMax-M2.7 for profiling...")
    builder = Builder(**MiniMax_M2_7)
    builder.build_graph(ids_shape)
    builder.add_kv_cache(context_length, 0)
    builder.graph.valid_shape = True

    # Prefill
    print(f"\n--- Prefill (seq_len={prefill_length}) ---")
    for device_key in ['Gaudi2H', 'H20']:
        builder.profile(RuntimeCfg, Devices[device_key])
        print(f"  {device_key}: Latency={builder.llm_profile[2]:.2f}ms")

    # Decode
    print(f"\n--- Decode (seq_len=1, past_kv={prefill_length}) ---")
    builder.set_past_kv_length(prefill_length)
    builder.graph.shape_infer(inputs={'ids': create_ndarray_int64([bs, 1])})
    builder.graph.profile()
    for device_key in ['Gaudi2H', 'H20']:
        builder.profile(RuntimeCfg, Devices[device_key])
        print(f"  {device_key}: Latency={builder.llm_profile[2]:.2f}ms")

    # Memory compression
    print(f"\n--- Memory Compression ---")
    compress_memory_with_kv_cache(builder.graph)


if __name__ == '__main__':
    # export_with_pytorch_weight_name()
    # add_hugging_face_model()
    # build_onnx_models()
    # profile_models()
    # add_kv_cache()
    # gpt2_kv_cache()
    # profile_model()
    # profile_model_with_devices()
    # profile_model_multicards()
    # export_minimax_m2_7()
    # export_qwen3_5_4b_with_kv_cache()
    # profile_qwen3_5_4b() 
    export_qwen3_5_4b_vision()