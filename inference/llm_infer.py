"""
LLM 推理脚本：使用 GraphInfer 推理 LLM 模型（Qwen3、GPT2 等）

支持：
  - 通过 GraphInfer + safetensors 加载权重
  - 自回归生成（prefill + decode）
  - 与 HuggingFace Transformers 对比验证

用法：
  # Qwen3-0.6B 推理
  python inference/llm_infer.py --model-name Qwen/Qwen3-0.6B-Base --prompt "Hello"

  # GPT2 推理
  python inference/llm_infer.py --model-name gpt2 --onnx-path ./gpt2-10_shaped.onnx --prompt "Hello"

  # 与 HF 对比
  python inference/llm_infer.py --model-name Qwen/Qwen3-0.6B-Base --prompt "Hello" --compare-hf

  # 使用 XPU
  python inference/llm_infer.py --model-name Qwen/Qwen3-0.6B-Base --prompt "Hello" --device xpu
"""

import argparse
import time
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# 模型配置
# ===========================================================================

# Qwen3-0.6B 配置（与 benchmark/llm_test.py 保持一致）
QWEN3_0_6B_CONFIG = {
    "name": "Qwen3-0.6B-Base",
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "num_hidden_layers": 28,
    "intermediate_size": 3072,
    "head_dim": 128,
    "vocab_size": 151936,
    "hidden_act": "silu",
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
    "eos_token_id": 151643,
    "pad_token_id": 151643,
}

# 支持的模型列表
SUPPORTED_MODELS = {
    "Qwen/Qwen3-0.6B-Base": QWEN3_0_6B_CONFIG,
    "Qwen/Qwen3-0.6B": QWEN3_0_6B_CONFIG,
}


# ===========================================================================
# Tokenizer 工具
# ===========================================================================

def get_tokenizer(model_name: str, model_dir: str = None):
    """获取 HuggingFace tokenizer"""
    from transformers import AutoTokenizer
    if model_dir and os.path.isdir(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except OSError:
            base_name = model_name + "-Base"
            tokenizer = AutoTokenizer.from_pretrained(base_name, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


# ===========================================================================
# LLM 推理引擎
# ===========================================================================

class LLMInfer:
    """
    基于 GraphInfer 的 LLM 自回归推理引擎。

    支持两种模式：
      1. Qwen3 模式：通过 onnx_path + safetensors 加载（Builder 导出的 ONNX + safetensors 权重）
      2. 通用模式：直接加载 ONNX（含内嵌权重）
    """

    def __init__(
        self,
        onnx_path: str,
        model_name: str = None,
        safetensors_path: str = None,
        weight_map: Dict[str, str] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        max_seq_len: int = 128,
        config: dict = None,
        model_dir: str = None,
    ):
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.model_name = model_name
        self.config = config or {}
        self.model_dir = model_dir

        # 初始化 tokenizer
        if model_dir and os.path.isdir(model_dir):
            print(f"Loading tokenizer from: {model_dir}")
            self.tokenizer = get_tokenizer(model_name or "", model_dir=model_dir)
        elif model_name:
            print(f"Loading tokenizer: {model_name}")
            self.tokenizer = get_tokenizer(model_name)
        else:
            self.tokenizer = get_tokenizer("gpt2")

        # 自动检测 ONNX 模型的 input names
        from infer import GraphInfer
        import onnx_tool
        tmp_model = onnx_tool.Model(onnx_path, {'constant_folding': True})
        tmp_graph = tmp_model.graph

        # 找出输入名称（只保留真正的动态输入，过滤 weight/bias）
        all_inputs = list(tmp_graph.input)
        # weight/bias 等静态 tensor 在 initials 中
        initials_set = set(tmp_graph.initials)
        input_names = [n for n in all_inputs if n not in initials_set]
        print(f"ONNX inputs: {input_names}")

        # 检测是否为 KV-cache 模型
        self.has_kv_cache = 'kv_cache' in input_names
        self.has_n_past = 'n_past' in input_names
        if self.has_kv_cache:
            print(f"Detected KV-cache model (kv_cache + n_past inputs)")

        # 构建 input_desc 和 input_range（只包含动态输入，排除 weight/bias）
        self.input_names = input_names
        input_desc = {}
        input_range = {}

        # 动态输入名称：ids, position, n_past, kv_cache
        dynamic_input_names = {'ids', 'position', 'n_past', 'kv_cache'}
        for name in input_names:
            if name not in dynamic_input_names:
                continue
            if name in tmp_graph.tensormap:
                shape = tmp_graph.tensormap[name].get_shape()
                if name == 'n_past':
                    # n_past 是标量，shape 为空，但仍需加入 input_desc
                    input_desc[name] = ()
                elif shape:
                    desc = []
                    for i, s in enumerate(shape):
                        if isinstance(s, int) and s > 0:
                            desc.append(s)
                        else:
                            var_name = f'dim_{name}_{i}'
                            desc.append(var_name)
                            if name == 'kv_cache':
                                input_range[var_name] = (1, max_seq_len * 32)
                            else:
                                input_range[var_name] = (1, max_seq_len)
                    input_desc[name] = tuple(desc)

        # 对于 KV-cache 模型，确保 kv_cache 的 shape 使用正确的固定值
        # （避免 shape_regress 错误推导 kv_cache 的维度）
        if self.has_kv_cache and 'kv_cache' in input_desc:
            # kv_cache shape: [bs, 2*num_layers, context_length, num_kv_heads*head_dim]
            kv_shape = tmp_graph.tensormap['kv_cache'].get_shape()
            if kv_shape and all(isinstance(s, int) and s > 0 for s in kv_shape):
                input_desc['kv_cache'] = tuple(kv_shape)

        if not input_desc:
            # fallback: 假设是 ids + position，seq 维度设为变量
            input_desc = {input_names[0]: (1, 'seq')}
            if len(input_names) > 1:
                input_desc[input_names[1]] = (1, 'seq')
            input_range = {'seq': (1, max_seq_len)}
        elif not input_range:
            # shape 全是固定值（如 [1, 128]），把 ids 和 position 的 seq 维度设为变量
            for name in list(input_desc.keys()):
                if name in ('kv_cache', 'n_past'):
                    continue  # 不修改 kv_cache 和 n_past 的 shape
                desc = list(input_desc[name])
                if len(desc) >= 2 and isinstance(desc[-1], int):
                    var_name = f'seq'
                    input_range[var_name] = (1, desc[-1])
                    desc[-1] = var_name
                    input_desc[name] = tuple(desc)
                    # 不 break，继续处理其他 input（如 position）

        print(f"Input desc: {input_desc}")
        print(f"Input range: {input_range}")

        # 找出输出名称
        self.logits_name = list(tmp_graph.output)[0] if tmp_graph.output else input_names[0]
        print(f"Output name: {self.logits_name}")

        # 初始化 GraphInfer
        self.engine = GraphInfer(
            onnx_path,
            input_desc,
            input_range,
            dtype=self.dtype,
            device=self.device,
            safetensors_path=safetensors_path,
            weight_map=weight_map or {},
        )

        print(f"LLMInfer initialized: {len(self.engine._node_names)} nodes")

    def generate(
        self,
        prompt: str,
        max_length: int = 30,
        temperature: float = 0.0,
        top_k: int = 0,
        verbose: bool = False,
        profile: bool = False,
    ) -> str:
        """
        自回归生成文本。

        Args:
            prompt: 输入提示文本
            max_length: 最大生成 token 数（含 prompt）
            temperature: 采样温度（0.0 = greedy）
            top_k: top-k 采样（0 = 不使用）
            verbose: 是否打印详细信息
            profile: 是否对 prefill + 第一个 decode step 做 op-level profiling

        Returns:
            生成的完整文本
        """
        input_ids = self.tokenizer.encode(prompt)
        prompt_len = len(input_ids)
        if verbose:
            print(f"Prompt: {prompt} ({prompt_len} tokens)")

        generated = list(input_ids)
        device = self.device
        t_start = time.perf_counter()
        total_tokens = 0

        # KV-cache 状态
        n_past = 0

        # TTFT / TPOT 统计
        ttft = None          # Time To First Token (ms)
        decode_times = []    # 每个 decode step 的耗时 (ms)

        # === Warmup: 跑一次 prefill + decode，触发 XPU kernel JIT 编译 ===
        if verbose:
            print("Warming up...")
        warm_prompt_len = min(prompt_len, 4)  # 用少量 token 做 warmup
        warm_ids = input_ids[:warm_prompt_len]
        warm_input = torch.tensor([warm_ids], device=device, dtype=torch.int64)
        warm_pos = torch.arange(warm_prompt_len, device=device).unsqueeze(0)
        warm_feed = {}
        for name in self.input_names:
            if name == 'ids' or 'input' in name.lower() or 'token' in name.lower():
                warm_feed[name] = warm_input
            elif name == 'position':
                warm_feed[name] = warm_pos
            elif name == 'n_past':
                warm_feed[name] = torch.tensor(0, device=device, dtype=torch.int64)
            elif name == 'kv_cache':
                kv_shape = self.engine._cg.tensormap['kv_cache'].get_shape()
                warm_feed[name] = torch.zeros(*kv_shape, device=device, dtype=self.dtype)
        self.engine.forward(warm_feed, debug=False)
        # Warmup decode: 单 token
        warm_feed_decode = {}
        for name in self.input_names:
            if name == 'ids' or 'input' in name.lower() or 'token' in name.lower():
                warm_feed_decode[name] = torch.tensor([[input_ids[0]]], device=device, dtype=torch.int64)
            elif name == 'position':
                warm_feed_decode[name] = torch.tensor([[warm_prompt_len]], device=device)
            elif name == 'n_past':
                warm_feed_decode[name] = torch.tensor(warm_prompt_len, device=device, dtype=torch.int64)
            elif name == 'kv_cache':
                pass  # 复用 pool 中的 kv_cache
        self.engine.forward(warm_feed_decode, debug=False)
        if device == 'xpu':
            torch.xpu.synchronize()
        if verbose:
            print("Warmup done.\n")

        for step in range(max_length - prompt_len + 1):
            if step == 0:
                # Prefill: 输入所有 prompt tokens
                step_input = torch.tensor([input_ids], device=device, dtype=torch.int64)
                pos = torch.arange(prompt_len, device=device).unsqueeze(0)
                seq_len = prompt_len
            else:
                if self.has_kv_cache:
                    # KV-cache 模式：只输入单个 token
                    step_input = torch.tensor([[last_token]], device=device, dtype=torch.int64)
                    pos = torch.tensor([[prompt_len + step - 1]], device=device)
                    seq_len = 1
                else:
                    # 非 KV-cache 模式：输入完整序列（所有已生成的 token）
                    step_input = torch.tensor([generated], device=device, dtype=torch.int64)
                    pos = torch.arange(len(generated), device=device).unsqueeze(0)
                    seq_len = len(generated)

            # 构建 feed dict（只传 input_desc 中的动态输入，weight/bias 由 safetensors 加载）
            feed = {}
            for name in self.input_names:
                if name == 'ids' or 'input' in name.lower() or 'token' in name.lower():
                    feed[name] = step_input
                elif name == 'position':
                    feed[name] = pos
                elif name == 'n_past':
                    feed[name] = torch.tensor(n_past, device=device, dtype=torch.int64)
                elif name == 'kv_cache':
                    if step == 0:
                        # Prefill: 传入零初始化的 kv_cache
                        kv_shape = self.engine._cg.tensormap['kv_cache'].get_shape()
                        feed[name] = torch.zeros(
                            *kv_shape, device=device, dtype=self.dtype
                        )
                    else:
                        # Decode: 不传 kv_cache，memory pool 中已有上一轮更新后的值
                        pass
                # 其他输入（weight/bias 等）不传，由 GraphInfer 的 _constant_tensors 处理

            t0 = time.perf_counter()
            # Prefill (step=0) 和第一个 decode (step=1) 做 profiling
            do_profile = profile and step <= 1
            outputs = self.engine.forward(feed, debug=False, profile=do_profile)
            if device == 'xpu':
                torch.xpu.synchronize()
            step_time = (time.perf_counter() - t0) * 1000

            # 打印 profile 结果
            if do_profile and '__profile__' in outputs:
                label = "PREFILL" if step == 0 else "DECODE (1st token)"
                print(f"\n{'='*70}")
                print(f" Profile: {label}")
                print(f"{'='*70}")
                self.engine.print_profile(outputs.pop('__profile__'))

            # 记录 TTFT（第一个 decode token 的时刻）
            if step == 1 and ttft is None:
                ttft = (time.perf_counter() - t_start) * 1000
            # 记录 decode 耗时（step >= 1 为 decode）
            if step >= 1:
                decode_times.append(step_time)

            logits = outputs[self.logits_name]
            next_logits = logits[0, -1, :]

            if temperature > 0:
                next_logits = next_logits / temperature
                probs = F.softmax(next_logits, dim=-1)
                if top_k > 0:
                    top_probs, top_indices = torch.topk(probs, top_k)
                    probs = torch.zeros_like(probs).scatter_(0, top_indices, top_probs)
                    probs = probs / probs.sum()
                last_token = int(torch.multinomial(probs, 1).item())
            else:
                last_token = int(torch.argmax(next_logits).item())

            generated.append(last_token)
            total_tokens += 1

            # 更新 KV-cache 状态
            if self.has_kv_cache:
                n_past += seq_len

            if verbose:
                decoded = self.tokenizer.decode([last_token])
                print(f"  Step {step}: token={last_token} '{decoded}' ({step_time:.1f}ms)")

            if last_token == self.tokenizer.eos_token_id:
                if verbose:
                    print(f"  EOS at step {step}")
                break

        total_time = (time.perf_counter() - t_start) * 1000
        generated_text = self.tokenizer.decode(generated, skip_special_tokens=True)

        if verbose:
            print(f"\n{total_tokens} tokens in {total_time:.1f}ms "
                  f"({total_time/total_tokens:.1f}ms/token)")

        # 打印 TTFT / TPOT
        if ttft is not None:
            tpot = sum(decode_times) / len(decode_times) if decode_times else 0
            prefill_time = ttft - decode_times[0] if decode_times else ttft
            print(f"\n{'='*50}")
            print(f"Performance Metrics")
            print(f"{'='*50}")
            print(f"  TTFT (Time To First Token): {ttft:>8.1f} ms")
            print(f"  TPOT (Time Per Output Token): {tpot:>6.1f} ms/token")
            print(f"  Prefill time:                 {prefill_time:>8.1f} ms")
            print(f"  Decode tokens:                {len(decode_times):>8}")
            print(f"  Total time:                   {total_time:>8.1f} ms")
            print(f"{'='*50}")

        return generated_text

    def generate_with_hf_comparison(
        self,
        prompt: str,
        max_length: int = 30,
    ):
        """生成文本并与 HuggingFace Transformers 对比"""
        from transformers import AutoModelForCausalLM

        model_name = self.model_name or "gpt2"

        print("=" * 60)
        print(f"Prompt: '{prompt}'")
        print(f"Model: {model_name}")
        print("=" * 60)

        # GraphInfer
        print("\n--- GraphInfer ---")
        t0 = time.perf_counter()
        gi_text = self.generate(prompt, max_length=max_length, verbose=True)
        gi_time = (time.perf_counter() - t0) * 1000
        print(f"Result: '{gi_text}'")
        print(f"Time: {gi_time:.1f}ms")

        # HuggingFace
        print("\n--- HuggingFace Transformers ---")
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        hf_model = hf_model.to(self.device)
        hf_model.eval()

        hf_tokenizer = get_tokenizer(model_name)
        input_ids = hf_tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            hf_output = hf_model.generate(
                input_ids,
                max_length=max_length + input_ids.shape[1],
                do_sample=False,
                pad_token_id=hf_tokenizer.eos_token_id,
            )
        if self.device == 'xpu':
            torch.xpu.synchronize()
        hf_time = (time.perf_counter() - t0) * 1000
        hf_text = hf_tokenizer.decode(hf_output[0], skip_special_tokens=True)
        print(f"Result: '{hf_text}'")
        print(f"Time: {hf_time:.1f}ms")

        # 对比
        print("\n--- Comparison ---")
        print(f"GraphInfer:  '{gi_text}'")
        print(f"HF:          '{hf_text}'")
        print(f"Time: GI={gi_time:.1f}ms vs HF={hf_time:.1f}ms")


# ===========================================================================
# 工具函数
# ===========================================================================

def get_safetensors_path(model_name: str, model_dir: str = None) -> Optional[str]:
    """获取 HuggingFace 模型的 safetensors 路径"""
    import json

    # 优先使用本地文件夹
    if model_dir and os.path.isdir(model_dir):
        st_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(st_path):
            return st_path
        for f in os.listdir(model_dir):
            if f.endswith('.safetensors'):
                return os.path.join(model_dir, f)

    # 尝试多个可能的模型名称（如 Qwen/Qwen3-0.6B 和 Qwen/Qwen3-0.6B-Base）
    names_to_try = [model_name]
    if not model_name.endswith('-Base'):
        names_to_try.append(model_name + '-Base')

    for name in names_to_try:
        cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{name.replace('/', '--')}")
        if os.path.exists(cache_dir):
            snapshots = os.path.join(cache_dir, "snapshots")
            if os.path.exists(snapshots):
                for snap in sorted(os.listdir(snapshots), reverse=True):
                    snap_path = os.path.join(snapshots, snap)
                    st_path = os.path.join(snap_path, "model.safetensors")
                    if os.path.exists(st_path):
                        return st_path
                    for f in os.listdir(snap_path):
                        if f.endswith('.safetensors'):
                            return os.path.join(snap_path, f)
    return None


def get_qwen3_weight_map(config: dict) -> Dict[str, str]:
    """构建 Qwen2/Qwen3 的 weight_map（ONNX 名称 -> safetensors 名称）"""
    weight_map = {}
    num_layers = config['num_hidden_layers']

    # embed_tokens
    weight_map['model.embed_tokens.weight'] = 'model.embed_tokens.weight'

    # lm_head: tied 时共享 embed_tokens，否则独立
    if config.get('tie_word_embeddings', False):
        weight_map['lm_head.weight'] = 'model.embed_tokens.weight'
    else:
        weight_map['lm_head.weight'] = 'lm_head.weight'

    # per-layer weights
    for i in range(num_layers):
        prefix = f'model.layers.{i}'
        st_prefix = f'model.layers.{i}'
        weight_map[f'{prefix}.input_layernorm.weight'] = f'{st_prefix}.input_layernorm.weight'
        weight_map[f'{prefix}.self_attn.q_proj.weight'] = f'{st_prefix}.self_attn.q_proj.weight'
        weight_map[f'{prefix}.self_attn.q_proj.bias'] = f'{st_prefix}.self_attn.q_proj.bias'
        weight_map[f'{prefix}.self_attn.k_proj.weight'] = f'{st_prefix}.self_attn.k_proj.weight'
        weight_map[f'{prefix}.self_attn.k_proj.bias'] = f'{st_prefix}.self_attn.k_proj.bias'
        weight_map[f'{prefix}.self_attn.v_proj.weight'] = f'{st_prefix}.self_attn.v_proj.weight'
        weight_map[f'{prefix}.self_attn.v_proj.bias'] = f'{st_prefix}.self_attn.v_proj.bias'
        weight_map[f'{prefix}.self_attn.o_proj.weight'] = f'{st_prefix}.self_attn.o_proj.weight'
        weight_map[f'{prefix}.self_attn.q_norm.weight'] = f'{st_prefix}.self_attn.q_norm.weight'
        weight_map[f'{prefix}.self_attn.k_norm.weight'] = f'{st_prefix}.self_attn.k_norm.weight'
        weight_map[f'{prefix}.post_attention_layernorm.weight'] = f'{st_prefix}.post_attention_layernorm.weight'
        weight_map[f'{prefix}.mlp.gate_proj.weight'] = f'{st_prefix}.mlp.gate_proj.weight'
        weight_map[f'{prefix}.mlp.up_proj.weight'] = f'{st_prefix}.mlp.up_proj.weight'
        weight_map[f'{prefix}.mlp.down_proj.weight'] = f'{st_prefix}.mlp.down_proj.weight'

    # final norm
    weight_map['model.norm.weight'] = 'model.norm.weight'

    return weight_map


# ===========================================================================
# 命令行入口
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM GraphInfer 推理")
    parser.add_argument("--model-name", type=str, default=None,
                        help="HuggingFace 模型名称（如 Qwen/Qwen3-0.6B-Base）")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="本地模型文件夹路径（包含 config.json 和 safetensors）")
    parser.add_argument("--onnx-path", type=str, default=None,
                        help="ONNX 模型路径（不指定则自动导出）")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                        help="输入提示文本")
    parser.add_argument("--max-len", type=int, default=30,
                        help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度（0=greedy）")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Top-K 采样")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备（xpu/cpu）")
    parser.add_argument("--compare-hf", action="store_true",
                        help="与 HuggingFace Transformers 对比")
    parser.add_argument("--verbose", action="store_true",
                        help="打印详细信息")
    parser.add_argument("--profile", action="store_true",
                        help="对 prefill + 第一个 decode step 做 op-level profiling")
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "xpu" if torch.xpu.is_available() else "cpu"

    print(f"Device: {device}")

    # 确定模型配置
    model_name = args.model_name
    model_dir = args.model_dir
    onnx_path = args.onnx_path
    safetensors_path = None
    weight_map = None
    config = None

    # 如果 --model-name 是本地路径，自动当作 --model-dir
    if model_name and not model_dir and os.path.isdir(model_name):
        model_dir = model_name
        model_name = None
        print(f"Detected local model directory: {model_dir}")

    # 从本地文件夹或 HF cache 加载 config
    if model_dir and os.path.isdir(model_dir):
        import json
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            config.setdefault('head_dim', config.get('hidden_size', 1024) // config.get('num_attention_heads', 16))
            # 注意：不要 setdefault attention_bias！
            # Qwen2 硬编码 bias=True（不受 attention_bias 控制），
            # Qwen3 的 bias 由 config.attention_bias 控制。
            # Builder 会根据 ArchMap 和 config 中是否有 attention_bias 来决定。
            if not model_name:
                model_name = config.get('_name_or_path', os.path.basename(model_dir))
            print(f"Loaded config from {config_path}")
        else:
            raise RuntimeError(f"config.json not found in {model_dir}")
    elif model_name and model_name in SUPPORTED_MODELS:
        config = SUPPORTED_MODELS[model_name]
    elif model_name:
        raise RuntimeError(
            f"Unsupported model: '{model_name}'. "
            f"Supported: {list(SUPPORTED_MODELS.keys())}. "
            f"For local models, use --model-dir <path> or --model-name <path>."
        )

    if config is None:
        raise RuntimeError("No model specified. Use --model-name or --model-dir.")

    # 根据 architectures 判断是否支持
    arch = config.get('architectures', [''])[0]
    supported_archs = {'Qwen2ForCausalLM', 'Qwen3ForCausalLM', 'LlamaForCausalLM'}
    if arch not in supported_archs:
        raise RuntimeError(
            f"Unsupported architecture: '{arch}'. Supported: {sorted(supported_archs)}"
        )
    print(f"Architecture: {arch}")

    # 自动找 safetensors
    safetensors_path = get_safetensors_path(model_name or "", model_dir=model_dir)
    if safetensors_path:
        print(f"Found safetensors: {safetensors_path}")
    elif model_name and model_name in SUPPORTED_MODELS:
        print(f"Downloading safetensors for {model_name}...")
        from huggingface_hub import snapshot_download
        snapshot_download(model_name)
        safetensors_path = get_safetensors_path(model_name)
        print(f"Downloaded: {safetensors_path}")
    else:
        raise RuntimeError(f"safetensors not found. Put them in {model_dir or 'HF cache'}")

    # 构建 weight_map
    weight_map = get_qwen3_weight_map(config)

    # 自动找 ONNX 或导出
    if not onnx_path:
        import glob
        # 1) 在 model_dir 中查找
        if model_dir and os.path.isdir(model_dir):
            for f in sorted(os.listdir(model_dir)):
                if f.endswith('.onnx'):
                    onnx_path = os.path.join(model_dir, f)
                    print(f"Found ONNX in model_dir: {onnx_path}")
                    break
        # 2) 当前目录查找（仅当没有 model_dir 时）
        if not onnx_path and not model_dir:
            candidates = glob.glob('*kvcache.onnx') or glob.glob('*.onnx')
            if candidates:
                onnx_path = candidates[0]
                print(f"Found ONNX in current dir: {onnx_path}")
        # 3) 导出
        if not onnx_path:
            print("ONNX not found, exporting KV-cache model...")
            from onnx_tool.llm import Builder
            builder = Builder(**config)
            builder.build_graph([1, 128])
            builder.add_kv_cache(8192, 0)
            model_base = os.path.basename(model_dir) if model_dir else config.get("name", "model")
            onnx_path = os.path.join(model_dir, f'{model_base}_kvcache.onnx') if model_dir else f'{model_base.lower().replace("-", "_")}_kvcache.onnx'
            builder.save_graph(onnx_path)
            print(f"Exported to {onnx_path}")

    print(f"ONNX: {onnx_path}")
    print(f"Model: {model_name or 'unknown'}")

    # 导入 GraphInfer
    from infer import GraphInfer

    engine = LLMInfer(
        onnx_path=onnx_path,
        model_name=model_name,
        safetensors_path=safetensors_path,
        weight_map=weight_map,
        device=device,
        dtype=torch.float32,
        max_seq_len=128,
        config=config,
        model_dir=model_dir,
    )

    if args.compare_hf:
        engine.generate_with_hf_comparison(args.prompt, max_length=args.max_len)
    else:
        result = engine.generate(
            args.prompt,
            max_length=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            verbose=args.verbose,
            profile=args.profile,
        )
        print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
