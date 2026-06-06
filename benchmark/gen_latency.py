"""Generate latency estimation tables for README.

Computes prefill/decode throughput (tokens/s) across devices
using 4-bit weights and 16-bit KV cache at 1k input length.
"""
import importlib
import onnx_tool.llm as _llm
importlib.reload(_llm)
from onnx_tool.llm import *
from onnx_tool.device import Devices

# Qwen3.5 model configs (defined here to avoid circular import from llm_test.py)
Qwen3_5_4B = {
    "name": "Qwen3.5-4B-Instruct", "architectures": ["Qwen3_5ForConditionalGeneration"],
    "attention_bias": False, "attention_dropout": 0.0, "head_dim": 256,
    "hidden_act": "silu", "hidden_size": 2560, "intermediate_size": 9216,
    "max_position_embeddings": 262144, "model_type": "qwen3_5",
    "num_attention_heads": 16, "num_hidden_layers": 32, "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06, "rope_theta": 10000000.0, "tie_word_embeddings": True,
    "use_cache": True, "vocab_size": 248320,
    "linear_num_key_heads": 16, "linear_num_value_heads": 32,
    "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
    "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention"] * 8,
}
Qwen3_5_35B_A3B = {
    "name": "Qwen3.5-35B-A3B-Instruct", "architectures": ["Qwen3_5MoeForCausalLM"],
    "attention_bias": False, "attention_dropout": 0.0, "head_dim": 256,
    "hidden_act": "silu", "hidden_size": 2048, "intermediate_size": 512,
    "max_position_embeddings": 32768, "model_type": "qwen3_5_moe",
    "num_attention_heads": 16, "num_hidden_layers": 40, "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06, "rope_theta": 10000000.0, "tie_word_embeddings": False,
    "use_cache": True, "vocab_size": 248320,
    "num_experts": 256, "num_experts_per_tok": 8,
    "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512,
    "linear_num_key_heads": 16, "linear_num_value_heads": 32,
    "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
    "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention"] * 10,
}


def gen_latency_tables():
    RuntimeCfg = {
        'Compute': {'MM': 'FP16', 'MHA': 'FP16', 'Others': 'FP16'},
        'Bits': {'MM': 4, 'MHA': 16, 'Others': 4},
    }
    bs = 1
    prefill = 1024
    ctx = 4096

    models = [
        ('gpt-j-6b', gptj_6b),
        ('yi-1.5-34B', yi_34B),
        ('phi-2', phi2),
        ('Phi-3-mini-4k', phi3_mini),
        ('Phi-3-small-8k', Phi_3_small_8k_instruct),
        ('Phi-3-medium-4k', Phi_3_medium_4k_instruct),
        ('Llama3-8B', Llama3_8B),
        ('Llama-3.1-70B', llama_31_70B),
        ('QWen-7B', QWen_7B),
        ('Qwen2-72B', Qwen2_72B_Instruct),
        ('Qwen3.5-4B', Qwen3_5_4B),
        ('Qwen3.5-35B-A3B', Qwen3_5_35B_A3B),
    ]

    devices = ['Ultra-358H', 'Arc-B70', 'RTX-4090', 'RTX-5090']

    # ---- Prefill ----
    print("### Prefill Throughput (tokens/s, 1k input)")
    header = f"{'model':30s} | " + " | ".join(f"{d:>10s}" for d in devices)
    print(header)
    print('-' * len(header))
    for name, cfg in models:
        b = Builder(**cfg)
        b.build_graph([bs, prefill])
        b.add_kv_cache(ctx, 0)
        b.graph.valid_shape = True
        b.profile(RuntimeCfg, None)
        tps = []
        for d in devices:
            b.profile(RuntimeCfg, Devices[d])
            ttft_s = b.llm_profile[2]
            tps.append(prefill / ttft_s if ttft_s > 0 else float('inf'))
        row = f"{name:30s} | " + " | ".join(f"{v:10.1f}" for v in tps)
        print(row)

    print()

    # ---- Decode ----
    print("### Decode Throughput (tokens/s)")
    header = f"{'model':30s} | " + " | ".join(f"{d:>10s}" for d in devices)
    print(header)
    print('-' * len(header))
    for name, cfg in models:
        b = Builder(**cfg)
        b.build_graph([bs, prefill])
        b.add_kv_cache(ctx, 0)
        b.graph.valid_shape = True
        b.set_past_kv_length(prefill)
        b.graph.shape_infer(inputs={'ids': create_ndarray_int64([bs, 1])})
        b.graph.profile()
        tps = []
        for d in devices:
            b.profile(RuntimeCfg, Devices[d])
            tpot = b.llm_profile[2]
            tps.append(1 / tpot if tpot > 0 else float('inf'))
        row = f"{name:30s} | " + " | ".join(f"{v:10.1f}" for v in tps)
        print(row)


if __name__ == '__main__':
    gen_latency_tables()
