---
name: llm-model-support
description: 'Support a new LLM model architecture in onnx-tool. Use when: adding a new transformer model (Qwen, DeepSeek, LLaMA, GPT, Phi, etc.) for graph building, profiling (MACs/params/memory), KV-cache analysis, or device latency projection. Covers the full workflow from HF config.json to ONNX export and performance profiling.'
argument-hint: '[model-name] [huggingface-url-or-config]'
user-invocable: true
disable-model-invocation: false
---

# New LLM Model Support for onnx-tool

Add support for a new LLM architecture in the onnx-tool profiling & analysis framework. This skill guides you through the complete process: analyzing the HuggingFace model config, registering the architecture, building the compute graph, exporting ONNX, and running performance profiling.

## When to Use

- Adding a new transformer-based LLM (dense, MoE, MLA, hybrid) to the benchmark suite
- User provides a HuggingFace model name or `config.json` URL
- Need to profile MACs, parameters, KV-cache memory, or device-projected latency
- Model uses a novel attention mechanism (Gated DeltaNet, MLA, etc.) requiring new builder methods

## Architecture Overview

The onnx-tool LLM framework has three layers:

```
┌─────────────────────────────────────────────────┐
│  benchmark/llm_test.py                          │
│  Model config dicts + export/profile functions  │
├─────────────────────────────────────────────────┤
│  onnx_tool/llm.py                               │
│  Builder class + ArchMap + WeightMap            │
├─────────────────────────────────────────────────┤
│  onnx_tool/graph.py + node.py + tensor.py       │
│  Core graph IR, shape inference, profiling      │
└─────────────────────────────────────────────────┘
```

Key files:
- `onnx_tool/llm.py` — `Builder` class, `ArchMap` (architecture flags), `WeightMap` (weight name mapping)
- `onnx_tool/model_configs.py` — Pre-defined model config dicts
- `benchmark/llm_test.py` — Export/profile functions, dual-pool memory compression
- `inference/infer.py` — `GraphInfer` runtime engine
- `inference/kernels.py` — Kernel registry (50+ ops)

## Procedure

### Step 1: Gather Model Configuration

Obtain the HuggingFace `config.json` for the target model. Key fields needed:

| Field | Required | Notes |
|-------|----------|-------|
| `architectures` | **Yes** | e.g. `["Qwen3ForCausalLM"]` — determines `ArchMap` entry |
| `hidden_size` | **Yes** | Model dimension |
| `num_hidden_layers` | **Yes** | Number of transformer layers |
| `num_attention_heads` | **Yes** | Query heads |
| `num_key_value_heads` | **Yes** | KV heads (for GQA) |
| `head_dim` | No | Defaults to `hidden_size / num_attention_heads` |
| `intermediate_size` | **Yes** | FFN/MLP intermediate dimension |
| `hidden_act` | **Yes** | Activation: `silu`, `gelu_new`, `gegelu` |
| `rms_norm_eps` / `layer_norm_eps` | **Yes** | Norm epsilon |
| `rope_theta` | No | RoPE base frequency (default 10000) |
| `max_position_embeddings` | No | Max context length |
| `vocab_size` | **Yes** | Vocabulary size |
| `tie_word_embeddings` | No | Whether lm_head shares embedding weights |
| `attention_bias` | No | Whether Q/K/V/O projections have bias |
| `mlp_bias` | No | Whether MLP projections have bias |

**MoE-specific** (if applicable):
- `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `shared_expert_intermediate_size`

**MLA-specific** (DeepSeek-V4):
- `q_lora_rank`, `o_lora_rank`, `o_groups`, `rope_head_dim`

**Hybrid architecture** (Qwen3.5):
- `layer_types` — list of `"linear_attention"` / `"full_attention"` per layer
- `linear_num_key_heads`, `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_conv_kernel_dim`

### Step 2: Classify the Architecture

Determine which architectural features the model uses by comparing against existing `ArchMap` entries in `onnx_tool/llm.py`:

| Feature | Flag | Models |
|---------|------|--------|
| SwiGLU MLP | `mlp_gate: True` | LLaMA, Qwen2/3, DeepSeek |
| Standard MLP | `mlp_gate: False` | GPT-2, Phi, GPT-J |
| RMSNorm | `norm_scale: True, norm_bias: False` | Most modern LLMs |
| LayerNorm | `norm_scale: True, norm_bias: True` | GPT-2, Phi-2 |
| Fused QKV | `fuse_qkv: True` | Phi-3, GPT-2 |
| Separate Q/K/V | `fuse_qkv: False` | LLaMA, Qwen, DeepSeek |
| Q/K/V bias | `qkv_bias: True` | Qwen2/3, GPT-2 |
| Per-head Q/K norm | `qk_norm: True` | Qwen3, Qwen3.5 |
| Gated attention | `qkv_gated: True` | Qwen3.5 |
| Sparse MoE | `is_moe: True` | Qwen3.5-MoE, DeepSeek-V4 |
| MLA (low-rank attn) | `is_mla: True` | DeepSeek-V4 |
| Shared expert no gate | `no_shared_gate: True` | DeepSeek-V4 |
| Absolute position embedding | `pos_embedding: True` | GPT-2 |
| No RoPE | `qk_rope: False` | GPT-2 |

**Decision tree for new architectures:**

1. Does it match an existing `ArchMap` entry exactly? → Reuse it
2. Is it a variant of an existing architecture? → Copy and modify the closest entry
3. Is it a completely new architecture? → Create a new entry with all flags

### Step 3: Register the Architecture

#### 3a. Add to `ArchMap` in `onnx_tool/llm.py`

```python
ArchMap['NewModelForCausalLM'] = {
    "mlp_gate": True,        # SwiGLU if silu activation
    "norm_scale": True,      # RMSNorm if rms_norm_eps present
    "norm_bias": False,
    "fuse_qkv": False,       # True only if Q/K/V are a single projection
    "qkv_bias": True,        # From config.attention_bias
    "o_bias": False,
    "mlp_bias": False,
    "lm_head_bias": False,
    # Add new flags only if needed:
    # "qk_norm": True,       # Per-head Q/K RMSNorm
    # "qkv_gated": True,     # Gated attention
    # "is_moe": True,        # Sparse MoE
    # "is_mla": True,        # Multi-head Latent Attention
    # "no_shared_gate": True,# MoE shared expert without sigmoid gate
}
```

#### 3b. Update `WeightMap` (if weight naming differs)

The default `WeightMap` in `onnx_tool/llm.py` follows the LLaMA/Qwen convention:
```python
WeightMap = {
    'embedding': {'embed': 'model.embed_tokens'},
    'layer_prefix': 'model.layers.',
    'attention': {
        'input_norm': 'input_layernorm',
        'q': 'self_attn.q_proj',
        'k': 'self_attn.k_proj',
        'v': 'self_attn.v_proj',
        'o': 'self_attn.o_proj',
    },
    'mlp': {
        'gate': 'mlp.gate_proj',
        'up': 'mlp.up_proj',
        'down': 'mlp.down_proj',
    },
    'lm_head': {'input_norm': 'model.norm', 'lm': 'lm_head'},
}
```

If the new model uses different naming (e.g., GPT-2 uses `transformer.h.*`), pass a custom `weight_map` to `builder.build_graph(ids_shape, weight_map=custom_map)`.

#### 3c. Add activation mapping (if needed)

If the model uses a non-standard activation, add it to `ActMap`:
```python
ActMap = {
    'silu': 'Silu',
    'gelu_new': 'Gelu',
    'gegelu': 'GeGelu',
    # Add new: 'new_act': 'NewAct',
}
```

### Step 4: Add New Builder Methods (if needed)

If the model introduces a novel operator not covered by existing builder methods, add it to the `Builder` class in `onnx_tool/llm.py`.

**Existing builder methods:**
- `add_embedding()` — Token/position embedding (Gather)
- `add_layernorm()` — RMSNorm or LayerNorm
- `add_mm()` — MatMul with optional bias
- `add_rope()` — Rotary position embedding
- `add_mha()` — Multi-head attention (SDPA node)
- `add_qkv()` — Standard Q/K/V projections
- `add_qkv_gated()` — Gated QKV (Qwen3.5)
- `add_mlp()` — Standard/SwiGLU MLP
- `add_moe()` — Sparse MoE (Qwen3.5-MoE, DeepSeek-V4)
- `add_mla()` — Multi-head Latent Attention (DeepSeek-V4)
- `add_gdn()` — Gated DeltaNet (Qwen3.5)
- `add_qk_norm()` — Per-head Q/K RMSNorm
- `add_layers()` — Layer loop dispatcher

**Pattern for new builder methods:**
1. Create weight tensors with `create_tensor(name, STATIC_TENSOR, shape, dtype)`
2. Add to `self.graph.initials` and `self.graph.tensormap`
3. Create the node with `create_node(TmpNodeProto(name, op_type, attrs))`
4. Set `node.input` and `node.output`
5. Register output tensors in `self.graph.tensormap`
6. Register the node in `self.graph.nodemap`

### Step 5: Create Model Config Dict

Add the model configuration to `benchmark/llm_test.py` (or `onnx_tool/model_configs.py` for shared configs):

```python
NewModel_7B = {
    "name": "NewModel-7B",
    "architectures": ["NewModelForCausalLM"],
    "attention_bias": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "max_position_embeddings": 32768,
    "model_type": "new_model",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": 128256,
    # MoE/MLA/hybrid params if applicable
}
```

The `Builder.__init__` automatically handles field name normalization:
- `ff_intermediate_size` → `intermediate_size`
- `layer_norm_epsilon` → `layer_norm_eps`
- `n_embd` → `hidden_size`
- `n_head` → `num_attention_heads`
- `n_layer` → `num_hidden_layers`
- `n_inner` → `intermediate_size`
- `activation_function` → `hidden_act`

### Step 6: Create Export & Profile Functions

Add to `benchmark/llm_test.py`:

```python
def export_new_model():
    """Export NewModel to ONNX format"""
    bs = 1
    seq_len = 2048
    ids_shape = [bs, seq_len]

    print("Building NewModel graph...")
    builder = Builder(**NewModel_7B)
    builder.build_graph(ids_shape)

    onnx_path = 'new_model.onnx'
    print(f"Saving to {onnx_path}...")
    builder.save_graph(onnx_path)

    print("Profiling...")
    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()

    macs = int(builder.graph.macs[0] / 1e9)
    params = builder.graph.params / 1e9
    print(f"\nNewModel: MACs={macs}G, Parameters={params:.3f}G")
    return onnx_path


def export_new_model_with_kv_cache():
    """Export NewModel with KV-cache"""
    bs = 1
    seq_len = 2048
    ids_shape = [bs, seq_len]
    context_length = 8192

    builder = Builder(**NewModel_7B)
    builder.build_graph(ids_shape)
    builder.add_kv_cache(context_length, 0)

    onnx_path = 'new_model_kvcache.onnx'
    builder.save_graph(onnx_path)

    builder.graph.valid_shape = True
    builder.graph.profile()
    builder.graph.print_node_map()
    return onnx_path


def profile_new_model():
    """Detailed prefill + decode profiling"""
    from onnx_tool.device import Devices

    RuntimeCfg = {
        'Compute': {'MM': 'FP16', 'MHA': 'FP16', 'Others': 'FP16'},
        'Bits': {'MM': 16, 'MHA': 16, 'Others': 16},
    }

    bs = 1
    prefill_length = 2048
    context_length = 8192
    ids_shape = [bs, prefill_length]

    builder = Builder(**NewModel_7B)
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
```

### Step 7: Validate

Run the export function to verify:

```python
python -c "from benchmark.llm_test import export_new_model; export_new_model()"
```

Checklist:
- [ ] ONNX file generated without errors
- [ ] `graph.profile()` completes — MACs and parameter counts are reasonable
- [ ] `graph.print_node_map()` shows correct op types and shapes
- [ ] KV-cache variant exports correctly (if applicable)
- [ ] Device profiling runs without NaN/inf latency values
- [ ] Parameter count matches HuggingFace reported value (±1%)

### Step 8: (Optional) Add Inference Kernel Support

If the model uses custom ops not in `inference/kernels.py`, register new kernels:

```python
@KernelRegistry.register("NewOp")
class NewOpKernel(Kernel):
    @staticmethod
    def run(inputs, outputs, attrs):
        # inputs: List[torch.Tensor] — already reshaped
        # outputs: List[torch.Tensor] — write results here
        result = torch.some_operation(inputs[0], ...)
        outputs[0].copy_(result)
```

Then test with `GraphInfer`:
```python
from inference.infer import GraphInfer
engine = GraphInfer(
    onnx_path='new_model.onnx',
    input_desc={'ids': (1, 's')},
    input_range={'s': (1, 2048)},
    safetensors_path='path/to/model.safetensors',
)
output = engine.forward({'ids': input_ids_tensor})
```

## Common Pitfalls

1. **`head_dim` mismatch**: When `head_dim` is explicitly configured and differs from `hidden_size / num_attention_heads`, Q/K/V output dimensions use `num_heads * head_dim`, not `hidden_size`. The `Builder.__init__` handles this via `self.head_size`.

2. **MoE `intermediate_size`**: For MoE models, `intermediate_size` in config is the *per-expert* size. The `add_moe()` method uses `moe_intermediate_size` (defaults to `intermediate_size`). Set both explicitly.

3. **MLA KV dimension**: MLA uses `head_dim` (single head) for KV, not `num_kv_heads * head_dim`. The `build_graph()` method checks `is_mla` to compute `kv_size` correctly.

4. **Weight naming**: If the model's safetensors weight names don't match `WeightMap`, inference with `GraphInfer` will fail. Check the actual weight names in the safetensors file.

5. **`layer_types` length**: Must equal `num_hidden_layers`. Each element is either `"linear_attention"` (GDN) or `"full_attention"` (standard SDPA).

6. **Config field name normalization**: Some HuggingFace configs use non-standard field names. Add mappings in `Builder.__init__` if needed:
   - `num_local_experts` → `num_experts` (MiniMax-M2)
   - `ff_intermediate_size` → `intermediate_size`
   - `n_embd` → `hidden_size` (GPT-2 style)
   - `n_head` → `num_attention_heads`
   - `n_layer` → `num_hidden_layers`
   - `activation_function` → `hidden_act`

7. **No shared expert in MoE**: Some MoE models (MiniMax-M2) have `shared_intermediate_size: 0`, meaning no shared expert at all. Add `"no_shared_expert": True` to `ArchMap` and ensure `add_moe()` and `MoENode.profile()` both handle this flag — skip shared expert weight creation, MACs, and `static_params`.

8. **Partial RoPE (`rotary_dim`)**: When `rotary_dim < head_dim`, only the first `rotary_dim` dimensions participate in RoPE. Update `add_rope()` to use `rotary_dim` instead of `head_size` for cos/sin table generation. The `RopeNode.profile()` must also scale `static_params` by `seq_len` (not full `max_position_embeddings`), otherwise decode latency will be overestimated due to counting the entire cos/sin table.

9. **MoE `static_params` must account for sparse activation**: `MoENode.profile()` already scales routed expert `static_params` by `min(B*S*k, E)`. Verify this works correctly for the new model's `num_experts_per_tok`. For decode (S=1), this dramatically reduces the counted weight memory vs total params.

10. **GDN `static_params` double-counting**: GDN nodes receive Q/K/V activations as inputs — the Q/K/V projection weights are separate `MatMul` nodes. Do NOT include projection weights in GDN's `static_params` (set to 0), or activated params will exceed total params at large seq_len.

11. **MLA `log2(0)` crash at small seq_len**: When `compress_ratio == 4` and `seq_len < 4`, `compressed_tokens = S // 4 = 0`, causing `log2(0)` in `MLANode.profile()`. Guard with `if compressed_tokens > 1:` before the log2 call.

12. **MoE/MLA/GDN should use `c_mm` not `c_others`**: These fused nodes are MatMul-heavy. In `Builder.profile()`, they should use `c_mm` (tensor core throughput) for compute latency, not `c_others`. They already use `cfg['Bits']['MM']` for weight memory — ensure the compute path matches.

13. **Activated params vs total params**: `graph.params` counts all static weights. Activated params = `sum(node.static_params)` across all nodes. For MoE at small seq_len, activated ≪ total. At S=32 with top-8, 256 experts are fully activated (B×S×k=256=E). The remaining gap is from embedding/lm_head which only access `seq_len` rows of the full vocab table.

14. **Validate with `benchmark/moe_activated_params.py`**: After adding a new MoE model, run this script to verify activated params are ≤ total params at all seq_len, and that saturation occurs at the expected seq_len.

## Reference: Supported Architectures

| Architecture | Key Features | Example Models |
|---|---|---|
| `LlamaForCausalLM` | SwiGLU, RMSNorm, GQA, RoPE | LLaMA 3, LLaMA 3.1 |
| `Qwen2ForCausalLM` | SwiGLU, RMSNorm, GQA, QKV bias | Qwen2-7B/72B |
| `Qwen3ForCausalLM` | + per-head Q/K norm | Qwen3-0.6B |
| `Qwen3_5ForConditionalGeneration` | + gated attention, hybrid GDN | Qwen3.5-4B |
| `Qwen3_5MoeForCausalLM` | + sparse MoE | Qwen3.5-35B-A3B |
| `DeepSeekV4ForCausalLM` | MLA + MoE | DeepSeek-V4-Flash/Pro |
| `Phi3ForCausalLM` | Fused QKV, GeGeLU | Phi-3-mini |
| `PhiForCausalLM` | LayerNorm, partial RoPE | Phi-2 |
| `GPT2LMHeadModel` | Fused QKV, absolute position | GPT-2 |
| `GPTJForCausalLM` | Standard MLP, LayerNorm | GPT-J |
| `MiniMaxM2ForCausalLM` | MoE (no shared expert), per-layer QK norm, partial RoPE, sigmoid routing | MiniMax-M2.7 |
