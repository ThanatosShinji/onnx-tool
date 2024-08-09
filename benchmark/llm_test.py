from onnx_tool.llm import *


# Export the model with pytorch tensor names
# Not necessary to convert safetensors to ONNX format
def export_with_pytorch_weight_name():
    bs = 1
    seq_len = 1024
    ids_shape = [bs, seq_len]
    builder = Builder(**QWen_7B)
    builder.build_graph(ids_shape, WeightMap)
    for name in builder.graph.initials:
        print(name)
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
    import tabulate
    RuntimeCfg = {
        'Compute': {
            'MM': 'INT8',
            'MHA': 'FP16',
            'Others': 'FP16',
        },
        'Bits': {
            'MM': 4.5,
            'MHA': 16,
            'Others': 32,
        }
    }
    bs = 1
    seq_len = 1024
    ids_shape = [bs, seq_len]
    models = [gptj_6b, yi_34B, phi2, phi3_mini, Phi_3_small_8k_instruct, Phi_3_medium_4k_instruct, Llama3_8B,
              llama_31_70B, QWen_7B, Qwen2_72B_Instruct]

    # export model profile
    header = ['model_type', 'MACs', 'Parameters', 'KV Cache']
    rows = []
    for model in models:
        builder = Builder(**model)
        builder.build_graph(ids_shape)
        builder.graph.valid_shape = True
        builder.graph.profile()
        row = [builder.name, int(builder.graph.macs[0]), builder.graph.params, builder.kv_params]
        rows.append(row)
    print(tabulate.tabulate(rows, headers=header))

    # estimate latencies from hardware specs in onnx_tool.device
    from onnx_tool.device import Devices
    header = ['model_type', 'memory_size']
    rows = []
    dkeys = Devices.keys()
    for key in dkeys:
        header.append(key + '_first_latency')
        header.append(key + '_next_latency')
    for model in models:
        builder = Builder(**model)
        builder.build_graph(ids_shape)
        builder.graph.valid_shape = True
        builder.profile(RuntimeCfg, None)
        row = [builder.name, builder.MemSizes[3] / 1e9]
        for key in dkeys:
            builder.profile(RuntimeCfg, Devices[key])
            row.append(builder.first_latency)
            row.append(builder.next_latency)
        rows.append(row)

    print(tabulate.tabulate(rows, headers=header))


if __name__ == '__main__':
    export_with_pytorch_weight_name()
    add_hugging_face_model()
    build_onnx_models()
    profile_models()
