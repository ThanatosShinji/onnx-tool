from onnx_tool.llm import *
import tabulate


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


if __name__ == '__main__':
    export_with_pytorch_weight_name()
    add_hugging_face_model()
    build_onnx_models()
    profile_models()
    add_kv_cache()
    gpt2_kv_cache()
    profile_model()
    profile_model_with_devices()
    profile_model_multicards()
