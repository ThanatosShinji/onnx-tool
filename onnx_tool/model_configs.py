"""预定义的模型 config（从 llm.py 分离出来）"""

false = False
true = True
null = None

phi3_mini = {
    "name": 'Phi-3-mini-4k',
    "_name_or_path": "Phi-3-mini-4k-instruct",
    "architectures": [
        "Phi3ForCausalLM"
    ],
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoConfig": "configuration_phi3.Phi3Config",
        "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM"
    },
    "bos_token_id": 1,
    "embd_pdrop": 0.0,
    "eos_token_id": 32000,
    "hidden_act": "silu",
    "hidden_size": 3072,
    "initializer_range": 0.02,
    "intermediate_size": 8192,
    "max_position_embeddings": 4096,
    "model_type": "phi3",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 32,
    "original_max_position_embeddings": 4096,
    "pad_token_id": 32000,
    "resid_pdrop": 0.0,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "rope_theta": 10000.0,
    "sliding_window": 2047,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.40.2",
    "use_cache": true,
    "attention_bias": false,
    "vocab_size": 32064
}

QWen_7B = {
    "name": 'QWen-7B',
    "architectures": [
        "Qwen2ForCausalLM"
    ],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "hidden_act": "silu",
    "hidden_size": 3584,
    "initializer_range": 0.02,
    "intermediate_size": 18944,
    "max_position_embeddings": 32768,
    "max_window_layers": 28,
    "model_type": "qwen2",
    "num_attention_heads": 28,
    "num_hidden_layers": 28,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": 131072,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.41.2",
    "use_cache": true,
    "use_sliding_window": false,
    "vocab_size": 152064
}

Llama3_8B = {
    "name": 'Llama3-8B',
    "_name_or_path": "./dpo_1_000005_07",
    "architectures": [
        "LlamaForCausalLM"
    ],
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 14336,
    "max_position_embeddings": 8192,
    "mlp_bias": false,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "rope_theta": 500000.0,
    "tie_word_embeddings": false,
    "torch_dtype": "float32",
    "transformers_version": "4.42.3",
    "use_cache": true,
    "vocab_size": 128256
}

phi2 = {
    "name": 'microsoft/phi-2',
    "_name_or_path": "microsoft/phi-2",
    "architectures": [
        "PhiForCausalLM"
    ],
    "attention_dropout": 0.0,
    "bos_token_id": 50256,
    "embd_pdrop": 0.0,
    "eos_token_id": 50256,
    "hidden_act": "gelu_new",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 10240,
    "layer_norm_eps": 1e-05,
    "max_position_embeddings": 2048,
    "model_type": "phi",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 32,
    "partial_rotary_factor": 0.4,
    "qk_layernorm": false,
    "resid_pdrop": 0.1,
    "rope_scaling": null,
    "rope_theta": 10000.0,
    "tie_word_embeddings": false,
    "torch_dtype": "float16",
    "transformers_version": "4.37.0",
    "use_cache": true,
    "vocab_size": 51200
}

Qwen2_72B_Instruct = {
    "name": 'Qwen2_72B_Instruct',
    "architectures": [
        "Qwen2ForCausalLM"
    ],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "hidden_act": "silu",
    "hidden_size": 8192,
    "initializer_range": 0.02,
    "intermediate_size": 29568,
    "max_position_embeddings": 32768,
    "max_window_layers": 80,
    "model_type": "qwen2",
    "num_attention_heads": 64,
    "num_hidden_layers": 80,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": 131072,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.40.1",
    "use_cache": true,
    "use_sliding_window": false,
    "vocab_size": 152064
}

llama_31_70B = {
    "name": 'Llama-3.1-70B-Japanese-Instruct-2407',
    "_name_or_path": "Llama-3.1-70B-Japanese-Instruct-2407",
    "architectures": [
        "LlamaForCausalLM"
    ],
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": [
        128001,
        128008,
        128009
    ],
    "hidden_act": "silu",
    "hidden_size": 8192,
    "initializer_range": 0.02,
    "intermediate_size": 28672,
    "max_position_embeddings": 131072,
    "mlp_bias": false,
    "model_type": "llama",
    "num_attention_heads": 64,
    "num_hidden_layers": 80,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.44.0.dev0",
    "use_cache": true,
    "vocab_size": 128256
}

Phi_3_medium_4k_instruct = {
    "name": "Phi-3-medium-4k-instruct",
    "_name_or_path": "Phi-3-medium-4k-instruct",
    "architectures": [
        "Phi3ForCausalLM"
    ],
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoConfig": "configuration_phi3.Phi3Config",
        "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM"
    },
    "bos_token_id": 1,
    "embd_pdrop": 0.0,
    "eos_token_id": 32000,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 17920,
    "max_position_embeddings": 4096,
    "model_type": "phi3",
    "num_attention_heads": 40,
    "num_hidden_layers": 40,
    "num_key_value_heads": 10,
    "original_max_position_embeddings": 4096,
    "pad_token_id": 32000,
    "resid_pdrop": 0.0,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "rope_theta": 10000.0,
    "sliding_window": 2047,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.39.3",
    "use_cache": true,
    "attention_bias": false,
    "vocab_size": 32064
}

Phi_3_small_8k_instruct = {
    "name": "Phi-3-small-8k-instruct",
    "_name_or_path": "Phi-3-small-8k-instruct",
    "architectures": [
        "Phi3SmallForCausalLM"
    ],
    "attention_dropout_prob": 0.0,
    "auto_map": {
        "AutoConfig": "configuration_phi3_small.Phi3SmallConfig",
        "AutoModelForCausalLM": "modeling_phi3_small.Phi3SmallForCausalLM",
        "AutoModelForSequenceClassification": "modeling_phi3_small.Phi3SmallForSequenceClassification",
        "AutoTokenizer": "tokenization_phi3_small.Phi3SmallTokenizer"
    },
    "blocksparse_block_size": 64,
    "blocksparse_homo_head_pattern": false,
    "blocksparse_num_local_blocks": 16,
    "blocksparse_triton_kernel_block_size": 64,
    "blocksparse_vert_stride": 8,
    "bos_token_id": 100257,
    "dense_attention_every_n_layers": 2,
    "embedding_dropout_prob": 0.1,
    "eos_token_id": 100257,
    "ff_dim_multiplier": null,
    "ff_intermediate_size": 14336,
    "ffn_dropout_prob": 0.1,
    "gegelu_limit": 20.0,
    "gegelu_pad_to_256": true,
    "hidden_act": "gegelu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "max_position_embeddings": 8192,
    "model_type": "phi3small",
    "mup_attn_multiplier": 1.0,
    "mup_embedding_multiplier": 10.0,
    "mup_use_scaling": true,
    "mup_width_multiplier": 8.0,
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "pad_sequence_to_multiple_of_64": true,
    "reorder_and_upcast_attn": false,
    "rope_embedding_base": 1000000,
    "rope_position_scale": 1.0,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.38.1",
    "use_cache": true,
    "attention_bias": false,
    "vocab_size": 100352
}

gptj_6b = {
    'name': "gpt-j-6b",
    "activation_function": "gelu_new",
    "architectures": [
        "GPTJForCausalLM"
    ],
    "attn_pdrop": 0.0,
    "bos_token_id": 50256,
    "embd_pdrop": 0.0,
    "eos_token_id": 50256,
    "gradient_checkpointing": false,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gptj",
    "n_embd": 4096,
    "n_head": 16,
    "n_inner": null,
    "n_layer": 28,
    "n_positions": 2048,
    "resid_pdrop": 0.0,
    "rotary": true,
    "rotary_dim": 64,
    "scale_attn_weights": true,
    "summary_activation": null,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": true,
    "summary_type": "cls_index",
    "summary_use_proj": true,
    "task_specific_params": {
        "text-generation": {
            "do_sample": true,
            "max_length": 50,
            "temperature": 1.0
        }
    },
    "tie_word_embeddings": false,
    "tokenizer_class": "GPT2Tokenizer",
    "transformers_version": "4.18.0.dev0",
    "use_cache": true,
    "vocab_size": 50400
}

yi_34B = {
    'name': "yi-1.5-34B",
    "architectures": [
        "LlamaForCausalLM"
    ],
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 7168,
    "initializer_range": 0.02,
    "intermediate_size": 20480,
    "max_position_embeddings": 4096,
    "model_type": "llama",
    "num_attention_heads": 56,
    "num_hidden_layers": 60,
    "num_key_value_heads": 8,
    "pad_token_id": 0,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-06,
    "rope_scaling": null,
    "rope_theta": 5000000.0,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.40.0",
    "use_cache": false,
    "vocab_size": 64000
}

gpt2 = {
    "activation_function": "gelu_new",
    "architectures": [
        "GPT2LMHeadModel"
    ],
    "attn_pdrop": 0.1,
    "bos_token_id": 50256,
    "embd_pdrop": 0.1,
    "eos_token_id": 50256,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "resid_pdrop": 0.1,
    "summary_activation": null,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": true,
    "summary_type": "cls_index",
    "summary_use_proj": true,
    "task_specific_params": {
        "text-generation": {
            "do_sample": true,
            "max_length": 50
        }
    },
    "vocab_size": 50257
}

llama2_7b = {
    "_name_or_path": "meta-llama/Llama-2-7b-chat-hf",
    "architectures": [
        "LlamaForCausalLM"
    ],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_position_embeddings": 4096,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 32,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-06,
    "rope_scaling": null,
    "tie_word_embeddings": false,
    "torch_dtype": "float16",
    "transformers_version": "4.32.0.dev0",
    "use_cache": true,
    "vocab_size": 32000
}
