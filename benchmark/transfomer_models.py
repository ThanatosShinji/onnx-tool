import transformers
import torch
import onnx_tool

tmpfile = 'tmp.onnx'


def transfomer_llama():
    config = {"bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu", "hidden_size": 4096,
              "intermediate_size": 11008, "initializer_range": 0.02, "max_sequence_length": 2048, "model_type": "llama",
              "num_attention_heads": 32, "num_hidden_layers": 1, "pad_token_id": -1, "rms_norm_eps": 1e-06,
              "torch_dtype": "float16", "transformers_version": "4.27.0.dev0", "use_cache": True, "vocab_size": 32000,
              "max_position_embeddings": 2048}
    modelname = f"{config['model_type']}_{config['hidden_size']}_{config['num_attention_heads']}_{config['num_hidden_layers']}.onnx"
    config = transformers.PretrainedConfig(**config)
    m = transformers.LlamaForCausalLM(config)
    ids = torch.zeros((1, 512), dtype=torch.long)
    torch.onnx.export(m, ids, tmpfile)
    onnx_tool.model_profile(tmpfile, save_profile='llama-1layer.csv', mcfg={'constant_folding': True, 'verbose': True},
                            shape_only=True, save_model=modelname)


def transfomer_gptj():
    config = {"activation_function": "gelu_new",
              "architectures": [
                  "GPTJForCausalLM"
              ],
              "attn_pdrop": 0.0,
              "bos_token_id": 50256,
              "embd_pdrop": 0.0,
              "eos_token_id": 50256,
              "gradient_checkpointing": False,
              "initializer_range": 0.02,
              "layer_norm_epsilon": 1e-05,
              "model_type": "gptj",
              "n_embd": 2048,
              "hidden_size": 2048,
              "n_head": 16,
              "num_attention_heads": 16,
              "n_inner": None,
              "n_layer": 1,
              "n_positions": 2048,
              "resid_pdrop": 0.0,
              "rotary": True,
              "rotary_dim": 64,
              "scale_attn_weights": True,
              "summary_activation": None,
              "summary_first_dropout": 0.1,
              "summary_proj_to_labels": True,
              "summary_type": "cls_index",
              "summary_use_proj": True,
              "task_specific_params": {
                  "text-generation": {
                      "do_sample": True,
                      "max_length": 50,
                      "temperature": 1.0
                  }
              },
              "tie_word_embeddings": False,
              "tokenizer_class": "GPT2Tokenizer",
              "transformers_version": "4.18.0.dev0",
              "use_cache": True,
              "vocab_size": 50400, "max_position_embeddings": 2048}
    modelname = f"{config['model_type']}_{config['n_embd']}_{config['n_head']}_{config['n_layer']}.onnx"
    config = transformers.PretrainedConfig(**config)
    m = transformers.GPTJForCausalLM(config)
    ids = torch.ones((1, 8), dtype=torch.long)
    # out = m(ids)
    # print(out)
    torch.onnx.export(m, ids, tmpfile)
    onnx_tool.model_profile(tmpfile, save_profile='gptj-1layer.csv', mcfg={'constant_folding': True, 'verbose': True},
                            shape_only=True, save_model=modelname)


def transformer_mpt():
    from mpt.configuration_mpt import MPTConfig
    from mpt.modeling_mpt import MPTForCausalLM
    config = MPTConfig(n_layers=1, attn_config={'attn_impl': 'torch'})
    m = MPTForCausalLM(config)
    modelname = f"mpt_{config.d_model}_{config.n_heads}_{config.n_layers}.onnx"
    ids = torch.zeros((1, 512), dtype=torch.long)
    torch.onnx.export(m, ids, tmpfile)
    onnx_tool.model_profile(tmpfile, save_profile='mpt'
                                                  '-1layer.csv', mcfg={'constant_folding': True, 'verbose': True},
                            shape_only=True, save_model=modelname)


def transformer_qwen():
    from onnx_tool.llm import QWen_7B
    QWen_7B['num_hidden_layers'] = 1
    modelname = f"{QWen_7B['model_type']}_{QWen_7B['hidden_size']}_{QWen_7B['num_attention_heads']}_{QWen_7B['num_hidden_layers']}.onnx"
    config = transformers.PretrainedConfig(**QWen_7B)
    m = transformers.Qwen2ForCausalLM(config)
    ids = torch.zeros((1, 512), dtype=torch.long)
    torch.onnx.export(m, ids, tmpfile)
    onnx_tool.model_profile(tmpfile, save_profile='llama-1layer.csv', mcfg={'constant_folding': True, 'verbose': True},
                            shape_only=True, save_model=modelname)


def transformer_llama3():
    from onnx_tool.llm import Llama3_8B
    Llama3_8B['num_hidden_layers'] = 2
    modelname = f"{Llama3_8B['model_type']}_{Llama3_8B['hidden_size']}_{Llama3_8B['num_attention_heads']}_{Llama3_8B['num_hidden_layers']}.onnx"
    config = transformers.PretrainedConfig(**Llama3_8B)
    m = transformers.LlamaForCausalLM(config)
    ids = torch.zeros((1, 512), dtype=torch.long)
    torch.onnx.export(m, ids, tmpfile)
    # onnx_tool.model_profile(tmpfile, save_profile='llama-1layer.csv', mcfg={'constant_folding': True, 'verbose': True},
    #                         shape_only=True, save_model=modelname)


def transformer_llama3():
    from onnx_tool.llm import Llama3_8B
    Llama3_8B['num_hidden_layers'] = 2
    modelname = f"{Llama3_8B['model_type']}_{Llama3_8B['hidden_size']}_{Llama3_8B['num_attention_heads']}_{Llama3_8B['num_hidden_layers']}.onnx"
    config = transformers.PretrainedConfig(**Llama3_8B)
    m = transformers.LlamaForCausalLM(config)
    ids = torch.zeros((1, 512), dtype=torch.long)
    torch.onnx.export(m, ids, tmpfile)
    # onnx_tool.model_profile(tmpfile, save_profile='llama-1layer.csv', mcfg={'constant_folding': True, 'verbose': True},
    #                         shape_only=True, save_model=modelname)


def transformer_phi3():
    from onnx_tool.llm import phi3_mini
    phi3_mini['num_hidden_layers'] = 1
    modelname = f"{phi3_mini['model_type']}_{phi3_mini['hidden_size']}_{phi3_mini['num_attention_heads']}_{phi3_mini['num_hidden_layers']}.onnx"
    config = transformers.PretrainedConfig(**phi3_mini)
    m = transformers.Phi3ForCausalLM(config)
    ids = torch.zeros((1, 512), dtype=torch.long)
    torch.onnx.export(m, ids, tmpfile)
    # onnx_tool.model_profile(tmpfile, save_profile='llama-1layer.csv', mcfg={'constant_folding': True, 'verbose': True},
    #                         shape_only=True, save_model=modelname)


def transformer_phi2():
    from onnx_tool.llm import Phi_3_small_8k_instruct
    cfg = Phi_3_small_8k_instruct
    cfg['num_hidden_layers'] = 1
    modelname = f"{cfg['model_type']}_{cfg['hidden_size']}_{cfg['num_attention_heads']}_{cfg['num_hidden_layers']}.onnx"
    config = transformers.PretrainedConfig(**cfg)
    m = transformers.Phi3SmallForCausalLM(config)
    ids = torch.zeros((1, 512), dtype=torch.long)
    torch.onnx.export(m, ids, tmpfile)
    # onnx_tool.model_profile(tmpfile, save_profile='llama-1layer.csv', mcfg={'constant_folding': True, 'verbose': True},
    #                         shape_only=True, save_model=modelname)


def transformer_gpt2():
    from onnx_tool.llm import null, true, false
    cfg = {
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
        "num_hidden_layers": 2,
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
    config = transformers.PretrainedConfig(**cfg)
    m = transformers.Gemma2ForCausalLM(config)
    ids = torch.zeros((1, 3000), dtype=torch.long)
    torch.onnx.export(m, ids, tmpfile)


# transfomer_llama()
# transfomer_gptj()
# transformer_mpt()
# transformer_phi2()
transformer_gpt2()
# transformer_llama3()
# transformer_qwen()
