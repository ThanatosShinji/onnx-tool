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
    onnx_tool.model_profile(tmpfile, shapesonly=True, saveshapesmodel=modelname)


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
              "n_embd": 4096,
              "hidden_size": 4096,
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
    ids = torch.zeros((1, 512), dtype=torch.long)
    torch.onnx.export(m, ids, tmpfile)
    onnx_tool.model_profile(tmpfile, shapesonly=True, saveshapesmodel=modelname)

def transformer_mpt():
    from mpt.configuration_mpt import MPTConfig
    from mpt.modeling_mpt import MPTForCausalLM
    config = MPTConfig(n_layers=1,attn_config={'attn_impl':'torch'})
    m = MPTForCausalLM(config)
    modelname = f"mpt_{config.d_model}_{config.n_heads}_{config.n_layers}.onnx"
    ids = torch.zeros((1, 512), dtype=torch.long)
    torch.onnx.export(m, ids, tmpfile)
    onnx_tool.model_profile(tmpfile, shapesonly=True, saveshapesmodel=modelname)

transfomer_llama()
transfomer_gptj()
transformer_mpt()