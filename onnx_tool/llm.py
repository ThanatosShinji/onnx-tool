from .graph import Graph
from .node import *
from .node import _max_shape
from .tensor import *

false = False
true = True
null = None

ActMap = {
    'silu': 'Silu',
    'gelu_new': 'Gelu',
    'gegelu': 'GeGelu'
}

ArchMap = {
    'LlamaForCausalLM': {
        "ffn_num_mm": 3,
        "norm_scale": True,
        "norm_bias": False,
        "fuse_qkv": False,
        "qkv_bias": False,
        "o_bias": False,
        "mlp_bias": False,
        "lm_head_bias": False,
    }
}


class Builder():
    def __init__(self, **kwargs):
        for k in kwargs.keys():
            newk = k
            newv = kwargs[k]
            print(k, kwargs[k])
            if k == 'ff_intermediate_size':
                newk = 'intermediate_size'
            if k == 'layer_norm_epsilon':
                newk = 'layer_norm_eps'
            if k == 'n_embd':
                newk = 'hidden_size'
            if k == 'n_head':
                newk = 'num_attention_heads'
                setattr(self, 'num_key_value_heads', newv)
            if k == 'n_layer':
                newk = 'num_hidden_layers'
            if k == 'n_inner':
                newv = newv if newv is not None else 4 * kwargs['n_embd']
                newk = 'intermediate_size'
            if k == 'activation_function':
                newk = 'hidden_act'
            setattr(self, newk, newv)
        if ArchMap.__contains__(self.architectures[0]):
            self.arch_config = ArchMap[self.architectures[0]]
        else:
            self.arch_config = ArchMap['LlamaForCausalLM']
        if hasattr(self, 'mlp_bias'):
            self.arch_config['mlp_bias'] = self.mlp_bias
        if hasattr(self, 'attention_bias'):
            self.arch_config['qkv_bias'] = self.attention_bias
            self.arch_config['o_bias'] = self.attention_bias
        self.graph = Graph(None, None)
        self.node_count = 0
        self.tensor_count = 0
        self.head_size = self.hidden_size // self.num_attention_heads
        self.hidden_kv_size = self.head_size * self.num_key_value_heads

    def get_init_tensor_name(self):
        name = f'InitTensor_{self.tensor_count}'
        self.tensor_count += 1
        self.graph.initials.append(name)
        return name

    def get_dynamic_tensor_name(self):
        name = f'DynaTensor_{self.tensor_count}'
        self.tensor_count += 1
        self.graph.dynamics.append(name)
        return name

    def get_node_name(self):
        name = f'Node_{self.node_count}'
        self.node_count += 1
        return name

    def add_embedding(self, inp):
        emd = create_node(TmpNodeProto(self.get_node_name(), 'Gather', {'axis': 0}))
        w = create_tensor(self.get_init_tensor_name(), STATIC_TENSOR, [self.vocab_size, self.hidden_size],
                          numpy.float32)
        emd.input = [w.name, inp.name]
        o = create_tensor(self.get_dynamic_tensor_name(), DYNAMIC_TENSOR, [self.batch, self.seq_len, self.hidden_size],
                          numpy.float32)
        emd.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.tensormap[w.name] = w
        self.graph.nodemap[f'Node_{self.node_count}'] = emd
        return o

    def add_layernorm(self, inp):
        if hasattr(self, 'rms_norm_eps'):
            nod = create_node(TmpNodeProto(self.get_node_name(), 'LayerNormalization',
                                           {'epsilon': self.rms_norm_eps, 'typoe': 'rms'}))
        elif hasattr(self, 'layer_norm_eps'):
            nod = create_node(
                TmpNodeProto(self.get_node_name(), 'LayerNormalization', {'epsilon': self.layer_norm_eps}))
        else:
            assert 0
        nod.input = [inp.name]
        if self.arch_config['norm_scale']:
            s = create_tensor(self.get_init_tensor_name(), STATIC_TENSOR, [self.hidden_size],
                              numpy.float32)
            nod.input.append(s.name)
            self.graph.tensormap[s.name] = s
            if self.arch_config['norm_bias']:
                b = create_tensor(self.get_init_tensor_name(), STATIC_TENSOR, [self.hidden_size],
                                  numpy.float32)
                nod.input.append(b.name)
                self.graph.tensormap[b.name] = b

        o = create_tensor(self.get_dynamic_tensor_name(), DYNAMIC_TENSOR, inp.get_shape(),
                          numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.nodemap[f'Node_{self.node_count}'] = nod
        return o

    def add_act(self, inp, act):
        nod = create_node(TmpNodeProto(self.get_node_name(), ActMap[act], {}))
        nod.input = [inp.name]
        o = create_tensor(self.get_dynamic_tensor_name(), DYNAMIC_TENSOR, inp.get_shape(),
                          numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.nodemap[f'Node_{self.node_count}'] = nod
        return o

    def add_rope(self, inp):
        nod = create_node(TmpNodeProto(self.get_node_name(), 'Rope',
                                       {'rope_theta': self.rope_theta if hasattr(self,
                                                                                 'rope_theta') and self.rope_theta is not None else 0,
                                        'rope_scaling': self.rope_scaling if hasattr(self,
                                                                                     'rope_scaling') and self.rope_scaling is not None else 0}))
        nod.input = [inp.name]
        o = create_tensor(self.get_dynamic_tensor_name(), DYNAMIC_TENSOR, inp.get_shape(),
                          numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.nodemap[f'Node_{self.node_count}'] = nod
        return o

    def add_eltop(self, inp, inp1, op):
        nod = create_node(TmpNodeProto(self.get_node_name(), op, {}))
        nod.input = [inp.name, inp1.name]
        s = inp.get_shape()
        s1 = inp1.get_shape()
        s = _max_shape([s, s1])
        o = create_tensor(self.get_dynamic_tensor_name(), DYNAMIC_TENSOR, s,
                          numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.nodemap[f'Node_{self.node_count}'] = nod
        return o

    def add_mm(self, inp, fin, fout, bias):
        nod = create_node(TmpNodeProto(self.get_node_name(), 'MatMul', {}))
        w = create_tensor(self.get_init_tensor_name(), STATIC_TENSOR, [fin, fout],
                          numpy.float32)
        nod.input = [inp.name, w.name]

        if bias:
            b = create_tensor(self.get_init_tensor_name(), STATIC_TENSOR, [fout],
                              numpy.float32)
            nod.input.append(b.name)
            self.graph.tensormap[b.name] = b

        o = create_tensor(self.get_dynamic_tensor_name(), DYNAMIC_TENSOR, [self.batch, self.seq_len, fout],
                          numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.tensormap[w.name] = w
        self.graph.nodemap[f'Node_{self.node_count}'] = nod
        return o

    def add_mha(self, inps):
        nod = create_node(TmpNodeProto(self.get_node_name(), 'MHA',
                                       {'head_num': self.num_attention_heads, "head_size": self.head_size,
                                        "kv_head_num": self.num_key_value_heads}))
        nod.input = [inp.name for inp in inps]
        o = create_tensor(self.get_dynamic_tensor_name(), DYNAMIC_TENSOR, [self.batch, self.seq_len, self.hidden_size],
                          numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.nodemap[f'Node_{self.node_count}'] = nod
        return o

    def add_slice(self, inp, axis, off, size, step):
        nod = create_node(TmpNodeProto(self.get_node_name(), 'Slice', {}))
        tstart = self.graph.add_initial(self.get_init_tensor_name(), numpy.array([off], dtype=numpy.int64))
        tend = self.graph.add_initial(self.get_init_tensor_name(), numpy.array([off + size], dtype=numpy.int64))
        taxis = self.graph.add_initial(self.get_init_tensor_name(), numpy.array([axis], dtype=numpy.int64))
        tstep = self.graph.add_initial(self.get_init_tensor_name(), numpy.array([step], dtype=numpy.int64))
        nod.input = [inp.name, tstart.name, tend.name, taxis.name, tstep.name]
        s = inp.get_shape()
        s[axis] = size
        o = create_tensor(self.get_dynamic_tensor_name(), DYNAMIC_TENSOR, s, numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.nodemap[f'Node_{self.node_count}'] = nod
        return o

    def add_qkv(self, inp):
        bias = self.arch_config['qkv_bias']
        if self.arch_config['fuse_qkv']:
            qkv = self.add_mm(inp, self.hidden_size, self.hidden_size + self.hidden_kv_size * 2, bias)
            q = self.add_slice(qkv, 2, 0, self.hidden_size, 1)
            k = self.add_slice(qkv, 2, self.hidden_size, self.hidden_kv_size, 1)
            v = self.add_slice(qkv, 2, self.hidden_size + self.hidden_kv_size, self.hidden_kv_size, 1)
        else:
            q = self.add_mm(inp, self.hidden_size, self.hidden_size, bias)
            k = self.add_mm(inp, self.hidden_size, self.hidden_kv_size, bias)
            v = self.add_mm(inp, self.hidden_size, self.hidden_kv_size, bias)
        return [q, k, v]

    def add_mlp(self, inp):
        bias = self.arch_config['mlp_bias']
        if self.model_type == 'phi3' or self.model_type == 'phi3small':
            o0 = self.add_mm(inp, self.hidden_size, self.intermediate_size * 2, bias)
            o0 = self.add_act(o0, 'gegelu')
            s = o0.get_shape()
            s[-1] = s[-1] // 2
            o0.update_shape(s)
            o2 = self.add_mm(o0, self.intermediate_size, self.hidden_size, bias)
        else:
            o0 = self.add_mm(inp, self.hidden_size, self.intermediate_size, bias)
            o0 = self.add_act(o0, self.hidden_act)
            if self.arch_config['ffn_num_mm'] == 3:
                o1 = self.add_mm(inp, self.hidden_size, self.intermediate_size, bias)
                o0 = self.add_eltop(o0, o1, 'Mul')
            o2 = self.add_mm(o0, self.intermediate_size, self.hidden_size, bias)
        return o2

    def add_lm_head(self, inp):
        cur = self.add_layernorm(inp)
        cur = self.add_mm(cur, self.hidden_size, self.vocab_size, self.arch_config['lm_head_bias'])
        return cur

    def add_layers(self, inp):
        for i in range(self.num_hidden_layers):
            cur = inp
            cur = self.add_layernorm(cur)
            q, k, v = self.add_qkv(cur)
            q = self.add_rope(q)
            k = self.add_rope(k)
            cur = self.add_mha([q, k, v])
            cur = self.add_mm(cur, self.hidden_size, self.hidden_size, self.arch_config['o_bias'])
            cur = self.add_eltop(inp, cur, 'Add')
            inp = cur
            cur = self.add_layernorm(cur)
            cur = self.add_mlp(cur)
            inp = self.add_eltop(inp, cur, 'Add')
        return inp

    def build_graph(self, ids_shape: List):
        self.batch, self.seq_len = ids_shape
        ids = create_tensor('ids', DYNAMIC_TENSOR, [self.batch, self.seq_len], numpy.int64)
        self.graph.input.append('ids')
        self.graph.tensormap[ids.name] = ids
        cur = self.add_embedding(ids)
        cur = self.add_layers(cur)
        cur = self.add_lm_head(cur)
        self.graph.output.append(cur.name)

    def save_graph(self, path):
        self.graph.graph_reorder_nodes()
        self.graph.save_model(path, shape_only=True)


phi3_mini = {
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

ArchMap['Phi3ForCausalLM'] = {
    "ffn_num_mm": 2,
    "norm_scale": True,
    "norm_bias": False,
    "fuse_qkv": True,
    "qkv_bias": False,
    "o_bias": False,
    "mlp_bias": False,
    "lm_head_bias": False,
}

QWen_7B = {
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

ArchMap['Qwen2ForCausalLM'] = {
    "ffn_num_mm": 3,
    "norm_scale": True,
    "norm_bias": False,
    "fuse_qkv": False,
    "qkv_bias": True,
    "o_bias": False,
    "mlp_bias": False,
    "lm_head_bias": False,
}

Llama3_8B = {
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

ArchMap['PhiForCausalLM'] = {
    "ffn_num_mm": 2,
    "norm_scale": True,
    "norm_bias": True,
    "fuse_qkv": False,
    "qkv_bias": True,
    "o_bias": True,
    "mlp_bias": False,
    "lm_head_bias": True,
}

Qwen2_72B_Instruct = {
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

ArchMap['Phi3SmallForCausalLM'] = {
    "ffn_num_mm": 2,
    "norm_scale": True,
    "norm_bias": False,
    "fuse_qkv": True,
    "qkv_bias": False,
    "o_bias": False,
    "mlp_bias": False,
    "lm_head_bias": False,
}

gptj_6b = {
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

ArchMap['GPTJForCausalLM'] = {
    "ffn_num_mm": 2,
    "norm_scale": True,
    "norm_bias": True,
    "fuse_qkv": False,
    "qkv_bias": False,
    "o_bias": False,
    "mlp_bias": False,
    "lm_head_bias": False,
}

yi_34B={
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