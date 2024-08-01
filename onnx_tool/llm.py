from .graph import Graph
from .node import *
from .tensor import *

false = False
true = True
null = None

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
    "vocab_size": 152064,
    "arch_config": {
        "ffn_num_mm": 3,
        "norm_scale": True,
        "norm_bias": False
    }
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
    "vocab_size": 128256,
    "arch_config": {
        "ffn_num_mm": 3,
        "norm_scale": True,
        "norm_bias": False
    }
}

ActMap = {
    'silu': 'Silu'
}


class Builder():
    def __init__(self, **kwargs):
        for k in kwargs.keys():
            setattr(self, k, kwargs[k])
            print(k, self.__getattribute__(k))
        self.ffn_num_mm = self.arch_config['ffn_num_mm']
        self.norm_scale = self.arch_config['norm_scale']
        self.norm_bias = self.arch_config['norm_bias']
        self.graph = Graph(None, None)
        self.node_count = 0
        self.tensor_count = 0
        self.head_size = self.hidden_size // self.num_attention_heads
        self.hidden_kv_size = self.head_size * self.num_key_value_heads

    def get_init_tensor_name(self):
        name = f'InitTensor_{self.tensor_count}'
        self.tensor_count += 1
        return name

    def get_dynamic_tensor_name(self):
        name = f'DynaTensor_{self.tensor_count}'
        self.tensor_count += 1
        return name

    def get_node_name(self):
        name = f'Node_{self.node_count}'
        self.node_count += 1
        return name

    def add_embedding(self, inp):
        emd = Node(TmpNodeProto(self.get_node_name(), 'Gather', {'axis': 0}))
        w = create_tensor(self.get_init_tensor_name(), STATIC_TENSOR, [self.vocab_size, self.hidden_size],
                          numpy.float32)
        emd.input = [w.name, inp.name]
        o = create_tensor(self.get_dynamic_tensor_name(), DYNAMIC_TENSOR, [self.batch, self.seq_len, self.hidden_size],
                          numpy.float32)
        emd.output = [o.name]
        self.graph.initials.append(w.name)
        self.graph.tensormap[o.name] = o
        self.graph.tensormap[w.name] = w
        self.graph.nodemap[f'Node_{self.node_count}'] = emd
        return o

    def add_layernorm(self, inp):
        nod = Node(TmpNodeProto(self.get_node_name(), 'LayerNormalization', {'epsilon': self.rms_norm_eps}))
        nod.input = [inp.name]
        if self.norm_scale:
            s = create_tensor(self.get_init_tensor_name(), STATIC_TENSOR, [self.hidden_size],
                              numpy.float32)
            nod.input.append(s.name)
            self.graph.initials.append(s.name)
            self.graph.tensormap[s.name] = s
            if self.norm_bias:
                b = create_tensor(self.get_init_tensor_name(), STATIC_TENSOR, [self.hidden_size],
                                  numpy.float32)
                nod.input.append(b.name)
                self.graph.initials.append(b.name)
                self.graph.tensormap[b.name] = b

        o = create_tensor(self.get_dynamic_tensor_name(), DYNAMIC_TENSOR, [self.batch, self.seq_len, self.hidden_size],
                          numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.nodemap[f'Node_{self.node_count}'] = nod
        return o

    def add_act(self, inp, act):
        nod = Node(TmpNodeProto(self.get_node_name(), ActMap[act], {}))
        nod.input = [inp.name]
        o = create_tensor(self.get_dynamic_tensor_name(), DYNAMIC_TENSOR, [self.batch, self.seq_len, self.hidden_size],
                          numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.nodemap[f'Node_{self.node_count}'] = nod
        return o

    def add_eltop(self, inp, inp1, op):
        nod = Node(TmpNodeProto(self.get_node_name(), op, {}))
        nod.input = [inp.name, inp1.name]
        o = create_tensor(self.get_dynamic_tensor_name(), DYNAMIC_TENSOR, [self.batch, self.seq_len, self.hidden_size],
                          numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.nodemap[f'Node_{self.node_count}'] = nod
        return o

    def add_mm(self, inp, fin, fout):
        nod = Node(TmpNodeProto(self.get_node_name(), 'MatMul', {}))
        w = create_tensor(self.get_init_tensor_name(), STATIC_TENSOR, [fin, fout],
                          numpy.float32)
        nod.input = [inp.name, w.name]
        o = create_tensor(self.get_dynamic_tensor_name(), DYNAMIC_TENSOR, [self.batch, self.seq_len, fout],
                          numpy.float32)
        nod.output = [o.name]
        self.graph.initials.append(w.name)
        self.graph.tensormap[o.name] = o
        self.graph.tensormap[w.name] = w
        self.graph.nodemap[f'Node_{self.node_count}'] = nod
        return o

    def add_mha(self, inps):
        nod = Node(TmpNodeProto(self.get_node_name(), 'MHA',
                                {'head_num': self.num_attention_heads, "head_size": self.head_size,
                                 "kv_head_num": self.num_key_value_heads}))
        nod.input = [inp.name for inp in inps]
        o = create_tensor(self.get_dynamic_tensor_name(), DYNAMIC_TENSOR, [self.batch, self.seq_len, self.hidden_size],
                          numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.nodemap[f'Node_{self.node_count}'] = nod
        return o

    def add_qkv(self, inp):
        q = self.add_mm(inp, self.hidden_size, self.hidden_size)
        k = self.add_mm(inp, self.hidden_size, self.hidden_kv_size)
        v = self.add_mm(inp, self.hidden_size, self.hidden_kv_size)
        return [q, k, v]

    def add_ffn(self, inp):
        o0 = self.add_mm(inp, self.hidden_size, self.intermediate_size)
        o0 = self.add_act(o0, self.hidden_act)
        if self.ffn_num_mm == 3:
            o1 = self.add_mm(inp, self.hidden_size, self.intermediate_size)
            o0 = self.add_eltop(o0, o1, 'Mul')
        o2 = self.add_mm(o0, self.intermediate_size, self.hidden_size)
        return o2

    def add_lm_head(self, inp):
        cur = self.add_layernorm(inp)
        cur = self.add_mm(cur, self.hidden_size, self.vocab_size)
        return cur

    def add_layers(self, inp):
        for i in range(self.num_hidden_layers):
            cur = inp
            cur = self.add_layernorm(cur)
            q, k, v = self.add_qkv(cur)
            cur = self.add_mha([q, k, v])
            cur = self.add_mm(cur, self.hidden_size, self.hidden_size)
            cur = self.add_eltop(inp, cur, 'Add')
            inp = cur
            cur = self.add_layernorm(cur)
            cur = self.add_ffn(cur)
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
