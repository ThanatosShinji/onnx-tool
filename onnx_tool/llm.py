import numpy

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
        "mlp_gate": 3,
        "norm_scale": True,
        "norm_bias": False,
        "fuse_qkv": False,
        "qkv_bias": False,
        "o_bias": False,
        "mlp_bias": False,
        "lm_head_bias": False,
        "post_mlp_norm": False,
        "post_attn_norm": False
    }
}

WeightMap = {
    'embedding': {
        'embed': 'model.embed_tokens'
    },
    'layer_prefix': 'model.layers.',
    'attention': {
        'input_norm': 'input_layernorm',
        'qkv': 'self_attn.qkv_proj',
        'q': 'self_attn.q_proj',
        'k': 'self_attn.k_proj',
        'v': 'self_attn.v_proj',
        'o': 'self_attn.o_proj',
        'output_norm': 'post_attention_layernorm'
    },
    'mlp': {
        'input_norm': 'post_attention_layernorm',
        'gate': 'mlp.gate_proj',
        'up': 'mlp.up_proj',
        'down': 'mlp.down_proj',
        'gate_up': 'mlp.gate_up_proj',
        'output_norm': 'post_feedforward_layernorm',
    },
    'lm_head': {
        'input_norm': 'model.norm',
        'lm': 'lm_head'
    }
}


class Builder():
    def __init__(self, **kwargs):
        for k in kwargs.keys():
            newk = k
            newv = kwargs[k]
            # print(k, kwargs[k])
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
            if k == '_name_or_path':
                newk = 'name'
            setattr(self, newk, newv)
        if kwargs['architectures'][0] == 'GPT2LMHeadModel':
            if not hasattr(self, 'intermediate_size'):
                setattr(self, 'intermediate_size', self.hidden_size * 4)
            if not hasattr(self, 'n_positions'):
                setattr(self, 'n_positions', 1024)
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

    def get_filename(self):
        name = self.name
        name = name.replace('/','_')
        return name

    def add_embedding(self, inp, insize, name):
        emd = create_node(TmpNodeProto(name, 'Gather', {'axis': 0}))
        w = create_tensor(name + '.weight', STATIC_TENSOR, [insize, self.hidden_size],
                          numpy.float32)
        self.graph.initials.append(w.name)
        emd.input = [w.name, inp.name]
        o = create_tensor(name + '.output', DYNAMIC_TENSOR, [self.batch, self.seq_len, self.hidden_size],
                          numpy.float32)
        emd.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.tensormap[w.name] = w
        self.graph.nodemap[emd.name] = emd
        return o

    def add_layernorm(self, inp, name):
        if hasattr(self, 'rms_norm_eps'):
            attrs = {'epsilon': self.rms_norm_eps, 'type': 'rms'}
        elif hasattr(self, 'layer_norm_eps'):
            attrs = {'epsilon': self.layer_norm_eps}
        else:
            assert 0
        nod = create_node(TmpNodeProto(name, 'LayerNormalization', attrs))
        nod.input = [inp.name]
        if self.arch_config['norm_scale']:
            s = create_tensor(name + '.weight', STATIC_TENSOR, [self.hidden_size],
                              numpy.float32)
            self.graph.initials.append(s.name)
            nod.input.append(s.name)
            self.graph.tensormap[s.name] = s
            if self.arch_config['norm_bias']:
                b = create_tensor(name + '.bias', STATIC_TENSOR, [self.hidden_size],
                                  numpy.float32)
                nod.input.append(b.name)
                self.graph.initials.append(b.name)
                self.graph.tensormap[b.name] = b

        o = create_tensor(name + '.output', DYNAMIC_TENSOR, inp.get_shape(),
                          numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.nodemap[nod.name] = nod
        return o

    def add_act(self, inp, act, name):
        nod = create_node(TmpNodeProto(name, ActMap[act], {}))
        nod.input = [inp.name]
        o = create_tensor(name + '.output', DYNAMIC_TENSOR, inp.get_shape(),
                          numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.nodemap[nod.name] = nod
        return o

    def add_rope(self, inp, name):
        nod = create_node(TmpNodeProto(name, 'Rope',
                                       {'rope_theta': self.rope_theta if hasattr(self,
                                                                                 'rope_theta') and self.rope_theta is not None else 0,
                                        'rope_scaling': self.rope_scaling if hasattr(self,
                                                                                     'rope_scaling') and self.rope_scaling is not None else 0}))
        nod.input = [inp.name]
        o = create_tensor(name + '.output', DYNAMIC_TENSOR, inp.get_shape(),
                          numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.nodemap[nod.name] = nod
        return o

    def add_eltop(self, inp, inp1, op, name):
        nod = create_node(TmpNodeProto(name, op, {}))
        nod.input = [inp.name, inp1.name]
        s = inp.get_shape()
        s1 = inp1.get_shape()
        s = _max_shape([s, s1])
        o = create_tensor(name + '.output', DYNAMIC_TENSOR, s,
                          numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.nodemap[nod.name] = nod
        return o

    def add_mm(self, inp, fin, fout, bias, name):
        nod = create_node(TmpNodeProto(name, 'MatMul', {}))
        w = create_tensor(name + '.weight', STATIC_TENSOR, [fin, fout],
                          numpy.float32)
        self.graph.initials.append(w.name)
        nod.input = [inp.name, w.name]

        if bias:
            b = create_tensor(name + '.bias', STATIC_TENSOR, [fout],
                              numpy.float32)
            nod.input.append(b.name)
            self.graph.initials.append(b.name)
            self.graph.tensormap[b.name] = b

        o = create_tensor(name + '.output', DYNAMIC_TENSOR, [self.batch, self.seq_len, fout],
                          numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.tensormap[w.name] = w
        self.graph.nodemap[nod.name] = nod
        return o

    def add_mha(self, inps, name):
        attrs = {'head_num': self.num_attention_heads, "head_size": self.head_size,
                 "kv_head_num": self.num_key_value_heads}
        attrs['layer_i'] = self.layer_i
        if getattr(self, 'attn_logit_softcapping', None) is not None:
            attrs['attn_logit_softcapping'] = self.attn_logit_softcapping
        nod = create_node(TmpNodeProto(name, 'MHA', attrs))
        nod.input = [inp.name for inp in inps]
        o = create_tensor(name + '.output', DYNAMIC_TENSOR, [self.batch, self.seq_len, self.hidden_size],
                          numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.nodemap[nod.name] = nod
        return o

    def add_slice(self, inp, axis, off, size, step, name):
        nod = create_node(TmpNodeProto(name, 'Slice', {}))
        tstart = self.graph.add_initial(name + '.start', numpy.array([off], dtype=numpy.int64))
        tend = self.graph.add_initial(name + '.end', numpy.array([off + size], dtype=numpy.int64))
        taxis = self.graph.add_initial(name + '.axis', numpy.array([axis], dtype=numpy.int64))
        tstep = self.graph.add_initial(name + '.step', numpy.array([step], dtype=numpy.int64))
        nod.input = [inp.name, tstart.name, tend.name, taxis.name, tstep.name]
        s = inp.get_shape()
        s[axis] = size
        o = create_tensor(name + '.output', DYNAMIC_TENSOR, s, numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.nodemap[nod.name] = nod
        return o

    def add_qkv(self, inp):
        bias = self.arch_config['qkv_bias']
        if self.arch_config['fuse_qkv']:
            qkv = self.add_mm(inp, self.hidden_size, self.hidden_size + self.hidden_kv_size * 2, bias,
                              self.layer_prefix + self.w_map['attention']['qkv'])
            q = self.add_slice(qkv, 2, 0, self.hidden_size, 1, self.layer_prefix + 'q.slice')
            k = self.add_slice(qkv, 2, self.hidden_size, self.hidden_kv_size, 1, self.layer_prefix + 'k.slice')
            v = self.add_slice(qkv, 2, self.hidden_size + self.hidden_kv_size, self.hidden_kv_size, 1,
                               self.layer_prefix + 'v.slice')
        else:
            nameq = self.layer_prefix + self.w_map['attention']['q']
            namek = self.layer_prefix + self.w_map['attention']['k']
            namev = self.layer_prefix + self.w_map['attention']['v']
            q = self.add_mm(inp, self.hidden_size, self.hidden_size, bias, nameq)
            k = self.add_mm(inp, self.hidden_size, self.hidden_kv_size, bias, namek)
            v = self.add_mm(inp, self.hidden_size, self.hidden_kv_size, bias, namev)
        return [q, k, v]

    def add_mlp(self, inp):
        bias = self.arch_config['mlp_bias']
        mlp_gate = self.arch_config['mlp_gate']
        nameup = self.layer_prefix + self.w_map['mlp']['up']
        namedown = self.layer_prefix + self.w_map['mlp']['down']
        if mlp_gate:
            namegate = self.layer_prefix + self.w_map['mlp']['gate']
        if self.model_type == 'phi3' or self.model_type == 'phi3small':
            name_gu = self.layer_prefix + self.w_map['mlp']['gate_up']
            o0 = self.add_mm(inp, self.hidden_size, self.intermediate_size * 2, bias, name_gu)
            o0 = self.add_act(o0, 'gegelu', self.layer_prefix + 'mlp.activation')
            s = o0.get_shape()
            s[-1] = s[-1] // 2
            o0.update_shape(s)
            o2 = self.add_mm(o0, self.intermediate_size, self.hidden_size, bias, namedown)
        else:
            o0 = self.add_mm(inp, self.hidden_size, self.intermediate_size, bias, nameup)
            o0 = self.add_act(o0, self.hidden_act, self.layer_prefix + 'mlp.activation')
            if mlp_gate:
                o1 = self.add_mm(inp, self.hidden_size, self.intermediate_size, bias, namegate)
                o0 = self.add_eltop(o0, o1, 'Mul', self.layer_prefix + 'mlp.gate_mul')
            o2 = self.add_mm(o0, self.intermediate_size, self.hidden_size, bias, namedown)
        return o2

    def add_lm_head(self, inp):
        cur = self.add_layernorm(inp, self.w_map['lm_head']['input_norm'])
        cur = self.add_mm(cur, self.hidden_size, self.vocab_size, self.arch_config['lm_head_bias'],
                          self.w_map['lm_head']['lm'])
        return cur

    def add_layers(self, inp):
        for i in range(self.num_hidden_layers):
            self.layer_i = i
            self.layer_prefix = self.w_map['layer_prefix'] + str(self.layer_i) + '.'
            cur = inp
            cur = self.add_layernorm(cur, self.layer_prefix + self.w_map['attention']['input_norm'])
            q, k, v = self.add_qkv(cur)
            if self.arch_config.get('qk_rope', True):
                q = self.add_rope(q, self.layer_prefix + 'rope_q')
                k = self.add_rope(k, self.layer_prefix + 'rope_k')
            cur = self.add_mha([q, k, v], self.layer_prefix + 'mha')
            cur = self.add_mm(cur, self.hidden_size, self.hidden_size, self.arch_config['o_bias'],
                              self.layer_prefix + self.w_map['attention']['o'])
            if self.arch_config.get('post_attn_norm', False):
                cur = self.add_layernorm(cur, self.layer_prefix + self.w_map['attention']['output_norm'])
            cur = self.add_eltop(inp, cur, 'Add', self.layer_prefix + 'attention_add')
            inp = cur
            cur = self.add_layernorm(cur, self.layer_prefix + self.w_map['mlp']['input_norm'])
            cur = self.add_mlp(cur)
            if self.arch_config.get('post_mlp_norm', False):
                cur = self.add_layernorm(cur, self.layer_prefix + self.w_map['mlp']['output_norm'])
            inp = self.add_eltop(inp, cur, 'Add', self.layer_prefix + 'mlp_add')
        return inp

    def add_softcapping(self, inp, value, name):
        b = self.graph.add_initial(name + '.weight', numpy.array(value, dtype=numpy.float32))
        nod = create_node(TmpNodeProto(name, 'LogitSoftCapping', {}))
        nod.input = [inp.name, b.name]
        o = create_tensor(name + '.output', DYNAMIC_TENSOR, inp.get_shape(), numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.nodemap[nod.name] = nod
        return o

    DefaultCfg = {
        'Compute': {
            'MM': 'FP32',
            'MHA': 'FP32',
            'Others': 'FP32',
        },
        'Bits': {
            'MM': 4.5,
            'MHA': 16,
            'Others': 32,
        }
    }

    def profile(self, Config: {} = None, Device: {} = None):
        self.graph.valid_shape = True
        self.graph.profile()
        cfg = Config if Config is not None else self.DefaultCfg
        if Device is not None:
            c_mm = Device.get(cfg['Compute']['MM'], Device['FP32']) * 1e6
            c_mha = Device.get(cfg['Compute']['MHA'], Device['FP32']) * 1e6
            c_others = Device.get(cfg['Compute']['Others'], Device['FP32'])* 1e6
            mw = Device.get('Bandwidth') * 1e6
        self.graph.Device = Device
        sum = [0, 0, 0]
        for n in self.graph.nodemap.keys():
            node = self.graph.nodemap[n]
            flops = node.macs[0] * 2
            mem = 0
            c_latency = 0
            l_latency = 0
            if node.op_type == 'MatMul':
                mem += volume(self.graph.tensormap[node.input[0]].get_shape())
                mem += volume(self.graph.tensormap[node.output[0]].get_shape())
                mem = mem * cfg['Bits']['Others'] / 8
                mem += volume(self.graph.tensormap[node.input[1]].get_shape()) * cfg['Bits']['MM'] / 8
                if Device is not None:
                    c_latency = flops / c_mm
                    l_latency = mem / mw
            elif node.op_type == 'MHA':
                mem += volume(self.graph.tensormap[node.input[0]].get_shape())
                mem += volume(self.graph.tensormap[node.output[0]].get_shape())
                mem = mem * cfg['Bits']['Others'] / 8
                mem += volume(self.graph.tensormap[node.input[1]].get_shape()) * cfg['Bits']['MHA'] / 8
                mem += volume(self.graph.tensormap[node.input[2]].get_shape()) * cfg['Bits']['MHA'] / 8
                if Device is not None:
                    c_latency = flops / c_mha
                    l_latency = mem / mw
            else:
                if node.op_type == 'Gather':
                    mem = 0
                else:
                    for inp in node.input:
                        mem += volume(self.graph.tensormap[inp].get_shape())
                mem += volume(self.graph.tensormap[node.output[0]].get_shape())
                mem = mem * cfg['Bits']['Others'] / 8
                if Device is not None:
                    c_latency = flops / c_others
                    l_latency = mem / mw
            sum[0] += flops
            sum[1] += mem
            llm_profile = {}
            llm_profile['FLOPs']=flops
            llm_profile['Memory']=mem
            llm_profile['Device']=None
            if Device is not None:
                n_latency = max(c_latency, l_latency)
                bottle = 'Compute' if c_latency > l_latency else 'Memory'
                sum[2] += n_latency
                llm_profile['Device'] = {'latency':[c_latency, l_latency, n_latency],'Bottleneck':bottle}
            node.llm_profile = llm_profile

        self.graph.llm_profile = sum

    def print_profile(self, f=None):
        from .utils import tuple2str, print_table, num2str

        metric = 'FLOPs'
        splitch = 'x'
        if f is not None and '.csv' in f:
            csvformat = True
        else:
            csvformat = False
        ptable = []
        for n in self.graph.nodemap.keys():
            node = self.graph.nodemap[n]
            row = [n, node.op_type]
            row.append(num2str(int(node.llm_profile['FLOPs']), csvformat))
            row.append(num2str(int(node.llm_profile['Memory']), csvformat))
            Device = node.llm_profile['Device']
            if Device is not None:
                row.append('{:.5f}'.format(Device['latency'][2]))
                row.append(Device['Bottleneck'])
            row.append(tuple2str(node.inshape, splitch))
            row.append(tuple2str(node.outshape, splitch))
            ptable.append(row)

        row = ['Total', '_']
        row.append(num2str(self.graph.llm_profile[0], csvformat))
        row.append(num2str(self.graph.llm_profile[1], csvformat))
        if self.graph.Device is not None:
            row.append(num2str(self.graph.llm_profile[2], csvformat))
            row.append('_')
        row.append('_')
        row.append('_')
        ptable.append(row)
        header = ['Name', 'Type']
        header.extend(
            ['Forward_' + metric, 'Memory Bytes'])
        if self.graph.Device is not None:
            header.extend(['Latency(ms)', 'Bottleneck'])
        header.extend(
            ['InShape',
             'OutShape'])
        print_table(ptable,header,f)


    def build_graph(self, ids_shape: List, weight_map: {} = None):
        self.batch, self.seq_len = ids_shape
        self.w_map = weight_map if weight_map is not None else WeightMap
        self.kv_params = self.batch * self.seq_len * self.hidden_kv_size * 2 * self.num_hidden_layers
        ids = create_tensor('ids', DYNAMIC_TENSOR, [self.batch, self.seq_len], numpy.int64)
        self.graph.input.append('ids')
        self.graph.tensormap[ids.name] = ids

        cur = self.add_embedding(ids, self.vocab_size, self.w_map['embedding']['embed'])
        if self.arch_config.get('pos_embedding', False):
            pos = create_tensor('position', DYNAMIC_TENSOR, [self.batch, self.seq_len], numpy.int64)
            self.graph.input.append('position')
            self.graph.tensormap[pos.name] = pos
            pos_out = self.add_embedding(pos, self.n_positions, self.w_map['embedding']['pos'])
            cur = self.add_eltop(pos_out, cur, 'Add', 'embedding_pos_add')

        cur = self.add_layers(cur)
        cur = self.add_lm_head(cur)
        if getattr(self, 'final_logit_softcapping', None) is not None:
            cur = self.add_softcapping(cur, self.final_logit_softcapping, 'models.final_logit_softcapping')
        self.graph.output.append(cur.name)

    def add_kv_cache(self, n_context, n_past):
        t_n_past = create_tensor('n_past', DYNAMIC_TENSOR, [self.batch], numpy.int64)
        t_n_past.update_tensor(numpy.array(n_past, dtype=numpy.int64))
        self.graph.input.append('n_past')
        self.graph.tensormap[t_n_past.name] = t_n_past
        self.kv_params = self.batch * (self.seq_len + n_past) * self.hidden_kv_size * 2 * self.num_hidden_layers
        kv_cache = create_tensor('kv_cache', DYNAMIC_TENSOR,
                                 [self.batch, self.num_hidden_layers, n_context, self.hidden_kv_size],
                                 numpy.float32)
        self.graph.input.append('kv_cache')
        self.graph.tensormap[kv_cache.name] = kv_cache
        for n in self.graph.nodemap:
            if self.graph.nodemap[n].op_type == 'MHA':
                self.graph.nodemap[n].input.append('n_past')
                self.graph.nodemap[n].input.append('kv_cache')

    def save_graph(self, path):
        self.graph.graph_reorder_nodes()
        self.graph.save_model(path, shape_only=False)


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

ArchMap['Phi3ForCausalLM'] = {
    "mlp_gate": False,
    "norm_scale": True,
    "norm_bias": False,
    "fuse_qkv": True,
    "qkv_bias": False,
    "o_bias": False,
    "mlp_bias": False,
    "lm_head_bias": False,
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

ArchMap['Qwen2ForCausalLM'] = {
    "mlp_gate": True,
    "norm_scale": True,
    "norm_bias": False,
    "fuse_qkv": False,
    "qkv_bias": True,
    "o_bias": False,
    "mlp_bias": False,
    "lm_head_bias": False,
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

ArchMap['PhiForCausalLM'] = {
    "mlp_gate": False,
    "norm_scale": True,
    "norm_bias": True,
    "fuse_qkv": False,
    "qkv_bias": True,
    "o_bias": True,
    "mlp_bias": False,
    "lm_head_bias": True,
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

ArchMap['Phi3SmallForCausalLM'] = {
    "mlp_gate": False,
    "norm_scale": True,
    "norm_bias": False,
    "fuse_qkv": True,
    "qkv_bias": False,
    "o_bias": False,
    "mlp_bias": False,
    "lm_head_bias": False,
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

ArchMap['GPTJForCausalLM'] = {
    "mlp_gate": False,
    "norm_scale": True,
    "norm_bias": True,
    "fuse_qkv": False,
    "qkv_bias": False,
    "o_bias": False,
    "mlp_bias": False,
    "lm_head_bias": False,
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

ArchMap['GPT2LMHeadModel'] = {
    "mlp_gate": False,
    "norm_scale": True,
    "norm_bias": True,
    "fuse_qkv": True,
    "qkv_bias": True,
    "o_bias": True,
    "mlp_bias": True,
    "lm_head_bias": False,
    'qk_rope': False,
    'pos_embedding': True
}

llama2_7b={
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