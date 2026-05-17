import numpy

from .graph import Graph
from .node import *
from .node import _max_shape
from .tensor import *
from .utils import ModelConfig, tuple2str, print_table, num2str

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
        "mlp_gate": True,
        "norm_scale": True,
        "norm_bias": False,
        "fuse_qkv": False,
        "qkv_bias": False,
        "o_bias": False,
        "mlp_bias": False,
        "lm_head_bias": False,
        "post_mlp_norm": False,
        "post_attn_norm": False
    },
    'Qwen2ForCausalLM': {
        # Qwen2Attention: q_proj/k_proj/v_proj bias=True, o_proj bias=False
        # Qwen2MLP: gate_proj/up_proj/down_proj bias=False
        # No qk_norm (Q/K per-head norm is Qwen3 only)
        "mlp_gate": True,
        "norm_scale": True,
        "norm_bias": False,
        "fuse_qkv": False,
        "qkv_bias": True,
        "o_bias": False,
        "mlp_bias": False,
        "lm_head_bias": False,
    },
    'Qwen3ForCausalLM': {
        # Qwen3Attention: bias controlled by config.attention_bias
        # Has q_norm/k_norm (per-head RMSNorm)
        "mlp_gate": True,
        "norm_scale": True,
        "norm_bias": False,
        "fuse_qkv": False,
        "qkv_bias": True,
        "o_bias": False,
        "mlp_bias": False,
        "lm_head_bias": False,
        "qk_norm": True,
    },
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
            if k == '_name_or_path' and not hasattr(self, 'name'):
                newk = 'name'
            setattr(self, newk, newv)
        if kwargs['architectures'][0] == 'GPT2LMHeadModel':
            if not hasattr(self, 'intermediate_size'):
                setattr(self, 'intermediate_size', self.hidden_size * 4)
            if not hasattr(self, 'n_positions'):
                setattr(self, 'n_positions', 1024)
        if not hasattr(self, 'model_type'):
            setattr(self, 'model_type', '')
        if ArchMap.__contains__(self.architectures[0]):
            self.arch_config = ArchMap[self.architectures[0]]
        else:
            self.arch_config = ArchMap['LlamaForCausalLM']
        if hasattr(self, 'mlp_bias'):
            self.arch_config['mlp_bias'] = self.mlp_bias
        if hasattr(self, 'attention_bias'):
            self.arch_config['qkv_bias'] = self.attention_bias
            self.arch_config['o_bias'] = self.attention_bias
        self.graph = Graph(None, ModelConfig())
        self.node_count = 0
        self.tensor_count = 0
        # head_size: 优先使用配置中的 head_dim，否则用 hidden_size / num_heads
        if hasattr(self, 'head_dim'):
            self.head_size = self.head_dim
        else:
            self.head_size = self.hidden_size // self.num_attention_heads
        self.hidden_kv_size = self.head_size * self.num_key_value_heads
        self.device_perf = []

    def get_filename(self):
        name = self.name
        name = name.replace('/', '_')
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
        # 全局共享的 cos/sin constant tensor（所有 Rope 节点共用）
        if not hasattr(self, '_rope_cos_sin_added'):
            max_pos = getattr(self, 'max_position_embeddings', 4096)
            head_dim = self.head_size
            half = head_dim // 2
            theta = getattr(self, 'rope_theta', 10000.0)

            pos = numpy.arange(max_pos, dtype=numpy.float32)
            dim_idx = numpy.arange(half, dtype=numpy.float32)
            freq = 1.0 / (theta ** (2 * dim_idx / head_dim))
            angles = pos.reshape(-1, 1) * freq.reshape(1, -1)
            cos = numpy.cos(angles).reshape(1, 1, max_pos, half).astype(numpy.float32)
            sin = numpy.sin(angles).reshape(1, 1, max_pos, half).astype(numpy.float32)

            t_cos = create_tensor('rope.cos', STATIC_TENSOR, list(cos.shape), numpy.float32)
            t_cos.update_tensor(cos)
            t_sin = create_tensor('rope.sin', STATIC_TENSOR, list(sin.shape), numpy.float32)
            t_sin.update_tensor(sin)
            self.graph.initials.append('rope.cos')
            self.graph.initials.append('rope.sin')
            self.graph.tensormap['rope.cos'] = t_cos
            self.graph.tensormap['rope.sin'] = t_sin
            self._rope_cos_sin_added = True

        # position 输入（每个 token 的位置索引，支持 KV cache 偏移）
        if not hasattr(self, '_position_added'):
            pos_tensor = create_tensor('position', DYNAMIC_TENSOR, [self.batch, self.seq_len], numpy.int64)
            self.graph.input.append('position')
            self.graph.tensormap['position'] = pos_tensor
            self._position_added = True

        nod = create_node(TmpNodeProto(name, 'Rope', {}))
        nod.input = [inp.name, 'rope.cos', 'rope.sin', 'position']
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
        nod = create_node(TmpNodeProto(name, 'SDPA', attrs))
        nod.input = [inp.name for inp in inps]
        # SDPA 输出维度: num_attention_heads * head_dim（可能不同于 hidden_size）
        attn_out = self.num_attention_heads * self.head_size
        o = create_tensor(name + '.output', DYNAMIC_TENSOR, [self.batch, self.seq_len, attn_out],
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
        # Q/K 输出维度：当 head_dim 显式配置且与 hidden_size/num_heads 不同时使用
        q_out = self.num_attention_heads * self.head_size
        k_out = self.num_key_value_heads * self.head_size
        v_out = self.num_key_value_heads * self.head_size
        if self.arch_config['fuse_qkv']:
            qkv = self.add_mm(inp, self.hidden_size, q_out + k_out + v_out, bias,
                              self.layer_prefix + self.w_map['attention']['qkv'])
            q = self.add_slice(qkv, 2, 0, q_out, 1, self.layer_prefix + 'q.slice')
            k = self.add_slice(qkv, 2, q_out, k_out, 1, self.layer_prefix + 'k.slice')
            v = self.add_slice(qkv, 2, q_out + k_out, v_out, 1,
                               self.layer_prefix + 'v.slice')
        else:
            nameq = self.layer_prefix + self.w_map['attention']['q']
            namek = self.layer_prefix + self.w_map['attention']['k']
            namev = self.layer_prefix + self.w_map['attention']['v']
            q = self.add_mm(inp, self.hidden_size, q_out, bias, nameq)
            k = self.add_mm(inp, self.hidden_size, k_out, bias, namek)
            v = self.add_mm(inp, self.hidden_size, v_out, bias, namev)
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
            o_up = self.add_mm(inp, self.hidden_size, self.intermediate_size, bias, nameup)
            if mlp_gate:
                o_gate = self.add_mm(inp, self.hidden_size, self.intermediate_size, bias, namegate)
                o_act = self.add_act(o_gate, self.hidden_act, self.layer_prefix + 'mlp.activation')
                o0 = self.add_eltop(o_up, o_act, 'Mul', self.layer_prefix + 'mlp.gate_mul')
            else:
                o0 = self.add_act(o_up, self.hidden_act, self.layer_prefix + 'mlp.activation')
            o2 = self.add_mm(o0, self.intermediate_size, self.hidden_size, bias, namedown)
        return o2

    def add_lm_head(self, inp):
        cur = self.add_layernorm(inp, self.w_map['lm_head']['input_norm'])
        cur = self.add_mm(cur, self.hidden_size, self.vocab_size, self.arch_config['lm_head_bias'],
                          self.w_map['lm_head']['lm'])
        return cur

    def add_qk_norm(self, inp, name, num_heads):
        """Per-head Q/K RMS norm (Qwen3 特有).
        
        输入: [B, S, num_heads * head_dim]
        输出: [B, S, num_heads * head_dim]
        对每个 head 独立做 RMS norm，weight shape = [head_dim]
        """
        head_dim = self.head_size
        # Reshape: [B, S, num_heads * head_dim] -> [B, S, num_heads, head_dim]
        # 做 LayerNormalization（沿最后一个维度，即 head_dim）
        attrs = {'epsilon': self.rms_norm_eps, 'type': 'rms'}
        nod = create_node(TmpNodeProto(name, 'LayerNormalization', attrs))
        nod.input = [inp.name]
        # scale weight: [head_dim]（每个 head 共享）
        s = create_tensor(name + '.weight', STATIC_TENSOR, [head_dim], numpy.float32)
        self.graph.initials.append(s.name)
        nod.input.append(s.name)
        self.graph.tensormap[s.name] = s
        o = create_tensor(name + '.output', DYNAMIC_TENSOR, inp.get_shape(), numpy.float32)
        nod.output = [o.name]
        self.graph.tensormap[o.name] = o
        self.graph.nodemap[nod.name] = nod
        return o

    def add_layers(self, inp):
        for i in range(self.num_hidden_layers):
            self.layer_i = i
            self.layer_prefix = self.w_map['layer_prefix'] + str(self.layer_i) + '.'
            cur = inp
            cur = self.add_layernorm(cur, self.layer_prefix + self.w_map['attention']['input_norm'])
            q, k, v = self.add_qkv(cur)
            # Q/K per-head norm (Qwen3 特有)
            if self.arch_config.get('qk_norm', False):
                q = self.add_qk_norm(q, self.layer_prefix + 'self_attn.q_norm', self.num_attention_heads)
                k = self.add_qk_norm(k, self.layer_prefix + 'self_attn.k_norm', self.num_key_value_heads)
            if self.arch_config.get('qk_rope', True):
                q = self.add_rope(q, self.layer_prefix + 'rope_q')
                k = self.add_rope(k, self.layer_prefix + 'rope_k')
            cur = self.add_mha([q, k, v], self.layer_prefix + 'mha')
            attn_out = self.num_attention_heads * self.head_size
            cur = self.add_mm(cur, attn_out, self.hidden_size, self.arch_config['o_bias'],
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
            link_bw = Device.get('LinkBandwidth', 0) * 1e6
            d_num = 1 if link_bw == 0 else Device.get('Number', 1)
            c_mm = Device.get(cfg['Compute']['MM'], Device['FP32']) * 1e6
            c_mha = Device.get(cfg['Compute']['MHA'], Device['FP32']) * 1e6
            c_others = Device.get(cfg['Compute']['Others'], Device['FP32']) * 1e6
            mw = Device.get('Bandwidth') * 1e6
        sum = [0, 0, 0]
        MM_mem = 0
        Other_mem = 0
        MHA_mem = 0
        for n in self.graph.nodemap.keys():
            node = self.graph.nodemap[n]
            flops = node.macs[0] * 2
            mem = 0
            c_latency = 0
            l_latency = 0
            comm_mem = 0  # TP for MatMul and MHA only, column split
            if node.op_type == 'MatMul':
                mem += volume(self.graph.tensormap[node.input[0]].get_shape())
                mem += volume(self.graph.tensormap[node.output[0]].get_shape())
                mem = mem * cfg['Bits']['Others'] / 8
                comm_mem += mem
                w_mem = volume(self.graph.tensormap[node.input[1]].get_shape()) * cfg['Bits']['MM'] / 8
                mem += w_mem
                MM_mem += w_mem
                if Device is not None:
                    c_latency = flops / c_mm / d_num
                    l_latency = mem / mw / d_num
            elif node.op_type == 'SDPA':
                mem += volume(self.graph.tensormap[node.input[0]].get_shape())
                mem += volume(self.graph.tensormap[node.output[0]].get_shape())
                mem = mem * cfg['Bits']['Others'] / 8
                comm_mem += mem
                kv_mem = node.kv_size * cfg['Bits']['MHA'] / 8
                mem += kv_mem
                MHA_mem += kv_mem
                if Device is not None:
                    c_latency = flops / c_mha / d_num
                    l_latency = mem / mw / d_num
            else:
                tmp_sum = 0
                for inp in node.input:
                    if self.graph.tensormap[inp].type == STATIC_TENSOR:
                        tmp_sum += volume(self.graph.tensormap[inp].get_shape())
                Other_mem += tmp_sum * cfg['Bits']['Others'] / 8
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
            llm_profile = {'FLOPs': flops, 'Memory': mem, 'Device': None}
            if Device is not None:
                if d_num > 1:
                    sync_latency = comm_mem / link_bw
                else:
                    sync_latency = 0
                n_latency = max(c_latency, l_latency) + sync_latency
                bottle = 'Compute' if c_latency > l_latency else 'Memory'
                sum[2] += n_latency
                llm_profile['Device'] = {'latency': [c_latency, l_latency, n_latency, sync_latency], 'Bottleneck': bottle}
            node.llm_profile = llm_profile
        self.llm_profile = sum
        self.context_mem = [MM_mem, MHA_mem, Other_mem, MM_mem + MHA_mem + Other_mem]

    def print_profile(self, f=None):
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
            if self.llm_profile[2] > 0:
                row.append('{:.5f}'.format(Device['latency'][2]))
                row.append(Device['Bottleneck'])
            row.append(tuple2str(node.inshape, splitch))
            row.append(tuple2str(node.outshape, splitch))
            ptable.append(row)

        row = ['Total', '_']
        row.append(num2str(self.llm_profile[0], csvformat))
        row.append(num2str(self.llm_profile[1], csvformat))
        if self.llm_profile[2] > 0:
            row.append(num2str(self.llm_profile[2], csvformat))
            row.append('_')
        row.append('_')
        row.append('_')
        ptable.append(row)
        header = ['Name', 'Type']
        header.extend(
            [metric, 'Memory(bytes)'])
        if self.llm_profile[2] > 0:
            header.extend(['Projected Latency(ms)', 'Bottleneck'])
        header.extend(
            ['InShape',
             'OutShape'])
        print_table(ptable, header, f)

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

    def set_past_kv_length(self, n_past):
        self.graph.tensormap['n_past'].update_tensor(numpy.array(n_past, dtype=numpy.int64))

    def add_kv_cache(self, n_context, n_past):
        t_n_past = create_tensor('n_past', DYNAMIC_TENSOR, [self.batch], numpy.int64)
        t_n_past.update_tensor(numpy.array(n_past, dtype=numpy.int64))
        self.graph.input.append('n_past')
        self.graph.tensormap[t_n_past.name] = t_n_past
        self.kv_params = self.batch * (self.seq_len + n_past) * self.hidden_kv_size * 2 * self.num_hidden_layers
        kv_cache = create_tensor('kv_cache', DYNAMIC_TENSOR,
                                 [self.batch, self.num_hidden_layers * 2, n_context, self.hidden_kv_size],
                                 numpy.float32)
        self.graph.input.append('kv_cache')
        self.graph.tensormap[kv_cache.name] = kv_cache
        for n in self.graph.nodemap:
            if self.graph.nodemap[n].op_type == 'SDPA':
                self.graph.nodemap[n].input.append('n_past')
                self.graph.nodemap[n].input.append('kv_cache')

    def save_graph(self, path):
        self.graph.graph_reorder_nodes()
        self.graph.save_model(path, shape_only=False)


from .model_configs import *
