import warnings

import numpy

import onnx_tool

'''
f32 layout: OCxIC ONNX default, len(shape)==2 2-d weight only

block: 0 per tensor
       -1 per oc
       >0 per block on ic
'''


def find_min_max(f32: numpy.ndarray, block):
    if block == 0:
        return numpy.array(f32.min(keepdims=False)), numpy.array(f32.max(keepdims=False))
    if block == -1:
        fmin = [f32[i, :].min() for i in numpy.ndindex(*f32.shape[:-1])]
        fmax = [f32[i, :].max() for i in numpy.ndindex(*f32.shape[:-1])]
        return numpy.array(fmin), numpy.array(fmax)
    if block > 0:
        fmin = []
        fmax = []
        for i in range(f32.shape[0]):
            tmin = []
            tmax = []
            for j in range(0, f32.shape[1], block):
                tmin.append(f32[i, j:j + block].min())
                tmax.append(f32[i, j:j + block].max())
            fmin.append(tmin)
            fmax.append(tmax)
        return numpy.array(fmin), numpy.array(fmax)


def get_symmetric_parameter(fmin, fmax, bits):
    alpha = max(abs(fmin), abs(fmax))
    scale = alpha / (2 ** (bits - 1) - 1)
    return scale, 0


def get_asymmetric_u8_parameter(fmin, fmax, bits):
    fmin = min(fmin, 0)
    fmax = max(fmax, 0)
    delta = fmax - fmin
    scale = delta / (2 ** bits - 1)
    zp = round(-fmin / scale)
    return scale, zp


def symmetric_quant(fval, scale, *agrs):
    return numpy.clip(round(fval / scale), -128, 127)


def asymmetric_quant(fval, scale, zp):
    return numpy.clip(round(fval / scale) + zp, 0, 255)


def symmetric_dequant(qval, scale, *agrs):
    return qval * scale


def asymmetric_dequant(qval, scale, zp):
    return (qval - zp) * scale


def quantize(f32: numpy.ndarray, block: int = -1, type: str = 'sym', bits: int = 8):
    fmin, fmax = find_min_max(f32, block)
    if type == 'sym':
        get_param = get_symmetric_parameter
        quant_val = symmetric_quant
    elif type == 'asym':
        get_param = get_asymmetric_u8_parameter
        quant_val = asymmetric_quant
    if len(fmin.shape) == 0:
        Q = numpy.zeros_like(f32, dtype=numpy.int32)
        scale, zp = get_param(fmin, fmax, bits)
        for i in numpy.ndindex(*f32.shape):
            Q[i] = quant_val(f32[i], scale, zp)
        return Q, numpy.array(scale), numpy.array(zp)
    if len(fmin.shape) == 1:
        Q = numpy.zeros_like(f32, dtype=numpy.int32)
        scale = numpy.ones_like(fmin, dtype=numpy.float32)
        zp = numpy.zeros_like(fmin, dtype=numpy.int32)
        for i in range(len(fmin)):
            scale[i], zp[i] = get_param(fmin[i], fmax[i], bits)
        for i in numpy.ndindex(*f32.shape):
            Q[i] = quant_val(f32[i], scale[i[0]], zp[i[0]])
        return Q, scale, zp
    if len(fmin.shape) == 2:
        Q = numpy.zeros_like(f32, dtype=numpy.int32)
        scale = numpy.ones_like(fmin, dtype=numpy.float32)
        zp = numpy.zeros_like(fmin, dtype=numpy.int32)
        for i in numpy.ndindex(*fmin.shape):
            scale[i], zp[i] = get_param(fmin[i], fmax[i], bits)
        for i in numpy.ndindex(*f32.shape):
            blk_idx = i[1] // block
            Q[i] = quant_val(f32[i], scale[i[0], blk_idx], zp[i[0], blk_idx])
        return Q, scale, zp

def pack_4bits(Q_int:numpy.ndarray):
    shape=Q_int.shape
    s4shape=list(shape)
    s4shape[-1]=shape[-1]//2
    s4arr=numpy.ones(s4shape,dtype=numpy.uint8)
    import ctypes
    def cvt_s8bits_to_s4bits(v8):
        sigbit=v8&0x80
        sigbit=sigbit>>4
        v8=v8&0x7
        return v8|sigbit
    for i in numpy.ndindex(*s4shape):
        rawi=list(i)
        rawi[-1]*=2
        rawi1=list(rawi)
        rawi1[-1]+=1
        lo4=cvt_s8bits_to_s4bits(ctypes.c_uint8(Q_int[tuple(rawi)]).value)
        hi4=cvt_s8bits_to_s4bits(ctypes.c_uint8(Q_int[tuple(rawi1)]).value)
        val=lo4|(hi4<<4)
        s4arr[i]=val
    return s4arr

def graph_quantize(g: onnx_tool.Graph, tname: str, block: int = -1, type: str = 'sym', bits: int = 8):
    if tname not in g.initials:
        warnings.warn("Quantize activation tensor is useless")
        return
    quantized_suffix=['_ot_q','_ot_s','_ot_z']
    for suf in quantized_suffix:
        if tname.endswith(suf):
            return
    f32arr = g.tensormap[tname].numpy
    if f32arr.dtype not in [numpy.float32, numpy.float16, numpy.float64]:
        return
    if len(f32arr.shape) != 2:
        return
    if f32arr.dtype is not numpy.float32:
        f32arr = f32arr.astype(numpy.float32)
    name=g.consumedby[tname][0]
    node=g.nodemap[name]
    if isinstance(node,onnx_tool.node.GemmNode):
        if node.transB==0:
            f32arr=f32arr.transpose()
    node.set_attr('OTQ_Block',block)
    node.set_attr('OTQ_Type',type)
    node.set_attr('OTQ_Bits',bits)
    Q, scale, zp = quantize(f32arr, block, type, bits)
    tname_q = tname + '_ot_q'
    tname_s = tname + '_ot_s'
    tname_z = tname + '_ot_z'
    if bits == 8:
        if type == 'asym':
            Q = Q.astype(numpy.uint8)
            zp = zp.astype(numpy.uint8)
        else:
            Q = Q.astype(numpy.int8)
    if bits == 4:
        scale = scale.astype(numpy.float16)
        if type == 'asym':
            Q = pack_4bits(Q)
            zp = zp.astype(numpy.uint8)
        else:
            Q = pack_4bits(Q)

    g.add_initial(tname_q, Q)
    g.add_initial(tname_s, scale)
    if type == 'asym':
        g.add_initial(tname_z, zp)
    for nname in g.consumedby[tname]:
        node = g.nodemap[nname]
        idx = node.input.index(tname)
        node.input.remove(tname)
        if type == 'asym':
            node.input.insert(idx, tname_z)
        node.input.insert(idx, tname_s)
        node.input.insert(idx, tname_q)
        if tname_q not in g.consumedby.keys():
            g.consumedby[tname_q] = nname
            g.consumedby[tname_s] = nname
            g.consumedby[tname_z] = nname
        else:
            g.consumedby[tname_q].append(nname)
            g.consumedby[tname_s].append(nname)
            g.consumedby[tname_z].append(nname)
    g.consumedby.pop(tname)
    g.initials.remove(tname)
    g.tensormap.pop(tname)
