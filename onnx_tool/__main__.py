import argparse

import numpy

import onnx_tool


def get_parser():
    parser = argparse.ArgumentParser(
        "onnx_tool",
        description="Profile MACs(FLOPs) and perform shape inference on fixed-shape ONNX model"
    )
    parser.add_argument(
        "-m", "--mode",
        choices=['profile', 'export_tensors'],
        default='profile',
        help="path of input ONNX model")
    parser.add_argument(
        "-i", "--in", dest='in_', required=True,
        help="path of input ONNX model")
    parser.add_argument(
        "-o", "--out",
        help="path to save the ONNX model with shapes")
    parser.add_argument(
        '--names',
        nargs='+',
        default=None,
        help='tensor names as: --names 410 420'
    )
    parser.add_argument(
        '--dynamic_inputs',
        nargs='+',
        default=None,
        help='dynamic shapes for inputs as: --dynamic_inputs input:f32:1x3x224x224 scale:f32:1x2:0.25x0.25'
    )
    parser.add_argument(
        "--fp16",
        action='store_true',
        help="path to save the ONNX model with shapes")
    parser.add_argument(
        "-f", "--file", default=None,
        help="file to store the MACs result for each node. None: print to console.")
    return parser


parser = get_parser()
args = parser.parse_args()


def __str2numpytype__(strtype):
    if strtype == 'f32':
        return numpy.float32
    if strtype == 'int32':
        return numpy.int32


def __str2list__(strshape, type):
    strs = strshape.split('x')
    shape = []
    for s in strs:
        shape.append(type(s))
    return shape


def __args2dynamicshapes__(args: [str]):
    dic = {}
    for arg in args:
        strs = arg.split(':')
        dtype = __str2numpytype__(strs[1])
        shape = __str2list__(strs[2], int)
        if len(strs) > 3:
            arr = numpy.array(__str2list__(strs[3], dtype), dtype=dtype).reshape(shape)
        else:
            arr = numpy.zeros(shape, dtype=dtype)
        dic[strs[0]] = arr
    return dic


if args.mode == 'profile':
    if args.dynamic_inputs is not None:
        dynamic = __args2dynamicshapes__(args.dynamic_inputs)
    else:
        dynamic = None
    onnx_tool.model_profile(args.in_, dynamic, args.file, args.out, dump_outputs=args.names)
elif args.mode == 'export_tensors':
    onnx_tool.model_export_tensors_numpy(args.in_, tensornames=args.names, savefolder=args.out, fp16=args.fp16)
