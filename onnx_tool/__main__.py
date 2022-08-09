import argparse

import onnx_tool


def get_parser():
    parser = argparse.ArgumentParser(
        "onnx_tool",
        description="Profile MACs(FLOPs) and perform shape inference on fixed-shape ONNX model"
        )
    parser.add_argument(
        "-m", "--mode",
        choices=['profile','export_tensors'],
        default='profile',
        help="path of input ONNX model")
    parser.add_argument(
        "-i", "--in", dest='in_',required=True,
        help="path of input ONNX model")
    parser.add_argument(
        "-o", "--out",
        help="path to save the ONNX model with shapes")
    parser.add_argument(
        '--names',
        nargs='+',
        default=None,
        help='tensor names'
    )
    parser.add_argument(
        "--fp16",
        action='store_true',
        help="path to save the ONNX model with shapes")
    parser.add_argument(
        "-f", "--file", default=None,
        help="file to store the MACs result for each node. None: print to console.")
    return parser

parser=get_parser()
args=parser.parse_args()
if args.mode=='profile':
    onnx_tool.model_profile(args.in_,None,args.file,args.out,dump_outputs=args.names)
elif args.mode=='export_tensors':
    onnx_tool.model_export_tensors_numpy(args.in_,tensornames=args.names,savefolder=args.out,fp16=args.fp16)