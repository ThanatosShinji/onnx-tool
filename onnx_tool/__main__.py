import argparse

import onnx_tool


def get_parser():
    parser = argparse.ArgumentParser(
        "onnx_tool",
        description="Profile MACs(FLOPs) and perform shape inference on fixed-shape ONNX model"
        )
    parser.add_argument(
        "-i", "--in", dest='in_',
        help="path of input ONNX model")
    parser.add_argument(
        "-o", "--out",
        help="path to save the ONNX model with shapes")
    parser.add_argument(
        "-f", "--file", default=None,
        help="file to store the MACs result for each node. None: print to console.")
    return parser

parser=get_parser()
args=parser.parse_args()
onnx_tool.model_profile(args.in_,None,args.file,args.out)