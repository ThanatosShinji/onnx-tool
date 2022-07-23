import onnx
import numpy

from .node_profilers import NodeBase, graph_profile, infer_shapes, create_ndarray_f32, create_ndarray_int64, print_node_map
from .utils import NODEPROFILER_REGISTRY

def __remove_initilisers(model:onnx.ModelProto):
    model.graph.ClearField('initializer')

def model_profile(m: [onnx.ModelProto | str], dynamic_shapes: {str: tuple} = None, savenode: str = None,
                  saveshapesmodel: str = None, shapesonly:bool=False, verbose:bool=False)-> None:
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        macs,params,nodemap=graph_profile(m.graph, dynamic_shapes,verbose)
        print_node_map(nodemap,savenode)
        if saveshapesmodel is not None:
            if shapesonly:
                __remove_initilisers(m)
            onnx.save_model(m,saveshapesmodel)

def model_shape_infer(m: [onnx.ModelProto | str], dynamic_shapes: {str: tuple} = None,
                  saveshapesmodel: str = None, shapesonly:bool=False, verbose:bool=False):
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        _,_=infer_shapes(m.graph, dynamic_shapes,verbose)
        if saveshapesmodel is not None:
            if shapesonly:
                __remove_initilisers(m)
            onnx.save_model(m,saveshapesmodel)
