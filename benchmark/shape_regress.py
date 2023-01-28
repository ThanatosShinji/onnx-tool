from onnx_tool import Graph
from onnx_tool.utils import timer
import onnx


def bert_base():
    onnxfile = 'data/public/BERT_quan95.onnx'
    input_desc = {
        'input_ids': ('batch', 'seq'),
        'attention_mask': ('batch', 'seq'),
        'token_type_ids': ('batch', 'seq'),
    }
    input_range = {
        'batch': (1, 4),
        'seq': (16, 384)
    }
    model = onnx.load_model(onnxfile)
    graph = Graph(model.graph)
    shapeengine = graph.shape_regress(input_desc, input_range)

    # try update shape
    print_tensor = '1979'
    tm = timer()
    count = 0
    for b in range(1, 33, 1):
        shapeengine.update_variable('batch', b)# update batch size
        for i in range(16, 385, 16):
            shapeengine.update_variable('seq', i)# update sequence length
            shapeengine.update_variables()# update all shape variables
            for tensor in graph.dynamics:
                shape = shapeengine.get_tensorshape(tensor)# query tensor shapes
                if tensor == print_tensor:
                    print(b, i, shape)
            count += 1
    t = tm.stop()
    print(f'Total:{t} Time per reshape:{t / count}')# less than 1ms


def resnet18():
    onnxfile = 'data/public/resnet18-v1-7.onnx'
    input_desc = {
        'data': ['batch', 3, 'h', 'w']# channel is fixed to 3
    }
    input_range = {
        'batch': (1, 4),
        'h': (224, 299),
        'w': (224, 299),
    }
    model = onnx.load_model(onnxfile)
    graph = Graph(model.graph)
    shapeengine = graph.shape_regress(input_desc, input_range)

    # try update shape
    print_tensor = 'resnetv15_stage4__plus1'
    tm = timer()
    count = 0
    for b in range(1, 5, 1):
        shapeengine.update_variable('batch', b)# update the batch size
        for j in range(224, 299, 8):
            shapeengine.update_variable('h', j)# update the height of input image
            for i in range(224, 257, 16):
                shapeengine.update_variable('w', i)# update the width of input image
                shapeengine.update_variables()# update all shape variables
                for tensor in graph.dynamics:
                    shape = shapeengine.get_tensorshape(tensor)# query tensor shapes
                    if tensor == print_tensor:
                        print(b, j, i, shape)
                count += 1
    t = tm.stop()
    print(f'Total:{t} Time per reshape:{t / count}')# less than 1us


bert_base()
resnet18()
