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
        shapeengine.update_variable('batch', b)
        for i in range(16, 385, 16):
            shapeengine.update_variable('seq', i)
            shapeengine.update_variables()
            for tensor in graph.dynamics:
                shape = shapeengine.get_tensorshape(tensor)
                if tensor == print_tensor:
                    print(b, i, shape)
            count += 1
    t = tm.stop()
    print(f'Total:{t} Time per reshape:{t / count}')


def resnet18():
    onnxfile = 'data/public/resnet18-v1-7.onnx'
    input_desc = {
        'data': ['batch', 3, 'h', 'w']
    }
    input_range = {
        'batch': (1, 4),
        'h': (224, 256),
        'w': (224, 256),
    }
    model = onnx.load_model(onnxfile)
    graph = Graph(model.graph)
    shapeengine = graph.shape_regress(input_desc, input_range)

    # try update shape
    print_tensor = 'resnetv15_stage4__plus1'
    tm = timer()
    count = 0
    for b in range(1, 5, 1):
        shapeengine.update_variable('batch', b)
        for j in range(224, 257, 16):
            shapeengine.update_variable('h', j)
            for i in range(224, 257, 16):
                shapeengine.update_variable('w', i)
                shapeengine.update_variables()
                for tensor in graph.dynamics:
                    shape = shapeengine.get_tensorshape(tensor)
                    if tensor == print_tensor:
                        print(b, j, i, shape)
                count += 1
    t = tm.stop()
    print(f'Total:{t} Time per reshape:{t / count}')


bert_base()
resnet18()
