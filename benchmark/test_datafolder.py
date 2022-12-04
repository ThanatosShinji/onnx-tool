import os.path
import onnx
from data.private.config import private_models
from data.public.config import public_models
from onnx_tool import model_api_test

folder = public_models['folder']
for modelinfo in public_models['models']:
    print('-' * 64)
    print(modelinfo['name'])
    model = onnx.load_model(os.path.join(folder, modelinfo['name']))
    basen = os.path.basename(modelinfo['name'])
    name = os.path.splitext(basen)[0]
    model_api_test(model, modelinfo['dynamic_input'])
    print('-' * 64)

folder = private_models['folder']
for modelinfo in private_models['models']:
    print('-' * 64)
    print(modelinfo['name'])
    model = onnx.load_model(os.path.join(folder, modelinfo['name']))
    basen = os.path.basename(modelinfo['name'])
    name = os.path.splitext(basen)[0]
    model_api_test(model, modelinfo['dynamic_input'])
    print('-' * 64)
