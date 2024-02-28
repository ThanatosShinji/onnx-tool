import os.path

import onnx

from data.private.config import private_models
from data.public.config import public_models
from onnx_tool import model_profile

folder = public_models['folder']
for modelinfo in public_models['models']:
    print('-' * 64)
    print(modelinfo['name'])
    if '.onnx' not in modelinfo['name']:
        modelinfo['name'] = f"{modelinfo['name']}/{modelinfo['name']}.onnx"
    model = onnx.load_model(os.path.join(folder, modelinfo['name']))
    basen = os.path.basename(modelinfo['name'])
    name = os.path.splitext(basen)[0]
    model_profile(model, modelinfo['dynamic_input'], save_profile=os.path.join(folder, name + '_info.log'),
                  mcfg={'constant_folding': True, 'verbose': True}
                  , save_model=os.path.join(folder, name + '_shapes_only.onnx'), shape_only=True)
    print('-' * 64)

folder = private_models['folder']
for modelinfo in private_models['models']:
    print('-' * 64)
    print(modelinfo['name'])
    if '.onnx' not in modelinfo['name']:
        modelinfo['name'] = f"{modelinfo['name']}/{modelinfo['name']}.onnx"
    model = onnx.load_model(os.path.join(folder, modelinfo['name']))
    basen = os.path.basename(modelinfo['name'])
    name = os.path.splitext(basen)[0]
    model_profile(model, modelinfo['dynamic_input'], save_profile=os.path.join(folder, name + '_info.log'),
                  mcfg={'constant_folding': True, 'verbose': True}
                  , save_model=os.path.join(folder, name + '_shapes_only.onnx'), shape_only=True)
    print('-' * 64)
