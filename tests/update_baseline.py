"""
Update baseline CSV files for model profile tests.
Run this script to generate/refresh all baseline files under tests/baseline/.

Reference: benchmark/profile_datafolder.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import onnx

from data.private.config import private_models
from data.public.config import public_models
from onnx_tool import model_profile

BASELINE_DIR = os.path.join(os.path.dirname(__file__), 'baseline')


def profile_model(model_path, dynamic_shapes):
    """Run model_profile and save baseline CSV."""
    basename = os.path.basename(model_path)
    name = os.path.splitext(basename)[0]
    csv_path = os.path.join(BASELINE_DIR, name + '.csv')

    model = onnx.load_model(model_path)
    model_profile(model, dynamic_shapes, save_profile=csv_path,
                  mcfg={'constant_folding': True, 'verbose': False})
    print(f'  -> saved to {csv_path}')


def process_models(config):
    """Process all models from a config dict (same format as public_models/private_models)."""
    folder = config['folder']
    for modelinfo in config['models']:
        print('-' * 64)
        print(modelinfo['name'])
        model_path = modelinfo['name']
        if '.onnx' not in model_path:
            model_path = f"{model_path}/{model_path}.onnx"
        model_path = os.path.join(folder, model_path)

        if not os.path.exists(model_path):
            print(f'  SKIPPED: file not found - {model_path}')
            continue

        try:
            profile_model(model_path, modelinfo['dynamic_input'])
        except Exception as e:
            print(f'  FAILED: {e}')


def main():
    os.makedirs(BASELINE_DIR, exist_ok=True)
    print(f'Baseline directory: {BASELINE_DIR}')
    print('=' * 64)

    print('Processing public models...')
    process_models(public_models)

    print()
    print('Processing private models...')
    process_models(private_models)

    print('=' * 64)
    print('Done.')


if __name__ == '__main__':
    main()
