"""
Model Profile Regression Tests.

Compares current model profiling output against saved baseline CSV files.
Both baseline generation and testing use the same model_profile() pipeline.
Comparison: Total row (Forward_MACs, Memory, Params) + each node's OutShape.

Run `python tests/update_baseline.py` to generate/refresh baseline files.

Reference: benchmark/profile_datafolder.py
"""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import onnx

from data.private.config import private_models
from data.public.config import public_models
from onnx_tool import model_profile

BASELINE_DIR = os.path.join(os.path.dirname(__file__), 'baseline')

RTOL = 0.01
ATOL = 1


def parse_csv(csv_path):
    """Parse a CSV file and return (header, rows)."""
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def normalize_value(val_str):
    """Parse a numeric value string (possibly with commas) to int."""
    return int(val_str.replace(',', ''))


def profile_to_csv(model_path, dynamic_shapes):
    """Run model_profile and return (header, rows) from a temp CSV."""
    model = onnx.load_model(model_path)
    name = os.path.splitext(os.path.basename(model_path))[0]
    tmp_csv = os.path.join(BASELINE_DIR, f'__tmp_{name}.csv')
    try:
        model_profile(model, dynamic_shapes, save_profile=tmp_csv,
                      mcfg={'constant_folding': True, 'verbose': False})
        return parse_csv(tmp_csv)
    finally:
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)


def find_baseline(model_name):
    """Find the baseline CSV file for a given model name."""
    name = os.path.splitext(model_name)[0]
    csv_path = os.path.join(BASELINE_DIR, name + '.csv')
    return csv_path if os.path.exists(csv_path) else None


def collect_test_cases(config):
    """Collect (model_path, dynamic_shapes, model_name) test cases."""
    cases = []
    folder = config['folder']
    for modelinfo in config['models']:
        model_name = modelinfo['name']
        model_path = model_name
        if '.onnx' not in model_path:
            model_path = f"{model_path}/{model_path}.onnx"
        model_path = os.path.join(folder, model_path)
        if not os.path.exists(model_path):
            continue
        if find_baseline(model_name) is None:
            continue
        cases.append((model_path, modelinfo['dynamic_input'], model_name))
    return cases


# Collect all test cases
_test_cases = (collect_test_cases(public_models) +
               collect_test_cases(private_models))


@pytest.mark.parametrize("model_path,dynamic_shapes,model_name", _test_cases)
def test_model_profile_regression(model_path, dynamic_shapes, model_name):
    """Test that model profiling output matches the saved baseline."""
    baseline_path = find_baseline(model_name)
    assert baseline_path is not None

    # Profile model to CSV (same pipeline as update_baseline.py)
    actual_header, actual_rows = profile_to_csv(model_path, dynamic_shapes)
    baseline_header, baseline_rows = parse_csv(baseline_path)

    # --- Compare Total row ---
    total_diffs = []
    for i, col in enumerate(baseline_header):
        if col not in ('Forward_MACs', 'Memory', 'Params'):
            continue
        try:
            bv = normalize_value(baseline_rows[-1][i])
            av = normalize_value(actual_rows[-1][i])
        except (ValueError, IndexError):
            if baseline_rows[-1][i] != actual_rows[-1][i]:
                total_diffs.append((col, baseline_rows[-1][i], actual_rows[-1][i]))
            continue
        if bv == 0 and av == 0:
            continue
        if bv == 0 or av == 0:
            if abs(bv - av) > ATOL:
                total_diffs.append((col, str(bv), str(av)))
            continue
        if abs(bv - av) / max(abs(bv), abs(av)) > RTOL and abs(bv - av) > ATOL:
            total_diffs.append((col, str(bv), str(av)))

    if total_diffs:
        msg = f"Total row mismatch for {model_name}:\n"
        for col, bv, av in total_diffs:
            msg += f"  {col}: baseline={bv}, actual={av}\n"
        pytest.fail(msg)

    # --- Compare OutShape of each node ---
    # Build {node_name: outshape} maps from both CSVs
    try:
        out_idx = baseline_header.index('OutShape')
    except ValueError:
        return

    baseline_map = {row[0]: row[out_idx] for row in baseline_rows[:-1]}
    actual_map = {row[0]: row[out_idx] for row in actual_rows[:-1]}

    matched = 0
    missing = []
    incorrect = []
    extra = []

    for name, shape in baseline_map.items():
        if name not in actual_map:
            missing.append((name, shape))
        elif actual_map[name] != shape:
            incorrect.append((name, shape, actual_map[name]))
        else:
            matched += 1

    for name, shape in actual_map.items():
        if name not in baseline_map:
            extra.append((name, shape))

    err_lines = []
    if missing:
        err_lines.append(f"  Missing ({len(missing)}):")
        for name, shape in missing[:10]:
            err_lines.append(f"    {name}: shape={shape}")
        if len(missing) > 10:
            err_lines.append(f"    ... and {len(missing) - 10} more")
    if incorrect:
        err_lines.append(f"  Shape mismatch ({len(incorrect)}):")
        for name, b_shape, a_shape in incorrect[:10]:
            err_lines.append(f"    {name}: baseline={b_shape}, actual={a_shape}")
        if len(incorrect) > 10:
            err_lines.append(f"    ... and {len(incorrect) - 10} more")
    if extra:
        err_lines.append(f"  Extra ({len(extra)}):")
        for name, shape in extra[:10]:
            err_lines.append(f"    {name}: shape={shape}")
        if len(extra) > 10:
            err_lines.append(f"    ... and {len(extra) - 10} more")

    if err_lines:
        summary = (f"Node mismatch for {model_name}:\n"
                   f"  Matched: {matched}, Missing: {len(missing)}, "
                   f"Incorrect: {len(incorrect)}, Extra: {len(extra)}\n" +
                   "\n".join(err_lines))
        pytest.fail(summary)
