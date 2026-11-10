"""CLI-level test: a malformed --dynamic_shapes value must be a usage error."""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_malformed_dynamic_shapes_is_usage_error():
    proc = subprocess.run(
        [
            sys.executable, "-m", "onnx_tool",
            "-i", "dummy.onnx",
            "--dynamic_shapes", "not-a-valid-spec",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "dynamic shape" in proc.stderr
