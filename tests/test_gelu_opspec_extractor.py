from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_forge.benchmark import extract_opspec


GELU_CASE = ROOT / "third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/t1/gelu.py"
GELU_EXPERIMENT = ROOT / "experiments/runs/2026-06-30-gelu-manual-baseline.yaml"


def test_extracts_gelu_opspec_core_fields():
    spec = extract_opspec(GELU_CASE, experiment_path=GELU_EXPERIMENT, repo_root=ROOT)

    assert spec["id"] == "t1/gelu"
    assert spec["name"] == "gelu"
    assert spec["tier"] == "t1"
    assert spec["category"] == "elementwise"
    assert spec["reference_class"] == "Model"
    assert spec["candidate_class"] == "ModelNew"
    assert spec["inputs"] == [
        {
            "name": "input_tensor",
            "shape": [32, 512, 1024],
            "dtype": "float32",
            "layout": "contiguous",
        }
    ]
    assert spec["outputs"][0]["shape"] == [32, 512, 1024]
    assert spec["outputs"][0]["dtype"] == "float32"
    assert spec["semantics"]["expression"] == "torch.nn.functional.gelu(input_tensor)"
    assert spec["performance"]["baseline_latency_ms"] == 0.0438129500253126


def test_extracts_gelu_sketch_fields():
    spec = extract_opspec(GELU_CASE, repo_root=ROOT)
    sketch = spec["sketch"]

    assert sketch["operator_category"] == "elementwise"
    assert sketch["compute_pattern"] == "unary_pointwise_gelu"
    assert sketch["tile_plan"]["shape"] == [16777216]
    assert sketch["boundary_mask"]["required"] is True
    assert sketch["backend_target"] == "triton_ascend"


def test_cli_writes_valid_yaml(tmp_path):
    output = tmp_path / "t1_gelu.yaml"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/extract_opspec.py"),
        "--case",
        str(GELU_CASE),
        "--experiment",
        str(GELU_EXPERIMENT),
        "--repo-root",
        str(ROOT),
        "--output",
        str(output),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)

    data = yaml.safe_load(output.read_text())
    assert data["id"] == "t1/gelu"
    assert data["inputs"][0]["shape"] == [32, 512, 1024]
    assert data["performance"]["baseline_latency_ms"] == 0.0438129500253126


def test_committed_yaml_round_trips():
    for path in [
        ROOT / "benchmarks/parsed/t1_gelu.yaml",
        GELU_EXPERIMENT,
    ]:
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict)
        assert data["id"]
