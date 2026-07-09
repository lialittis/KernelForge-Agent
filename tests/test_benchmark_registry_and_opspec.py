from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_forge.benchmark import extract_opspec, scan_benchmark_cases


BENCH = ROOT / "third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite"
EXPECTED_CASES = {
    "t1/gelu",
    "t1/fused_silu_and_mul",
    "t1/matmul_basic",
    "t1/matmul_biasadd",
    "t1/sigmoid_scale_sum",
    "t1/softmax",
    "t2/add_rmsnorm_cast",
    "t2/add_rmsnorm_quant",
    "t2/moe_topk_softmax",
    "t2/rope",
    "t3/causal_conv1d",
    "t3/decode_mla",
    "t3/layernorm_gated",
}
REQUIRED_OPSPEC_KEYS = {
    "id",
    "name",
    "tier",
    "category",
    "source_path",
    "reference_class",
    "candidate_class",
    "inputs",
    "outputs",
    "semantics",
    "validation",
    "performance",
    "sketch",
    "submission",
}
REQUIRED_SKETCH_KEYS = {
    "operator_category",
    "compute_pattern",
    "parallel_axes",
    "tile_plan",
    "memory_plan",
    "pipeline_plan",
    "boundary_mask",
    "accumulation_dtype",
    "backend_target",
    "performance_knobs",
    "known_risks",
}


def test_scanner_finds_official_cases_and_supported_subset():
    registry = scan_benchmark_cases(BENCH, repo_root=ROOT)

    assert registry["summary"]["total_cases"] == 13
    assert registry["summary"]["by_tier"] == {"t1": 6, "t2": 4, "t3": 3}

    supported = {
        case["id"]
        for case in registry["cases"]
        if case["support"]["status"] == "opspec_supported"
    }
    assert supported == EXPECTED_CASES
    assert registry["summary"]["by_support"]["opspec_supported"] == 13
    assert "parse_failed" not in registry["summary"]["by_support"]
    assert "unsupported" not in registry["summary"]["by_support"]


def test_extracts_fused_silu_opspec():
    spec = extract_opspec(BENCH / "t1/fused_silu_and_mul.py", repo_root=ROOT)

    assert spec["id"] == "t1/fused_silu_and_mul"
    assert spec["category"] == "elementwise"
    assert spec["inputs"][0]["shape"] == [4096, 8192]
    assert spec["outputs"][0]["shape"] == [4096, 4096]
    assert spec["semantics"]["layout_transform"] == "split_last_dim"
    assert spec["sketch"]["compute_pattern"] == "fused_silu_and_mul"


def test_extracts_sigmoid_scale_sum_opspec():
    spec = extract_opspec(BENCH / "t1/sigmoid_scale_sum.py", repo_root=ROOT)

    assert spec["id"] == "t1/sigmoid_scale_sum"
    assert spec["category"] == "reduction"
    assert spec["inputs"][1]["shape"] == [8192]
    assert spec["outputs"][0]["shape"] == [1000, 1]
    assert spec["semantics"]["reduction_axes"] == [-1]
    assert spec["sketch"]["tile_plan"]["shape"] == [1000, 8192]


def test_extracts_matmul_basic_opspec():
    spec = extract_opspec(BENCH / "t1/matmul_basic.py", repo_root=ROOT)

    assert spec["id"] == "t1/matmul_basic"
    assert spec["category"] == "matmul_like"
    assert spec["inputs"][0]["shape"] == [32, 8192]
    assert spec["inputs"][0]["dtype"] == "bfloat16"
    assert spec["inputs"][1]["shape"] == [8192, 8192]
    assert spec["outputs"][0]["shape"] == [32, 8192]
    assert spec["outputs"][0]["dtype"] == "bfloat16"
    assert spec["semantics"]["reduction_axes"] == ["K"]
    assert spec["semantics"]["accumulation_dtype"] == "float32"
    assert spec["sketch"]["compute_pattern"] == "matmul_basic"
    assert spec["sketch"]["tile_plan"]["shape"] == [32, 8192, 8192]
    assert spec["sketch"]["tile_plan"]["axes"] == ["M", "N", "K"]
    assert spec["sketch"]["axis_map"]["M"]["extent"] == 32
    assert spec["sketch"]["axis_map"]["K"]["lhs"]["extent"] == 8192
    assert spec["sketch"]["dtype_plan"] == {
        "lhs": "bfloat16",
        "rhs": "bfloat16",
        "bias": None,
        "accumulator": "float32",
        "output": "bfloat16",
    }
    assert "matmul_backend_selection" in spec["sketch"]["known_risks"]


def test_extracts_matmul_biasadd_opspec():
    spec = extract_opspec(BENCH / "t1/matmul_biasadd.py", repo_root=ROOT)

    assert spec["id"] == "t1/matmul_biasadd"
    assert spec["category"] == "matmul_like"
    assert spec["inputs"][0]["shape"] == [4096, 4096]
    assert spec["inputs"][2]["shape"] == [1, 4096]
    assert spec["outputs"][0]["shape"] == [4096, 4096]
    assert spec["outputs"][0]["dtype"] == "float16"
    assert spec["semantics"]["broadcast"] == "bias broadcasts over the M axis"
    assert spec["semantics"]["reduction_axes"] == ["K"]
    assert spec["sketch"]["compute_pattern"] == "matmul_biasadd"
    assert spec["sketch"]["axis_map"]["N"]["extent"] == 4096
    assert spec["sketch"]["memory_plan"]["bias"]["broadcast"] == "bias[0, n] broadcasts over M"
    assert spec["sketch"]["dtype_plan"]["bias"] == "float16"
    assert "bias_broadcast_correctness" in spec["sketch"]["known_risks"]


def test_extracts_softmax_opspec():
    spec = extract_opspec(BENCH / "t1/softmax.py", repo_root=ROOT)

    assert spec["id"] == "t1/softmax"
    assert spec["inputs"][0]["shape"] == [32, 512, 4096]
    assert spec["outputs"][0]["shape"] == [32, 512, 4096]
    assert spec["semantics"]["normalization_axes"] == [-1]
    assert spec["sketch"]["compute_pattern"] == "rowwise_softmax"


def test_extracts_add_rmsnorm_cast_opspec():
    spec = extract_opspec(BENCH / "t2/add_rmsnorm_cast.py", repo_root=ROOT)

    assert spec["id"] == "t2/add_rmsnorm_cast"
    assert spec["category"] == "normalization"
    assert spec["inputs"][0]["shape"] == [32, 1024, 4096]
    assert spec["inputs"][2]["shape"] == [4096]
    assert spec["outputs"][0]["shape"] == [32, 1024, 4096]
    assert spec["outputs"][0]["dtype"] == "float16"
    assert spec["semantics"]["target_dtype"] == "float16"
    assert spec["semantics"]["normalization_axes"] == [-1]
    assert spec["sketch"]["compute_pattern"] == "add_rmsnorm_cast"
    assert spec["sketch"]["tile_plan"]["shape"] == [32768, 4096]


def test_extracts_add_rmsnorm_quant_opspec():
    spec = extract_opspec(BENCH / "t2/add_rmsnorm_quant.py", repo_root=ROOT)

    assert spec["id"] == "t2/add_rmsnorm_quant"
    assert spec["category"] == "normalization"
    assert [item["name"] for item in spec["inputs"]] == ["x", "residual", "gamma", "scale", "zero_point"]
    assert spec["inputs"][3]["shape"] == [1]
    assert spec["outputs"][0]["dtype"] == "int8"
    assert spec["semantics"]["quantization"]["clamp"] == [-128, 127]
    assert spec["sketch"]["compute_pattern"] == "add_rmsnorm_quant"


def test_extracts_moe_topk_softmax_opspec():
    spec = extract_opspec(BENCH / "t2/moe_topk_softmax.py", repo_root=ROOT)

    assert spec["id"] == "t2/moe_topk_softmax"
    assert spec["category"] == "reduction"
    assert spec["inputs"][0]["shape"] == [1024, 8]
    assert [item["name"] for item in spec["outputs"]] == ["top_k_probs", "top_k_indices"]
    assert spec["outputs"][0]["shape"] == [1024, 2]
    assert spec["outputs"][0]["dtype"] == "float32"
    assert spec["outputs"][1]["shape"] == [1024, 2]
    assert spec["outputs"][1]["dtype"] == "int64"
    assert spec["semantics"]["top_k"] == 2
    assert spec["semantics"]["layout_transform"] == "topk_tuple_output"
    assert spec["sketch"]["compute_pattern"] == "moe_topk_softmax"
    assert spec["sketch"]["tile_plan"]["shape"] == [1024, 8, 2]
    assert spec["sketch"]["axis_map"]["experts"]["extent"] == 8
    assert spec["sketch"]["axis_map"]["top_k"] == 2
    assert spec["sketch"]["output_contract"][0]["name"] == "top_k_probs"
    assert spec["sketch"]["output_contract"][1]["dtype"] == "int64"
    assert spec["sketch"]["numerical_plan"]["topk_renormalization"] == (
        "divide_selected_probabilities_by_selected_sum"
    )
    assert "stable_topk_ordering" in spec["sketch"]["known_risks"]


def test_extracts_rope_opspec():
    spec = extract_opspec(BENCH / "t2/rope.py", repo_root=ROOT)

    assert spec["id"] == "t2/rope"
    assert spec["category"] == "transpose_layout"
    assert spec["inputs"][0]["shape"] == [16, 48, 1000, 128]
    assert spec["inputs"][1]["shape"] == [1, 1, 1000, 128]
    assert spec["outputs"][0]["dtype"] == "float16"
    assert spec["semantics"]["layout_transform"] == "rotate_half_last_dim"
    assert spec["sketch"]["compute_pattern"] == "rotary_position_embedding"
    assert spec["sketch"]["tile_plan"]["shape"] == [768000, 128]


def test_extracts_layernorm_gated_opspec():
    spec = extract_opspec(BENCH / "t3/layernorm_gated.py", repo_root=ROOT)

    assert spec["id"] == "t3/layernorm_gated"
    assert spec["category"] == "normalization"
    assert spec["inputs"][0]["dtype"] == "float16"
    assert spec["inputs"][1]["shape"] == [4096]
    assert spec["outputs"][0]["shape"] == [32, 512, 4096]
    assert spec["semantics"]["norm_before_gate"] is True
    assert spec["semantics"]["is_rms_norm"] is True
    assert spec["sketch"]["compute_pattern"] == "gated_rmsnorm"


def test_extracts_causal_conv1d_opspec():
    spec = extract_opspec(BENCH / "t3/causal_conv1d.py", repo_root=ROOT)

    assert spec["id"] == "t3/causal_conv1d"
    assert spec["category"] == "convolution"
    assert spec["inputs"][1]["shape"] == [32, 2048, 3]
    assert spec["inputs"][4]["dtype"] == "int32"
    assert spec["outputs"][0]["shape"] == [32, 2048]
    assert spec["semantics"]["activation"] == "silu"
    assert spec["semantics"]["state_update"] == "conv_state.copy_(x_padded[:, :, -(width - 1):])"
    assert spec["sketch"]["compute_pattern"] == "causal_depthwise_conv1d_silu"
    assert spec["sketch"]["tile_plan"]["shape"] == [32, 2048, 4]


def test_extracts_decode_mla_opspec():
    spec = extract_opspec(BENCH / "t3/decode_mla.py", repo_root=ROOT)

    assert spec["id"] == "t3/decode_mla"
    assert spec["category"] == "matmul_like"
    assert spec["inputs"][0]["shape"] == [16, 128, 576]
    assert spec["inputs"][4]["dtype"] == "int32"
    assert spec["inputs"][5]["shape"] == [16, 8]
    assert spec["outputs"][0]["shape"] == [16, 128, 512]
    assert spec["semantics"]["page_size"] == 128
    assert spec["semantics"]["sm_scale"] == 0.041666666666666664
    assert spec["sketch"]["compute_pattern"] == "paged_mla_decode_attention"
    assert spec["sketch"]["tile_plan"]["shape"] == [16, 128, 1024, 512]


def test_all_supported_cases_have_parsed_opspec_files():
    parsed = {_case_to_parsed_filename(case_id) for case_id in EXPECTED_CASES}
    assert {path.name for path in (ROOT / "benchmarks/parsed").glob("*.yaml")} == parsed


def test_all_parsed_opspecs_have_required_fields_and_non_generic_sketches():
    for path in sorted((ROOT / "benchmarks/parsed").glob("*.yaml")):
        spec = yaml.safe_load(path.read_text())
        assert set(spec) >= REQUIRED_OPSPEC_KEYS, path
        assert spec["id"] in EXPECTED_CASES, path
        assert spec["name"]
        assert spec["tier"] in {"t1", "t2", "t3"}
        assert spec["source_path"].endswith(f"{spec['tier']}/{spec['name']}.py")
        assert spec["reference_class"] == "Model"
        assert spec["candidate_class"] == "ModelNew"

        assert spec["inputs"], path
        assert spec["outputs"], path
        for tensor in [*spec["inputs"], *spec["outputs"]]:
            assert {"name", "shape", "dtype", "layout"} <= set(tensor), (path, tensor)
            assert tensor["name"]
            assert isinstance(tensor["shape"], list)
            assert tensor["dtype"]

        assert {"rtol", "atol", "shape_cases", "dtype_cases"} <= set(spec["validation"]), path
        assert spec["validation"]["shape_cases"]
        assert spec["validation"]["dtype_cases"]
        assert {"metric", "warmup", "repeats", "num_trials"} <= set(spec["performance"]), path

        sketch = spec["sketch"]
        assert set(sketch) >= REQUIRED_SKETCH_KEYS, path
        assert sketch["operator_category"] != "unknown", path
        assert sketch["compute_pattern"] != "unsupported_pending_classification", path
        assert sketch["tile_plan"]["strategy"] != "manual_required", path
        assert sketch["tile_plan"]["shape"], path
        assert sketch["pipeline_plan"]["stages"], path
        assert sketch["known_risks"], path
        assert spec["submission"]["entrypoint"] == "ModelNew"
        assert spec["submission"]["required_files"] == [f"{spec['tier']}/{spec['name']}.py"]


def test_tuple_output_opspecs_describe_all_outputs_explicitly():
    tuple_specs = []
    for path in sorted((ROOT / "benchmarks/parsed").glob("*.yaml")):
        spec = yaml.safe_load(path.read_text())
        if len(spec["outputs"]) > 1:
            tuple_specs.append((path, spec))

    assert [spec["id"] for _, spec in tuple_specs] == ["t2/moe_topk_softmax"]
    for path, spec in tuple_specs:
        output_names = [item["name"] for item in spec["outputs"]]
        semantic_outputs = spec["semantics"].get("outputs")
        sketch_outputs = spec["sketch"].get("output_contract")

        assert semantic_outputs is not None, path
        assert sketch_outputs is not None, path
        assert [item["name"] for item in semantic_outputs] == output_names
        assert [item["name"] for item in sketch_outputs] == output_names

        for output, contract in zip(spec["outputs"], sketch_outputs):
            assert contract["shape"] == output["shape"]
            assert contract["dtype"] == output["dtype"]
            assert contract["semantics"]


def test_scan_cli_writes_registry_yaml(tmp_path):
    output = tmp_path / "registry.yaml"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/scan_benchmark_cases.py"),
            "--bench-dir",
            str(BENCH),
            "--repo-root",
            str(ROOT),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )

    data = yaml.safe_load(output.read_text())
    assert data["summary"]["total_cases"] == 13
    assert data["summary"]["by_support"]["opspec_supported"] == 13
    assert "parse_failed" not in data["summary"]["by_support"]
    assert "unsupported" not in data["summary"]["by_support"]


def _case_to_parsed_filename(case_id: str) -> str:
    return f"{case_id.replace('/', '_')}.yaml"
