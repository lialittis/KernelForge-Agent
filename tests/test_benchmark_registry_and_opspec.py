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


def test_scanner_finds_official_cases_and_supported_subset():
    registry = scan_benchmark_cases(BENCH, repo_root=ROOT)

    assert registry["summary"]["total_cases"] == 13
    assert registry["summary"]["by_tier"] == {"t1": 6, "t2": 4, "t3": 3}

    supported = {
        case["id"]
        for case in registry["cases"]
        if case["support"]["status"] == "opspec_supported"
    }
    assert supported == {
        "t1/gelu",
        "t1/fused_silu_and_mul",
        "t1/sigmoid_scale_sum",
        "t1/softmax",
        "t2/add_rmsnorm_cast",
        "t2/add_rmsnorm_quant",
        "t2/rope",
        "t3/causal_conv1d",
        "t3/decode_mla",
        "t3/layernorm_gated",
    }
    assert "parse_failed" not in registry["summary"]["by_support"]

    matmul = {case["id"]: case for case in registry["cases"]}["t1/matmul_basic"]
    assert matmul["category"] == "matmul_like"
    assert matmul["support"]["status"] == "unsupported"


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


def test_unsupported_case_can_emit_metadata():
    spec = extract_opspec(
        BENCH / "t1/matmul_basic.py",
        repo_root=ROOT,
        allow_unsupported=True,
    )

    assert spec["id"] == "t1/matmul_basic"
    assert spec["category"] == "matmul_like"
    assert spec["support"]["status"] == "unsupported"
    assert spec["inputs"][0]["dtype"] == "bfloat16"


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
    assert data["summary"]["by_support"]["opspec_supported"] == 10
    assert "parse_failed" not in data["summary"]["by_support"]
