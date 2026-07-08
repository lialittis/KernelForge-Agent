from __future__ import annotations

import ast
import json
import os
import py_compile
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CASES = [
    ("reference_t1_matmul_basic", "t1/matmul_basic.py", "kernel_forge/candidates/reference_matmul_basic.py"),
    ("reference_t1_matmul_biasadd", "t1/matmul_biasadd.py", "kernel_forge/candidates/reference_matmul_biasadd.py"),
    ("reference_t2_rope", "t2/rope.py", "kernel_forge/candidates/reference_rope.py"),
    ("reference_t2_add_rmsnorm_cast", "t2/add_rmsnorm_cast.py", "kernel_forge/candidates/reference_add_rmsnorm_cast.py"),
    ("reference_t2_add_rmsnorm_quant", "t2/add_rmsnorm_quant.py", "kernel_forge/candidates/reference_add_rmsnorm_quant.py"),
    ("reference_t2_moe_topk_softmax", "t2/moe_topk_softmax.py", "kernel_forge/candidates/reference_moe_topk_softmax.py"),
    ("reference_t3_causal_conv1d", "t3/causal_conv1d.py", "kernel_forge/candidates/reference_causal_conv1d.py"),
    ("reference_t3_decode_mla", "t3/decode_mla.py", "kernel_forge/candidates/reference_decode_mla.py"),
    ("reference_t3_layernorm_gated", "t3/layernorm_gated.py", "kernel_forge/candidates/reference_layernorm_gated.py"),
]


def test_remaining_reference_candidates_are_valid_python():
    for _, _, source in CASES:
        py_compile.compile(str(ROOT / source), doraise=True)


def test_remaining_reference_candidates_define_modelnew_and_safe_imports():
    blocked = {
        "subprocess",
        "shutil",
        "socket",
        "http",
        "urllib",
        "ftplib",
        "smtplib",
        "ctypes",
        "multiprocessing",
    }

    for _, _, source in CASES:
        tree = ast.parse((ROOT / source).read_text(encoding="utf-8"))
        class_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }
        assert "ModelNew" in class_names

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".")[0])

        assert not (set(imports) & blocked)


def test_remaining_reference_submission_script_creates_flat_runner_layout(tmp_path):
    output_root = tmp_path / "remaining_reference"

    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/create_remaining_reference_submissions.sh"),
        ],
        cwd=ROOT,
        env={"OUTPUT_ROOT": str(output_root), **dict(os.environ)},
        check=True,
    )

    for team, case_path, source in CASES:
        team_dir = output_root / team
        generated_case = team_dir / case_path
        meta = json.loads((team_dir / "meta.json").read_text())

        assert meta["team_name"] == team
        assert meta["candidate"] == team
        assert generated_case.read_text() == (ROOT / source).read_text()
