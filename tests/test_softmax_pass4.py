from __future__ import annotations

import ast
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = [
    ROOT / "kernel_forge/candidates/softmax_v1.py",
    ROOT / "kernel_forge/candidates/softmax_v2.py",
    ROOT / "kernel_forge/candidates/softmax_v3.py",
    ROOT / "kernel_forge/candidates/softmax_v4.py",
]


def test_softmax_candidates_are_valid_python():
    for candidate in CANDIDATES:
        py_compile.compile(str(candidate), doraise=True)


def test_softmax_candidates_define_modelnew_and_safe_imports():
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

    for candidate in CANDIDATES:
        tree = ast.parse(candidate.read_text(encoding="utf-8"))
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


def test_softmax_pass4_submission_script_creates_flat_runner_layout(tmp_path):
    output_root = tmp_path / "softmax_pass4"

    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/create_softmax_pass4_submissions.sh"),
        ],
        cwd=ROOT,
        env={"OUTPUT_ROOT": str(output_root), **dict(os.environ)},
        check=True,
    )

    for index in range(1, 5):
        team = f"softmax_v{index}"
        team_dir = output_root / team
        generated_case = team_dir / "t1/softmax.py"
        meta = json.loads((team_dir / "meta.json").read_text())

        assert meta["team_name"] == team
        assert meta["candidate"] == team
        assert generated_case.read_text() == CANDIDATES[index - 1].read_text()
