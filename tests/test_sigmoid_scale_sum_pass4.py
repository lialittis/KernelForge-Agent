from __future__ import annotations

import ast
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_forge.experiments import summarize_passn


CANDIDATES = [
    ROOT / "kernel_forge/candidates/sigmoid_scale_sum_v1.py",
    ROOT / "kernel_forge/candidates/sigmoid_scale_sum_v2.py",
    ROOT / "kernel_forge/candidates/sigmoid_scale_sum_v3.py",
    ROOT / "kernel_forge/candidates/sigmoid_scale_sum_v4.py",
]


def test_sigmoid_scale_sum_candidates_are_valid_python():
    for candidate in CANDIDATES:
        py_compile.compile(str(candidate), doraise=True)


def test_sigmoid_scale_sum_candidates_define_modelnew_and_safe_imports():
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


def test_pass4_submission_script_creates_flat_runner_layout(tmp_path):
    output_root = tmp_path / "sigmoid_scale_sum_pass4"

    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/create_sigmoid_scale_sum_pass4_submissions.sh"),
        ],
        cwd=ROOT,
        env={"OUTPUT_ROOT": str(output_root), **dict(os.environ)},
        check=True,
    )

    for index in range(1, 5):
        team = f"sigmoid_scale_sum_v{index}"
        team_dir = output_root / team
        generated_case = team_dir / "t1/sigmoid_scale_sum.py"
        meta = json.loads((team_dir / "meta.json").read_text())

        assert meta["team_name"] == team
        assert meta["candidate"] == team
        assert generated_case.read_text() == CANDIDATES[index - 1].read_text()


def test_passn_summary_computes_pass_at_1_and_pass_at_4(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_result(results_dir / "sigmoid_scale_sum_v1.json", False, 0.0)
    _write_result(results_dir / "sigmoid_scale_sum_v2.json", True, 1.25)
    _write_result(results_dir / "sigmoid_scale_sum_v3.json", True, 0.9)

    summary = summarize_passn(
        results_dir,
        case_id="t1/sigmoid_scale_sum",
        candidates=[
            "sigmoid_scale_sum_v1",
            "sigmoid_scale_sum_v2",
            "sigmoid_scale_sum_v3",
            "sigmoid_scale_sum_v4",
        ],
    )

    assert summary["pass_at_1"] is False
    assert summary["pass_at_n"] is True
    assert summary["n"] == 4
    assert summary["passed_count"] == 2
    assert summary["best_candidate"]["team_name"] == "sigmoid_scale_sum_v2"
    assert summary["best_candidate"] is not summary["candidates"][1]
    assert summary["candidates"][3]["status"] == "missing_result"


def test_passn_cli_writes_yaml(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_result(results_dir / "sigmoid_scale_sum_v1.json", True, 1.0)
    output = tmp_path / "pass4.yaml"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/summarize_passn.py"),
            "--results-dir",
            str(results_dir),
            "--case",
            "t1/sigmoid_scale_sum",
            "--candidate",
            "sigmoid_scale_sum_v1",
            "--candidate",
            "sigmoid_scale_sum_v2",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )

    output_text = output.read_text()
    data = yaml.safe_load(output_text)
    assert data["pass_at_1"] is True
    assert data["pass_at_n"] is True
    assert data["best_candidate"]["team_name"] == "sigmoid_scale_sum_v1"
    assert "&id" not in output_text
    assert "*id" not in output_text


def _write_result(path: Path, passed: bool, speedup: float) -> None:
    status = "pass" if passed else "fail"
    result = {
        "team_name": path.stem,
        "cases": [
            {
                "case": "t1/sigmoid_scale_sum",
                "status": status,
                "correctness": passed,
                "max_abs_diff": 0.0 if passed else 1.0,
                "max_rel_diff": 0.0 if passed else 1.0,
                "baseline_ms": 1.0 if passed else None,
                "solution_ms": 1.0 / speedup if passed and speedup else None,
                "speedup": speedup if passed else None,
                "weighted_score": 60.0 * speedup if passed else 0.0,
                "error": None if passed else "failed",
            }
        ],
    }
    path.write_text(json.dumps(result), encoding="utf-8")
