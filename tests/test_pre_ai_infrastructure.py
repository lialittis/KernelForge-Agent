from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_forge.agents.prompts import render_code_prompt
from kernel_forge.agents.skills import load_retrieved_skills, retrieve_skill_paths
from kernel_forge.benchmark import read_yaml, validate_opspec_dir, validate_opspec_file, validate_sketch
from scripts.prepare_gitlink_package import EXCLUDE_DIR_NAMES, INCLUDE_PATHS


def test_all_parsed_opspecs_pass_validator():
    report = validate_opspec_dir(ROOT / "benchmarks/parsed")

    assert report["status"] == "pass"
    assert report["total_files"] == 13
    assert report["passed_files"] == 13
    assert report["failed_files"] == 0
    assert report["issues"] == []


def test_opspec_validator_reports_structural_errors(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "id": "t1/bad",
                "name": "bad",
                "tier": "t1",
                "category": "reduction",
                "source_path": "t1/bad.py",
                "reference_class": "Model",
                "candidate_class": "ModelNew",
                "inputs": [],
                "outputs": [{"name": "out", "shape": [], "dtype": "", "layout": "contiguous"}],
                "semantics": {},
                "validation": {},
                "performance": {},
                "sketch": {"operator_category": "unknown"},
                "submission": {"entrypoint": "ModelNew", "required_files": ["t1/bad.py"]},
            }
        ),
        encoding="utf-8",
    )

    messages = [issue.message for issue in validate_opspec_file(bad)]

    assert "inputs must be a non-empty list" in messages
    assert "sketch.operator_category must be non-generic" in messages
    assert any("validation missing required key" in message for message in messages)


def test_sketch_validator_catches_matmul_contract_errors():
    spec = read_yaml(ROOT / "benchmarks/parsed/t1_matmul_biasadd.yaml")
    sketch = dict(spec["sketch"])
    sketch["tile_plan"] = {**sketch["tile_plan"], "axes": ["M", "K", "N"]}
    sketch["memory_plan"] = {**sketch["memory_plan"], "bias": "none"}

    messages = [issue.message for issue in validate_sketch(sketch, spec=spec, path="matmul")]

    assert "matmul sketch tile_plan.axes must be ['M', 'N', 'K']" in messages
    assert "matmul_biasadd sketch must describe bias broadcast" in messages


def test_validate_opspecs_cli_is_provider_independent_gate():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_opspecs.py"),
            "--opspec-dir",
            str(ROOT / "benchmarks/parsed"),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(completed.stdout)
    assert report["status"] == "pass"
    assert report["total_files"] == 13


def test_prompt_context_snapshot_for_matmul_and_moe_topk():
    cases = {
        "benchmarks/parsed/t1_matmul_biasadd.yaml": [
            "skills/matmul_like/SKILL.md",
            "blocked_mnk_matmul",
            "bias[0, n] broadcasts over M",
            "tile_plan:",
        ],
        "benchmarks/parsed/t2_moe_topk_softmax.yaml": [
            "skills/reduction/SKILL.md",
            "moe_topk_softmax",
            "top_k_probs",
            "top_k_indices",
            "topk_probability_renormalization",
        ],
    }

    for rel_path, expected_fragments in cases.items():
        opspec = read_yaml(ROOT / rel_path)
        skill_paths = retrieve_skill_paths(opspec)
        skills = load_retrieved_skills(skill_paths, repo_root=ROOT)
        prompt, version = render_code_prompt(
            opspec=opspec,
            backend="triton_ascend",
            candidate_index=1,
            pass_n=4,
            skills=skills,
            repo_root=ROOT,
        )

        assert version == "code_agent.v1"
        assert f"Case: {opspec['id']}" in prompt
        assert "Backend: triton_ascend" in prompt
        assert "Candidate: 1 / 4" in prompt
        for fragment in expected_fragments:
            assert fragment in prompt


def test_package_hygiene_contract_excludes_outputs_and_runtime_caches():
    assert "outputs" in EXCLUDE_DIR_NAMES
    assert ".git" in EXCLUDE_DIR_NAMES
    assert "__pycache__" in EXCLUDE_DIR_NAMES
    assert "outputs" not in INCLUDE_PATHS
    assert all(not item.startswith("outputs/") for item in INCLUDE_PATHS)
    assert "skills" in INCLUDE_PATHS
    assert "prompts" in INCLUDE_PATHS
    assert "benchmarks/parsed" in INCLUDE_PATHS


def test_prepare_gitlink_package_to_tmp_manifest_is_clean(tmp_path):
    output_root = tmp_path / "package"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_gitlink_package.py"),
            "--team",
            "test-team",
            "--output-root",
            str(output_root),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    package_root = output_root / "projects/test-team/SketchSkill-AKG"
    manifest = json.loads((package_root / "PACKAGE_MANIFEST.json").read_text())

    assert "Package prepared:" in completed.stdout
    assert manifest["copied_file_count"] > 0
    assert not any("/outputs/" in path for path in manifest["copied_files"])
    assert (package_root / "README.md").exists()
    assert (package_root / "PROJECT_README.md").exists()


def test_replay_regression_cli_exposes_akg_commit_guard():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_replay_regression.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--expected-akg-commit" in completed.stdout
    assert "--allow-akg-commit-mismatch" in completed.stdout
    assert "47aa428fcdc8c68f78d331dc578bc6c74fb9d91d" in completed.stdout


def test_result_comparison_preserves_multi_output_candidate_details(tmp_path):
    standalone = tmp_path / "standalone.yaml"
    akg = tmp_path / "akg.json"
    standalone.write_text(
        yaml.safe_dump(
            {
                "case": "t2/moe_topk_softmax",
                "pass_at_1": True,
                "pass_at_n": True,
                "n": 1,
                "passed_count": 1,
                "best_candidate": {
                    "team_name": "moe_topk_replay_v1",
                    "correctness": True,
                    "speedup": 0.8,
                    "outputs": [
                        {"name": "top_k_probs", "dtype": "float32", "max_abs_diff": 1.0e-6},
                        {"name": "top_k_indices", "dtype": "int64", "max_abs_diff": 0.0},
                    ],
                },
                "candidates": [
                    {
                        "team_name": "moe_topk_replay_v1",
                        "correctness": True,
                        "speedup": 0.8,
                        "outputs": [
                            {"name": "top_k_probs", "dtype": "float32", "max_abs_diff": 1.0e-6},
                            {"name": "top_k_indices", "dtype": "int64", "max_abs_diff": 0.0},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    akg.write_text(
        json.dumps(
            {
                "runner_version": "benchmark-lite-v2",
                "mode": "full",
                "config": {"backend": "npu", "pass_n": 1, "cases": ["moe_topk_softmax"]},
                "summary": {"total_cases": 1, "total_attempts": 1, "successful_attempts": 1},
                "performance_results": [
                    {
                        "case": "t2/moe_topk_softmax",
                        "team_name": "moe_topk_replay_v1",
                        "status": "pass",
                        "correctness": True,
                        "speedup": 0.8,
                        "outputs": [
                            {"name": "top_k_probs", "dtype": "float32", "max_abs_diff": 1.0e-6},
                            {"name": "top_k_indices", "dtype": "int64", "max_abs_diff": 0.0},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/compare_runner_results.py"),
            "--standalone-report",
            str(standalone),
            "--akg-agents-json",
            str(akg),
            "--case",
            "t2/moe_topk_softmax",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    comparison = yaml.safe_load(completed.stdout)
    assert comparison["standalone_runner"]["best_candidate"]["output_count"] == 2
    assert comparison["akg_agents_runner"]["best_candidate"]["outputs"][1]["dtype"] == "int64"
