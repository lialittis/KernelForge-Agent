from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_forge.experiments import import_benchmark_result


def test_importer_updates_single_case_experiment(tmp_path):
    result_json = _write_result_json(tmp_path)
    probe_json = tmp_path / "probe.json"
    probe_json.write_text(
        json.dumps(
            {
                "last_backend": "triton_tanh_sigmoid_form_bs16384x2",
                "last_error": None,
                "allclose": True,
                "max_abs_diff": 4.7e-7,
                "max_rel_diff": 3.0e-6,
            }
        ),
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "id": "example",
                "status": "planned",
                "benchmark": {"task_id": "t1/gelu"},
                "results": {
                    "runtime": {"status": "not_run"},
                    "correctness": {"status": "not_run"},
                    "performance": {"status": "not_run"},
                },
            }
        ),
        encoding="utf-8",
    )

    updated = import_benchmark_result(
        result_json,
        experiment_path=experiment,
        probe_json=probe_json,
    )

    assert updated["status"] == "pass_but_slow"
    assert updated["results"]["correctness"]["status"] == "pass"
    assert updated["results"]["correctness"]["max_abs_diff"] == 4.76837158203125e-07
    assert updated["results"]["performance"]["speedup_vs_baseline"] == 0.6059
    assert updated["results"]["runtime"]["last_backend"] == "triton_tanh_sigmoid_form_bs16384x2"
    assert updated["artifacts"]["results"] == result_json.as_posix()


def test_importer_summarizes_result_without_experiment(tmp_path):
    result_json = _write_result_json(tmp_path)

    summary = import_benchmark_result(result_json)

    assert summary["team_name"] == "gelu_triton_v13"
    assert summary["summary"]["passed"] == 1
    assert summary["cases"][0]["case"] == "t1/gelu"
    assert summary["cases"][0]["weighted_score"] == 36.35


def test_import_result_cli_writes_yaml(tmp_path):
    result_json = _write_result_json(tmp_path)
    output = tmp_path / "summary.yaml"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/import_benchmark_result.py"),
            "--result-json",
            str(result_json),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )

    data = yaml.safe_load(output.read_text())
    assert data["team_name"] == "gelu_triton_v13"
    assert data["cases"][0]["speedup"] == 0.6059


def _write_result_json(tmp_path: Path) -> Path:
    result = {
        "team_name": "gelu_triton_v13",
        "device": "npu",
        "timestamp": "2026-07-03T00:00:00",
        "bench_config": {
            "rtol": 0.01,
            "atol": 0.01,
            "warmup_runs": 10,
            "iterations": 100,
            "num_trials": 3,
        },
        "cases": [
            {
                "case": "t1/gelu",
                "tier": "t1",
                "correctness": True,
                "max_abs_diff": 4.76837158203125e-07,
                "max_rel_diff": 3.0e-6,
                "correctness_detail": "PASS",
                "status": "pass",
                "baseline_ms": 0.0438129500253126,
                "solution_ms": 0.07231052983217132,
                "speedup": 0.6059,
                "weighted_score": 36.35,
                "error": None,
            }
        ],
        "summary": {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "total_weighted_score": 36.35,
            "avg_speedup": 0.6059,
        },
    }
    path = tmp_path / "gelu_triton_v13.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path
