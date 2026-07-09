from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compare_runner_results_reports_blocked_akg_agents(tmp_path):
    standalone = tmp_path / "standalone.yaml"
    akg = tmp_path / "akg.json"
    output = tmp_path / "comparison.yaml"
    standalone.write_text(yaml.safe_dump(_standalone_report()), encoding="utf-8")
    akg.write_text(
        json.dumps(
            {
                "runner_version": "benchmark-lite-v2",
                "mode": "correctness",
                "config": {"backend": "npu", "pass_n": 1, "cases": ["sigmoid_scale_sum"]},
                "summary": {"total_cases": 1, "total_attempts": 1, "successful_attempts": 0},
                "results": [
                    {
                        "case": "t1/sigmoid_scale_sum",
                        "failure_category": "provider_config_missing",
                        "failure_detail": "模型级别 'standard' 未配置，无法创建 LLM 客户端",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/compare_runner_results.py"),
            "--standalone-report",
            str(standalone),
            "--akg-agents-json",
            str(akg),
            "--case",
            "t1/sigmoid_scale_sum",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )

    comparison = yaml.safe_load(output.read_text())
    assert comparison["akg_agents_runner"]["status"] == "blocked_pre_key_provider_config"
    assert comparison["comparison"]["comparable"] is False
    assert comparison["decision"]["authoritative_runner"] == "standalone_tools_run_bench_py"


def test_compare_runner_results_computes_full_mode_deltas(tmp_path):
    standalone = tmp_path / "standalone.yaml"
    akg = tmp_path / "akg.json"
    standalone.write_text(yaml.safe_dump(_standalone_report()), encoding="utf-8")
    akg.write_text(
        json.dumps(
            {
                "runner_version": "benchmark-lite-v2",
                "mode": "full",
                "config": {"backend": "npu", "pass_n": 4, "cases": ["sigmoid_scale_sum"]},
                "summary": {"total_cases": 1, "total_attempts": 4, "successful_attempts": 4},
                "performance_results": [
                    {
                        "case": "t1/sigmoid_scale_sum",
                        "team_name": "sigmoid_scale_sum_replay_v1",
                        "status": "pass",
                        "correctness": True,
                        "speedup": 1.0,
                        "weighted_score": 60.0,
                    },
                    {
                        "case": "t1/sigmoid_scale_sum",
                        "team_name": "sigmoid_scale_sum_replay_v2",
                        "status": "pass",
                        "correctness": True,
                        "speedup": 1.98,
                        "weighted_score": 69.8,
                    },
                ],
                "performance_summary": {"avg_speedup": 1.49},
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
            "t1/sigmoid_scale_sum",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    comparison = yaml.safe_load(completed.stdout)
    assert comparison["akg_agents_runner"]["status"] == "comparable_performance"
    assert comparison["comparison"]["comparable"] is True
    assert comparison["comparison"]["delta"]["best_candidate_agree"] is True
    assert comparison["comparison"]["delta"]["speedup_delta"] == 0.010000000000000009
    assert comparison["decision"]["authoritative_runner"] == "both_compare_before_final_claim"


def _standalone_report():
    return {
        "case": "t1/sigmoid_scale_sum",
        "pass_at_1": True,
        "pass_at_n": True,
        "n": 4,
        "passed_count": 4,
        "best_candidate": {
            "team_name": "sigmoid_scale_sum_replay_v2",
            "status": "pass",
            "correctness": True,
            "speedup": 1.97,
            "weighted_score": 69.7,
        },
        "candidates": [
            {
                "team_name": "sigmoid_scale_sum_replay_v1",
                "status": "pass",
                "correctness": True,
                "speedup": 1.0,
                "weighted_score": 60.0,
            },
            {
                "team_name": "sigmoid_scale_sum_replay_v2",
                "status": "pass",
                "correctness": True,
                "speedup": 1.97,
                "weighted_score": 69.7,
            },
        ],
    }
