#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_forge.benchmark import scan_benchmark_cases

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_akg_agents_model_config import inspect_model_config


BENCH_DIR = "third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite"
UPDATED_AKG_COMMIT = "47aa428fcdc8c68f78d331dc578bc6c74fb9d91d"

REQUIRED_GROUPS = {
    "runner_path_comparison": [
        "experiments/reports/2026-07-09-runner-path-comparison-sigmoid-scale-sum.yaml",
        "experiments/runs/2026-07-09-runner-path-comparison-sigmoid-scale-sum.yaml",
        "scripts/check_akg_agents_model_config.py",
        "scripts/run_akg_agents_full_comparison.sh",
        "scripts/compare_runner_results.py",
        "scripts/run_akg_agents_verifier_probe.py",
        "scripts/run_ascend_verifier_probe.sh",
    ],
    "generated_result_import": [
        "scripts/run_replay_regression.py",
        "kernel_forge/experiments/passn.py",
        "tests/test_experiment_result_import.py",
    ],
    "t2_t3_opspecs": [
        "benchmarks/parsed/t2_add_rmsnorm_cast.yaml",
        "benchmarks/parsed/t2_add_rmsnorm_quant.yaml",
        "benchmarks/parsed/t2_rope.yaml",
        "benchmarks/parsed/t3_layernorm_gated.yaml",
        "benchmarks/parsed/t3_causal_conv1d.yaml",
        "benchmarks/parsed/t3_decode_mla.yaml",
    ],
    "priority_operator_reports": [
        "experiments/reports/2026-07-09-add-rmsnorm-cast-pass4.yaml",
        "experiments/reports/2026-07-09-add-rmsnorm-quant-pass4.yaml",
        "experiments/reports/2026-07-09-rope-pass4.yaml",
        "experiments/reports/2026-07-09-layernorm-gated-pass4.yaml",
    ],
    "deterministic_replay_regression": [
        "experiments/reports/2026-07-09-replay-sigmoid-scale-sum-pass4-updated-akg.yaml",
        "experiments/runs/2026-07-09-replay-sigmoid-scale-sum-pass4-updated-akg.yaml",
        "scripts/run_replay_regression.py",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit deterministic pre-key readiness for SketchSkill-AKG."
    )
    parser.add_argument("--repo-root", default=str(ROOT), help="Repository root to audit.")
    parser.add_argument("--output", default=None, help="Optional YAML output path.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of YAML.")
    parser.add_argument(
        "--require-standard-config",
        action="store_true",
        help="Return exit code 2 when deterministic work is complete but AKG Agents standard config is missing.",
    )
    parser.add_argument(
        "--check-ascend-ssh",
        action="store_true",
        help="Optionally check BatchMode SSH access to the Ascend worker.",
    )
    parser.add_argument("--ascend-host", default="ascend-kf", help="SSH host alias for the Ascend worker.")
    parser.add_argument(
        "--remote-dir",
        default="/data/KernelForge-Agent",
        help="Repository path on the Ascend worker.",
    )
    parser.add_argument("--ssh-timeout", type=int, default=10, help="SSH connection timeout in seconds.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    audit = audit_pre_key_readiness(
        repo_root,
        environ=os.environ,
        home=Path.home(),
        check_ascend_ssh=args.check_ascend_ssh,
        ascend_host=args.ascend_host,
        remote_dir=args.remote_dir,
        ssh_timeout=args.ssh_timeout,
    )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(audit, sort_keys=False, allow_unicode=True), encoding="utf-8")
    elif args.json:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
    else:
        yaml.safe_dump(audit, sys.stdout, sort_keys=False, allow_unicode=True)

    overall = audit["overall"]
    if not overall["deterministic_complete"]:
        return 1
    if args.require_standard_config and not overall["full_runner_comparison_ready"]:
        return 2
    return 0


def audit_pre_key_readiness(
    repo_root: Path,
    *,
    environ: Mapping[str, str],
    home: Path,
    check_ascend_ssh: bool = False,
    ascend_host: str = "ascend-kf",
    remote_dir: str = "/data/KernelForge-Agent",
    ssh_timeout: int = 10,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    for check_id, paths in REQUIRED_GROUPS.items():
        checks.append(_file_group_check(repo_root, check_id, paths))

    checks.append(_registry_check(repo_root))
    checks.append(_runner_decision_check(repo_root))
    checks.append(_replay_report_check(repo_root))
    checks.append(_priority_report_check(repo_root))
    checks.append(_standard_model_config_check(repo_root, environ=environ, home=home))
    if check_ascend_ssh:
        checks.append(
            _ascend_ssh_check(
                repo_root,
                host=ascend_host,
                remote_dir=remote_dir,
                timeout=ssh_timeout,
            )
        )

    blocking = [check for check in checks if check["status"] == "blocked"]
    failing = [check for check in checks if check["status"] == "fail"]
    deterministic_complete = not failing
    model_check = next(check for check in checks if check["id"] == "akg_agents_standard_model_config")
    full_runner_ready = model_check["status"] == "pass"
    ascend_check = next((check for check in checks if check["id"] == "ascend_batchmode_ssh"), None)

    if not deterministic_complete:
        status = "incomplete"
    elif full_runner_ready:
        status = "ready_for_full_runner_comparison"
    else:
        status = "pre_key_deterministic_complete_provider_config_missing"

    return {
        "id": "pre_key_readiness_audit",
        "status": status,
        "overall": {
            "deterministic_complete": deterministic_complete,
            "full_runner_comparison_ready": full_runner_ready,
            "failing_checks": [check["id"] for check in failing],
            "blocked_checks": [check["id"] for check in blocking],
            "authoritative_pre_key_runner": "standalone_tools_run_bench_py",
            "ascend_batchmode_ssh_ready": None if ascend_check is None else ascend_check["status"] == "pass",
            "next_unblock": (
                "Configure AKG Agents standard model credentials, then run "
                "scripts/run_akg_agents_full_comparison.sh and compare with "
                "scripts/compare_runner_results.py."
            ),
        },
        "checks": checks,
    }


def _file_group_check(repo_root: Path, check_id: str, paths: list[str]) -> dict[str, Any]:
    missing = [path for path in paths if not (repo_root / path).exists()]
    return {
        "id": check_id,
        "status": "pass" if not missing else "fail",
        "required_files": paths,
        "missing_files": missing,
    }


def _registry_check(repo_root: Path) -> dict[str, Any]:
    bench_dir = repo_root / BENCH_DIR
    if not bench_dir.exists():
        return {
            "id": "benchmark_registry_t2_t3_coverage",
            "status": "fail",
            "reason": f"Benchmark directory not found: {BENCH_DIR}",
        }

    try:
        registry = scan_benchmark_cases(bench_dir, repo_root=repo_root)
    except Exception as exc:
        return {
            "id": "benchmark_registry_t2_t3_coverage",
            "status": "fail",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    summary = registry["summary"]
    by_support = summary.get("by_support", {})
    total_cases = summary.get("total_cases")
    supported = by_support.get("opspec_supported", 0)
    parse_failed = by_support.get("parse_failed", 0)
    unsupported = by_support.get("unsupported", 0)
    ok = total_cases == 13 and supported == 13 and unsupported == 0 and parse_failed == 0

    return {
        "id": "benchmark_registry_full_lite_coverage",
        "status": "pass" if ok else "fail",
        "total_cases": total_cases,
        "opspec_supported": supported,
        "unsupported": unsupported,
        "parse_failed": parse_failed,
        "by_tier": summary.get("by_tier", {}),
    }


def _runner_decision_check(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "experiments/reports/2026-07-09-runner-path-comparison-sigmoid-scale-sum.yaml"
    try:
        report = _load_yaml(path)
    except Exception as exc:
        return {"id": "runner_path_decision", "status": "fail", "reason": str(exc)}

    standalone = report.get("standalone_runner") or {}
    akg_agents = report.get("akg_agents_runner") or {}
    decision = report.get("decision") or {}
    ok = (
        report.get("akg_commit") == UPDATED_AKG_COMMIT
        and standalone.get("pass_at_4") is True
        and standalone.get("passed_count") == 4
        and decision.get("current_authoritative_pre_key_runner") == "standalone_tools_run_bench_py"
        and akg_agents.get("status") in {"blocked_pre_key_provider_config", "comparable_performance"}
    )
    return {
        "id": "runner_path_decision",
        "status": "pass" if ok else "fail",
        "akg_commit": report.get("akg_commit"),
        "standalone_status": standalone.get("status"),
        "standalone_pass_at_4": standalone.get("pass_at_4"),
        "standalone_best_candidate": standalone.get("best_candidate"),
        "akg_agents_status": akg_agents.get("status"),
        "authoritative_pre_key_runner": decision.get("current_authoritative_pre_key_runner"),
    }


def _replay_report_check(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "experiments/reports/2026-07-09-replay-sigmoid-scale-sum-pass4-updated-akg.yaml"
    try:
        report = _load_yaml(path)
    except Exception as exc:
        return {"id": "updated_akg_replay_pass4_import", "status": "fail", "reason": str(exc)}

    candidates = report.get("candidates") or []
    rows_have_probe = all(
        isinstance(row, dict)
        and row.get("observed_backend")
        and isinstance(row.get("backend_probe"), dict)
        and isinstance(row.get("provider_metadata"), dict)
        for row in candidates
    )
    ok = (
        report.get("case") == "t1/sigmoid_scale_sum"
        and report.get("akg_commit") == UPDATED_AKG_COMMIT
        and report.get("pass_at_1") is True
        and report.get("pass_at_n") is True
        and report.get("n") == 4
        and report.get("passed_count") == 4
        and rows_have_probe
    )
    best = report.get("best_candidate") or {}
    return {
        "id": "updated_akg_replay_pass4_import",
        "status": "pass" if ok else "fail",
        "case": report.get("case"),
        "pass_at_1": report.get("pass_at_1"),
        "pass_at_n": report.get("pass_at_n"),
        "passed_count": report.get("passed_count"),
        "best_candidate": best.get("team_name"),
        "best_speedup": best.get("speedup"),
        "candidate_rows_have_probe_and_provider_metadata": rows_have_probe,
    }


def _priority_report_check(repo_root: Path) -> dict[str, Any]:
    reports = [
        "experiments/reports/2026-07-09-add-rmsnorm-cast-pass4.yaml",
        "experiments/reports/2026-07-09-add-rmsnorm-quant-pass4.yaml",
        "experiments/reports/2026-07-09-rope-pass4.yaml",
        "experiments/reports/2026-07-09-layernorm-gated-pass4.yaml",
    ]
    rows = []
    missing_or_bad = []
    for report_path in reports:
        try:
            report = _load_yaml(repo_root / report_path)
        except Exception as exc:
            missing_or_bad.append(f"{report_path}: {exc}")
            continue
        best = report.get("best_candidate") or {}
        row = {
            "path": report_path,
            "case": report.get("case"),
            "pass_at_1": report.get("pass_at_1"),
            "pass_at_n": report.get("pass_at_n"),
            "n": report.get("n"),
            "passed_count": report.get("passed_count"),
            "best_candidate": best.get("team_name"),
            "best_speedup": best.get("speedup"),
            "best_observed_backend": best.get("observed_backend"),
        }
        rows.append(row)
        if row["pass_at_1"] is not True or row["pass_at_n"] is not True or row["n"] != 4:
            missing_or_bad.append(report_path)

    return {
        "id": "priority_operator_pass4_reports",
        "status": "pass" if not missing_or_bad else "fail",
        "reports": rows,
        "issues": missing_or_bad,
    }


def _standard_model_config_check(
    repo_root: Path,
    *,
    environ: Mapping[str, str],
    home: Path,
) -> dict[str, Any]:
    result = inspect_model_config("standard", repo_root=repo_root, environ=environ, home=home)
    return {
        "id": "akg_agents_standard_model_config",
        "status": "pass" if result["configured"] else "blocked",
        "configured": result["configured"],
        "source": result["source"],
        "missing": result["missing"],
        "available_levels": result["available_levels"],
        "checked_paths": result["checked_paths"],
    }


def _ascend_ssh_check(
    repo_root: Path,
    *,
    host: str,
    remote_dir: str,
    timeout: int,
) -> dict[str, Any]:
    remote_cmd = f"cd {shlex.quote(remote_dir)} && git status --short --branch"
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={timeout}",
        host,
        remote_cmd,
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=max(timeout + 5, 10),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "id": "ascend_batchmode_ssh",
            "status": "blocked",
            "host": host,
            "remote_dir": remote_dir,
            "command": _redacted_command(cmd),
            "returncode": None,
            "reason": f"timeout after {exc.timeout} seconds",
        }
    except OSError as exc:
        return {
            "id": "ascend_batchmode_ssh",
            "status": "blocked",
            "host": host,
            "remote_dir": remote_dir,
            "command": _redacted_command(cmd),
            "returncode": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    output = (completed.stdout or "").strip()
    error = (completed.stderr or "").strip()
    return {
        "id": "ascend_batchmode_ssh",
        "status": "pass" if completed.returncode == 0 else "blocked",
        "host": host,
        "remote_dir": remote_dir,
        "command": _redacted_command(cmd),
        "returncode": completed.returncode,
        "branch_status": output if completed.returncode == 0 else None,
        "reason": None if completed.returncode == 0 else (error or output or "ssh command failed"),
    }


def _redacted_command(cmd: list[str]) -> list[str]:
    return [str(part) for part in cmd]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object: {path}")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
