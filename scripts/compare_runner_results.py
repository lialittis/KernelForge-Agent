#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare standalone run_bench.py Pass@N evidence with AKG Agents runner JSON."
    )
    parser.add_argument("--standalone-report", required=True, help="Standalone Pass@N YAML report.")
    parser.add_argument("--akg-agents-json", required=True, help="AKG Agents runner JSON output.")
    parser.add_argument("--case", required=True, help="Case id, for example t1/sigmoid_scale_sum.")
    parser.add_argument("--akg-log-dir", default=None, help="Optional AKG Agents log directory to record.")
    parser.add_argument("--output", default=None, help="Optional YAML comparison output path.")
    args = parser.parse_args()

    standalone = _load_yaml(Path(args.standalone_report))
    akg_agents = _load_json(Path(args.akg_agents_json))
    comparison = compare_runner_results(
        standalone,
        akg_agents,
        case_id=args.case,
        standalone_path=args.standalone_report,
        akg_agents_path=args.akg_agents_json,
        akg_log_dir=args.akg_log_dir,
    )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(comparison, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        yaml.safe_dump(comparison, sys.stdout, sort_keys=False, allow_unicode=True)
    return 0


def compare_runner_results(
    standalone: dict[str, Any],
    akg_agents: dict[str, Any],
    *,
    case_id: str,
    standalone_path: str,
    akg_agents_path: str,
    akg_log_dir: str | None = None,
) -> dict[str, Any]:
    standalone_summary = _standalone_summary(standalone, case_id=case_id, source_path=standalone_path)
    akg_summary = _akg_agents_summary(akg_agents, case_id=case_id, source_path=akg_agents_path)
    comparable = akg_summary["status"] == "comparable_performance"
    delta = _comparison_delta(standalone_summary, akg_summary) if comparable else None
    decision = _decision(standalone_summary, akg_summary, comparable=comparable)

    return {
        "case": case_id,
        "standalone_runner": standalone_summary,
        "akg_agents_runner": akg_summary,
        "comparison": {
            "comparable": comparable,
            "delta": delta,
            "schema": {
                "standalone": "passn_yaml_candidates",
                "akg_agents": akg_summary["schema"],
            },
            "logs": {
                "akg_agents_log_dir": akg_log_dir,
                "status": "provided" if akg_log_dir else "not_provided",
            },
        },
        "decision": decision,
    }


def _standalone_summary(data: dict[str, Any], *, case_id: str, source_path: str) -> dict[str, Any]:
    candidates = data.get("candidates") or []
    best = data.get("best_candidate") or {}
    return {
        "source": source_path,
        "status": "complete" if candidates else "missing_candidates",
        "case": data.get("case") or case_id,
        "akg_commit": data.get("akg_commit"),
        "pass_at_1": data.get("pass_at_1"),
        "pass_at_n": data.get("pass_at_n"),
        "n": data.get("n"),
        "passed_count": data.get("passed_count"),
        "best_candidate": _candidate_summary(best),
        "candidates": [_candidate_summary(row) for row in candidates],
    }


def _akg_agents_summary(data: dict[str, Any], *, case_id: str, source_path: str) -> dict[str, Any]:
    schema_fields = sorted(data.keys())
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    performance_results = data.get("performance_results")
    if performance_results is None:
        performance_results = []
    if not isinstance(performance_results, list):
        performance_results = []

    rows = [_normalise_akg_performance_row(row) for row in performance_results if _matches_case(row, case_id)]
    rows = [row for row in rows if row is not None]
    best = _best_candidate(rows)

    failure = _find_failure(data)
    if failure == "provider_config_missing":
        status = "blocked_pre_key_provider_config"
    elif data.get("mode") in {"performance", "full"} and rows:
        status = "comparable_performance"
    elif data.get("mode") in {"performance", "full"}:
        status = "missing_performance_results"
    else:
        status = "schema_only_or_correctness_only"

    pass_n = _get_nested(data, "config", "pass_n")
    successful_attempts = summary.get("successful_attempts")
    total_attempts = summary.get("total_attempts")

    return {
        "source": source_path,
        "status": status,
        "mode": data.get("mode"),
        "runner_version": data.get("runner_version"),
        "schema": {
            "top_level_fields": schema_fields,
            "has_performance_results": "performance_results" in data,
            "has_performance_summary": "performance_summary" in data,
        },
        "config": {
            "backend": _get_nested(data, "config", "backend"),
            "pass_n": pass_n,
            "cases": _get_nested(data, "config", "cases"),
        },
        "summary": {
            "total_cases": summary.get("total_cases"),
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "pass_at_n": _pass_at_n(total_attempts, successful_attempts),
        },
        "failure": failure,
        "best_candidate": best,
        "candidates": rows,
    }


def _candidate_summary(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    return {
        "team_name": row.get("team_name") or row.get("candidate") or row.get("name"),
        "status": row.get("status"),
        "correctness": row.get("correctness"),
        "speedup": row.get("speedup"),
        "weighted_score": row.get("weighted_score"),
        "baseline_ms": row.get("baseline_ms"),
        "solution_ms": row.get("solution_ms"),
        "max_abs_diff": row.get("max_abs_diff"),
        "max_rel_diff": row.get("max_rel_diff"),
    }


def _normalise_akg_performance_row(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    return {
        "team_name": row.get("team_name") or row.get("team") or row.get("candidate") or row.get("case"),
        "case": row.get("case") or row.get("case_id") or row.get("name"),
        "status": row.get("status"),
        "correctness": row.get("correctness"),
        "speedup": row.get("speedup") or row.get("speedup_vs_baseline"),
        "weighted_score": row.get("weighted_score") or row.get("score"),
        "baseline_ms": row.get("baseline_ms") or row.get("baseline_latency_ms"),
        "solution_ms": row.get("solution_ms") or row.get("solution_latency_ms"),
        "max_abs_diff": row.get("max_abs_diff"),
        "max_rel_diff": row.get("max_rel_diff"),
    }


def _comparison_delta(standalone: dict[str, Any], akg_agents: dict[str, Any]) -> dict[str, Any]:
    standalone_best = standalone.get("best_candidate") or {}
    akg_best = akg_agents.get("best_candidate") or {}
    return {
        "pass_at_n_agree": standalone.get("pass_at_n") == _get_nested(akg_agents, "summary", "pass_at_n"),
        "standalone_best": standalone_best.get("team_name"),
        "akg_agents_best": akg_best.get("team_name"),
        "best_candidate_agree": standalone_best.get("team_name") == akg_best.get("team_name"),
        "speedup_delta": _numeric_delta(akg_best.get("speedup"), standalone_best.get("speedup")),
        "weighted_score_delta": _numeric_delta(akg_best.get("weighted_score"), standalone_best.get("weighted_score")),
    }


def _decision(
    standalone: dict[str, Any],
    akg_agents: dict[str, Any],
    *,
    comparable: bool,
) -> dict[str, Any]:
    if comparable:
        return {
            "authoritative_runner": "both_compare_before_final_claim",
            "reason": (
                "Both standalone and AKG Agents full-mode outputs include "
                "Pass@N/performance fields; inspect deltas before final evidence claims."
            ),
        }
    if akg_agents.get("status") == "blocked_pre_key_provider_config":
        return {
            "authoritative_runner": "standalone_tools_run_bench_py",
            "reason": (
                "AKG Agents runner is schema-producing but blocked before full-mode "
                "performance by missing standard model configuration."
            ),
        }
    return {
        "authoritative_runner": "standalone_tools_run_bench_py_pending_akg_agents_full_results",
        "reason": "AKG Agents JSON is not yet comparable with standalone Pass@N performance output.",
    }


def _best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    passed = [row for row in rows if row.get("correctness") is not False and row.get("speedup") is not None]
    if not passed:
        return None
    return max(
        passed,
        key=lambda row: (
            row.get("speedup") if row.get("speedup") is not None else -1.0,
            row.get("weighted_score") if row.get("weighted_score") is not None else -1.0,
        ),
    )


def _matches_case(row: Any, case_id: str) -> bool:
    if not isinstance(row, dict):
        return False
    values = {row.get("case"), row.get("case_id"), row.get("name")}
    case_name = case_id.split("/")[-1]
    return case_id in values or case_name in values or not any(values)


def _find_failure(data: dict[str, Any]) -> str | None:
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        for key in ("failure_category", "error_category", "category"):
            if item.get(key) == "provider_config_missing":
                return "provider_config_missing"
        detail = str(item.get("failure_detail") or item.get("error") or "")
        if "standard" in detail and ("未配置" in detail or "not configured" in detail):
            return "provider_config_missing"
    text = json.dumps(data, ensure_ascii=False)
    if "模型级别 'standard' 未配置" in text:
        return "provider_config_missing"
    return None


def _pass_at_n(total_attempts: Any, successful_attempts: Any) -> bool | None:
    if successful_attempts is None:
        return None
    try:
        return int(successful_attempts) > 0
    except Exception:
        return None


def _numeric_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    try:
        return float(left) - float(right)
    except Exception:
        return None


def _get_nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object: {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
