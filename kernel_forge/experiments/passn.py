from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def summarize_passn(
    result_dir: str | Path,
    *,
    case_id: str,
    candidates: list[str],
) -> dict[str, Any]:
    result_path = Path(result_dir)
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        path = result_path / f"{candidate}.json"
        row = {
            "candidate_index": index,
            "team_name": candidate,
            "result_path": path.as_posix(),
            "status": "missing_result",
            "correctness": False,
            "speedup": None,
            "weighted_score": None,
            "error": None,
        }
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            case = _find_case(data.get("cases", []), case_id)
            if case is None:
                row["status"] = "missing_case"
                row["error"] = f"case {case_id} not found"
            else:
                case_update = {
                    "status": case.get("status"),
                    "correctness": bool(case.get("correctness")),
                    "max_abs_diff": case.get("max_abs_diff"),
                    "max_rel_diff": case.get("max_rel_diff"),
                    "baseline_ms": case.get("baseline_ms"),
                    "solution_ms": case.get("solution_ms"),
                    "speedup": case.get("speedup"),
                    "weighted_score": case.get("weighted_score"),
                    "error": case.get("error"),
                }
                output_summaries = _output_summaries(case)
                if output_summaries:
                    case_update["outputs"] = output_summaries
                    case_update["output_count"] = len(output_summaries)
                row.update(case_update)
        rows.append(row)

    passed = [row for row in rows if row["correctness"]]
    best = None
    if passed:
        best = dict(
            max(
                passed,
                key=lambda row: (
                    row["speedup"] if row["speedup"] is not None else -1.0,
                    row["weighted_score"] if row["weighted_score"] is not None else -1.0,
                ),
            ),
        )

    return {
        "case": case_id,
        "candidate_count": len(candidates),
        "pass_at_1": bool(rows[0]["correctness"]) if rows else False,
        "pass_at_n": bool(passed),
        "n": len(candidates),
        "passed_count": len(passed),
        "best_candidate": best,
        "candidates": rows,
    }


def enrich_passn_summary(
    summary: dict[str, Any],
    *,
    generated_candidates: list[dict[str, Any]] | None = None,
    probes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attach generation metadata and backend probes to a Pass@N summary."""
    enriched = deepcopy(summary)
    generated_by_team = {
        str(candidate.get("team_name")): candidate
        for candidate in generated_candidates or []
        if candidate.get("team_name")
    }
    probes_by_team = probes or {}

    for row in enriched.get("candidates", []):
        team_name = str(row.get("team_name"))
        generated = generated_by_team.get(team_name)
        if generated:
            for key in ("provider", "model", "prompt_version", "candidate_path", "submission_root"):
                if generated.get(key) is not None:
                    row[key] = generated[key]
            if generated.get("provider_metadata") is not None:
                row["provider_metadata"] = generated["provider_metadata"]

        probe = probes_by_team.get(team_name)
        if probe:
            row["observed_backend"] = probe.get("last_backend")
            row["backend_probe"] = _probe_summary(probe)

    best = enriched.get("best_candidate")
    if isinstance(best, dict):
        team_name = str(best.get("team_name"))
        probe = probes_by_team.get(team_name)
        generated = generated_by_team.get(team_name)
        if generated and generated.get("provider_metadata") is not None:
            best["provider_metadata"] = generated["provider_metadata"]
        if probe:
            best["observed_backend"] = probe.get("last_backend")
            best["backend_probe"] = _probe_summary(probe)

    return enriched


def apply_passn_to_generated_experiment(
    experiment: dict[str, Any],
    passn_summary: dict[str, Any],
    *,
    results_dir: str,
    probes_dir: str | None = None,
    report_path: str | None = None,
    date: str | None = None,
    machine: str | None = None,
    branch: str | None = None,
    commit: str | None = None,
    akg_commit: str | None = None,
) -> dict[str, Any]:
    """Update a generated experiment YAML after benchmark/probe completion."""
    updated = deepcopy(experiment)
    if date is not None:
        updated["date"] = date
    if machine is not None:
        updated["machine"] = machine
    if branch is not None:
        updated["branch"] = branch
    if commit is not None:
        updated["commit"] = commit
    if akg_commit is not None:
        updated.setdefault("benchmark", {})["source_commit"] = akg_commit

    updated["status"] = "completed" if passn_summary.get("pass_at_n") else "completed_failed"
    pass_n = passn_summary.get("n")
    best = passn_summary.get("best_candidate") or {}
    results = updated.setdefault("results", {})
    results["pass_n"] = {
        "pass_at_1": passn_summary.get("pass_at_1"),
        "pass_at_n": passn_summary.get("pass_at_n"),
        "passed_count": passn_summary.get("passed_count"),
        "n": pass_n,
        "best_candidate": best.get("team_name"),
    }
    if pass_n is not None:
        results["pass_n"][f"pass_at_{pass_n}"] = passn_summary.get("pass_at_n")

    candidates = passn_summary.get("candidates", [])
    correctness_pass = bool(candidates) and all(bool(row.get("correctness")) for row in candidates)
    results["correctness"] = {
        "status": "pass" if correctness_pass else "fail",
        "passed_count": passn_summary.get("passed_count"),
        "candidate_count": passn_summary.get("candidate_count"),
    }
    results["performance"] = {
        "status": "measured" if best else "not_measured",
        "best_candidate": best.get("team_name"),
        "baseline_latency_ms": best.get("baseline_ms"),
        "best_solution_latency_ms": best.get("solution_ms"),
        "best_speedup_vs_baseline": best.get("speedup"),
        "best_weighted_score": best.get("weighted_score"),
    }
    results["completed_cases"] = [
        {
            "candidate_index": row.get("candidate_index"),
            "team_name": row.get("team_name"),
            "status": row.get("status"),
            "correctness": row.get("correctness"),
            "baseline_latency_ms": row.get("baseline_ms"),
            "solution_latency_ms": row.get("solution_ms"),
            "speedup_vs_baseline": row.get("speedup"),
            "weighted_score": row.get("weighted_score"),
            "max_abs_diff": row.get("max_abs_diff"),
            "max_rel_diff": row.get("max_rel_diff"),
            "observed_backend": row.get("observed_backend"),
        }
        for row in candidates
    ]
    for completed_case, row in zip(results["completed_cases"], candidates):
        if row.get("outputs"):
            completed_case["outputs"] = row["outputs"]
            completed_case["output_count"] = len(row["outputs"])

    generated_candidates = updated.setdefault("generation", {}).setdefault("candidates", [])
    generated_by_team = {
        str(candidate.get("team_name")): candidate
        for candidate in generated_candidates
        if candidate.get("team_name")
    }
    for row in candidates:
        target = generated_by_team.get(str(row.get("team_name")))
        if target is None:
            continue
        if row.get("observed_backend") is not None:
            target["observed_backend"] = row["observed_backend"]
        if row.get("backend_probe") is not None:
            target["backend_probe"] = row["backend_probe"]
        target["benchmark_result"] = {
            "result_path": row.get("result_path"),
            "status": row.get("status"),
            "correctness": row.get("correctness"),
            "speedup": row.get("speedup"),
            "weighted_score": row.get("weighted_score"),
            "max_abs_diff": row.get("max_abs_diff"),
            "max_rel_diff": row.get("max_rel_diff"),
        }
        if row.get("outputs"):
            target["benchmark_result"]["outputs"] = row["outputs"]
            target["benchmark_result"]["output_count"] = len(row["outputs"])

    artifacts = updated.setdefault("artifacts", {})
    artifacts["results"] = results_dir
    if probes_dir is not None:
        artifacts["probes"] = probes_dir
    if report_path is not None:
        artifacts["reports"] = report_path

    updated["notes"] = (
        "Replay regression completed; generated experiment metadata was updated "
        "with benchmark results and backend probes."
    )
    return updated


def _find_case(cases: list[dict[str, Any]], case_id: str) -> dict[str, Any] | None:
    for case in cases:
        if case.get("case") == case_id:
            return case
    return None


def _output_summaries(case: dict[str, Any]) -> list[dict[str, Any]]:
    raw_outputs = None
    for key in (
        "outputs",
        "output_details",
        "per_output",
        "per_output_details",
        "output_diffs",
        "correctness_outputs",
    ):
        value = case.get(key)
        if isinstance(value, list):
            raw_outputs = value
            break
    if not raw_outputs:
        return []

    outputs: list[dict[str, Any]] = []
    for index, output in enumerate(raw_outputs):
        if not isinstance(output, dict):
            outputs.append({"index": index, "detail": output})
            continue
        item: dict[str, Any] = {"index": output.get("index", index)}
        for key in (
            "name",
            "status",
            "correctness",
            "shape",
            "dtype",
            "max_abs_diff",
            "max_rel_diff",
            "detail",
            "error",
        ):
            if key in output:
                item[key] = output[key]
        outputs.append(item)
    return outputs


def _probe_summary(probe: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": probe.get("candidate"),
        "shape": probe.get("shape"),
        "output_device": probe.get("output_device"),
        "output_dtype": probe.get("output_dtype"),
        "last_backend": probe.get("last_backend"),
        "last_error": probe.get("last_error"),
        "allclose": probe.get("allclose"),
        "max_abs_diff": probe.get("max_abs_diff"),
        "max_rel_diff": probe.get("max_rel_diff"),
    }
