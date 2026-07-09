from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from kernel_forge.benchmark import read_yaml


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def import_benchmark_result(
    result_json: str | Path,
    *,
    experiment_path: str | Path | None = None,
    probe_json: str | Path | None = None,
) -> dict[str, Any]:
    result = load_json(result_json)
    probe = load_json(probe_json) if probe_json else None
    if experiment_path:
        experiment = read_yaml(experiment_path)
        return apply_result_to_experiment(
            experiment,
            result,
            result_path=Path(result_json).as_posix(),
            probe=probe,
            probe_path=Path(probe_json).as_posix() if probe_json else None,
        )
    return summarize_result(result, result_path=Path(result_json).as_posix(), probe=probe)


def summarize_result(
    result: dict[str, Any],
    *,
    result_path: str | None = None,
    probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cases = [_case_summary(case) for case in result.get("cases", [])]
    summary = {
        "team_name": result.get("team_name"),
        "device": result.get("device"),
        "timestamp": result.get("timestamp"),
        "result_path": result_path,
        "bench_config": result.get("bench_config", {}),
        "summary": result.get("summary", {}),
        "cases": cases,
    }
    if probe is not None:
        summary["probe"] = _probe_summary(probe)
    return summary


def apply_result_to_experiment(
    experiment: dict[str, Any],
    result: dict[str, Any],
    *,
    result_path: str | None = None,
    probe: dict[str, Any] | None = None,
    probe_path: str | None = None,
) -> dict[str, Any]:
    updated = deepcopy(experiment)
    cases = result.get("cases", [])
    task_id = updated.get("benchmark", {}).get("task_id")
    selected = _select_case(cases, task_id)
    bench_config = result.get("bench_config", {})

    if selected is not None:
        output_summaries = _output_summaries(selected)
        correctness = updated.setdefault("results", {}).setdefault("correctness", {})
        correctness["status"] = "pass" if selected.get("correctness") else "fail"
        correctness["rtol"] = bench_config.get("rtol")
        correctness["atol"] = bench_config.get("atol")
        correctness["max_abs_diff"] = selected.get("max_abs_diff")
        correctness["max_rel_diff"] = selected.get("max_rel_diff")
        correctness["detail"] = selected.get("correctness_detail")
        if output_summaries:
            correctness["outputs"] = output_summaries
            correctness["output_count"] = len(output_summaries)

        performance = updated.setdefault("results", {}).setdefault("performance", {})
        performance["status"] = (
            "measured" if selected.get("correctness") else "not_measured_failed_correctness"
        )
        performance["warmup"] = bench_config.get("warmup_runs")
        performance["repeats"] = bench_config.get("iterations")
        performance["num_trials"] = bench_config.get("num_trials")
        performance["baseline_latency_ms"] = selected.get("baseline_ms")
        performance["solution_latency_ms"] = selected.get("solution_ms")
        performance["speedup_vs_baseline"] = selected.get("speedup")
        performance["weighted_score"] = selected.get("weighted_score")

        if selected.get("correctness"):
            speedup = selected.get("speedup") or 0
            updated["status"] = "pass" if speedup >= 1.0 else "pass_but_slow"
        else:
            updated["status"] = "fail"

    runtime = updated.setdefault("results", {}).setdefault("runtime", {})
    if probe is not None:
        runtime["status"] = "probed"
        runtime["last_backend"] = probe.get("last_backend")
        runtime["last_error"] = probe.get("last_error")
    elif selected is not None:
        runtime.setdefault("status", "benchmark_imported")

    artifacts = updated.setdefault("artifacts", {})
    if result_path is not None:
        artifacts["results"] = result_path
    if probe_path is not None:
        artifacts["probe"] = probe_path

    return updated


def _select_case(
    cases: list[dict[str, Any]],
    task_id: str | None,
) -> dict[str, Any] | None:
    if not cases:
        return None
    if task_id is None:
        return cases[0] if len(cases) == 1 else None
    for case in cases:
        if case.get("case") == task_id:
            return case
    return None


def _case_summary(case: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "case": case.get("case"),
        "tier": case.get("tier"),
        "status": case.get("status"),
        "correctness": case.get("correctness"),
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
        summary["outputs"] = output_summaries
        summary["output_count"] = len(output_summaries)
    return summary


def _output_summaries(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve per-output correctness details from richer runner schemas.

    The current AKG Bench Lite runner emits aggregate max diffs at case level,
    while tuple-output cases are checked internally. This accepts several
    likely future key names so imports stay deterministic if the runner starts
    emitting per-output detail.
    """
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
