from __future__ import annotations

import json
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
                row.update(
                    {
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
                )
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


def _find_case(cases: list[dict[str, Any]], case_id: str) -> dict[str, Any] | None:
    for case in cases:
        if case.get("case") == case_id:
            return case
    return None
