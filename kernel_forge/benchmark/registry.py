from __future__ import annotations

from pathlib import Path
from typing import Any

from .extractor import ExtractionError, case_support, classify_case, inspect_case
from .opspec import TensorSpec


def scan_benchmark_cases(
    bench_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    bench_path = Path(bench_dir)
    root = Path(repo_root) if repo_root else Path.cwd()
    cases = []
    for case_file in sorted(bench_path.glob("t[0-9]/*.py")):
        try:
            parsed = inspect_case(case_file, repo_root=root)
            support = case_support(parsed["id"])
            cases.append(_registry_case(parsed, support))
        except (ExtractionError, SyntaxError, ValueError) as exc:
            tier = case_file.parent.name
            name = case_file.stem
            case_id = f"{tier}/{name}"
            cases.append(
                {
                    "id": case_id,
                    "name": name,
                    "tier": tier,
                    "category": classify_case(case_id),
                    "source_path": _display_path(case_file, root),
                    "support": {
                        "status": "parse_failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    },
                    "inputs": [],
                    "init_inputs": [],
                    "forward_args": [],
                    "forward_expression": None,
                }
            )
    return {
        "benchmark": {
            "name": "akg_kernels_bench_lite",
            "path": _display_path(bench_path, root),
        },
        "summary": _summary(cases),
        "cases": cases,
    }


def _registry_case(
    parsed: dict[str, Any],
    support: dict[str, str],
) -> dict[str, Any]:
    return {
        "id": parsed["id"],
        "name": parsed["name"],
        "tier": parsed["tier"],
        "category": parsed["category"],
        "source_path": parsed["source_path"],
        "support": support,
        "inputs": [_tensor_dict(item) for item in parsed["inputs"]],
        "init_inputs": parsed["init_inputs"],
        "forward_args": parsed["forward_args"],
        "forward_expression": parsed["forward_expression"],
        "reference_class": parsed["reference_class"],
        "candidate_class": parsed["candidate_class"],
    }


def _tensor_dict(spec: TensorSpec) -> dict[str, Any]:
    return spec.to_dict()


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_tier: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_support: dict[str, int] = {}
    for case in cases:
        by_tier[case["tier"]] = by_tier.get(case["tier"], 0) + 1
        by_category[case["category"]] = by_category.get(case["category"], 0) + 1
        status = case["support"]["status"]
        by_support[status] = by_support.get(status, 0) + 1
    return {
        "total_cases": len(cases),
        "by_tier": dict(sorted(by_tier.items())),
        "by_category": dict(sorted(by_category.items())),
        "by_support": dict(sorted(by_support.items())),
    }


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
