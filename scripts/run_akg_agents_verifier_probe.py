#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AKG_AGENTS_PYTHON = ROOT / "third_party/akg/akg_agents/python"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one existing candidate through AKG Agents verifier_only_workflow. "
            "This is a no-LLM probe, not a full run_torch_bench_lite.py comparison."
        )
    )
    parser.add_argument("--case", default="t1/sigmoid_scale_sum")
    parser.add_argument("--candidate", default="kernel_forge/candidates/sigmoid_scale_sum_v2.py")
    parser.add_argument("--bench-dir", default="third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite")
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-dir", default="outputs/akg_agents_verifier_logs")
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--backend", default="ascend")
    parser.add_argument("--arch", default="ascend910b4")
    parser.add_argument("--dsl", default="triton_ascend")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--task-type", choices=["precision_only", "profile"], default="precision_only")
    parser.add_argument("--profile-warmup", type=int, default=5)
    parser.add_argument("--profile-runs", type=int, default=50)
    parser.add_argument("--verify-timeout", type=int, default=600)
    parser.add_argument("--workflow-timeout", type=int, default=1800)
    args = parser.parse_args()

    root = ROOT
    if str(AKG_AGENTS_PYTHON) not in sys.path:
        sys.path.insert(0, str(AKG_AGENTS_PYTHON))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        payload = asyncio.run(run_probe(args, root=root))
    except Exception as exc:
        parser.exit(1, f"run_akg_agents_verifier_probe failed: {type(exc).__name__}: {exc}\n")

    output_path = _resolve_output(args, root, payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": _display_path(output_path, root), **payload["result"]}, indent=2, ensure_ascii=False))
    return 0 if payload["result"]["success"] else 1


async def run_probe(args: argparse.Namespace, *, root: Path) -> dict[str, Any]:
    from akg_agents.core.worker.manager import register_local_worker
    from akg_agents.op.config.config_validator import load_config
    from akg_agents.op.langgraph_op.task import LangGraphTask
    from akg_agents.utils.environment_check import check_env_for_task

    bench_dir = _resolve_path(args.bench_dir, root)
    reference_path = bench_dir / f"{args.case}.py"
    candidate_path = _resolve_path(args.candidate, root)
    if not reference_path.is_file():
        raise FileNotFoundError(f"Benchmark reference not found: {reference_path}")
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate not found: {candidate_path}")

    op_name = args.case.split("/")[-1]
    task_id = args.task_id or f"akg_verifier_probe_{op_name}_{candidate_path.stem}_{date.today().isoformat()}"
    log_dir = _resolve_path(args.log_dir, root)

    config = load_config(args.dsl, backend=args.backend)
    config["log_dir"] = str(log_dir)
    config["task_label"] = task_id
    config["skip_kernel_gen"] = True
    config["skip_kernel_designer"] = True
    config["max_step"] = 1
    config["verify_timeout"] = args.verify_timeout
    config["workflow_timeout"] = args.workflow_timeout
    config["profile_settings"] = {
        "warmup_times": args.profile_warmup,
        "run_times": args.profile_runs,
    }

    await register_local_worker([args.device], backend=args.backend, arch=args.arch)
    check_env_for_task("torch", args.backend, args.dsl, config)

    framework_code = reference_path.read_text(encoding="utf-8")
    candidate_code = candidate_path.read_text(encoding="utf-8")

    task = LangGraphTask(
        op_name=op_name,
        task_desc=framework_code,
        task_id=task_id,
        dsl=args.dsl,
        backend=args.backend,
        arch=args.arch,
        config=config,
        framework="torch",
        workflow="verifier_only_workflow",
        task_type=args.task_type,
        bench_type="kernelbench",
    )

    started = time.time()
    _, success, final_state = await task.run({"coder_code": candidate_code})
    elapsed_s = time.time() - started
    verifier = task.agents.get("verifier") if hasattr(task, "agents") else None

    return {
        "runner_path": "akg_agents_verifier_only_workflow",
        "comparable_with_full_runner": False,
        "comparison_scope": (
            "No-LLM AKG Agents verifier probe for an existing candidate. "
            "It exercises LangGraphTask verifier/project generation/logging, "
            "but it does not replace run_torch_bench_lite.py --mode full."
        ),
        "case": args.case,
        "candidate": {
            "path": _display_path(candidate_path, root),
            "team_name": candidate_path.stem.replace("_v", "_replay_v")
            if candidate_path.stem.startswith("sigmoid_scale_sum_v")
            else candidate_path.stem,
        },
        "akg_commit": _git_rev_parse(root / "third_party/akg"),
        "project_commit": _git_rev_parse(root),
        "config": {
            "workflow": "verifier_only_workflow",
            "task_type": args.task_type,
            "backend": args.backend,
            "arch": args.arch,
            "dsl": args.dsl,
            "device": args.device,
            "profile_warmup": args.profile_warmup,
            "profile_runs": args.profile_runs,
            "verify_timeout": args.verify_timeout,
        },
        "result": {
            "success": bool(success),
            "verifier_result": bool(final_state.get("verifier_result", False)),
            "elapsed_s": round(elapsed_s, 3),
            "verifier_error_excerpt": _excerpt(final_state.get("verifier_error", "")),
            "profile_res": _json_safe(final_state.get("profile_res", {})),
            "verify_dir": _display_path(Path(getattr(verifier, "last_verify_dir", "")), root)
            if verifier and getattr(verifier, "last_verify_dir", "")
            else None,
            "verify_sidecar": _json_safe(getattr(verifier, "last_verify_sidecar", None))
            if verifier
            else None,
        },
        "logs": {
            "log_dir": _display_path(log_dir, root),
            "task_id": task_id,
        },
    }


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _resolve_output(args: argparse.Namespace, root: Path, payload: dict[str, Any]) -> Path:
    if args.output:
        return _resolve_path(args.output, root)
    op_name = args.case.split("/")[-1]
    candidate_name = payload["candidate"]["team_name"]
    return root / "outputs/results" / f"akg_agents_verifier_probe_{op_name}_{candidate_name}_{date.today().isoformat()}.json"


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def _excerpt(value: Any, limit: int = 2000) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _git_rev_parse(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return None
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
