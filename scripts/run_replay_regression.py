#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_forge.agents import generate_passn_candidates
from kernel_forge.benchmark import read_yaml, write_yaml
from kernel_forge.experiments import (
    apply_passn_to_generated_experiment,
    enrich_passn_summary,
    summarize_passn,
)
from kernel_forge.submission import display_path

EXPECTED_AKG_COMMIT = "47aa428fcdc8c68f78d331dc578bc6c74fb9d91d"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic replay generation, benchmark, probes, and Pass@N reporting."
    )
    parser.add_argument("--opspec", default="benchmarks/parsed/t1_sigmoid_scale_sum.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--backend", default="triton_ascend")
    parser.add_argument("--pass-n", type=int, default=4)
    parser.add_argument("--output-root", default="outputs/generated")
    parser.add_argument("--bench-dir", default="third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--report-output", default=None)
    parser.add_argument("--probe-script", default="scripts/probe_sigmoid_scale_sum_backend.py")
    parser.add_argument("--probe-shape", type=int, nargs=2, default=[1000, 8192])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--machine", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--expected-akg-commit",
        default=EXPECTED_AKG_COMMIT,
        help="Expected third_party/akg commit for benchmark claims.",
    )
    parser.add_argument(
        "--allow-akg-commit-mismatch",
        action="store_true",
        help="Allow replay to continue when third_party/akg is not at --expected-akg-commit.",
    )
    args = parser.parse_args()

    root = ROOT
    opspec = read_yaml(root / args.opspec)
    case_id = str(opspec["id"])
    if case_id != "t1/sigmoid_scale_sum":
        parser.error("The replay provider currently has deterministic templates only for t1/sigmoid_scale_sum")

    generated = generate_passn_candidates(
        opspec_path=args.opspec,
        provider_name="replay",
        backend=args.backend,
        pass_n=args.pass_n,
        run_id=args.run_id,
        output_root=args.output_root,
        repo_root=root,
    )
    generated_root = root / generated.output_root
    submissions_dir = generated_root / "submissions"
    results_dir = _resolve_path(args.results_dir or f"outputs/results/{args.run_id}", root)
    report_output = _resolve_path(args.report_output, root) if args.report_output else generated_root / "passn_report.yaml"

    bench_script = root / args.bench_dir / "tools/run_bench.py"
    benchmark_cmd = [
        args.python,
        str(bench_script),
        str(submissions_dir),
        "--bench-dir",
        str(root / args.bench_dir),
        "--output",
        str(results_dir),
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--num-trials",
        str(args.num_trials),
    ]
    subprocess.run(benchmark_cmd, cwd=root, check=True)

    probe_dir = generated_root / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probes: dict[str, dict[str, Any]] = {}
    probe_script = root / args.probe_script
    for candidate in generated.candidates:
        candidate_case = root / candidate.submission_root / "t1/sigmoid_scale_sum.py"
        probe_cmd = [
            args.python,
            str(probe_script),
            "--candidate",
            str(candidate_case),
            "--shape",
            str(args.probe_shape[0]),
            str(args.probe_shape[1]),
        ]
        completed = subprocess.run(probe_cmd, cwd=root, text=True, capture_output=True, check=True)
        probe = json.loads(completed.stdout)
        probes[candidate.team_name] = probe
        (probe_dir / f"{candidate.team_name}.json").write_text(
            json.dumps(probe, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    candidates = [candidate.team_name for candidate in generated.candidates]
    summary = summarize_passn(results_dir, case_id=case_id, candidates=candidates)
    generated_candidates = [asdict(candidate) for candidate in generated.candidates]
    enriched = enrich_passn_summary(
        summary,
        generated_candidates=generated_candidates,
        probes=probes,
    )

    project_commit = _git_rev_parse(root)
    akg_commit = _git_rev_parse(root / "third_party/akg")
    if (
        args.expected_akg_commit
        and akg_commit != args.expected_akg_commit
        and not args.allow_akg_commit_mismatch
    ):
        parser.error(
            "third_party/akg commit mismatch: "
            f"expected {args.expected_akg_commit}, got {akg_commit}. "
            "Use --allow-akg-commit-mismatch only for explicitly labeled exploratory runs."
        )
    branch = _git_branch(root)
    report = {
        "case": case_id,
        "id": args.run_id,
        "akg_commit": akg_commit,
        "expected_akg_commit": args.expected_akg_commit,
        "project_commit": project_commit,
        "provider": "replay",
        "model": "replay-v1",
        "prompt_version": "code_agent.v1",
        "generated_root": generated.output_root,
        **enriched,
    }
    write_yaml(report, report_output)

    experiment_path = root / generated.experiment_path
    experiment = read_yaml(experiment_path)
    updated_experiment = apply_passn_to_generated_experiment(
        experiment,
        enriched,
        results_dir=display_path(results_dir, root),
        probes_dir=display_path(probe_dir, root),
        report_path=display_path(report_output, root),
        date=args.date,
        machine=args.machine,
        branch=branch,
        commit=project_commit,
        akg_commit=akg_commit,
    )
    write_yaml(updated_experiment, experiment_path)

    print(
        json.dumps(
            {
                "run_id": generated.run_id,
                "experiment_path": generated.experiment_path,
                "report_path": display_path(report_output, root),
                "results_dir": display_path(results_dir, root),
                "probes_dir": display_path(probe_dir, root),
                "pass_at_1": enriched.get("pass_at_1"),
                "pass_at_n": enriched.get("pass_at_n"),
                "best_candidate": enriched.get("best_candidate", {}).get("team_name"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _resolve_path(value: str | None, root: Path) -> Path:
    if value is None:
        raise ValueError("value must not be None")
    path = Path(value)
    return path if path.is_absolute() else root / path


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


def _git_branch(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "branch", "--show-current"],
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


if __name__ == "__main__":
    raise SystemExit(main())
