#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_forge.agents import generate_passn_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Pass@N candidates through a provider workflow.")
    parser.add_argument("--opspec", required=True, help="Path to OpSpec YAML.")
    parser.add_argument("--provider", default="replay", help="Provider name. Supported: replay.")
    parser.add_argument("--backend", default="triton_ascend", help="Backend target to record in prompts.")
    parser.add_argument("--pass-n", type=int, default=4, help="Number of candidates to generate.")
    parser.add_argument("--run-id", required=True, help="Stable run id for generated artifacts.")
    parser.add_argument(
        "--output-root",
        default="outputs/generated",
        help="Root directory for generated artifacts.",
    )
    args = parser.parse_args()

    try:
        generated = generate_passn_candidates(
            opspec_path=args.opspec,
            provider_name=args.provider,
            backend=args.backend,
            pass_n=args.pass_n,
            run_id=args.run_id,
            output_root=args.output_root,
            repo_root=ROOT,
        )
    except Exception as exc:
        parser.exit(1, f"generate_candidate failed: {type(exc).__name__}: {exc}\n")

    print(
        json.dumps(
            {
                "run_id": generated.run_id,
                "output_root": generated.output_root,
                "experiment_path": generated.experiment_path,
                "candidates": [asdict(candidate) for candidate in generated.candidates],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
