#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_forge.benchmark import write_yaml
from kernel_forge.experiments import summarize_passn


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Pass@N from official benchmark JSON files.")
    parser.add_argument("--results-dir", required=True, help="Directory containing <team>.json files.")
    parser.add_argument("--case", required=True, help="Case id, for example t1/sigmoid_scale_sum.")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate/team name in generation order. Repeat for Pass@N.",
    )
    parser.add_argument("--output", default=None, help="Optional YAML output path.")
    args = parser.parse_args()

    summary = summarize_passn(args.results_dir, case_id=args.case, candidates=args.candidate)
    if args.output:
        write_yaml(summary, args.output)
    else:
        yaml.safe_dump(summary, sys.stdout, sort_keys=False, allow_unicode=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
