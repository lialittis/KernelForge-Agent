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
from kernel_forge.experiments import import_benchmark_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Import official benchmark JSON into experiment YAML.")
    parser.add_argument("--result-json", required=True, help="Official run_bench.py JSON output.")
    parser.add_argument("--experiment", default=None, help="Optional existing experiment YAML to update.")
    parser.add_argument("--probe-json", default=None, help="Optional backend probe JSON.")
    parser.add_argument("--output", default=None, help="Output YAML path. Defaults to stdout.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write back to --experiment. Mutually exclusive with --output.",
    )
    args = parser.parse_args()

    if args.in_place and not args.experiment:
        parser.error("--in-place requires --experiment")
    if args.in_place and args.output:
        parser.error("--in-place cannot be combined with --output")

    imported = import_benchmark_result(
        args.result_json,
        experiment_path=args.experiment,
        probe_json=args.probe_json,
    )

    if args.in_place:
        write_yaml(imported, args.experiment)
    elif args.output:
        write_yaml(imported, args.output)
    else:
        yaml.safe_dump(imported, sys.stdout, sort_keys=False, allow_unicode=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
