#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_forge.benchmark import extract_opspec, write_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract an OpSpec from an AKG Bench Lite case file.")
    parser.add_argument("--case", required=True, help="Path to an official benchmark case file.")
    parser.add_argument("--output", required=True, help="YAML output path.")
    parser.add_argument("--experiment", default=None, help="Optional experiment YAML for performance enrichment.")
    parser.add_argument("--repo-root", default=".", help="Repository root used to make source paths relative.")
    parser.add_argument("--backend-target", default="triton_ascend", help="Sketch backend target.")
    parser.add_argument(
        "--allow-unsupported",
        action="store_true",
        help="Emit explicit unsupported-case metadata instead of failing.",
    )
    args = parser.parse_args()

    spec = extract_opspec(
        args.case,
        experiment_path=args.experiment,
        repo_root=Path(args.repo_root),
        backend_target=args.backend_target,
        allow_unsupported=args.allow_unsupported,
    )
    write_yaml(spec, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
