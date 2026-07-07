#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_forge.benchmark import scan_benchmark_cases, write_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan AKG Bench Lite cases into a registry YAML.")
    parser.add_argument(
        "--bench-dir",
        default="third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite",
        help="Path to the AKG Bench Lite directory.",
    )
    parser.add_argument("--output", required=True, help="YAML registry output path.")
    parser.add_argument("--repo-root", default=".", help="Repository root used for relative paths.")
    args = parser.parse_args()

    registry = scan_benchmark_cases(args.bench_dir, repo_root=Path(args.repo_root))
    write_yaml(registry, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
