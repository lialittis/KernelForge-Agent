#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_forge.benchmark import extract_opspec, scan_benchmark_cases, write_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch extract OpSpecs from AKG Bench Lite.")
    parser.add_argument(
        "--bench-dir",
        default="third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite",
        help="Path to the AKG Bench Lite directory.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for generated YAML files.")
    parser.add_argument("--repo-root", default=".", help="Repository root used for relative paths.")
    parser.add_argument("--backend-target", default="triton_ascend", help="Sketch backend target.")
    parser.add_argument(
        "--include-unsupported",
        action="store_true",
        help="Also write explicit unsupported metadata for deferred cases.",
    )
    args = parser.parse_args()

    registry = scan_benchmark_cases(args.bench_dir, repo_root=Path(args.repo_root))
    output_dir = Path(args.output_dir)
    written = 0
    for case in registry["cases"]:
        supported = case["support"]["status"] == "opspec_supported"
        if not supported and not args.include_unsupported:
            continue
        spec = extract_opspec(
            Path(args.repo_root) / case["source_path"],
            repo_root=Path(args.repo_root),
            backend_target=args.backend_target,
            allow_unsupported=not supported,
        )
        output_path = output_dir / f"{case['id'].replace('/', '_')}.yaml"
        write_yaml(spec, output_path)
        written += 1
    print(f"Wrote {written} files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
