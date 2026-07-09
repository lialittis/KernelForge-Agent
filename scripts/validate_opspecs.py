#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_forge.benchmark import validate_opspec_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate parsed OpSpec and Sketch YAML files.")
    parser.add_argument("--opspec-dir", default="benchmarks/parsed")
    parser.add_argument("--output", default=None, help="Optional YAML output path.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of YAML.")
    args = parser.parse_args()

    opspec_dir = Path(args.opspec_dir)
    if not opspec_dir.is_absolute():
        opspec_dir = ROOT / opspec_dir
    report = validate_opspec_dir(opspec_dir)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(report, sort_keys=False, allow_unicode=True), encoding="utf-8")
    elif args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        yaml.safe_dump(report, sys.stdout, sort_keys=False, allow_unicode=True)

    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
