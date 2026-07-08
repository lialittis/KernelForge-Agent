#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_forge.submission import create_submission_layout, parse_case_mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an official AKG Bench Lite submission layout.")
    parser.add_argument("--team", required=True, help="Team/submission name.")
    parser.add_argument("--candidate", required=True, help="Candidate label to record in meta.json.")
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="CASE_ID=SOURCE",
        help="Case source mapping, for example t1/gelu=kernel_forge/candidates/gelu_triton_v13.py.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/submissions",
        help="Directory that will contain team/team submission layout.",
    )
    parser.add_argument(
        "--layout",
        choices=("nested", "flat"),
        default="nested",
        help="Use nested output-root/team/team layout or flat output-root/team layout.",
    )
    args = parser.parse_args()

    try:
        mappings = [parse_case_mapping(item) for item in args.case]
        submission_root = create_submission_layout(
            team=args.team,
            candidate=args.candidate,
            cases=mappings,
            output_root=args.output_root,
            layout=args.layout,
            repo_root=ROOT,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Submission created: {submission_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
