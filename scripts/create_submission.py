#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEAM_RE = re.compile(r"^[A-Za-z0-9_-]+$")


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
    args = parser.parse_args()

    if not TEAM_RE.match(args.team):
        parser.error("--team must contain only letters, numbers, underscores, and hyphens")

    try:
        mappings = [_parse_case_mapping(item) for item in args.case]
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    submission_root = ROOT / args.output_root / args.team / args.team
    submission_root.mkdir(parents=True, exist_ok=True)

    cases_meta = []
    for case_id, source in mappings:
        tier, name = case_id.split("/", 1)
        target_dir = submission_root / tier
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{name}.py"
        source_path = (ROOT / source).resolve()
        shutil.copyfile(source_path, target_path)
        cases_meta.append(
            {
                "case": case_id,
                "source": _display_path(source_path),
                "target": _display_path(target_path),
            }
        )

    meta = {
        "team_name": args.team,
        "candidate": args.candidate,
        "cases": cases_meta,
    }
    (submission_root / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Submission created: {submission_root}")
    return 0


def _parse_case_mapping(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case mapping must use CASE_ID=SOURCE")
    case_id, source = value.split("=", 1)
    if "/" not in case_id:
        raise argparse.ArgumentTypeError("case id must look like tier/name")
    if not source:
        raise argparse.ArgumentTypeError("source path must not be empty")
    return case_id, source


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
