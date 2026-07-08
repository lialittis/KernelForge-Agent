from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEAM_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class CaseMapping:
    case_id: str
    source: str


def create_submission_layout(
    *,
    team: str,
    candidate: str,
    cases: list[CaseMapping],
    output_root: str | Path = "outputs/submissions",
    layout: str = "nested",
    repo_root: str | Path | None = None,
) -> Path:
    root = Path(repo_root) if repo_root is not None else ROOT
    if not TEAM_RE.match(team):
        raise ValueError("team must contain only letters, numbers, underscores, and hyphens")
    if layout not in {"nested", "flat"}:
        raise ValueError("layout must be 'nested' or 'flat'")

    output_root_path = Path(output_root)
    if not output_root_path.is_absolute():
        output_root_path = root / output_root_path
    if layout == "nested":
        submission_root = output_root_path / team / team
    else:
        submission_root = output_root_path / team
    submission_root.mkdir(parents=True, exist_ok=True)

    cases_meta = []
    for mapping in cases:
        tier, name = _split_case_id(mapping.case_id)
        target_dir = submission_root / tier
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{name}.py"
        source_path = (root / mapping.source).resolve()
        shutil.copyfile(source_path, target_path)
        cases_meta.append(
            {
                "case": mapping.case_id,
                "source": display_path(source_path, root),
                "target": display_path(target_path, root),
            }
        )

    meta = {
        "team_name": team,
        "candidate": candidate,
        "cases": cases_meta,
    }
    (submission_root / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return submission_root


def parse_case_mapping(value: str) -> CaseMapping:
    if "=" not in value:
        raise ValueError("case mapping must use CASE_ID=SOURCE")
    case_id, source = value.split("=", 1)
    _split_case_id(case_id)
    if not source:
        raise ValueError("source path must not be empty")
    return CaseMapping(case_id=case_id, source=source)


def display_path(path: Path, root: Path | None = None) -> str:
    root = root or ROOT
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _split_case_id(case_id: str) -> tuple[str, str]:
    if "/" not in case_id:
        raise ValueError("case id must look like tier/name")
    tier, name = case_id.split("/", 1)
    if not tier or not name:
        raise ValueError("case id must look like tier/name")
    return tier, name
