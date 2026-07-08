from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RetrievedSkill:
    path: str
    content: str


def retrieve_skill_paths(opspec: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    category = opspec.get("category")
    sketch_category = opspec.get("sketch", {}).get("operator_category")
    operator_category = sketch_category or category

    if operator_category == "reduction":
        paths.append("skills/reduction/SKILL.md")
    elif operator_category in {"elementwise", "fused_elementwise"}:
        paths.append("skills/elementwise/SKILL.md")
    elif operator_category == "normalization":
        paths.append("skills/normalization/SKILL.md")

    semantics = opspec.get("semantics", {})
    if semantics.get("broadcast") not in {None, "none"}:
        paths.append("skills/broadcast/SKILL.md")
    if semantics.get("layout_transform") not in {None, "none"}:
        paths.append("skills/transpose_layout/SKILL.md")

    paths.extend(
        [
            "skills/ascend_debug/SKILL.md",
            "skills/ascend_performance/SKILL.md",
            "skills/benchmark_evaluation/SKILL.md",
        ]
    )
    return _dedupe_existing(paths)


def load_retrieved_skills(paths: list[str], *, repo_root: str | Path | None = None) -> list[RetrievedSkill]:
    root = Path(repo_root) if repo_root is not None else ROOT
    skills = []
    for path in paths:
        skill_path = root / path
        skills.append(
            RetrievedSkill(
                path=path,
                content=skill_path.read_text(encoding="utf-8"),
            )
        )
    return skills


def _dedupe_existing(paths: list[str]) -> list[str]:
    seen = set()
    result = []
    for path in paths:
        if path in seen:
            continue
        if (ROOT / path).exists():
            result.append(path)
            seen.add(path)
    return result
