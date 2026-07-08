from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .skills import RetrievedSkill


ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = ROOT / "prompts"


def load_prompt_template(name: str, *, repo_root: str | Path | None = None) -> str:
    root = Path(repo_root) if repo_root is not None else ROOT
    return (root / "prompts" / name).read_text(encoding="utf-8")


def render_code_prompt(
    *,
    opspec: dict[str, Any],
    backend: str,
    candidate_index: int,
    pass_n: int,
    skills: list[RetrievedSkill],
    prompt_name: str = "code_agent.v1.md",
    repo_root: str | Path | None = None,
) -> tuple[str, str]:
    template = load_prompt_template(prompt_name, repo_root=repo_root)
    context = {
        "case_id": str(opspec.get("id")),
        "operator_name": str(opspec.get("name")),
        "backend": backend,
        "candidate_index": str(candidate_index),
        "pass_n": str(pass_n),
        "opspec_yaml": _dump_yaml(opspec),
        "sketch_yaml": _dump_yaml(opspec.get("sketch", {})),
        "retrieved_skill_paths": "\n".join(f"- {skill.path}" for skill in skills),
        "retrieved_skill_summaries": "\n\n".join(_skill_summary(skill) for skill in skills),
    }
    return _render(template, context), prompt_name.removesuffix(".md")


def _render(template: str, context: dict[str, str]) -> str:
    output = template
    for key, value in context.items():
        output = output.replace("{{" + key + "}}", value)
    return output


def _dump_yaml(value: dict[str, Any]) -> str:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)


def _skill_summary(skill: RetrievedSkill) -> str:
    lines = [line for line in skill.content.splitlines() if line.strip()]
    preview = "\n".join(lines[:20])
    return f"## {skill.path}\n{preview}"
