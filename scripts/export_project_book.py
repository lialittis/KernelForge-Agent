#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "outputs/submission/project_book_full_zh.md"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the Chinese full project book as a standalone Markdown artifact."
    )
    parser.add_argument(
        "--source",
        default="docs/project_book_full_zh.md",
        help="Main project-book Markdown source.",
    )
    parser.add_argument(
        "--technical-design",
        default="docs/technical_design.md",
        help="Technical design appendix source.",
    )
    parser.add_argument(
        "--package-notes",
        default="docs/submission_package_readme.md",
        help="Package/reproduction appendix source.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output Markdown artifact path.",
    )
    parser.add_argument(
        "--pr-link",
        default=os.environ.get("GITLINK_PR_URL"),
        help="Optional GitLink PR URL to fill into the project book.",
    )
    parser.add_argument(
        "--no-append-technical-design",
        action="store_true",
        help="Do not append docs/technical_design.md.",
    )
    parser.add_argument(
        "--no-append-package-notes",
        action="store_true",
        help="Do not append docs/submission_package_readme.md.",
    )
    args = parser.parse_args()

    source_path = _resolve(args.source)
    output_path = _resolve(args.output)
    parts = [_apply_pr_link(source_path.read_text(encoding="utf-8"), args.pr_link).rstrip()]
    included_sources = [_display_path(source_path)]

    if not args.no_append_technical_design:
        technical_path = _resolve(args.technical_design)
        parts.append(_appendix("附录 A：技术设计文档", technical_path))
        included_sources.append(_display_path(technical_path))

    if not args.no_append_package_notes:
        package_notes_path = _resolve(args.package_notes)
        parts.append(_appendix("附录 B：提交包与复现说明", package_notes_path))
        included_sources.append(_display_path(package_notes_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(parts).rstrip() + "\n", encoding="utf-8")

    metadata = {
        "output": _display_path(output_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pr_link": args.pr_link,
        "included_sources": included_sources,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Exported project book: {_display_path(output_path)}")
    print(f"Metadata: {_display_path(metadata_path)}")
    return 0


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return ROOT / value


def _apply_pr_link(text: str, pr_link: str | None) -> str:
    if not pr_link:
        return text
    return text.replace("| GitLink PR | 待提交后填写 |", f"| GitLink PR | {pr_link} |")


def _appendix(title: str, path: Path) -> str:
    content = path.read_text(encoding="utf-8").strip()
    return f"# {title}\n\n{_demote_markdown_headings(content)}"


def _demote_markdown_headings(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("#"):
            lines.append("#" + line)
        else:
            lines.append(line)
    return "\n".join(lines).strip()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
