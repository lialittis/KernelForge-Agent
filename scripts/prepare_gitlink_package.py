#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEAM_RE = re.compile(r"^[A-Za-z0-9_-]+$")
PACKAGE_README_SOURCE = "docs/submission/package_readme_zh.md"

INCLUDE_PATHS = [
    "README.md",
    "AGENTS.md",
    "benchmarks/README.md",
    "benchmarks/parsed",
    "benchmarks/raw/akg_kernels_bench_lite_registry.yaml",
    "docs",
    "experiments/README.md",
    "experiments/reports",
    "experiments/runs",
    "kernel_forge",
    "prompts",
    "scripts",
    "skills",
    "tasks/active.md",
    "tests",
    "third_party/README.md",
]

EXPLICIT_UNTRACKED_FILES = {
    "docs/project_book_full_zh.md",
    "docs/submission/gitlink_pr_body.md",
    "docs/submission/gitlink_pr_title.txt",
    PACKAGE_README_SOURCE,
    "docs/submission/project_book_email_zh.md",
    "docs/submission/step3_completion_audit.md",
    "docs/submission_package_readme.md",
    "docs/technical_design.md",
    "scripts/export_project_book.py",
    "scripts/prepare_gitlink_package.py",
}

EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
    ".venv",
    "venv",
    "env",
    "outputs",
    "build",
    "dist",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".tmp",
    ".log",
}
EXCLUDE_FILE_NAMES = {
    ".DS_Store",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a clean GitLink Step 3 project package under outputs/."
    )
    parser.add_argument(
        "--team",
        default="operator-alchemists",
        help="Team directory name for projects/<team>/SketchSkill-AKG.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/gitlink_package",
        help="Package output root. The selected package directory is recreated.",
    )
    args = parser.parse_args()

    if not TEAM_RE.match(args.team):
        parser.error("--team must contain only letters, numbers, underscores, and hyphens")

    output_root = _resolve_output_root(args.output_root)
    package_root = output_root / "projects" / args.team / "SketchSkill-AKG"
    _ensure_safe_package_root(package_root)
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    allowed_files = _package_file_set()
    for item in INCLUDE_PATHS:
        source = ROOT / item
        if not source.exists():
            raise FileNotFoundError(f"Required package path is missing: {item}")
        copied.extend(_copy_path(source, package_root / item, allowed_files))
    copied.extend(_install_package_readme(package_root))

    manifest = {
        "team": args.team,
        "package_root": _display_path(package_root),
        "source_root": ROOT.as_posix(),
        "included_paths": INCLUDE_PATHS,
        "package_readme": {
            "source": PACKAGE_README_SOURCE,
            "root_readme": "README.md",
            "original_project_readme": "PROJECT_README.md",
        },
        "copied_file_count": len(copied),
        "copied_files": copied,
        "excluded": {
            "dir_names": sorted(EXCLUDE_DIR_NAMES),
            "suffixes": sorted(EXCLUDE_SUFFIXES),
            "runtime_outputs": "outputs/ is intentionally excluded",
            "secrets": ".env and SSH/API credentials are intentionally excluded",
            "untracked_files": (
                "Only git-tracked files and explicit Step 3 draft artifacts are copied"
            ),
        },
    }
    manifest_path = package_root / "PACKAGE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Package prepared: {_display_path(package_root)}")
    print(f"Manifest: {_display_path(manifest_path)}")
    print(f"Files copied: {len(copied)}")
    return 0


def _resolve_output_root(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def _ensure_safe_package_root(package_root: Path) -> None:
    resolved = package_root.resolve()
    allowed_roots = [
        (ROOT / "outputs").resolve(),
        Path("/tmp").resolve(),
        Path("/private/tmp").resolve(),
    ]
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise ValueError(
            "Refusing to recreate package outside the project outputs/ tree or /tmp"
        )
    if resolved.name != "SketchSkill-AKG":
        raise ValueError("Refusing to recreate an unexpected package directory")


def _copy_path(source: Path, target: Path, allowed_files: set[str]) -> list[str]:
    copied: list[str] = []
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if _should_skip(path):
                continue
            rel = path.relative_to(source)
            out = target / rel
            if path.is_dir():
                out.mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                repo_rel = _repo_relative(path)
                if repo_rel not in allowed_files:
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)
                copied.append(_display_path(out))
        return copied

    if _should_skip(source):
        return copied
    repo_rel = _repo_relative(source)
    if repo_rel not in allowed_files:
        raise ValueError(f"Refusing to copy untracked package file: {repo_rel}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    copied.append(_display_path(target))
    return copied


def _install_package_readme(package_root: Path) -> list[str]:
    source = ROOT / PACKAGE_README_SOURCE
    if not source.exists():
        raise FileNotFoundError(f"Required package README is missing: {PACKAGE_README_SOURCE}")

    copied: list[str] = []
    root_readme = package_root / "README.md"
    project_readme = package_root / "PROJECT_README.md"
    if root_readme.exists():
        shutil.copy2(root_readme, project_readme)
        copied.append(_display_path(project_readme))

    shutil.copy2(source, root_readme)
    copied.append(_display_path(root_readme))
    return copied


def _should_skip(path: Path) -> bool:
    if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
        return True
    if path.name in EXCLUDE_FILE_NAMES:
        return True
    if path.name.startswith(".env"):
        return True
    return path.suffix in EXCLUDE_SUFFIXES


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _package_file_set() -> set[str]:
    tracked = subprocess.check_output(
        ["git", "ls-files"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    allowed = set(tracked)
    allowed.update(path for path in EXPLICIT_UNTRACKED_FILES if (ROOT / path).exists())
    return allowed


if __name__ == "__main__":
    raise SystemExit(main())
