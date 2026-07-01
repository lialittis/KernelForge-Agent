import ast
import json
import py_compile
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "kernel_forge/candidates/gelu_triton_v1.py"
SCRIPT = ROOT / "scripts/create_gelu_triton_submission.sh"


def test_candidate_source_is_valid_python():
    py_compile.compile(str(CANDIDATE), doraise=True)


def test_candidate_defines_modelnew_and_uses_safe_imports():
    tree = ast.parse(CANDIDATE.read_text(encoding="utf-8"))
    class_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    assert "ModelNew" in class_names

    blocked = {
        "subprocess",
        "shutil",
        "socket",
        "http",
        "urllib",
        "ftplib",
        "smtplib",
        "ctypes",
        "multiprocessing",
    }
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])

    assert not (set(imports) & blocked)


def test_generator_creates_official_submission_layout(monkeypatch):
    monkeypatch.setenv("TEAM_NAME", "gelu_triton_test")

    subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, check=True)

    submission_root = (
        ROOT / "outputs/submissions/gelu_triton_test/gelu_triton_test"
    )
    generated_case = submission_root / "t1/gelu.py"
    meta = json.loads((submission_root / "meta.json").read_text())

    assert meta["team_name"] == "gelu_triton_test"
    assert meta["candidate"] == "gelu_triton_v1"
    assert generated_case.read_text() == CANDIDATE.read_text()

