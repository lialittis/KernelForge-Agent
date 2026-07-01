import ast
import json
import py_compile
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "kernel_forge/candidates/gelu_triton_v1.py"
SCRIPT = ROOT / "scripts/create_gelu_triton_submission.sh"
V2_CANDIDATE = ROOT / "kernel_forge/candidates/gelu_triton_v2.py"
V2_SCRIPT = ROOT / "scripts/create_gelu_triton_v2_submission.sh"
V3_CANDIDATE = ROOT / "kernel_forge/candidates/gelu_triton_v3.py"
V3_SCRIPT = ROOT / "scripts/create_gelu_triton_v3_submission.sh"
V4_CANDIDATE = ROOT / "kernel_forge/candidates/gelu_triton_v4.py"
V4_SCRIPT = ROOT / "scripts/create_gelu_triton_v4_submission.sh"
V5_CANDIDATE = ROOT / "kernel_forge/candidates/gelu_triton_v5.py"
V5_SCRIPT = ROOT / "scripts/create_gelu_triton_v5_submission.sh"
V6_CANDIDATE = ROOT / "kernel_forge/candidates/gelu_triton_v6.py"
V6_SCRIPT = ROOT / "scripts/create_gelu_triton_v6_submission.sh"


def test_candidate_source_is_valid_python():
    py_compile.compile(str(CANDIDATE), doraise=True)
    py_compile.compile(str(V2_CANDIDATE), doraise=True)
    py_compile.compile(str(V3_CANDIDATE), doraise=True)
    py_compile.compile(str(V4_CANDIDATE), doraise=True)
    py_compile.compile(str(V5_CANDIDATE), doraise=True)
    py_compile.compile(str(V6_CANDIDATE), doraise=True)


def test_candidate_defines_modelnew_and_uses_safe_imports():
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

    for candidate in (
        CANDIDATE,
        V2_CANDIDATE,
        V3_CANDIDATE,
        V4_CANDIDATE,
        V5_CANDIDATE,
        V6_CANDIDATE,
    ):
        tree = ast.parse(candidate.read_text(encoding="utf-8"))
        class_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }
        assert "ModelNew" in class_names

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


def test_v2_generator_creates_official_submission_layout(monkeypatch):
    monkeypatch.setenv("TEAM_NAME", "gelu_triton_v2_test")

    subprocess.run(["bash", str(V2_SCRIPT)], cwd=ROOT, check=True)

    submission_root = (
        ROOT / "outputs/submissions/gelu_triton_v2_test/gelu_triton_v2_test"
    )
    generated_case = submission_root / "t1/gelu.py"
    meta = json.loads((submission_root / "meta.json").read_text())

    assert meta["team_name"] == "gelu_triton_v2_test"
    assert meta["candidate"] == "gelu_triton_v2"
    assert generated_case.read_text() == V2_CANDIDATE.read_text()


def test_v3_generator_creates_official_submission_layout(monkeypatch):
    monkeypatch.setenv("TEAM_NAME", "gelu_triton_v3_test")

    subprocess.run(["bash", str(V3_SCRIPT)], cwd=ROOT, check=True)

    submission_root = (
        ROOT / "outputs/submissions/gelu_triton_v3_test/gelu_triton_v3_test"
    )
    generated_case = submission_root / "t1/gelu.py"
    meta = json.loads((submission_root / "meta.json").read_text())

    assert meta["team_name"] == "gelu_triton_v3_test"
    assert meta["candidate"] == "gelu_triton_v3"
    assert generated_case.read_text() == V3_CANDIDATE.read_text()


def test_v4_generator_creates_official_submission_layout(monkeypatch):
    monkeypatch.setenv("TEAM_NAME", "gelu_triton_v4_test")

    subprocess.run(["bash", str(V4_SCRIPT)], cwd=ROOT, check=True)

    submission_root = (
        ROOT / "outputs/submissions/gelu_triton_v4_test/gelu_triton_v4_test"
    )
    generated_case = submission_root / "t1/gelu.py"
    meta = json.loads((submission_root / "meta.json").read_text())

    assert meta["team_name"] == "gelu_triton_v4_test"
    assert meta["candidate"] == "gelu_triton_v4"
    assert generated_case.read_text() == V4_CANDIDATE.read_text()


def test_v5_generator_creates_official_submission_layout(monkeypatch):
    monkeypatch.setenv("TEAM_NAME", "gelu_triton_v5_test")

    subprocess.run(["bash", str(V5_SCRIPT)], cwd=ROOT, check=True)

    submission_root = (
        ROOT / "outputs/submissions/gelu_triton_v5_test/gelu_triton_v5_test"
    )
    generated_case = submission_root / "t1/gelu.py"
    meta = json.loads((submission_root / "meta.json").read_text())

    assert meta["team_name"] == "gelu_triton_v5_test"
    assert meta["candidate"] == "gelu_triton_v5"
    assert generated_case.read_text() == V5_CANDIDATE.read_text()


def test_v6_generator_creates_official_submission_layout(monkeypatch):
    monkeypatch.setenv("TEAM_NAME", "gelu_triton_v6_test")

    subprocess.run(["bash", str(V6_SCRIPT)], cwd=ROOT, check=True)

    submission_root = (
        ROOT / "outputs/submissions/gelu_triton_v6_test/gelu_triton_v6_test"
    )
    generated_case = submission_root / "t1/gelu.py"
    meta = json.loads((submission_root / "meta.json").read_text())

    assert meta["team_name"] == "gelu_triton_v6_test"
    assert meta["candidate"] == "gelu_triton_v6"
    assert generated_case.read_text() == V6_CANDIDATE.read_text()
