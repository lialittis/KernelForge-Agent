from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_generic_submission_script_creates_official_layout(tmp_path):
    output_root = tmp_path / "submissions"
    source = ROOT / "kernel_forge/candidates/gelu_triton_v13.py"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/create_submission.py"),
            "--team",
            "generic_gelu_test",
            "--candidate",
            "gelu_triton_v13",
            "--case",
            "t1/gelu=kernel_forge/candidates/gelu_triton_v13.py",
            "--output-root",
            str(output_root),
        ],
        cwd=ROOT,
        check=True,
    )

    submission_root = output_root / "generic_gelu_test/generic_gelu_test"
    meta = json.loads((submission_root / "meta.json").read_text())
    generated_case = submission_root / "t1/gelu.py"

    assert meta["team_name"] == "generic_gelu_test"
    assert meta["candidate"] == "gelu_triton_v13"
    assert meta["cases"][0]["case"] == "t1/gelu"
    assert generated_case.read_text() == source.read_text()
