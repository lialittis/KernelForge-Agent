from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_pre_key_readiness.py"


def test_pre_key_audit_reports_deterministic_complete_without_standard_config(tmp_path):
    completed = _run_audit(tmp_path, "--json")

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "pre_key_deterministic_complete_provider_config_missing"
    assert payload["overall"]["deterministic_complete"] is True
    assert payload["overall"]["full_runner_comparison_ready"] is False
    assert "akg_agents_standard_model_config" in payload["overall"]["blocked_checks"]
    assert _check(payload, "runner_path_decision")["status"] == "pass"
    assert _check(payload, "updated_akg_replay_pass4_import")["status"] == "pass"


def test_pre_key_audit_require_standard_config_exits_two(tmp_path):
    completed = _run_audit(tmp_path, "--json", "--require-standard-config")

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["overall"]["deterministic_complete"] is True
    assert payload["overall"]["full_runner_comparison_ready"] is False


def test_pre_key_audit_detects_standard_config_from_env(tmp_path):
    completed = _run_audit(
        tmp_path,
        "--json",
        extra_env={
            "AKG_AGENTS_STANDARD_BASE_URL": "https://api.example.test/v1",
            "AKG_AGENTS_STANDARD_API_KEY": "standard-secret-key",
            "AKG_AGENTS_STANDARD_MODEL_NAME": "standard-model",
        },
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ready_for_full_runner_comparison"
    assert payload["overall"]["full_runner_comparison_ready"] is True
    model_check = _check(payload, "akg_agents_standard_model_config")
    assert model_check["status"] == "pass"
    assert model_check["source"] == "env: AKG_AGENTS_STANDARD_*"


def test_pre_key_audit_optional_ssh_check_reports_ready(tmp_path):
    fake_ssh = _write_fake_ssh(tmp_path, 'printf "## main...origin/main\\n"\nexit 0\n')
    completed = _run_audit(
        tmp_path,
        "--json",
        "--check-ascend-ssh",
        extra_env={"PATH": f"{fake_ssh.parent}{os.pathsep}{os.environ['PATH']}"},
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["overall"]["ascend_batchmode_ssh_ready"] is True
    ssh_check = _check(payload, "ascend_batchmode_ssh")
    assert ssh_check["status"] == "pass"
    assert ssh_check["branch_status"] == "## main...origin/main"


def test_pre_key_audit_optional_ssh_check_reports_blocked(tmp_path):
    fake_ssh = _write_fake_ssh(
        tmp_path,
        'printf "Permission denied (password).\\n" >&2\nexit 255\n',
    )
    completed = _run_audit(
        tmp_path,
        "--json",
        "--check-ascend-ssh",
        extra_env={"PATH": f"{fake_ssh.parent}{os.pathsep}{os.environ['PATH']}"},
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["overall"]["ascend_batchmode_ssh_ready"] is False
    assert "ascend_batchmode_ssh" in payload["overall"]["blocked_checks"]
    ssh_check = _check(payload, "ascend_batchmode_ssh")
    assert ssh_check["status"] == "blocked"
    assert ssh_check["returncode"] == 255
    assert "Permission denied" in ssh_check["reason"]


def _run_audit(
    tmp_path: Path,
    *args: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AKG_AGENTS_") and not key.startswith("AIKG_")
    }
    env["HOME"] = str(home)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(ROOT),
            *args,
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _check(payload: dict, check_id: str) -> dict:
    for check in payload["checks"]:
        if check["id"] == check_id:
            return check
    raise AssertionError(f"missing check {check_id}")


def _write_fake_ssh(tmp_path: Path, body: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake_ssh = bin_dir / "ssh"
    fake_ssh.write_text(f"#!/bin/sh\n{body}", encoding="utf-8")
    fake_ssh.chmod(0o755)
    return fake_ssh
