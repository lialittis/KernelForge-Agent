from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_akg_agents_model_config.py"


def test_missing_standard_model_config_exits_two(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    completed = _run_checker(tmp_path, repo_root, "--json")

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["configured"] is False
    assert payload["requested_level"] == "standard"
    assert payload["missing"] == ["standard"]


def test_env_standard_model_config_is_detected_and_masked(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    env = {
        "AKG_AGENTS_STANDARD_BASE_URL": "https://api.example.test/v1",
        "AKG_AGENTS_STANDARD_API_KEY": "sk-test-secret-123456",
        "AKG_AGENTS_STANDARD_MODEL_NAME": "test-coder-model",
    }
    completed = _run_checker(tmp_path, repo_root, "--json", extra_env=env)

    assert completed.returncode == 0
    assert "sk-test-secret-123456" not in completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["configured"] is True
    assert payload["resolved_level"] == "standard"
    assert payload["source"] == "env: AKG_AGENTS_STANDARD_*"
    assert payload["model"]["api_key"] == "sk-t***3456"
    assert payload["model"]["model_name"] == "test-coder-model"


def test_local_settings_model_config_is_detected(tmp_path):
    repo_root = tmp_path / "repo"
    settings_dir = repo_root / ".akg"
    settings_dir.mkdir(parents=True)
    (settings_dir / "settings.local.json").write_text(
        json.dumps(
            {
                "models": {
                    "standard": {
                        "base_url": "https://api.example.test/v1",
                        "api_key": "local-secret-key",
                        "model_name": "local-coder-model",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    completed = _run_checker(tmp_path, repo_root, "--json")

    assert completed.returncode == 0
    assert "local-secret-key" not in completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["configured"] is True
    assert payload["source"].startswith("local:")
    assert payload["model"]["api_key"] == "loca***-key"


def test_level_specific_env_overrides_single_env(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    env = {
        "AKG_AGENTS_BASE_URL": "https://api.single.test/v1",
        "AKG_AGENTS_API_KEY": "single-secret-key",
        "AKG_AGENTS_MODEL_NAME": "single-model",
        "AKG_AGENTS_STANDARD_BASE_URL": "https://api.standard.test/v1",
        "AKG_AGENTS_STANDARD_API_KEY": "standard-secret-key",
        "AKG_AGENTS_STANDARD_MODEL_NAME": "standard-model",
    }
    completed = _run_checker(tmp_path, repo_root, "--json", extra_env=env)

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["source"] == "env: AKG_AGENTS_STANDARD_*"
    assert payload["model"]["base_url"] == "https://api.standard.test/v1"
    assert payload["model"]["model_name"] == "standard-model"


def _run_checker(
    tmp_path: Path,
    repo_root: Path,
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
            str(repo_root),
            "--level",
            "standard",
            *args,
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
