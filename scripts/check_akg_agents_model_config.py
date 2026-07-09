#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("complex", "standard", "fast")
ENV_PREFIXES = ("AKG_AGENTS", "AIKG")
FALLBACK_ORDER = ("complex", "standard", "fast")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether AKG Agents has a complete model-level configuration."
    )
    parser.add_argument("--level", default="standard", help="Model level to check. Default: standard.")
    parser.add_argument("--repo-root", default=str(ROOT), help="Repository root used for project .akg files.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--quiet", action="store_true", help="Suppress output; use only the exit code.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    result = inspect_model_config(args.level, repo_root=repo_root, environ=os.environ, home=Path.home())

    if not args.quiet:
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            _print_human(result)

    return 0 if result["configured"] else 2


def inspect_model_config(
    requested_level: str,
    *,
    repo_root: Path,
    environ: os._Environ[str] | dict[str, str],
    home: Path,
) -> dict[str, Any]:
    settings = _load_settings(repo_root=repo_root, environ=environ, home=home)
    resolved_level = _resolve_level(requested_level, settings["default_model"], settings["models"])
    model = settings["models"].get(resolved_level or requested_level)

    missing: list[str] = []
    if model is None:
        missing.append(requested_level)
    else:
        for field in ("base_url", "api_key", "model_name"):
            if not model.get(field):
                missing.append(field)

    configured = model is not None and not missing
    source = settings["sources"].get(resolved_level or requested_level, "missing")

    return {
        "requested_level": requested_level,
        "resolved_level": resolved_level,
        "configured": configured,
        "source": source,
        "missing": missing,
        "model": _masked_model(model) if model else None,
        "available_levels": sorted(settings["models"]),
        "default_model": settings["default_model"],
        "checked_paths": settings["checked_paths"],
        "accepted_env": _accepted_env_names(requested_level),
    }


def _load_settings(
    *,
    repo_root: Path,
    environ: os._Environ[str] | dict[str, str],
    home: Path,
) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    default_model = "standard"

    checked_paths = [
        str(home / ".akg" / "settings.json"),
        str(repo_root / ".akg" / "settings.json"),
        str(repo_root / ".akg" / "settings.local.json"),
    ]
    layers = [
        ("user", Path(checked_paths[0]), True),
        ("project", Path(checked_paths[1]), False),
        ("local", Path(checked_paths[2]), False),
    ]

    for label, path, use_defaults in layers:
        data = _read_json(path)
        if not data:
            continue
        if data.get("default_model"):
            default_model = str(data["default_model"])
        for level, raw_model in (data.get("models") or {}).items():
            if not isinstance(raw_model, dict):
                continue
            incoming = _model_from_dict(raw_model, use_defaults=use_defaults)
            existing = models.get(level, {})
            models[level] = _merge_model(existing, incoming)
            sources[level] = f"{label}: {path}"

    env_default = _get_env(environ, "DEFAULT_MODEL")
    if env_default:
        default_model = env_default

    thinking_enabled = _parse_thinking(_get_env(environ, "MODEL_ENABLE_THINK"))
    single_env = any(_get_env(environ, name) for name in ("BASE_URL", "API_KEY", "MODEL_NAME"))
    if single_env:
        source = f"env: {_detect_prefix(environ, 'BASE_URL', 'API_KEY', 'MODEL_NAME')}_*"
        model = _model_from_env(environ, "", thinking_enabled=thinking_enabled)
        for level in LEVELS:
            models[level] = model
            sources[level] = source

    for level in LEVELS:
        prefix = f"{level.upper()}_"
        has_level_env = any(
            _get_env(environ, f"{prefix}{name}") for name in ("BASE_URL", "API_KEY", "MODEL_NAME")
        )
        if not has_level_env:
            continue
        source = (
            "env: "
            f"{_detect_prefix(environ, f'{prefix}BASE_URL', f'{prefix}API_KEY', f'{prefix}MODEL_NAME')}_{prefix}*"
        )
        models[level] = _model_from_env(environ, prefix, thinking_enabled=thinking_enabled)
        sources[level] = source

    return {
        "models": models,
        "sources": sources,
        "default_model": default_model,
        "checked_paths": checked_paths,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None


def _model_from_dict(data: dict[str, Any], *, use_defaults: bool) -> dict[str, Any]:
    if use_defaults:
        model = {
            "base_url": "https://api.openai.com/v1",
            "api_key": "",
            "model_name": "gpt-4",
            "temperature": 0.2,
            "max_tokens": 8192,
            "top_p": 0.9,
            "timeout": 300,
            "verify_ssl": True,
            "provider_type": "openai",
        }
    else:
        model = {}
    for key in (
        "base_url",
        "api_key",
        "model_name",
        "temperature",
        "max_tokens",
        "top_p",
        "timeout",
        "verify_ssl",
        "extra_body",
        "provider_type",
    ):
        if key in data:
            model[key] = data[key]
    if "thinking_enabled" in data and "extra_body" not in model:
        model["extra_body"] = {"thinking": {"type": "enabled" if data["thinking_enabled"] else "disabled"}}
    return model


def _model_from_env(
    environ: os._Environ[str] | dict[str, str],
    prefix: str,
    *,
    thinking_enabled: bool | None,
) -> dict[str, Any]:
    model = {
        "base_url": _get_env(environ, f"{prefix}BASE_URL") or "https://api.openai.com/v1",
        "api_key": _get_env(environ, f"{prefix}API_KEY") or "",
        "model_name": _get_env(environ, f"{prefix}MODEL_NAME") or "gpt-4",
        "temperature": _float_env(environ, f"{prefix}TEMPERATURE", 0.2),
        "max_tokens": _int_env(environ, f"{prefix}MAX_TOKENS", 8192),
        "timeout": _int_env(environ, f"{prefix}TIMEOUT", 300),
        "verify_ssl": _bool_env(environ, f"{prefix}VERIFY_SSL", True),
        "provider_type": _get_env(environ, f"{prefix}PROVIDER_TYPE") or "openai",
    }
    if thinking_enabled is not None:
        model["extra_body"] = {"thinking": {"type": "enabled" if thinking_enabled else "disabled"}}
    return model


def _merge_model(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if value not in ("", None):
            merged[key] = value
    return merged


def _resolve_level(requested: str, default_model: str, models: dict[str, dict[str, Any]]) -> str | None:
    if requested in models:
        return requested
    if default_model in models:
        return default_model
    for level in FALLBACK_ORDER:
        if level in models:
            return level
    return None


def _get_env(environ: os._Environ[str] | dict[str, str], name: str) -> str | None:
    for prefix in ENV_PREFIXES:
        value = environ.get(f"{prefix}_{name}")
        if value is not None:
            return value
    return None


def _detect_prefix(environ: os._Environ[str] | dict[str, str], *names: str) -> str:
    for name in names:
        if environ.get(f"AKG_AGENTS_{name}") is not None:
            return "AKG_AGENTS"
    return "AIKG"


def _parse_thinking(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.lower() in {"enabled", "true", "1", "yes", "on"}


def _bool_env(environ: os._Environ[str] | dict[str, str], name: str, default: bool) -> bool:
    value = _get_env(environ, name)
    if value is None:
        return default
    return value.lower() in {"enabled", "true", "1", "yes", "on"}


def _int_env(environ: os._Environ[str] | dict[str, str], name: str, default: int) -> int:
    value = _get_env(environ, name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(environ: os._Environ[str] | dict[str, str], name: str, default: float) -> float:
    value = _get_env(environ, name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _masked_model(model: dict[str, Any]) -> dict[str, Any]:
    masked = {
        "base_url": model.get("base_url"),
        "api_key": _mask_secret(str(model.get("api_key") or "")),
        "model_name": model.get("model_name"),
        "provider_type": model.get("provider_type", "openai"),
        "temperature": model.get("temperature", 0.2),
        "max_tokens": model.get("max_tokens", 8192),
        "timeout": model.get("timeout", 300),
        "verify_ssl": model.get("verify_ssl", True),
    }
    if model.get("extra_body"):
        masked["extra_body"] = model["extra_body"]
    return masked


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}***{value[-4:]}"


def _accepted_env_names(level: str) -> list[str]:
    upper = level.upper()
    names = []
    for prefix in ENV_PREFIXES:
        names.extend(
            [
                f"{prefix}_BASE_URL",
                f"{prefix}_API_KEY",
                f"{prefix}_MODEL_NAME",
                f"{prefix}_{upper}_BASE_URL",
                f"{prefix}_{upper}_API_KEY",
                f"{prefix}_{upper}_MODEL_NAME",
            ]
        )
    return names


def _print_human(result: dict[str, Any]) -> None:
    level = result["requested_level"]
    if result["configured"]:
        model = result["model"] or {}
        print(f"AKG Agents model level '{level}' is configured.")
        print(f"  resolved_level: {result['resolved_level']}")
        print(f"  source: {result['source']}")
        print(f"  model_name: {model.get('model_name')}")
        print(f"  base_url: {model.get('base_url')}")
        print(f"  api_key: {model.get('api_key')}")
        print(f"  provider_type: {model.get('provider_type')}")
        return

    print(f"AKG Agents model level '{level}' is not ready.", file=sys.stderr)
    print(f"  missing: {', '.join(result['missing'])}", file=sys.stderr)
    print(f"  available_levels: {result['available_levels']}", file=sys.stderr)
    print("  configure one of:", file=sys.stderr)
    print("    export AKG_AGENTS_STANDARD_BASE_URL=...", file=sys.stderr)
    print("    export AKG_AGENTS_STANDARD_API_KEY=...", file=sys.stderr)
    print("    export AKG_AGENTS_STANDARD_MODEL_NAME=...", file=sys.stderr)
    print("    or create .akg/settings.local.json / ~/.akg/settings.json", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
