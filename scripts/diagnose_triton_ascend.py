#!/usr/bin/env python3
"""Collect Triton-Ascend environment diagnostics as JSON."""

import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
from typing import Any


PACKAGE_NAMES = [
    "torch",
    "torch-npu",
    "torch_npu",
    "triton",
    "triton-ascend",
    "triton_ascend",
]

MODULE_NAMES = [
    "torch",
    "torch_npu",
    "triton",
    "triton.language",
    "triton_ascend",
]

ENV_KEYS = [
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_AICPU_PATH",
    "ASCEND_TOOLKIT_HOME",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
]


def _safe_call(fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _package_versions() -> dict[str, Any]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _module_imports() -> dict[str, Any]:
    imports = {}
    for name in MODULE_NAMES:
        try:
            module = importlib.import_module(name)
            imports[name] = {
                "ok": True,
                "file": getattr(module, "__file__", None),
                "version": getattr(module, "__version__", None),
            }
        except Exception as exc:
            imports[name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return imports


def _torch_npu_info() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return {"import_error": f"{type(exc).__name__}: {exc}"}

    info = {
        "torch_version": getattr(torch, "__version__", None),
        "has_torch_npu_attr": hasattr(torch, "npu"),
    }
    if not hasattr(torch, "npu"):
        return info

    info["npu_is_available"] = _safe_call(lambda: bool(torch.npu.is_available()))
    info["npu_device_count"] = _safe_call(lambda: int(torch.npu.device_count()))
    info["npu_current_device"] = _safe_call(lambda: int(torch.npu.current_device()))
    info["npu_device_name"] = _safe_call(lambda: str(torch.npu.get_device_name(0)))
    return info


def _triton_info() -> dict[str, Any]:
    info = {}
    try:
        triton = importlib.import_module("triton")
        tl = importlib.import_module("triton.language")
    except Exception as exc:
        return {"import_error": f"{type(exc).__name__}: {exc}"}

    info["triton_version"] = getattr(triton, "__version__", None)
    info["has_jit"] = hasattr(triton, "jit")
    info["has_language_erf"] = hasattr(tl, "erf")
    info["has_language_constexpr"] = hasattr(tl, "constexpr")

    def active_driver_repr():
        from triton.runtime import driver

        return repr(driver.active)

    def current_target_repr():
        from triton.runtime import driver

        return repr(driver.active.get_current_target())

    info["runtime_driver_active"] = _safe_call(active_driver_repr)
    info["runtime_current_target"] = _safe_call(current_target_repr)
    return info


def _ascend_paths() -> dict[str, Any]:
    roots = [
        Path("/usr/local/Ascend"),
        Path("/usr/local/Ascend/ascend-toolkit"),
        Path("/usr/local/Ascend/ascend-toolkit/latest"),
    ]
    paths = {}
    for path in roots:
        entry = {"exists": path.exists(), "children": []}
        if path.is_dir():
            try:
                entry["children"] = sorted(item.name for item in path.iterdir())[:20]
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
        paths[str(path)] = entry
    return {
        str(path): paths[str(path)]
        for path in roots
    }


def main() -> int:
    payload = {
        "packages": _package_versions(),
        "imports": _module_imports(),
        "torch_npu": _torch_npu_info(),
        "triton": _triton_info(),
        "ascend_paths": _ascend_paths(),
        "environment": {key: os.environ.get(key) for key in ENV_KEYS},
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
