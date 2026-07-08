#!/usr/bin/env python3
"""Probe a softmax candidate backend and numerical behavior."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch


DEFAULT_CANDIDATE = "outputs/submissions/softmax_pass4/softmax_v2/t1/softmax.py"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("softmax_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load candidate module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[32, 512, 4096],
        metavar="DIM",
        help="Input tensor shape for the probe.",
    )
    parser.add_argument("--dim", type=int, default=-1)
    args = parser.parse_args()

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable; run this on the Ascend worker")

    candidate_path = Path(args.candidate).resolve()
    module = load_module(candidate_path)
    model = module.ModelNew(args.dim).to("npu").eval()

    x = torch.randn(*args.shape, dtype=torch.float32).to("npu")
    with torch.no_grad():
        y = model(x)
        ref = torch.softmax(x, dim=args.dim)
    torch.npu.synchronize()

    abs_diff = (ref.float() - y.float()).abs()
    rel_diff = abs_diff / (ref.float().abs() + 1e-8)
    payload = {
        "candidate": str(candidate_path),
        "shape": args.shape,
        "dim": args.dim,
        "output_device": str(y.device),
        "output_dtype": str(y.dtype),
        "last_backend": getattr(model, "_last_backend", None),
        "last_error": getattr(model, "_last_error", None),
        "max_abs_diff": float(abs_diff.max().item()),
        "max_rel_diff": float(rel_diff.max().item()),
        "allclose": bool(torch.allclose(ref, y, rtol=1e-2, atol=1e-2)),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
