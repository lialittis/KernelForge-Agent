#!/usr/bin/env python3
"""Probe an add_rmsnorm_quant candidate backend and exact int8 behavior."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch


DEFAULT_CANDIDATE = (
    "outputs/submissions/add_rmsnorm_quant_pass4/"
    "add_rmsnorm_quant_v2/t2/add_rmsnorm_quant.py"
)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("add_rmsnorm_quant_candidate", path)
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
        nargs=2,
        default=[128, 4096],
        metavar=("ROWS", "HIDDEN"),
        help="Flattened x/residual tensor shape for the probe.",
    )
    args = parser.parse_args()

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable; run this on the Ascend worker")

    candidate_path = Path(args.candidate).resolve()
    module = load_module(candidate_path)
    model = module.ModelNew().to("npu").eval()

    rows, hidden = args.shape
    x = torch.randn(rows, hidden, dtype=torch.float32).to("npu")
    residual = torch.randn(rows, hidden, dtype=torch.float32).to("npu")
    gamma = torch.randn(hidden, dtype=torch.float32).to("npu")
    scale = (torch.randn(1, dtype=torch.float32) * 0.1 + 0.01).to("npu")
    zero_point = (torch.randn(1, dtype=torch.float32) * 10).to("npu")
    with torch.no_grad():
        y = model(x, residual, gamma, scale, zero_point)
        x_added = x + residual
        variance = x_added.pow(2).mean(dim=-1, keepdim=True)
        normalized = (x_added * torch.rsqrt(variance + 1e-6)) * gamma
        ref = torch.round(normalized / scale + zero_point).clamp(-128, 127).to(torch.int8)
    torch.npu.synchronize()

    abs_diff = (ref.float() - y.float()).abs()
    rel_diff = abs_diff / (ref.float().abs() + 1e-8)
    exact_match = bool(torch.equal(ref, y))
    mismatches = int((ref != y).sum().item())
    payload = {
        "candidate": str(candidate_path),
        "shape": args.shape,
        "output_device": str(y.device),
        "output_dtype": str(y.dtype),
        "last_backend": getattr(model, "_last_backend", None),
        "last_error": getattr(model, "_last_error", None),
        "max_abs_diff": float(abs_diff.max().item()),
        "max_rel_diff": float(rel_diff.max().item()),
        "allclose": bool(torch.allclose(ref.float(), y.float(), rtol=1e-2, atol=1e-2)),
        "exact_match": exact_match,
        "mismatches": mismatches,
        "scale": float(scale.cpu().item()),
        "zero_point": float(zero_point.cpu().item()),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
