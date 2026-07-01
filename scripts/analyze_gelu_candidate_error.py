#!/usr/bin/env python3
"""Locate worst GELU candidate errors on Ascend NPU."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("gelu_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load candidate module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[32, 512, 1024],
        help="Input tensor shape.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Manual seed for reproducible random input.",
    )
    args = parser.parse_args()

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable; run this on the Ascend worker")

    torch.manual_seed(args.seed)
    candidate_path = Path(args.candidate).resolve()
    module = load_module(candidate_path)
    model = module.ModelNew().to("npu").eval()

    x = torch.randn(*args.shape, dtype=torch.float32).to("npu")
    with torch.no_grad():
        actual = model(x)
        expected = torch.nn.functional.gelu(x)
    torch.npu.synchronize()

    expected_f = expected.float()
    actual_f = actual.float()
    abs_diff = (expected_f - actual_f).abs()
    rel_diff = abs_diff / (expected_f.abs() + 1e-8)

    flat_abs_idx = int(abs_diff.argmax().item())
    flat_rel_idx = int(rel_diff.argmax().item())
    x_flat = x.reshape(-1).float()
    expected_flat = expected_f.reshape(-1)
    actual_flat = actual_f.reshape(-1)
    abs_flat = abs_diff.reshape(-1)
    rel_flat = rel_diff.reshape(-1)

    def item_at(flat_idx: int) -> dict:
        return {
            "flat_index": flat_idx,
            "input": float(x_flat[flat_idx].item()),
            "expected": float(expected_flat[flat_idx].item()),
            "actual": float(actual_flat[flat_idx].item()),
            "abs_diff": float(abs_flat[flat_idx].item()),
            "rel_diff": float(rel_flat[flat_idx].item()),
        }

    payload = {
        "candidate": str(candidate_path),
        "shape": args.shape,
        "seed": args.seed,
        "last_backend": getattr(model, "_last_backend", None),
        "last_error": getattr(model, "_last_error", None),
        "max_abs": item_at(flat_abs_idx),
        "max_rel": item_at(flat_rel_idx),
        "allclose": bool(torch.allclose(expected, actual, rtol=1e-2, atol=1e-2)),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

