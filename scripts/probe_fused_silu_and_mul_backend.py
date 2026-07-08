#!/usr/bin/env python3
"""Probe a fused_silu_and_mul candidate backend and numerical behavior."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch

try:
    import torch_npu
except Exception:
    torch_npu = None


DEFAULT_CANDIDATE = (
    "outputs/submissions/fused_silu_and_mul_pass4/"
    "fused_silu_and_mul_v2/t1/fused_silu_and_mul.py"
)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("fused_silu_and_mul_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load candidate module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(combined):
    if torch_npu is not None and hasattr(torch_npu, "npu_swiglu"):
        return torch_npu.npu_swiglu(combined, dim=-1)
    x, y = torch.chunk(combined, 2, dim=-1)
    return torch.nn.functional.silu(x) * y


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=[4096, 8192],
        metavar=("ROWS", "COLS"),
        help="Input combined tensor shape for the probe.",
    )
    args = parser.parse_args()

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable; run this on the Ascend worker")
    if args.shape[1] % 2 != 0:
        raise ValueError("The last dimension must be even for fused_silu_and_mul")

    candidate_path = Path(args.candidate).resolve()
    module = load_module(candidate_path)
    model = module.ModelNew().to("npu").eval()

    rows, cols = args.shape
    x = torch.randn(rows, cols // 2, dtype=torch.float32)
    y = torch.randn(rows, cols // 2, dtype=torch.float32)
    combined = torch.cat([x, y], dim=-1).to("npu")
    with torch.no_grad():
        output = model(combined)
        ref = reference(combined)
    torch.npu.synchronize()

    abs_diff = (ref.float() - output.float()).abs()
    rel_diff = abs_diff / (ref.float().abs() + 1e-8)
    payload = {
        "candidate": str(candidate_path),
        "shape": args.shape,
        "output_device": str(output.device),
        "output_dtype": str(output.dtype),
        "last_backend": getattr(model, "_last_backend", None),
        "last_error": getattr(model, "_last_error", None),
        "max_abs_diff": float(abs_diff.max().item()),
        "max_rel_diff": float(rel_diff.max().item()),
        "allclose": bool(torch.allclose(ref, output, rtol=1e-2, atol=1e-2)),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
