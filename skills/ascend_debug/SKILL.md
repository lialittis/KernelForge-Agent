# Ascend Debug Skill

## Applicability

Use for compile errors, runtime errors, API incompatibilities, shape/dtype
failures, and environment issues on Ascend NPU.

## Error Categories

- syntax or API error
- missing include/import/dependency
- unsupported backend operation
- type mismatch
- shape mismatch
- out-of-bounds access
- runtime launch failure
- environment or version mismatch

## Routing Rules

- Syntax/API/type errors usually go back to Code Agent.
- Shape/broadcast/layout/tile-plan errors usually go back to Sketch Agent.
- Environment errors should be separated from generation failures.
- Numerical errors should involve dtype, boundary, mask, and reduction checks.

## Log Summary Format

```text
error_category:
first_error_line:
likely_owner:
minimal_context:
suggested_fix:
```

## Bad-To-Good Cases

### Triton Runtime Reports Zero Active Drivers

Observed in `gelu_triton_v1` on Ascend worker:

```text
last_backend: torch_fallback_after_error
last_error: RuntimeError: 0 active drivers ([]). There should only be one.
```

Interpretation:

- Triton imported, but no usable backend driver was registered for the current
  device.
- Treat any benchmark result from this path as PyTorch fallback behavior, not a
  custom Triton-Ascend kernel result.

First checks:

- `python scripts/diagnose_triton_ascend.py`
- `pip show triton triton-ascend torch torch-npu`
- Confirm the Ascend toolkit environment was sourced before running Python.
- Confirm the installed Triton-Ascend package is compatible with CANN, Python,
  torch, and torch_npu on the worker.

If the diagnosis shows `triton-ascend: null` and
`ModuleNotFoundError: No module named 'triton_ascend'`, the immediate issue is
missing backend installation. Install the backend package, then rerun both:

- `python scripts/diagnose_triton_ascend.py`
- `python scripts/probe_gelu_triton_backend.py`

### Preserve CANN Python Paths When Adding Repo Imports

Observed in `t3/causal_conv1d` on the Ascend worker:

```text
ModuleNotFoundError: No module named 'tbe'
```

Interpretation:

- The worker's CANN install already provided `tbe`, `te`, and `auto_tune` under
  `/usr/local/Ascend/cann-8.5.1/python/site-packages` after sourcing
  `set_env.sh`.
- The failure was caused by a benchmark command that ran Python as
  `PYTHONPATH=/data/KernelForge-Agent python ...`, which replaced the
  `PYTHONPATH` set by CANN and hid those modules from the ACL/TBE compile path.

First fix:

- Source `/usr/local/Ascend/ascend-toolkit/set_env.sh`.
- Activate the benchmark venv.
- Prepend project imports with:

```bash
export PYTHONPATH=/data/KernelForge-Agent:${PYTHONPATH:-}
```

- Verify imports before rerunning the failing case:

```bash
python -c "import importlib.util; print(importlib.util.find_spec('tbe'))"
```

For `t3/causal_conv1d`, this fixed the official reference path and the case
passed with speedup `0.9971x` and weighted score `119.65`.

### GELU Absolute Error Passes But Relative Error Fails

Observed in `gelu_triton_v1` after a real Triton-Ascend launch:

```text
max_abs_diff=4.737377e-04
max_rel_diff=4.803681e+00
```

Interpretation:

- The official benchmark checks max absolute error and max relative error
  separately.
- Small absolute differences around near-zero reference outputs can still fail
  the relative-error threshold.
- For exact GELU, prefer forms that avoid cancellation in negative-tail outputs,
  such as `0.5 * x * erfc(-x / sqrt(2))`, if the backend supports `erfc`.
- If `erfc` is unavailable, a temporary benchmark-safety strategy is to compute
  the bulk in Triton and repair the problematic tail with the framework exact
  GELU. Treat this as a diagnostic bridge, not a final pure-kernel solution.
- A repair threshold of `x < -3.0` reduced but did not eliminate the relative
  error. Use a wider threshold such as `x < -2.1` when validating the tail
  hypothesis against this benchmark.
- Framework tail repair can pass correctness but is extremely slow. Prefer a
  pure-kernel tail approximation such as the Abramowitz-Stegun erfc polynomial
  for `x < -2.1`.
- If a pure Triton approximation fails, run
  `scripts/analyze_gelu_candidate_error.py` to identify the input value at the
  worst relative-error element before changing formulas again.
- The NPU `torch.nn.functional.gelu` reference may follow tanh-approximate GELU
  numerics even when the source does not pass `approximate="tanh"`. Compare
  worst-error values against both erf-exact and tanh-approximate formulas before
  assuming exact GELU is the benchmark target.
- Avoid computing tanh-approximate GELU as `0.5 * x * (1 + tanh(u))` when
  `u << 0`; that reintroduces cancellation. Use the equivalent stable form
  `x / (1 + exp(-2u))`.

### Triton JIT Rejects Ordinary Python Globals

Observed in `gelu_triton_v17`:

```text
NameError: Cannot access global variable _NEG_TWO_LOG2E from within @jit'ed function.
```

Interpretation:

- Triton-Ascend can reject constants captured as ordinary Python globals inside
  `@triton.jit` functions.
- Inline scalar constants in the JIT expression or pass them as constexpr
  values.
- Retest with a backend probe before benchmarking; in the GELU case,
  `gelu_triton_v18` fixed this compile issue but still regressed in
  performance versus `gelu_triton_v13`.

### Fused Elementwise Tile Too Large Causes UB Overflow

Observed in early `fused_silu_and_mul` candidates:

```text
block size 16384: ub overflow, requires 3145728 bits while 1572864 bits available
block size 8192 x 2 chunks: ub overflow, requires about 2097152-2228224 bits while 1572864 bits available
```

Interpretation:

- Fused elementwise kernels with multiple loads and activation intermediates
  can exceed UB capacity at smaller tile sizes than a simple pointwise kernel.
- Sequential chunks inside one program can still increase live UB pressure if
  the compiler keeps chunk temporaries live.

First fix:

- Reduce the per-vector tile size and reprobe before changing the math.
- For this case, block size `4096` and `4096 x 2` chunks both compiled and
  passed correctness as real Triton paths.
