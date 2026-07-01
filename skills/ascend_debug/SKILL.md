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
