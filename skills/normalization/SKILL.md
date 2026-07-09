# Normalization Skill

## Applicability

Use for layer norm, batch norm-like, RMS norm-like, mean/variance
normalization, and operators that combine reductions with elementwise
post-processing.

## Non-Applicability

Do not use for simple reductions without normalization or for pure elementwise
operators.

## Shape And Dtype Constraints

- Record normalized axes.
- Record epsilon behavior.
- Record accumulation dtype.
- Record broadcast shape for scale and bias parameters.

## Recommended Sketch Focus

- reduction plan for statistics
- accumulation dtype
- reuse of mean/variance or RMS terms
- broadcast plan for scale/bias
- numerical stability
- multi-stage pipeline if needed

## Common Failures

- wrong normalized axes
- missing epsilon
- dtype instability
- incorrect parameter broadcasting
- multiple-pass implementation with inconsistent indexing

## Profiling And Tuning Notes

- Prioritize correctness and numerical stability first.
- Profile whether statistics computation or final elementwise pass dominates.
- Consider fusion only after each component is validated.
- For fixed hidden size 4096 RMSNorm-style rows on Ascend, first try one
  Triton program per row with a 4096-wide vector if UB pressure permits. Then
  try 2048x2 or 1024x4 chunking only if compilation or UB pressure fails.

## Bad-To-Good Cases

### `t2/add_rmsnorm_cast` Rowwise Fused RMSNorm Cast

- Source: `experiments/reports/2026-07-09-add-rmsnorm-cast-pass4.yaml`.
- Pattern: `x_added = x + residual`; compute `sum(x_added^2)` over the last
  axis in float32; apply `rsqrt(sum / hidden + eps)`; multiply by broadcast
  `gamma`; store directly to fp16 output.
- Best seed: `add_rmsnorm_cast_v2`, one row per Triton program with a
  4096-wide vector. It passed the official benchmark with speedup `2.0135x`
  and weighted score `105.2`.
- Correct but slightly slower variants split each row into `2048x2` and
  `1024x4` chunks. Use those as repair routes if the single 4096-wide program
  hits UB or compiler limits.
- Numerical note: official fp16 output tolerated max absolute error
  `0.00390625` and max relative error `0.0024865244049578905`, within
  `rtol=atol=1e-2`.

### `t2/add_rmsnorm_quant` Exact Int8 Boundary Failure

- Source: `experiments/reports/2026-07-09-add-rmsnorm-quant-pass4.yaml`.
- Pattern: `x_added = x + residual`; RMSNorm over hidden size 4096; multiply
  by `gamma`; then compute `round(output / scale + zero_point).clamp(-128,
  127).to(int8)`.
- Triton-Ascend 3.2.0 exposed `tl.floor`, `tl.minimum`, and `tl.maximum`, but
  not `tl.round`. Using `floor(x + 0.5)` plus int8 store launched successfully
  but failed the official benchmark with one-int8 boundary differences.
- For int8 outputs, the benchmark's `max_abs_diff <= 0.01` gate effectively
  requires exact equality. Near a quantization boundary, tiny RMSNorm/reduction
  differences are enough to fail even when probes sometimes match exactly.
- Treat quantized normalization as a poor pure-Triton target until there is a
  reliable exact-rounding path, a safe framework quantization fallback, or a
  backend-specific quantization primitive.

### `t3/layernorm_gated` Fp16 Gated RMSNorm

- Source: `experiments/reports/2026-07-09-layernorm-gated-pass4.yaml`.
- Pattern: RMSNorm over hidden size 4096 on fp16 `x`, multiply by fp16
  `weight`, then multiply by `sigmoid(z)`.
- PyTorch NPU keeps the visible intermediates as fp16 for `pow`, `mean`,
  `rsqrt`, `sigmoid`, and output. A fp32-heavy Triton implementation failed
  the official correctness gate with `max_abs_diff=0.01171875`.
- Matching fp16 intermediate rounding in Triton fixed correctness: cast the
  square, variance, rstd, sigmoid output, and each multiply stage back to
  `tl.float16`.
- Chunked reductions (`2048x2`, `1024x4`) passed small probes but failed the
  full-shape official runner. Preserve one 4096-wide reduction per row for
  numerical stability.
- Row grouping is useful after correctness is fixed. The best seed,
  `layernorm_gated_v4`, groups four rows per Triton program, passed the
  official benchmark exactly, and reached speedup `1.5137x` with weighted
  score `130.27`.
