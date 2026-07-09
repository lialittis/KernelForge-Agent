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
