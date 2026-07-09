# Transpose And Layout Skill

## Applicability

Use for transpose, reshape-with-layout-effect, layout conversion, permutation,
and operators dominated by non-contiguous reads or writes.

## Non-Applicability

Do not use for semantic reductions or arithmetic-only elementwise operators.

## Shape And Dtype Constraints

- Record input layout and output layout.
- Record permutation order.
- Distinguish metadata-only reshape from real memory reorder.
- Track alignment and tail behavior.

## Recommended Sketch Focus

- input/output index mapping
- read-contiguous versus write-contiguous choice
- tile reorder plan
- boundary mask
- bank/conflict risk notes when known

## Common Failures

- wrong permutation order
- treating layout conversion as metadata-only reshape
- non-contiguous indexing bugs
- poor performance from scattered reads and writes

## Profiling And Tuning Notes

- Start with correctness-oriented explicit index mapping.
- Tune tile shapes to improve locality.
- Compare read-contiguous and write-contiguous plans.
- For RoPE-style rotate-half kernels, compare against any available NPU
  intrinsic first. If implementing Triton, use fp32 accumulation before the
  final fp16 store to match `torch_npu.npu_rotary_mul` under strict max-relative
  error checks.

## Bad-To-Good Cases

### `t2/rope` Rotary Position Embedding

- Source: `experiments/reports/2026-07-09-rope-pass4.yaml`.
- Pattern: broadcast `cos` and `sin` over batch/head axes, rotate the last
  dimension by `concat(-x[..., D/2:], x[..., :D/2])`, then compute
  `cos * x + sin * rotated`.
- Initial fp16 arithmetic matched by absolute tolerance but exceeded the
  official max-relative-error gate near zero-valued outputs.
- Good fix: cast `x`, `cos`, `sin`, and the rotated partner to fp32 before the
  fused multiply-add, then store fp16. This matched `torch_npu.npu_rotary_mul`
  exactly in the recorded probe and official benchmark trials.
- Performance note: the NPU intrinsic wrapper remains best overall at
  `1.0006x` speedup and score `90.01`; the best Triton candidate
  (`rope_v4`, flat 1024-element tile) reaches parity at `1.0001x` and score
  `90.0` but does not beat the intrinsic.
