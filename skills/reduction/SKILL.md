# Reduction Skill

## Applicability

Use for operators that reduce one or more axes using sum, max, min, mean,
variance-like, or similar associative patterns.

## Non-Applicability

Do not use as the primary skill for pure elementwise, broadcast-only, or layout
transpose operators.

## Shape And Dtype Constraints

- Record reduction axes and keepdims behavior.
- Record accumulation dtype separately from input/output dtype.
- Record whether reduction axis is contiguous.
- Capture tolerance requirements because reductions amplify numerical error.

## Recommended Sketch Focus

- reduction axis mapping
- tile plan for partial reductions
- accumulation dtype
- parallel reduction strategy
- boundary mask for tail elements
- final writeback shape

## Common Failures

- wrong reduction axis
- missing keepdims handling
- accumulation dtype too narrow
- incorrect tail handling
- race or duplicate accumulation in parallel reduction

## Profiling And Tuning Notes

- Tune tile size and parallel axis after correctness.
- Compare contiguous versus non-contiguous reduction behavior.
- Watch UB pressure and partial-reduction writeback overhead.

## Bad-To-Good Cases

### Rowwise Sigmoid Sum Fits In One Reduction Tile

Observed for `t1/sigmoid_scale_sum` with input shape `[1000, 8192]`:

```text
sigmoid_scale_sum_v1: torch reference, pass, speedup 1.0006x
sigmoid_scale_sum_v2: Triton one row/program, 8192 tile, pass, speedup 2.0279x
sigmoid_scale_sum_v3: Triton one row/program, 2 x 4096 chunks, pass, speedup 1.9367x
sigmoid_scale_sum_v4: Triton one row/program, 4 x 2048 chunks, pass, speedup 1.5785x
```

Lessons:

- If the contiguous reduction axis fits in UB, start with one program per
  output row and one tile covering the full reduction axis.
- Use float32 accumulation for float32 rowwise sums unless the OpSpec requires
  a narrower output.
- Split the row into sequential chunks only when the full reduction tile causes
  compile failure or UB pressure. Chunking passed correctness here, but it
  reduced performance.
- Preserve `keepdim=True` output shape explicitly; for this case the output is
  `[1000, 1]`, not `[1000]`.
