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
- tuple-output contract when a reduction also selects indices
- explicit tie and ordering semantics for top-k or argmax-like reductions

## Common Failures

- wrong reduction axis
- missing keepdims handling
- accumulation dtype too narrow
- incorrect tail handling
- race or duplicate accumulation in parallel reduction
- unstable top-k ordering when values tie
- returning unnormalized selected probabilities after top-k softmax
- using the wrong index dtype for tuple-output selections

## Profiling And Tuning Notes

- Tune tile size and parallel axis after correctness.
- Compare contiguous versus non-contiguous reduction behavior.
- Watch UB pressure and partial-reduction writeback overhead.
- For small expert dimensions, a single row/program may be preferable to
  complicated cross-program reductions because top-k ordering and selected
  probability renormalization are easier to keep deterministic.

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

### Rowwise Softmax Needs More Than A Naive Full-Row Triton Kernel

Observed for `t1/softmax` with input shape `[32, 512, 4096]` and reduction over
the last dimension:

```text
softmax_v1: torch reference, pass, speedup 1.0006x
softmax_v2: Triton one row/program, 4096 tile, pass, speedup 0.7315x
softmax_v3: Triton two rows/program, sequential 4096 tiles, pass, speedup 0.8871x
softmax_v4: Triton four rows/program, sequential 4096 tiles, pass, speedup 0.9225x
```

Lessons:

- The stable rowwise softmax formula `exp(x - max(x)) / sum(exp(x - max(x)))`
  is numerically safe for the official tolerance when implemented with
  float32 Triton reductions.
- Reducing launch/program count by handling multiple rows per program improved
  latency, but the naive Triton implementation still did not beat the torch
  baseline.
- For future softmax attempts, prioritize memory-traffic reduction, backend
  softmax primitives, or a different lowering strategy before only increasing
  rows per program.

### `t2/moe_topk_softmax` Tuple Output And Top-K Renormalization

Source OpSpec: `benchmarks/parsed/t2_moe_topk_softmax.yaml`.

Pattern:

1. For each token row, compute stable softmax over expert logits with
   max-subtraction.
2. Select top-2 probabilities and corresponding expert indices.
3. Renormalize the selected probabilities by their selected sum.
4. Return a tuple: `top_k_probs: float32[1024, 2]` and
   `top_k_indices: int64[1024, 2]`.

Required generation constraints:

- Preserve tuple output order exactly: probabilities first, indices second.
- Match `torch.topk` tie behavior and ordering for equal probabilities.
- Store indices as int64, not int32.
- Preserve the Sketch `output_contract` and `numerical_plan` in prompt
  context so the generated code cannot treat this as ordinary softmax.

Repair rules:

- If probabilities pass but indices fail, inspect tie ordering and index dtype.
- If selected probabilities sum to something other than 1, add the
  selected-probability renormalization stage after top-k selection.
- If only near-zero rows fail relative tolerance, inspect max-subtraction and
  float32 accumulation for the softmax denominator.
