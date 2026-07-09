# Matmul-Like Skill

## Applicability

Use for matrix multiplication, batched matmul, attention-like inner products,
and operators dominated by M/N/K tiled compute.

## Non-Applicability

Do not use for simple elementwise, broadcast, or layout-only operations.

## Shape And Dtype Constraints

- Record M, N, K dimensions and batch dimensions.
- Record transposition flags.
- Record accumulation dtype.
- Record output dtype and tolerance.

## Recommended Sketch Focus

- M/N/K tile plan
- data reuse strategy
- memory movement plan
- double buffering
- compute/memory overlap
- boundary mask for non-multiple tile sizes
- explicit `axis_map` with M/N/K extents and source tensor axes
- `dtype_plan` with lhs/rhs/bias/accumulator/output dtypes
- bias broadcast contract for bias-add variants

## Common Failures

- wrong M/N/K interpretation
- missing transpose handling
- accumulation dtype mismatch
- incorrect boundary handling
- severe performance loss from poor data reuse
- UB/L1 overflow from overly large accumulator tiles
- using scalar/vector code when a backend matmul or dot intrinsic is available
- bias shape `[1, N]` accidentally indexed as `[M, N]`

## Profiling And Tuning Notes

- Treat this as a later-stage category unless official benchmark requires it
  early.
- Compare against baseline carefully.
- Search tile sizes and double buffering only after correctness.
- On Ascend 910B with Triton-Ascend, prefer a backend dot/matmul lowering
  path when available. A hand-written blocked K loop is mainly a debug path
  unless it can use hardware matrix instructions.
- Keep candidate tile shapes small enough that accumulator, A tile, B tile,
  and optional bias tile fit UB/L1. If compilation reports UB overflow, reduce
  `block_m` or `block_n` before reducing `block_k`.
- For bf16/fp16 inputs, accumulate in float32 unless the OpSpec or reference
  behavior proves a narrower accumulation is required.

## Bad-To-Good Cases

### `t1/matmul_basic` OpSpec And Sketch Coverage

- Source OpSpec: `benchmarks/parsed/t1_matmul_basic.yaml`.
- Shape: `A[32, 8192] @ B[8192, 8192] -> out[32, 8192]`.
- Dtype: bf16 inputs and output, float32 accumulator in the Sketch.
- Required Sketch fields: `axis_map` for M/N/K, `tile_plan.axes:
  [M, N, K]`, `dtype_plan.accumulator: float32`, and backend risk tags for
  bf16 accumulation, large K tiling, backend choice, and tail masks.

Generation rule:

- Do not emit a naive elementwise-style flat loop for this case. The correct
  generation plan is either a backend matmul/dot intrinsic path or a blocked
  M/N/K template with explicit accumulator tiles and masks.

### `t1/matmul_biasadd` Bias Broadcast Contract

- Source OpSpec: `benchmarks/parsed/t1_matmul_biasadd.yaml`.
- Shape: `A[4096, 4096] @ B[4096, 4096] + bias[1, 4096]`.
- Bias broadcasts over M, so the bias index is `bias[0, n]`.
- Required Sketch field: `memory_plan.bias.broadcast:
  "bias[0, n] broadcasts over M"`.

Repair rule:

- If correctness fails with row-dependent bias errors, inspect whether the
  generated code indexed bias with an M dimension. Bias must depend only on N.
