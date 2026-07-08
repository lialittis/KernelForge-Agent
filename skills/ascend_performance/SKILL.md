# Ascend Performance Skill

## Applicability

Use after a candidate passes correctness and needs latency or throughput
optimization on Ascend 910.

## Search Knobs

- tile size
- num cores
- parallel axis
- vector width
- unroll factor
- double buffer
- memory access order
- copyin/compute/copyout pipeline
- boundary strategy

## Profiling Checklist

- Record warmup and repeat counts.
- Compare mean and median latency.
- Compare against baseline.
- Identify whether bottleneck appears compute-bound, memory-bound,
  launch-bound, or indexing-bound.
- Save profile artifact path in the experiment record.

## Tuning Rules

- Optimize only correctness-passing kernels.
- Keep search space small and interpretable.
- Change one group of knobs per experiment when possible.
- Record both failed and successful tuning attempts.

## Bad-To-Good Cases

### Fused NPU Intrinsic Can Dominate Custom Triton Pointwise

Observed for `t1/fused_silu_and_mul`:

```text
fused_silu_and_mul_v1: torch_npu.npu_swiglu, pass, speedup 1.0027x
fused_silu_and_mul_v2: Triton bs8192, pass, speedup 0.0033x
fused_silu_and_mul_v3: Triton bs4096, pass, speedup 0.0033x
fused_silu_and_mul_v4: Triton bs4096 x 2 chunks, pass, speedup 0.0033x
```

Performance rule:

- If the official baseline is already a fused NPU intrinsic, a naive custom
  Triton pointwise implementation may be useful for correctness evidence but
  poor for performance.
- Do not spend many tuning cycles on this pattern unless testing a different
  backend, a substantially different memory strategy, or a generated candidate
  with a concrete reason to beat the intrinsic.
- Record it as a negative performance trajectory so the Retriever can avoid
  over-prioritizing similar manual Triton sketches.

### Rowwise Reduction Can Beat Framework Baseline

Observed for `t1/sigmoid_scale_sum`:

```text
sigmoid_scale_sum_v1: torch reference, pass, speedup 1.0006x
sigmoid_scale_sum_v2: Triton one row/program, 8192 tile, pass, speedup 2.0279x
sigmoid_scale_sum_v3: Triton one row/program, 2 x 4096 chunks, pass, speedup 1.9367x
sigmoid_scale_sum_v4: Triton one row/program, 4 x 2048 chunks, pass, speedup 1.5785x
```

Performance rule:

- When a rowwise reduction's full contiguous reduction axis fits in UB, prefer
  a single full-axis reduction tile before trying sequential chunking.
- If chunking is needed for UB pressure, benchmark each chunk count; smaller
  chunks are not automatically faster.
- Use backend probes to confirm the run used a real Triton path before
  interpreting speedup as custom-kernel speedup.

### Correct GELU Triton Kernel Still Slower Than Framework

Observed for `gelu_triton_v7` and improved through `gelu_triton_v10`:

```text
gelu_triton_v7: correctness pass, speedup 0.0728x, block size 1024
gelu_triton_v8: correctness pass, speedup 0.2856x, block size 4096
gelu_triton_v9: correctness pass, speedup 0.4869x, block size 8192
gelu_triton_v10: correctness pass, speedup 0.5635x, block size 16384
gelu_triton_v11: compile failure, UB overflow at block size 32768
gelu_triton_v12: correctness pass, speedup 0.5764x, block size 24576
gelu_triton_v13: correctness pass, speedup 0.6059x, block size 16384 x 2 chunks
gelu_triton_v14: correctness pass, speedup 0.5875x, block size 16384 x 3 chunks
gelu_triton_v15: correctness pass, speedup 0.5858x, block size 24576 x 2 chunks
gelu_triton_v16: correctness pass, speedup 0.5373x, tl.sigmoid lowering
gelu_triton_v17: compile failure, non-constexpr global used inside @jit function
gelu_triton_v18: correctness pass, speedup 0.5764x, inline exp2 lowering
```

First tuning axis:

- Reduce program count by increasing block size.
- Compare each candidate against the same official benchmark settings.
- Do not treat correctness-only probe timing as benchmark timing.
- Use UB overflow errors to infer the upper block-size boundary. For v11, the
  compiler reported `2097152` required bits and `1572864` available bits, which
  implies a boundary near `24576` elements.
- If the boundary candidate fails, stop increasing block size and reduce UB
  footprint instead.
- Once single-vector block size hits the UB boundary, try sequential chunks per
  program with a smaller per-vector block to reduce program count without
  increasing one vector tile.
- Increase chunks per program one step at a time and compare against the same
  official benchmark settings; fewer programs can help, but too much sequential
  work per program can reduce parallelism.
- If a higher chunk count regresses, combine the best successful chunk count
  with the best single-vector block size before abandoning the axis.
- After the tiling axis plateaus, keep the best tiling fixed and test backend
  math lowerings such as `tl.sigmoid` versus explicit reciprocal-exp sigmoid.
- If `tl.sigmoid` regresses, test equivalent explicit lowerings such as `exp2`
  before moving to a different operator or backend strategy.
- Triton JIT functions cannot read ordinary Python globals reliably. Inline
  scalar constants in the JIT expression or pass them as constexpr values.
- If a later math lowering compiles but does not beat the current best, freeze
  the operator and move to the next benchmark subset. For GELU, v13 remains the
  best tracked candidate and v18 did not justify more single-operator tuning.
