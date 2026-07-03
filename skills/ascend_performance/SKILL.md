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

### Correct GELU Triton Kernel Still Slower Than Framework

Observed for `gelu_triton_v7` and improved through `gelu_triton_v10`:

```text
gelu_triton_v7: correctness pass, speedup 0.0728x, block size 1024
gelu_triton_v8: correctness pass, speedup 0.2856x, block size 4096
gelu_triton_v9: correctness pass, speedup 0.4869x, block size 8192
gelu_triton_v10: correctness pass, speedup 0.5635x, block size 16384
```

First tuning axis:

- Reduce program count by increasing block size.
- Compare each candidate against the same official benchmark settings.
- Do not treat correctness-only probe timing as benchmark timing.
- Continue this axis with `32768` before changing the math or backend strategy.
