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

No recorded cases yet.

