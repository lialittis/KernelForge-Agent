# Benchmark Specification

This file is the project contract for official benchmark behavior. It is not
complete yet; fill it from the competition repository and official materials
before implementing the harness.

## Known From Current Proposal

- Target hardware: Ascend 910 NPU.
- Main benchmark source: community or competition-provided AKG benchmark,
  referred to in the proposal as `akg_kernels_bench_lite`.
- Main implementation path: AKG Agents + Triton-Ascend.
- Expected validation: compile, run, compare against standard/reference
  implementation, and benchmark latency/performance on real Ascend 910.
- Key correctness metrics: Pass@1 and Pass@4.
- Performance should be measured only after correctness passes.

## Open Items To Confirm

- Official repository URL and exact commit or release.
- Exact benchmark task directory layout.
- Exact input task format.
- Exact reference implementation format.
- Required candidate file layout.
- Required backend/runtime versions.
- CANN, MindSpore, AKG, Triton-Ascend, Python, and compiler versions.
- Build command.
- Run command.
- Correctness validation command.
- Performance/profiling command.
- rtol/atol or task-specific tolerance rules.
- Shape/dtype test set generation rules.
- Baseline latency or throughput format.
- Submission package format.
- Competition scoring formula.

## Expected OpSpec Fields

```yaml
id: null
name: null
category: null
source_path: null
reference_path: null

inputs:
  - name: null
    shape: null
    dtype: null
    layout: null

outputs:
  - name: null
    shape: null
    dtype: null
    layout: null

semantics:
  expression: null
  broadcast: null
  reduction_axes: null
  normalization_axes: null
  layout_transform: null
  boundary_conditions: null

validation:
  rtol: null
  atol: null
  max_error_required: null
  shape_cases: []
  dtype_cases: []

performance:
  baseline_latency_ms: null
  baseline_throughput: null
  metric: latency
  warmup: null
  repeats: null

submission:
  required_files: []
  entrypoint: null
```

## Benchmark Classification

Classify every benchmark task into one primary category:

- `elementwise`
- `broadcast`
- `reduction`
- `transpose_layout`
- `normalization`
- `matmul_like`
- `unknown`

Secondary tags may include:

- tail mask required
- non-contiguous access
- dtype accumulation risk
- layout reorder
- fusion opportunity
- UB pressure
- copyin-compute-copyout pipeline opportunity

## First Reproduction Checklist

1. Clone or obtain the official benchmark repository.
2. Record exact source location and commit.
3. Set up Ascend 910 environment.
4. Run official baseline for one simple operator.
5. Save command output summary in an experiment record.
6. Update this file with confirmed commands and formats.

