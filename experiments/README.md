# Experiments

This directory stores experiment metadata and summaries. It should make every
generated candidate reproducible and comparable across machines and agents.

## Rules

- Store structured experiment records under `experiments/runs/`.
- Use one file per experiment run.
- Commit metadata and concise summaries.
- Do not commit large build directories, raw logs, binaries, caches, profiling
  dumps, or full generated output trees by default.
- Record artifact paths even when artifacts are stored outside Git.
- Record OpSpec, Sketch, backend, prompt version, model, and retrieved skills.
- Promote reusable findings into `skills/`.

## Naming

Use a stable, sortable name:

```text
YYYY-MM-DD-short-task-or-method.yaml
```

Examples:

```text
2026-06-30-baseline-elementwise.yaml
2026-07-03-triton-ascend-pass4-v1.yaml
2026-07-05-reduction-repair-loop.yaml
2026-07-08-ascend910-profile-search.yaml
```

## Suggested Schema

```yaml
id: 2026-06-30-baseline-elementwise
date: 2026-06-30
agent: codex
machine: workstation-a
branch: task/example
commit: null
status: planned

benchmark:
  source: akg_kernels_bench_lite
  source_commit: null
  task_id: null
  operator_name: null
  operator_category: elementwise
  reference_path: null

environment:
  hardware: Ascend 910
  cann_version: null
  mindspore_version: null
  akg_version: null
  triton_ascend_version: null
  python_version: null

generation:
  provider: null
  model: null
  agent_role: null
  backend: triton_ascend
  prompt_version: null
  retrieved_skills: []
  opspec_path: null
  sketch_path: null
  candidate_path: null
  candidate_index: 0
  repair_iteration: 0

results:
  compile:
    status: not_run
    log: null
  runtime:
    status: not_run
    log: null
  correctness:
    status: not_run
    rtol: null
    atol: null
    max_error: null
    mean_error: null
    failed_shapes: []
  pass_n:
    pass_at_1: null
    pass_at_4: null
  performance:
    status: not_run
    warmup: null
    repeats: null
    mean_latency_ms: null
    median_latency_ms: null
    throughput: null
    baseline_latency_ms: null
    speedup_vs_baseline: null
  profiling:
    status: not_run
    bottleneck: null
    profile_path: null

search:
  method: null
  knobs:
    tile_size: null
    num_cores: null
    vector_width: null
    unroll_factor: null
    double_buffer: null
    parallel_axis: null
    boundary_strategy: null

failure:
  category: null
  summary: null
  routed_to: null

artifacts:
  code: null
  logs: null
  profiles: null
  reports: null

skill_updates:
  - path: null
    summary: null

notes: ""
next:
  - "Define the next concrete action."
```

## Failure Categories

Prefer consistent labels:

- `parse_error`
- `sketch_error`
- `generation_error`
- `compile_error`
- `runtime_error`
- `shape_error`
- `dtype_error`
- `broadcast_error`
- `layout_error`
- `boundary_mask_error`
- `reduction_error`
- `numerical_error`
- `performance_regression`
- `profile_error`
- `environment_error`
- `timeout`
- `unknown`

## Promotion To Skill Library

After an experiment, promote reusable findings into the relevant skill:

- elementwise behavior into `skills/elementwise/SKILL.md`
- broadcast behavior into `skills/broadcast/SKILL.md`
- reduction behavior into `skills/reduction/SKILL.md`
- layout behavior into `skills/transpose_layout/SKILL.md`
- normalization behavior into `skills/normalization/SKILL.md`
- matmul-like behavior into `skills/matmul_like/SKILL.md`
- compile/runtime fixes into `skills/ascend_debug/SKILL.md`
- profiling and tuning lessons into `skills/ascend_performance/SKILL.md`
- CUDA/Triton migration lessons into
  `skills/cuda_to_ascend_migration/SKILL.md`
- benchmark harness lessons into `skills/benchmark_evaluation/SKILL.md`
