# Experiments

This directory stores experiment metadata and summaries. It should make every
generated candidate reproducible and comparable across machines and agents.

## Rules

- Store structured experiment records under `experiments/runs/`.
- Use one file per experiment run.
- Commit metadata and concise summaries.
- Do not commit large build directories, raw logs, binaries, caches, or full
  generated output trees by default.
- Record artifact paths even when artifacts are stored outside Git.
- Record the exact prompt version and retrieved skills used by the run.

## Naming

Use a stable, sortable name:

```text
YYYY-MM-DD-short-task-or-method.yaml
```

Examples:

```text
2026-06-30-baseline-single-op.yaml
2026-07-03-pass-n-generation-v1.yaml
2026-07-05-repair-loop-ablation.yaml
```

## Suggested Schema

```yaml
id: 2026-06-30-baseline-single-op
date: 2026-06-30
task: unknown
agent: codex
machine: workstation-a
branch: task/example
commit: null
model: null
prompt_version: null
retrieved_skills: []
status: planned

candidate:
  path: null
  language: null
  repair_iteration: 0

results:
  compile:
    status: not_run
    log: null
  correctness:
    status: not_run
    tolerance: null
    log: null
  performance:
    status: not_run
    metric: null
    value: null
    baseline_value: null

failure:
  category: null
  summary: null

artifacts:
  code: null
  logs: null
  reports: null

notes: ""
next:
  - "Define the next concrete action."
```

## Failure Categories

Prefer consistent labels:

- `parse_error`
- `generation_error`
- `compile_error`
- `runtime_error`
- `numerical_error`
- `performance_regression`
- `timeout`
- `environment_error`
- `unknown`

## Promotion To Skill Library

After an experiment, promote reusable findings into the Skill Library:

- prompt changes into `skills/prompts/`
- dtype and shape constraints into `skills/rules/`
- recurring compiler/runtime fixes into `skills/fixes/`
- useful successful or failed examples into `skills/examples/`

