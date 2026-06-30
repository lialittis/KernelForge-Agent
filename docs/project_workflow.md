# Project Workflow

This document captures the shared research and implementation workflow for
KernelForge-Agent. It should evolve as experiments reveal better methods.

## Goal

Build a reproducible AI agent system for automatic operator generation and
optimization. The agent should close the loop from benchmark task description to
candidate code, validation, performance measurement, feedback repair, and
knowledge persistence.

## Core Loop

```text
benchmark task
-> parse operator semantics, inputs, outputs, shapes, and dtypes
-> retrieve relevant prompts, examples, rules, and prior fixes
-> generate one or more candidate implementations
-> compile and run candidates
-> verify correctness
-> benchmark performance
-> repair or optimize based on logs and metrics
-> persist useful findings into the Skill Library
```

## Development Stages

### Stage 1: Benchmark Specification

Create `docs/benchmark_spec.md` once official requirements and examples are
available. Capture:

- task input format
- required output format
- supported implementation language/runtime
- correctness tolerance
- performance metric
- submission constraints
- official sample operators

### Stage 2: Manual Baseline

Implement one known-good operator manually. The baseline should exercise the
same compile, run, validation, and profiling flow that generated candidates will
use later.

### Stage 3: Harness Automation

Build the benchmark harness before building complex agent behavior. Minimum
components:

- task parser
- candidate workspace manager
- compiler/build runner
- execution runner
- correctness validator
- performance profiler
- structured result writer

### Stage 4: Single-Shot Generation

Generate one candidate implementation from a benchmark task and run it through
the harness. Save the prompt, generated code, logs, and result metadata.

### Stage 5: Feedback Repair

Feed compiler errors, runtime errors, numerical mismatch reports, and relevant
source snippets back into a repair prompt. Track repair iterations separately
from the original generation.

### Stage 6: Multi-Candidate Generation

Generate multiple candidates for each task and report Pass@N. Keep each
candidate isolated so compile logs, runtime logs, and metrics do not overwrite
each other.

### Stage 7: Retrieval-Augmented Generation

Build a Skill Library from prompts, rules, examples, fixes, and tuning notes.
Retrieve only context relevant to the current task and record which skills were
used in each experiment.

### Stage 8: Performance Optimization

Only optimize candidates that already pass correctness checks. Compare optimized
versions against the manual baseline, official baseline, or previous best
candidate.

## Research Loop

Use short experiment cycles:

```text
choose 3-5 benchmark tasks
-> run current agent or harness
-> classify failures
-> improve one component
-> rerun the same task set
-> compare metrics
-> document results
-> promote reusable findings into the Skill Library
```

Avoid changing prompts, retrieval, repair logic, and benchmark scripts all in
the same experiment cycle. Mixed changes make results hard to explain.

## Metrics

Track separate metrics instead of only the final score:

- compile success rate
- correctness pass rate
- Pass@1, Pass@3, Pass@5
- average repair iterations
- performance compared with baseline
- generation time and cost
- failure categories
- retrieval hit rate once retrieval exists

## Skill Library Shape

Recommended initial structure:

```text
skills/
  prompts/
    generation.md
    repair.md
    optimization.md
  rules/
    dtype_rules.md
    shape_rules.md
    api_usage.md
    memory_layout.md
  examples/
    successful_ops/
    failed_ops/
  fixes/
    compile_errors.md
    runtime_errors.md
    numerical_errors.md
  scripts/
    validate.py
    benchmark.py
```

The Skill Library is not only documentation. It should become retrievable
context for future generations and repairs.

## Recommended Implementation Layout

This layout is a target, not a requirement for the current empty prototype:

```text
KernelForge-Agent/
  docs/
    benchmark_spec.md
    architecture.md
    experiment_plan.md
  kernel_forge/
    agent/
      planner.py
      generator.py
      repairer.py
      optimizer.py
    benchmark/
      parser.py
      runner.py
      validator.py
      profiler.py
    retrieval/
      indexer.py
      retriever.py
    skill_library/
      loader.py
    utils/
  skills/
  benchmarks/
    raw/
    parsed/
  experiments/
    runs/
  outputs/
    candidates/
    logs/
  tests/
```

## Documentation Rules

- Update `docs/status.md` when the active phase, blockers, or next actions
  change.
- Add a decision record for non-trivial architecture or workflow choices.
- Add or update experiment records for any generated or benchmarked candidate.
- Keep generated artifacts out of Git unless they are small, curated examples.

