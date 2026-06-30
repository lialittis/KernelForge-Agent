# Project Workflow

This document captures the shared research and implementation workflow for
SketchSkill-AKG. It should evolve as experiments reveal better methods.

## Goal

Build a reproducible AI agent system for automatic Ascend 910 NPU operator
generation and optimization. The system should close the loop from benchmark
task description to OpSpec, NPU-aware Sketch, backend candidate code,
correctness verification, hardware profiling, search, and Skill Library
write-back.

## Main Path

Use **AKG Agents + Triton-Ascend** as the initial path because it is expected to
match the benchmark repository and can provide the fastest end-to-end
prototype.

Enhancement paths are intentionally narrower:

- TileLang-Ascend for selected reduction, matmul-like, attention-like, or
  complex memory-access operators.
- Ascend C for a few representative performance-focused kernels.
- CCE/TBE and CANN documentation as knowledge sources for RAG and skills.
- MLIR/AscendNPU IR as a long-term alignment target for structured lowering.
- CUDA/Triton implementations as migration knowledge sources, not as direct
  line-by-line translation targets.

## Core Loop

```text
akg_kernels_bench_lite task or official sample
-> Benchmark Parser extracts raw task information
-> Spec Agent builds structured OpSpec
-> Sketch Agent builds NPU-aware Operator Sketch
-> Skill Retriever fetches relevant operator-pattern skills
-> Code Agent generates Triton-Ascend candidate kernels
-> Compile & Verify Agent compiles, runs, compares, and records Pass@N
-> Repair Agent routes failures to Code Agent or Sketch Agent
-> Profiler & Search Agent optimizes correct kernels on Ascend 910
-> Best Kernel Selector chooses correct and fastest candidate
-> Skill Writer persists bad-to-good trajectory and reusable rules
```

## Development Stages

### Stage 1: Benchmark And Environment Reproduction

Create and maintain `docs/benchmark_spec.md` from official materials. Capture:

- benchmark repository and task layout
- `akg_kernels_bench_lite` task format
- expected candidate file layout
- supported backend/runtime
- build, run, and validation commands
- correctness tolerances
- performance metrics
- sample operators and baselines
- submission packaging constraints

Exit condition: one official baseline or sample task can be run on the target
environment.

### Stage 2: OpSpec Parser

Implement the Benchmark Parser and Spec Agent. The output should be structured
and stable enough for later agents.

Minimum OpSpec fields:

- operator name and category
- input/output shapes and dtypes
- broadcasting and layout behavior
- reduction axes or normalization axes when relevant
- reference implementation path or callable
- tolerance rules
- baseline performance data if available
- test shape set

### Stage 3: NPU-Aware Sketch Templates

Define JSON/YAML-style Sketch templates per operator category. Sketch should be
easy for an LLM to generate, validate, repair, and lower to backend code.

Minimum Sketch fields:

- compute pattern
- parallel axes
- tile plan
- memory plan for GM/UB movement
- pipeline plan, including copyin-compute-copyout where applicable
- boundary and mask strategy
- accumulation dtype
- backend target
- performance knobs

### Stage 4: Triton-Ascend Candidate Generation

Use AKG Agents plus retrieved skills to generate multiple Triton-Ascend
candidates from OpSpec and Sketch. Save prompt, Sketch, retrieved skills,
generated code, and metadata for each candidate.

### Stage 5: Correctness Loop

Build compile/run/correctness verification before deep performance work. Track:

- compile errors
- runtime errors
- shape/dtype errors
- boundary/mask errors
- numerical mismatch
- Pass@1 and Pass@4
- shape/dtype coverage

Repair routing:

- syntax/API/type errors go to Code Agent
- shape/broadcast/layout/tile-design errors go to Sketch Agent
- numerical errors trigger dtype, boundary, reduction, and mask checks
- environment errors are marked separately and not treated as model failures

### Stage 6: Skill Library And RAG

Store skills by operator pattern. Each skill should include applicability,
shape/dtype constraints, Sketch templates, backend generation notes, common
failures, profiling interpretation, and bad-to-good examples.

Skill updates should be driven by experiments, not only by speculation.

### Stage 7: Hardware Profiling And Search

Only optimize candidates that already pass correctness checks. Use real Ascend
910 profiling and keep the search space small and interpretable.

Initial knobs:

- tile size
- num cores or parallel axis mapping
- vector width
- unroll factor
- double buffering
- boundary strategy
- memory access pattern
- copyin/compute/copyout pipeline

Search methods can include rule-based enumeration, UCB/adaptive search, and
small evolutionary search. Each search round must produce a structured
experiment record.

### Stage 8: Multi-Backend Enhancement

After the Triton-Ascend path works, select representative operators for:

- TileLang-Ascend experiments
- Ascend C experiments
- CUDA/Triton-to-NPU Sketch migration experiments

Do not let enhancement paths block the main benchmark pipeline.

## Research Loop

Use short experiment cycles:

```text
choose 3-5 benchmark tasks from one operator category
-> run current OpSpec/Sketch/generation/verification flow
-> classify failures
-> improve one component
-> rerun the same task set
-> compare metrics
-> document results
-> promote reusable lessons into skills/
```

Avoid changing prompts, Sketch templates, retrieval logic, repair routing, and
benchmark scripts in the same experiment cycle.

## Metrics

Track separate metrics instead of only the final score:

- compile success rate
- correctness pass rate
- Pass@1 and Pass@4
- operator-category pass rate
- shape/dtype coverage
- max error, mean error, rtol/atol status
- average repair iterations
- latency and throughput
- speedup or slowdown versus baseline
- profiling bottleneck category
- generation time and cost
- retrieved skill set
- failure categories

## Recommended Implementation Layout

```text
KernelForge-Agent/
  docs/
    architecture.md
    benchmark_spec.md
    project_workflow.md
  kernel_forge/
    benchmark/
      parser.py
      opspec.py
      runner.py
      validator.py
      profiler.py
    sketch/
      schema.py
      templates.py
      validator.py
    agent/
      spec_agent.py
      sketch_agent.py
      code_agent.py
      verify_agent.py
      repair_agent.py
      profile_search_agent.py
      skill_writer.py
    retrieval/
      indexer.py
      retriever.py
    backends/
      triton_ascend.py
      tilelang_ascend.py
      ascend_c.py
    skill_library/
      loader.py
      writer.py
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
    profiles/
  tests/
```

## Documentation Rules

- Update `docs/status.md` when the active phase, blockers, or next actions
  change.
- Add a decision record for non-trivial architecture or workflow choices.
- Add or update experiment records for any generated or benchmarked candidate.
- Keep generated artifacts out of Git unless they are small, curated examples.
- Update the relevant `skills/*/SKILL.md` file when a repair or optimization
  lesson becomes reusable.

