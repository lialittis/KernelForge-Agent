# Architecture

This document describes the active SketchSkill-AKG design. It supersedes the
earlier generic KernelForge-Agent draft.

## Positioning

SketchSkill-AKG targets AI/Agent-based operator generation for Ascend 910 NPU
benchmarks. The system is built around two constraints:

1. Low-level NPU kernels are hard for LLMs to generate directly and reliably.
2. Correctness and performance must be validated on real Ascend hardware.

The project therefore introduces a structured intermediate layer:
**NPU-aware Operator Sketch**. Agents generate and repair Sketches before
lowering them to backend implementations.

## Primary Objective

Given an official benchmark operator task, produce candidate NPU kernels that
are:

- compilable
- runnable
- correctness-verified against the reference implementation
- benchmarked on Ascend 910
- optimized through hardware feedback when possible
- recorded as reproducible experiments

## Final Product

The final product is not a single optimized kernel or a set of hand-written
benchmark submissions. It is a reusable research prototype:

**SketchSkill-AKG, a model-agnostic agent-driven operator generator for Ascend
910 NPU benchmarks.**

It should deliver two artifacts:

1. System artifact: the reusable pipeline that ingests benchmark tasks,
   extracts OpSpec, builds NPU-aware Sketches, retrieves skills, generates
   candidates, verifies correctness, repairs failures, profiles on hardware,
   searches tuning knobs, selects winners, and writes lessons back to skills.
2. Benchmark artifact: generated `ModelNew` submissions and experiment reports
   for official AKG Bench Lite cases, including correctness, Pass@1/Pass@4,
   latency, speedup, score, and failure analysis.

The research claim should be that the structured Sketch + Skill + hardware
feedback pipeline is more reliable and interpretable than one-shot low-level
kernel generation.

## Agent And LLM Strategy

SketchSkill-AKG owns the agent orchestration. The LLM is an interchangeable
backend, not the product.

Default stance:

- Agent framework: project-owned SketchSkill orchestration.
- Benchmark/harness base: AKG Agents and AKG Bench Lite.
- Main code backend: Triton-Ascend.
- Default LLM profile: a strong coding/reasoning model such as a
  Codex/GPT-5-class agent.
- Provider policy: keep the LLM backend pluggable so OpenAI/Codex-style models
  and local/open code models can be compared later.

LLMs should be used for Sketch planning, candidate code generation, repair
reasoning, and skill summarization. Deterministic project code should handle
AST parsing, schema validation, correctness checks, benchmark execution, score
import, profiling data import, and result comparison whenever practical.

## System Pipeline

```text
Benchmark OpSpec / Reference Implementation
  -> Spec Agent
  -> Sketch Agent
  -> Skill Retriever
  -> Code Agent
  -> Compile & Verify Agent
  -> Repair Agent
  -> Profiler & Search Agent
  -> Best Kernel Selector
  -> Skill Writer
```

## Modules

### Benchmark Parser

Reads official benchmark tasks, initially expected to come from
`akg_kernels_bench_lite`, and extracts raw task metadata.

Responsibilities:

- locate task files and reference implementations
- identify operator name and category
- extract input/output shape and dtype constraints
- record tolerance and baseline data when available
- write structured raw metadata for Spec Agent

### Spec Agent

Converts raw benchmark metadata into OpSpec.

OpSpec should capture:

- operator semantics
- input/output relationships
- broadcasting rules
- reduction or normalization axes
- layout transformations
- boundary conditions
- numerical precision requirements
- benchmark shape set
- reference implementation and baseline

### Sketch Agent

Generates an NPU-aware Operator Sketch from OpSpec and retrieved skills.

Sketch fields:

- `operator_category`
- `compute_pattern`
- `parallel_axes`
- `tile_plan`
- `memory_plan`
- `pipeline_plan`
- `boundary_mask`
- `accumulation_dtype`
- `backend_target`
- `performance_knobs`
- `known_risks`

The Sketch is intentionally JSON/YAML-like rather than a complete DSL. It
should be easy for agents to inspect, validate, repair, and lower.

### Skill Retriever

Retrieves relevant skills from `skills/` and external documentation indexes.

Retrieval sources:

- local operator-pattern `SKILL.md` files
- AKG Agents examples
- Triton-Ascend examples
- TileLang-Ascend examples
- Ascend C/CANN documentation notes
- CCE/TBE legacy knowledge
- historical successful and failed experiment records

### Code Agent

Lowers OpSpec plus Sketch plus retrieved skill context into candidate code.

Initial backend priority:

1. Triton-Ascend through AKG Agents
2. TileLang-Ascend for selected enhancement cases
3. Ascend C for selected performance-focused cases

Each task should produce multiple candidates when budget allows. Pass@4 is a
key correctness metric.

### Compile & Verify Agent

Runs build, execution, and correctness comparison.

Outputs:

- compile status and logs
- runtime status and logs
- numerical comparison status
- max error and mean error
- rtol/atol status
- failed shape/dtype cases
- Pass@1 and Pass@4 summaries

### Repair Agent

Routes failures by category:

- syntax/API/type errors: Code Agent
- shape/broadcast/layout errors: Sketch Agent
- tile/memory-plan errors: Sketch Agent
- boundary/mask errors: Sketch Agent and Code Agent
- accumulation/numerical errors: Sketch Agent with dtype and reduction checks
- environment errors: benchmark/harness owner

The Repair Agent must save bad-to-good trajectories when a fix succeeds.

### Profiler & Search Agent

Runs only after correctness passes. Uses real Ascend 910 profiling to search
small, interpretable performance spaces.

Search knobs:

- tile size
- parallel axis mapping
- number of cores
- vector width
- unroll factor
- double buffering
- memory access order
- copyin/compute/copyout overlap
- boundary strategy

Search strategies:

- rule-based enumeration
- UCB or adaptive search
- small evolutionary search

### Best Kernel Selector

Selects the best valid candidate based on correctness first and performance
second. It should preserve the full candidate history, not only the winner.

### Skill Writer

Writes reusable lessons into the Skill Library.

Skill write-back examples:

- common compile error and fix
- shape/dtype constraint discovered during repair
- tail-mask fix for a category
- tile size that improves latency for a shape family
- bad-to-good code and Sketch trajectory

## Backend Strategy

| Backend | Role | Use |
| --- | --- | --- |
| AKG Agents + Triton-Ascend | Main path | Fast benchmark coverage, compile/verify loop, Pass@4 |
| TileLang-Ascend | Enhancement path | Reduction, matmul-like, attention-like, complex memory access |
| Ascend C | Performance enhancement | Selected kernels where direct low-level control is valuable |
| CCE/TBE | Knowledge path | RAG, API history, compatibility, debugging patterns |
| MLIR/AscendNPU IR | Long-term path | Future structured lowering and optimization |
| CUDA/Triton | Migration source | Extract tiling, reduction, fusion, and memory patterns into Sketch |

## Operator Categories

Initial categories:

- elementwise
- broadcast
- reduction
- transpose/layout
- normalization
- matmul-like

Each category owns one `skills/<category>/SKILL.md` file and can add examples,
rules, and bad-to-good cases over time.

## Correctness Metrics

- Pass@1: first generated candidate passes correctness.
- Pass@4: at least one of four generated candidates passes correctness.
- compile success rate
- runtime success rate
- operator-category pass rate
- shape/dtype coverage
- max error, mean error, rtol/atol status
- failure category distribution

## Performance Metrics

- warmup count
- repeat count
- mean latency
- median latency
- throughput when applicable
- speedup or slowdown versus baseline
- profiling bottleneck category
- selected performance knobs

## Risk Controls

- Use Sketch and Skill constraints to reduce LLM API hallucination.
- Keep the initial backend fixed to AKG Agents + Triton-Ascend.
- Treat TileLang-Ascend and Ascend C as enhancement experiments.
- Prioritize correctness before performance.
- Keep search spaces small because NPU resources are finite.
- Classify benchmarks by operator pattern before expanding coverage.
