# 0004: Define Final Product And LLM Boundary

Date: 2026-07-03

## Status

Accepted

## Context

The project has already produced hand-written GELU candidates and benchmark
results. That work is useful, but it risks pulling the project toward
single-operator tuning instead of the intended research product.

We need a stable definition of the final product and a clear answer to which
agent or LLM the generator is based on.

## Decision

The final product is **SketchSkill-AKG**, a model-agnostic agent-driven operator
generation and optimization system for Ascend 910 NPU benchmarks.

The system is based on:

- project-owned SketchSkill orchestration for OpSpec, Sketch, retrieval,
  generation, verification, repair, profiling/search, selection, and skill
  write-back
- AKG Agents and AKG Bench Lite as the benchmark and harness base
- Triton-Ascend as the initial code-generation backend
- a strong coding/reasoning LLM, such as a Codex/GPT-5-class agent, as the
  default Sketch/Code/Repair/Skill Writer backend

The LLM backend must remain replaceable. OpenAI/Codex-style models and
local/open code models should be comparable under the same pipeline using
Pass@1, Pass@4, correctness rate, repair count, latency, speedup, and score.

LLMs should not be the source of truth for deterministic tasks such as benchmark
scanning, AST parsing, correctness comparison, score import, profiling data
import, or best-candidate selection.

## Consequences

Positive:

- The project contribution is the reusable agent system, not only individual
  kernels.
- Model comparisons become possible because OpSpec, Sketch, skills, harness,
  and scoring are stable.
- Deterministic infrastructure reduces hallucination risk and makes benchmark
  results auditable.

Negative:

- More infrastructure is needed before generation looks fully automated.
- Provider abstraction adds design work before local/open model comparisons can
  be meaningful.
- Prompt and model changes must be recorded carefully in experiment metadata.

## Follow-Up

- Add model/provider fields to generated experiment records.
- Build the first pluggable LLM adapter only after the deterministic T1
  non-matmul pipeline is stable.
- Keep GELU as a case study and move next to Pass@4 generation for
  `t1/sigmoid_scale_sum`.
