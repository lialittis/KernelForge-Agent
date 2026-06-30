# 0002: Adopt SketchSkill-AKG Design

Date: 2026-06-30

## Status

Accepted

## Context

The repository originally contained a generic KernelForge-Agent concept for
feedback-driven AI operator generation. A more developed proposal,
`SketchSkill_AKG_项目书基础版.pdf`, defines a stronger design:

- target Ascend 910 NPU
- build on AKG Kernel Agent
- use AKG Agents + Triton-Ascend as the main implementation path
- introduce NPU-aware Operator Sketch
- organize reusable knowledge as operator-pattern skills
- use correctness and performance dual-loop Agents
- use real hardware feedback and lightweight search for optimization

## Decision

Adopt SketchSkill-AKG as the active project design and update repository
guidance accordingly.

The project will prioritize:

1. official benchmark and environment reproduction
2. OpSpec extraction
3. NPU-aware Sketch generation
4. Triton-Ascend candidate generation through AKG Agents
5. compile/run/correctness verification and Pass@1/Pass@4
6. repair routing by failure type
7. Skill Library retrieval and write-back
8. Ascend 910 profiling and lightweight search
9. selected TileLang-Ascend and Ascend C enhancement cases

## Consequences

Positive:

- The design is more concrete and better aligned with the competition.
- The Sketch layer reduces direct low-level kernel-generation risk.
- Skill files become executable/retrievable project memory, not just notes.
- Experiments can measure correctness and performance separately.

Negative:

- The system has more moving parts than the generic draft.
- Ascend 910 access becomes essential for meaningful validation.
- The main path depends on AKG Agents and Triton-Ascend availability.

## Follow-Up

- Complete `docs/benchmark_spec.md` from official materials.
- Decide whether to commit the proposal PDF as source material.
- Reproduce one benchmark baseline before building agent automation.

