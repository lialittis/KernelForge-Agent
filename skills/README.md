# Skill Library

This directory stores reusable operator-generation, repair, debugging, and
optimization knowledge for SketchSkill-AKG.

The Skill Library is intended to be both human-readable and retrievable by
agents. Skills should be updated from real benchmark experiments, especially
bad-to-good trajectories.

## Layout

```text
skills/
  elementwise/SKILL.md
  broadcast/SKILL.md
  reduction/SKILL.md
  transpose_layout/SKILL.md
  normalization/SKILL.md
  matmul_like/SKILL.md
  ascend_debug/SKILL.md
  ascend_performance/SKILL.md
  cuda_to_ascend_migration/SKILL.md
  benchmark_evaluation/SKILL.md
```

## Skill Contract

Each `SKILL.md` should cover:

- applicable and non-applicable operator cases
- input/output shape and dtype constraints
- recommended NPU-aware Sketch fields
- tile, core mapping, memory, and pipeline strategy
- backend generation notes for Triton-Ascend, TileLang-Ascend, and Ascend C
- common compile/runtime/correctness failures
- profiling interpretation and performance bottlenecks
- bad-to-good examples with experiment links

Use `skills/TEMPLATE.md` when adding a new skill.

