# Historical Draft

This file previously described the generic KernelForge-Agent idea:

> KernelForge-Agent: feedback-loop-based AI operator generation and
> optimization.

The active project design has been updated to **SketchSkill-AKG**, based on
`SketchSkill_AKG_项目书基础版.pdf`.

## Active Direction

**SketchSkill-AKG: skill-driven operator generation and hardware-feedback
optimization for Ascend NPU.**

The project now plans to build on AKG Kernel Agent and combine:

- NPU-aware Operator Sketch
- operator-pattern Skill Library
- correctness and performance dual-loop Agents
- hardware-feedback search
- AKG Agents + Triton-Ascend as the main path
- TileLang-Ascend and Ascend C as selected enhancement paths

The target hardware is Ascend 910 NPU. The target benchmark is the
community/competition-provided AKG operator benchmark, referred to in the
proposal as `akg_kernels_bench_lite`.

See:

- `README.md`
- `docs/architecture.md`
- `docs/project_workflow.md`
- `docs/benchmark_spec.md`
- `skills/`
