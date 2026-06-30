# Status

Last updated: 2026-06-30

## Current Phase

Design pivot and benchmark-reproduction preparation.

## Completed

- Initial generic KernelForge-Agent idea drafted in `idea-draft.md`.
- Multi-agent project memory scaffold added.
- Read `SketchSkill_AKG_项目书基础版.pdf`.
- Updated repository design to SketchSkill-AKG:
  - AKG Agents + Triton-Ascend main path
  - NPU-aware Operator Sketch
  - operator-pattern Skill Library
  - correctness and performance dual-loop Agents
  - Ascend 910 hardware-feedback search
  - multi-backend enhancement strategy

## In Progress

- Confirming official benchmark requirements and environment constraints.
- Preparing the repo for OpSpec, Sketch, Skill Library, and experiment-driven
  implementation.

## Blockers

- Exact official benchmark repository, commit, and task layout are not yet
  captured in `docs/benchmark_spec.md`.
- Ascend 910 environment details, CANN/AKG/Triton-Ascend versions, and local
  commands still need confirmation.
- No official sample task has been reproduced yet.

## Next Actions

1. Fill `docs/benchmark_spec.md` from official competition and benchmark
   materials.
2. Obtain or clone the official `akg_kernels_bench_lite` benchmark source.
3. Reproduce one official baseline on Ascend 910.
4. Create the first OpSpec example under `benchmarks/parsed/`.
5. Draft the first elementwise or broadcast Sketch template.
6. Record the first reproduction run under `experiments/runs/`.

## Latest Handoff

Date: 2026-06-30
Agent: Codex
Branch: main
Summary:
- Read the SketchSkill-AKG proposal PDF and updated the repo design around it.
- Replaced the generic KernelForge-Agent workflow with the SketchSkill-AKG
  architecture and implementation plan.
- Added benchmark specification skeleton and operator-pattern Skill Library
  placeholders.

Changed Files:
- `README.md`
- `AGENTS.md`
- `idea-draft.md`
- `docs/architecture.md`
- `docs/benchmark_spec.md`
- `docs/project_workflow.md`
- `docs/status.md`
- `docs/roadmap.md`
- `docs/decisions/0002-adopt-sketchskill-akg-design.md`
- `experiments/README.md`
- `tasks/active.md`
- `skills/`
- `benchmarks/`

Verification:
- Documentation-only change. No code tests were run.

Open Issues:
- The PDF is present as `SketchSkill_AKG_项目书基础版.pdf` but is currently
  untracked; decide whether to commit it as source material.
- Need official benchmark details before implementation can begin.

Next Suggested Step:
- Complete `docs/benchmark_spec.md` from the official benchmark repository and
  run one baseline task on Ascend 910.
