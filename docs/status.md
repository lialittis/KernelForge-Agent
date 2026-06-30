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
- Confirmed the official benchmark source:
  `https://atomgit.com/mindspore/akg.git`, branch `br_agents`, path
  `akg_agents/benchmark/akg_kernels_bench_lite`, inspected at commit
  `bea77cb38db5713056a7e06e5e8a0cbe9d26954b`.
- Added the AKG repository as a pinned Git submodule at `third_party/akg` and
  configured local sparse checkout for `akg_kernels_bench_lite`.
- Added `scripts/setup_benchmark_submodule.sh` for new machines.
- Added `docs/dev_guide.md` to document the local-development plus Ascend-worker
  experiment loop.

## In Progress

- Confirming Ascend 910 environment constraints.
- Preparing the repo for OpSpec, Sketch, Skill Library, and experiment-driven
  implementation.

## Blockers

- Ascend 910 environment details, CANN/AKG/Triton-Ascend versions, and local
  commands still need confirmation.
- No official sample task has been reproduced yet.

## Next Actions

1. Reproduce `t1/gelu.py` from the official benchmark on Ascend 910.
2. Create a minimal single-case submission with `ModelNew`.
3. Run `tools/run_bench.py` against that submission.
4. Record the run under `experiments/runs/`.
5. Create `benchmarks/parsed/t1_gelu.yaml` as the first OpSpec example.
6. Draft the first elementwise Sketch template for GELU.

## Latest Handoff

Date: 2026-06-30
Agent: Codex
Branch: main
Summary:
- Added a dedicated development guide for the local-development plus
  Ascend-worker experiment loop.
- Linked the guide from `README.md` and `AGENTS.md`.
- Added a tracked script to recreate the ignored manual GELU submission on any
  worker machine.

Changed Files:
- `AGENTS.md`
- `README.md`
- `docs/dev_guide.md`
- `docs/status.md`
- `scripts/create_manual_gelu_submission.sh`

Verification:
- `bash -n scripts/create_manual_gelu_submission.sh` passes.
- `bash scripts/create_manual_gelu_submission.sh` creates the manual GELU
  submission.
- Official `validate_submission.py` passes for the generated submission.
- Documentation-only repo change. No project code tests were run.

Open Issues:
- The PDF is present as `SketchSkill_AKG_项目书基础版.pdf` but is currently
  untracked; decide whether to commit it as source material.
- Need Ascend 910 environment details and first local baseline run.

Next Suggested Step:
- Run `t1/gelu.py` through the official runner with a minimal `ModelNew`
  submission on Ascend 910.
