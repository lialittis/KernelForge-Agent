# Status

Last updated: 2026-07-01

## Current Phase

First custom GELU candidate execution.

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
- Ran the first official benchmark smoke test on the Ascend worker:
  `t1/gelu` passed correctness with speedup `0.9997x` and weighted score
  `59.98`.
- Added the first parsed OpSpec and Sketch for `t1/gelu`.
- Added the first experiment record for the manual GELU baseline.
- Captured Ascend worker CANN version: `8.5.1`.
- Implemented a GELU-only automated OpSpec extractor and CLI.
- Added the first custom GELU candidate source:
  `kernel_forge/candidates/gelu_triton_v1.py`.
- Added `scripts/create_gelu_triton_submission.sh` to materialize the candidate
  into the official benchmark submission layout.
- Added tests for the candidate source and generated submission layout.

## In Progress

- Preparing to run `gelu_triton_v1` on the Ascend worker.
- Verifying whether the candidate actually launches through Triton-Ascend or
  falls back to PyTorch GELU.

## Blockers

- Triton-Ascend availability is still unverified on the Ascend worker. The
  candidate has a correctness-preserving PyTorch fallback, but a fallback run is
  not a valid custom-kernel performance result.

## Next Actions

1. Pull the latest commit on the Ascend worker.
2. Run `bash scripts/create_gelu_triton_submission.sh`.
3. Run the optional smoke check in `docs/dev_guide.md` and record
   `last_backend`.
4. Run `gelu_triton_v1` through `tools/run_bench.py`.
5. Compare the custom candidate against the manual PyTorch baseline.
6. Record the custom candidate experiment.

## Latest Handoff

Date: 2026-07-01
Agent: Codex
Branch: main
Summary:
- Added a tracked `gelu_triton_v1` candidate source.
- Added a submission generator for the official benchmark layout.
- Documented the Ascend worker smoke check and benchmark command.
- Added tests that validate the candidate source and generated package shape
  without requiring NPU hardware.

Changed Files:
- `docs/dev_guide.md`
- `docs/status.md`
- `experiments/runs/2026-07-01-gelu-triton-v1-planned.yaml`
- `kernel_forge/candidates/`
- `scripts/create_gelu_triton_submission.sh`
- `tasks/active.md`
- `tests/test_gelu_triton_submission.py`

Verification:
- `python -m py_compile kernel_forge/candidates/gelu_triton_v1.py tests/test_gelu_triton_submission.py`
- `bash scripts/create_gelu_triton_submission.sh`
- `pytest -q tests/test_gelu_triton_submission.py`

Open Issues:
- The PDF is present as `SketchSkill_AKG_项目书基础版.pdf` but is currently
  untracked; decide whether to commit it as source material.
- Need `gelu_triton_v1` results from the Ascend worker.
- Need to confirm whether `tl.erf` lowers successfully on the worker's
  Triton-Ascend stack.

Next Suggested Step:
- Run `gelu_triton_v1` on the Ascend worker and paste the result JSON summary
  back into this repo.
