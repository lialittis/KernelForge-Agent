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
- Ran the first official benchmark smoke test on the Ascend worker:
  `t1/gelu` passed correctness with speedup `0.9997x` and weighted score
  `59.98`.
- Added the first parsed OpSpec and Sketch for `t1/gelu`.
- Added the first experiment record for the manual GELU baseline.
- Captured Ascend worker CANN version: `8.5.1`.

## In Progress

- Confirming Ascend 910 environment constraints.
- Preparing the repo for OpSpec, Sketch, Skill Library, and experiment-driven
  implementation.

## Blockers

- `triton-ascend` is not installed on the Ascend worker yet; this blocks custom
  Triton-Ascend candidate experiments but not the manual PyTorch baseline.

## Next Actions

1. Commit the first GELU experiment metadata and OpSpec.
2. Implement automated OpSpec extraction for `t1/gelu.py`.
3. Install/verify `triton-ascend` on the Ascend worker.
4. Generate a first custom Triton-Ascend GELU candidate.
5. Compare the custom candidate against the manual PyTorch baseline.

## Latest Handoff

Date: 2026-06-30
Agent: Codex
Branch: main
Summary:
- Recorded the first successful Ascend-worker benchmark run for `t1/gelu`.
- Added `benchmarks/parsed/t1_gelu.yaml` with OpSpec and initial Sketch fields.
- Added `experiments/runs/2026-06-30-gelu-manual-baseline.yaml`.
- Updated the experiment record with CANN `8.5.1`.

Changed Files:
- `benchmarks/parsed/t1_gelu.yaml`
- `docs/status.md`
- `experiments/runs/2026-06-30-gelu-manual-baseline.yaml`

Verification:
- Ascend worker ran official `tools/run_bench.py` for `t1/gelu`.
- Correctness passed with `max_abs_diff=0.0` and `max_rel_diff=0.0`.
- Median baseline latency: `0.0438129500253126 ms`.
- Median solution latency: `0.04382457991596311 ms`.
- Speedup: `0.9997x`; weighted score: `59.98`.

Open Issues:
- The PDF is present as `SketchSkill_AKG_项目书基础版.pdf` but is currently
  untracked; decide whether to commit it as source material.
- Need `triton-ascend` installed or otherwise confirmed for custom kernel work.

Next Suggested Step:
- Commit the GELU experiment metadata, then implement a small parser that emits
  `benchmarks/parsed/t1_gelu.yaml` automatically from the official case file.
