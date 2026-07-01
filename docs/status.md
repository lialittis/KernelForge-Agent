# Status

Last updated: 2026-07-01

## Current Phase

Triton-Ascend backend installation.

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
- Ran `gelu_triton_v1` through the official benchmark on the Ascend worker:
  `t1/gelu` passed correctness with speedup `1.0176x` and weighted score
  `60.18`.
- Ran the backend probe for `gelu_triton_v1`; it used PyTorch fallback after
  Triton raised `RuntimeError: 0 active drivers ([]). There should only be
  one.`.
- Ran `scripts/diagnose_triton_ascend.py` on the Ascend worker:
  - `torch_npu` is available with one `Ascend910B2C` device.
  - `triton` imports as version `3.6.0`.
  - `triton-ascend` is not installed.
  - `triton_ascend` import fails with `ModuleNotFoundError`.
- After installing Triton-Ascend in an isolated venv, `gelu_triton_v1` launched
  through Triton on NPU but failed official correctness due relative error:
  `max_abs_diff=4.737377e-04`, `max_rel_diff=4.803681e+00`.
- Added `gelu_triton_v2`, which uses an erfc-form GELU expression to avoid
  negative-tail cancellation when `tl.erfc` is available.

## In Progress

- Testing `gelu_triton_v2` against the official benchmark.

## Blockers

- `gelu_triton_v1` launches as a real Triton-Ascend kernel but fails the
  official benchmark's separate relative-error threshold.

## Next Actions

1. Pull the latest commit on the Ascend worker.
2. Generate `gelu_triton_v2` with
   `bash scripts/create_gelu_triton_v2_submission.sh`.
3. Probe v2 with `scripts/probe_gelu_triton_backend.py --candidate ...`.
4. Run v2 through the official benchmark if the probe reports Triton backend.

## Latest Handoff

Date: 2026-07-01
Agent: Codex
Branch: main
Summary:
- Added a tracked `gelu_triton_v1` candidate source.
- Added a submission generator for the official benchmark layout.
- Documented the Ascend worker backend probe and benchmark command.
- Added tests that validate the candidate source and generated package shape
  without requiring NPU hardware.
- Recorded the first `gelu_triton_v1` official benchmark result:
  correctness pass, speedup `1.0176x`, weighted score `60.18`.
- Recorded the backend probe result: `torch_fallback_after_error` due to
  Triton runtime reporting zero active drivers.
- Added `scripts/diagnose_triton_ascend.py` for environment diagnostics.
- Recorded the diagnostic result: `triton-ascend` is missing while `torch_npu`
  and the Ascend device are available.
- Recorded the first real Triton-Ascend launch for `gelu_triton_v1` and its
  official correctness failure.
- Added `gelu_triton_v2` and submission generator.

Changed Files:
- `docs/dev_guide.md`
- `docs/status.md`
- `experiments/runs/2026-07-01-gelu-triton-v1-planned.yaml`
- `kernel_forge/candidates/`
- `scripts/diagnose_triton_ascend.py`
- `scripts/probe_gelu_triton_backend.py`
- `scripts/create_gelu_triton_submission.sh`
- `scripts/create_gelu_triton_v2_submission.sh`
- `tasks/active.md`
- `tests/test_gelu_triton_submission.py`

Verification:
- `python -m py_compile kernel_forge/candidates/gelu_triton_v1.py tests/test_gelu_triton_submission.py`
- `python -m py_compile scripts/probe_gelu_triton_backend.py`
- `python -m py_compile scripts/diagnose_triton_ascend.py`
- `python -m py_compile kernel_forge/candidates/gelu_triton_v2.py`
- `bash scripts/create_gelu_triton_submission.sh`
- `pytest -q tests/test_gelu_triton_submission.py`

Open Issues:
- The PDF is present as `SketchSkill_AKG_项目书基础版.pdf` but is currently
  untracked; decide whether to commit it as source material.
- Need to confirm whether Triton-Ascend exposes `tl.erfc` and whether v2 passes
  the official relative-error check.

Next Suggested Step:
- Run the `gelu_triton_v2` probe and official benchmark on the Ascend worker.
