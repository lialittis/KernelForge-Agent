# Status

Last updated: 2026-06-30

## Current Phase

GELU OpSpec automation and first custom-kernel preparation.

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

## In Progress

- Preparing the first custom Triton-Ascend GELU candidate.
- Verifying Triton-Ascend availability on the Ascend worker.

## Blockers

- `triton-ascend` is not installed on the Ascend worker yet; this blocks custom
  Triton-Ascend candidate experiments but not the manual PyTorch baseline.

## Next Actions

1. Install/verify `triton-ascend` on the Ascend worker.
2. Generate a first custom Triton-Ascend GELU candidate.
3. Run the custom candidate through `tools/run_bench.py`.
4. Compare the custom candidate against the manual PyTorch baseline.
5. Record the custom candidate experiment.

## Latest Handoff

Date: 2026-06-30
Agent: Codex
Branch: main
Summary:
- Implemented the first automated OpSpec extraction path for `t1/gelu.py`.
- Added a CLI for extracting GELU OpSpecs from the official case file.
- Added focused tests for extraction, CLI YAML output, Sketch fields, and YAML
  round trips.

Changed Files:
- `docs/dev_guide.md`
- `docs/status.md`
- `kernel_forge/benchmark/`
- `scripts/extract_opspec.py`
- `tasks/active.md`
- `tests/test_gelu_opspec_extractor.py`

Verification:
- `python -m py_compile kernel_forge/benchmark/*.py scripts/extract_opspec.py tests/test_gelu_opspec_extractor.py`
- `python scripts/extract_opspec.py --case .../t1/gelu.py --experiment ... --output /tmp/t1_gelu.generated.yaml`
- `pytest -q tests/test_gelu_opspec_extractor.py`

Open Issues:
- The PDF is present as `SketchSkill_AKG_项目书基础版.pdf` but is currently
  untracked; decide whether to commit it as source material.
- Need `triton-ascend` installed or otherwise confirmed for custom kernel work.

Next Suggested Step:
- Install/verify `triton-ascend`, then create the first custom Triton-Ascend
  GELU `ModelNew` candidate.
