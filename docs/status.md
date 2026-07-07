# Status

Last updated: 2026-07-03

## Current Phase

T1 non-matmul benchmark pipeline: registry, OpSpec extraction, Sketch
templates, result import, and reusable workflow.

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
- Ran `gelu_triton_v2`; `tl.erfc` was unavailable, so it fell back to PyTorch
  GELU and passed with speedup `0.9998x` and weighted score `59.99`.
- Added `gelu_triton_v3`, a hybrid diagnostic candidate that computes the bulk
  GELU path in Triton and repairs `x < -3.0` with framework GELU.
- Ran `gelu_triton_v3`; it reduced max relative error to `0.1133512` but still
  failed the official `rtol=0.01` threshold.
- Added and ran `gelu_triton_v4`; it passed correctness with
  `max_rel_diff=0.004939538426697254`, but was too slow with speedup `0.0087x`
  and weighted score `0.52`.
- Added `gelu_triton_v5`, a pure Triton piecewise kernel using an
  Abramowitz-Stegun erfc tail approximation for `x < -2.1`.
- Ran `gelu_triton_v5`; it launched as a pure Triton kernel but failed the
  official relative-error check with `max_rel_diff=7.534710`.
- Added `scripts/analyze_gelu_candidate_error.py` to locate worst GELU errors
  by input value.
- Added and ran `gelu_triton_v7`; it passed official correctness as a pure
  Triton kernel but remained slow with speedup `0.0728x` and weighted score
  `4.37`.
- Added `gelu_triton_v8`, which keeps v7's stable tanh/sigmoid formula and
  increases block size from `1024` to `4096`.
- Ran `gelu_triton_v8`; it passed official correctness and improved over v7
  with speedup `0.2856x` and weighted score `17.13`.
- Added `gelu_triton_v9`, which keeps the v8 formula and increases block size
  from `4096` to `8192`.
- Added `scripts/bootstrap_ascend_env.sh` to quickly reconstruct a fresh
  Ascend worker environment with the pinned benchmark, system-site-packages
  venv, Triton-Ascend backend dependencies, and diagnostics.
- Reconstructed the Ascend worker enough to launch `gelu_triton_v9` through
  Triton-Ascend; probe passed with `max_abs_diff=4.76837158203125e-07` and
  `max_rel_diff=3.661129085230641e-06`.
- Ran `gelu_triton_v9`; it passed official correctness and improved over v8
  with speedup `0.4869x` and weighted score `29.22`.
- Added `gelu_triton_v10`, which keeps the v9 formula and increases block size
  from `8192` to `16384`.
- Ran `gelu_triton_v10`; it passed official correctness and improved over v9
  with speedup `0.5635x` and weighted score `33.81`.
- Added `gelu_triton_v11`, which keeps the v10 formula and increases block size
  from `16384` to `32768`.
- Probed `gelu_triton_v11`; Triton-Ascend compilation failed with UB overflow:
  `2097152` bits required versus `1572864` bits available.
- Added `gelu_triton_v12`, which tests the compiler-derived UB boundary block
  size `24576`.
- Ran `gelu_triton_v12`; it passed official correctness and improved over v10
  with speedup `0.5764x` and weighted score `34.59`.
- Added `gelu_triton_v13`, which keeps per-vector block size `16384` but
  computes two sequential chunks per Triton program.
- Ran `gelu_triton_v13`; it passed official correctness and improved over v12
  with speedup `0.6059x` and weighted score `36.35`.
- Added `gelu_triton_v14`, which keeps per-vector block size `16384` and
  increases sequential chunks per program from `2` to `3`.
- Ran `gelu_triton_v14`; it passed official correctness but regressed versus
  v13 with speedup `0.5875x` and weighted score `35.25`.
- Added `gelu_triton_v15`, which combines v12's `24576` block size with v13's
  two sequential chunks per program.
- Ran `gelu_triton_v15`; it passed official correctness but regressed versus
  v13 with speedup `0.5858x` and weighted score `35.15`.
- Added `gelu_triton_v16`, which keeps v13's tiling and tests Triton's
  `tl.sigmoid` lowering instead of explicit reciprocal-exp sigmoid.
- Ran `gelu_triton_v16`; it passed official correctness but regressed versus
  v13 with speedup `0.5373x` and weighted score `32.24`.
- Added `gelu_triton_v17`, which keeps v13's tiling and tests an `exp2`-based
  sigmoid lowering.
- Probed `gelu_triton_v17`; Triton JIT rejected the Python global
  `_NEG_TWO_LOG2E` inside a `@jit` function, so the candidate fell back to
  PyTorch before testing `tl.exp2`.
- Added `gelu_triton_v18`, which keeps v17's `exp2` lowering but inlines the
  numeric constant inside the JIT expression.
- Ran `gelu_triton_v18`; it passed official correctness but regressed versus
  v13 with speedup `0.5764x` and weighted score `34.58`.
- Froze GELU-only tuning with `gelu_triton_v13` as the current best tracked
  real Triton-Ascend candidate: speedup `0.6059x`, weighted score `36.35`.
- Added the benchmark registry workflow for all 13 official
  `akg_kernels_bench_lite` cases.
- Generalized OpSpec extraction beyond GELU for the initial T1 non-matmul
  subset:
  - `t1/gelu`
  - `t1/fused_silu_and_mul`
  - `t1/sigmoid_scale_sum`
  - `t1/softmax`
- Added Sketch templates for elementwise, fused elementwise, rowwise reduction,
  rowwise softmax, and unsupported placeholders.
- Added generic submission materialization with `scripts/create_submission.py`.
- Added benchmark result import tooling with
  `scripts/import_benchmark_result.py`.
- Added parsed OpSpecs for the four supported T1 non-matmul cases under
  `benchmarks/parsed/`.
- Added `experiments/reports/gelu_tuning_summary.md`.
- Promoted GELU backend, numerical, debug, and tuning lessons into `skills/`.
- Defined the final product as a model-agnostic SketchSkill-AKG agent system
  plus benchmark evidence, with AKG Agents + Triton-Ascend as the main path and
  a pluggable strong coding/reasoning LLM backend.
- Added decision record
  `docs/decisions/0004-define-final-product-and-llm-boundary.md`.
- Added `docs/competition_alignment.md` to explicitly map competition
  requirements to the SketchSkill-AKG design, implementation status, gaps, and
  final evidence checklist.

## In Progress

- Building the repeatable Pass@4 workflow for the T1 non-matmul subset.

## Blockers

- `gelu_triton_v13` remains much slower than the PyTorch baseline, so GELU is
  not a good next single-operator tuning target without a new backend strategy.
- T2/T3 cases with symbolic shape setup are currently discovered by the
  registry but marked `parse_failed` until the extractor supports local shape
  variables and more complex input construction.
- No non-GELU Triton candidates have been generated yet for the new supported
  OpSpecs.
- The first pluggable LLM adapter is not implemented yet; defer it until the
  deterministic T1 non-matmul loop is stable.
- A formal submission-oriented technical design document still needs to be
  assembled from the architecture, workflow, roadmap, and competition alignment
  docs.

## Next Actions

1. Review the committed OpSpecs for `fused_silu_and_mul`,
   `sigmoid_scale_sum`, and `softmax`.
2. Start Pass@4 candidate generation for one new T1 non-matmul case, preferably
   `sigmoid_scale_sum` because it exercises broadcast plus rowwise reduction.
3. Run the generated candidates on the Ascend worker with the official
   benchmark.
4. Import the result JSON with `scripts/import_benchmark_result.py`.
5. Promote reusable findings into the relevant `skills/` files.
6. Keep model/provider information explicit in every generated experiment
   record.
7. Draft `docs/technical_design.md` before final submission/report work.

## Latest Handoff

Date: 2026-07-03
Agent: Codex
Branch: main
Summary:
- Closed the current GELU tuning loop. `gelu_triton_v13` remains the best
  tracked real Triton-Ascend GELU candidate; `gelu_triton_v18` passed but
  regressed.
- Added a complete benchmark registry for all 13 official AKG Bench Lite cases.
- Generalized OpSpec extraction and Sketch generation for the T1 non-matmul
  subset.
- Added generic submission creation and official benchmark result import tools.
- Added parsed OpSpecs, a GELU tuning report, tests, and skill-library updates.
- Added decision record 0004 to pin the final product and LLM/provider
  boundary.
- Updated architecture, workflow, roadmap, active tasks, AGENTS guidance, and
  experiment schema to keep future work aligned with that product goal.
- Added explicit competition alignment matrix and final evidence checklist.

Changed Files:
- `AGENTS.md`
- `README.md`
- `benchmarks/parsed/`
- `benchmarks/raw/akg_kernels_bench_lite_registry.yaml`
- `docs/architecture.md`
- `docs/competition_alignment.md`
- `docs/project_workflow.md`
- `docs/dev_guide.md`
- `docs/decisions/0004-define-final-product-and-llm-boundary.md`
- `docs/roadmap.md`
- `docs/status.md`
- `experiments/README.md`
- `experiments/reports/gelu_tuning_summary.md`
- `experiments/runs/2026-07-03-gelu-triton-v18-planned.yaml`
- `kernel_forge/benchmark/`
- `kernel_forge/experiments/`
- `scripts/create_submission.py`
- `scripts/extract_opspec.py`
- `scripts/extract_opspec_batch.py`
- `scripts/import_benchmark_result.py`
- `scripts/scan_benchmark_cases.py`
- `skills/`
- `tasks/active.md`
- `tests/`

Verification:
- `python -m py_compile kernel_forge/benchmark/extractor.py kernel_forge/benchmark/sketch.py kernel_forge/benchmark/registry.py kernel_forge/experiments/results.py scripts/scan_benchmark_cases.py scripts/extract_opspec.py scripts/extract_opspec_batch.py scripts/import_benchmark_result.py scripts/create_submission.py`
- `pytest -q tests/test_benchmark_registry_and_opspec.py tests/test_experiment_result_import.py tests/test_generic_submission.py`
- `pytest -q tests/test_gelu_triton_submission.py`
- `pytest -q`

Open Issues:
- The PDF is present as `SketchSkill_AKG_项目书基础版.pdf` but is currently
  untracked; decide whether to commit it as source material.
- Need to generate first non-GELU Pass@4 candidates from the new OpSpecs.
- Need to extend parsing for symbolic shape construction in T2/T3 cases.
- Need to implement model/provider metadata flow before comparing LLMs.
- Need to write the final technical design document for competition submission.

Next Suggested Step:
- Start Pass@4 candidate generation for `t1/sigmoid_scale_sum`.
