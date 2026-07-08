# Status

Last updated: 2026-07-09

## Current Phase

Initial-round Step 3 local submission materials are ready: Chinese project
book, technical design, GitLink PR package, and email-ready report materials.
External PR and email submission are manual user actions.

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
  `akg_agents/benchmark/akg_kernels_bench_lite`.
- Added the AKG repository as a pinned Git submodule at `third_party/akg` and
  configured local sparse checkout for `akg_kernels_bench_lite`.
- Updated the AKG submodule pin from
  `bea77cb38db5713056a7e06e5e8a0cbe9d26954b` to latest `br_agents` commit
  `47aa428fcdc8c68f78d331dc578bc6c74fb9d91d` after being told the previous
  branch revision might have runner problems.
- Confirmed the updated benchmark case files and generated case registry are
  unchanged; only `RUNNER.md` and `tools/run_bench.py` changed under
  `akg_kernels_bench_lite`.
- Recorded the runner behavior change: standalone `run_bench.py` now uses
  independent seeded reference/solution inputs, three correctness trials, and
  NaN/Inf rejection before performance measurement.
- Added decision record
  `docs/decisions/0007-update-akg-br-agents-pin.md`.
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
- Added the first non-GELU Pass@4 candidate batch for `t1/sigmoid_scale_sum`:
  one torch reference candidate and three Triton-Ascend row-reduction variants.
- Added `scripts/create_sigmoid_scale_sum_pass4_submissions.sh`,
  `scripts/probe_sigmoid_scale_sum_backend.py`, and
  `scripts/summarize_passn.py`.
- Added completed experiment metadata at
  `experiments/runs/2026-07-07-sigmoid-scale-sum-pass4.yaml`.
- Synced the latest `main` branch to the Ascend worker at commit
  `234b4bae5d5107773efc6bb136d459014ec565e7`.
- Confirmed the Ascend worker environment:
  `Ascend910B2C`, CANN `8.5.1`, Python `3.11.14`, torch `2.9.0+cpu`,
  `torch_npu` `2.9.0rc1`, Triton-Ascend backend target `npu`.
- Probed all four `t1/sigmoid_scale_sum` Pass@4 candidates; v2, v3, and v4
  launched through real Triton-Ascend backend paths.
- Ran the official AKG Bench Lite benchmark for `t1/sigmoid_scale_sum`:
  - `sigmoid_scale_sum_v1`: pass, speedup `1.0006x`, score `60.01`
  - `sigmoid_scale_sum_v2`: pass, speedup `2.0279x`, score `70.28`
  - `sigmoid_scale_sum_v3`: pass, speedup `1.9367x`, score `69.37`
  - `sigmoid_scale_sum_v4`: pass, speedup `1.5785x`, score `65.79`
- Added Pass@4 report at
  `experiments/reports/2026-07-07-sigmoid-scale-sum-pass4.yaml`.
- Promoted the `sigmoid_scale_sum` row-reduction lesson into the Skill Library.
- Added the first fused-elementwise Pass@4 candidate batch for
  `t1/fused_silu_and_mul`: one reference candidate and three Triton-Ascend
  flattened-output tiling variants.
- Added `scripts/create_fused_silu_and_mul_pass4_submissions.sh`,
  `scripts/probe_fused_silu_and_mul_backend.py`, and experiment metadata for
  the fused Pass@4 cycle.
- Repaired early `fused_silu_and_mul` UB-overflow variants by reducing tile
  pressure from `16384` and `8192 x 2` to `4096` and `4096 x 2`.
- Ran the official AKG Bench Lite benchmark for `t1/fused_silu_and_mul`:
  - `fused_silu_and_mul_v1`: pass, speedup `1.0027x`, score `60.03`
  - `fused_silu_and_mul_v2`: pass, speedup `0.0033x`, score `0.2`
  - `fused_silu_and_mul_v3`: pass, speedup `0.0033x`, score `0.2`
  - `fused_silu_and_mul_v4`: pass, speedup `0.0033x`, score `0.2`
- Added completed experiment metadata at
  `experiments/runs/2026-07-08-fused-silu-and-mul-pass4.yaml` and Pass@4
  report at
  `experiments/reports/2026-07-08-fused-silu-and-mul-pass4.yaml`.
- Promoted the fused SwiGLU split-indexing, UB-pressure, and negative
  performance lessons into the Skill Library.
- Added the first pluggable provider scaffold with deterministic replay
  generation for `t1/sigmoid_scale_sum` Pass@4.
- Added prompt templates for Code, Repair, and Skill Writer agents.
- Added `scripts/generate_candidate.py` to generate provider-traceable
  candidates, prompts, submissions, and experiment metadata.
- Added decision record
  `docs/decisions/0005-replay-first-provider-adapter.md`.
- Ran replay-generated `t1/sigmoid_scale_sum` Pass@4 submissions on Ascend:
  - `sigmoid_scale_sum_replay_v1`: pass, speedup `1.0034x`, score `60.03`
  - `sigmoid_scale_sum_replay_v2`: pass, speedup `1.998x`, score `69.98`
  - `sigmoid_scale_sum_replay_v3`: pass, speedup `1.8731x`, score `68.73`
  - `sigmoid_scale_sum_replay_v4`: pass, speedup `1.5451x`, score `65.45`
- Added completed replay experiment metadata at
  `experiments/runs/2026-07-08-replay-sigmoid-scale-sum-pass4.yaml` and report
  at `experiments/reports/2026-07-08-replay-sigmoid-scale-sum-pass4.yaml`.
- Added the first live LLM provider adapter, `openai`, behind the same
  `ProviderRequest` and `ProviderResponse` interface used by deterministic
  replay generation.
- Added provider tests that validate OpenAI Responses API request construction,
  response text extraction, code-fence cleanup, and missing-configuration
  failure modes without real network calls or credentials.
- Added decision record
  `docs/decisions/0006-add-openai-responses-provider.md`.
- Pushed and synced the live provider adapter to the Ascend worker at commit
  `18f4b42e6c7a2acb1eefcd659084f09b60681be3`.
- Added the first softmax Pass@4 candidate batch for `t1/softmax`: one torch
  reference candidate and three Triton-Ascend rowwise softmax variants.
- Added `scripts/create_softmax_pass4_submissions.sh`,
  `scripts/probe_softmax_backend.py`, completed experiment metadata at
  `experiments/runs/2026-07-08-softmax-pass4.yaml`, and Pass@4 report at
  `experiments/reports/2026-07-08-softmax-pass4.yaml`.
- Probed all four `t1/softmax` Pass@4 candidates on Ascend; v2, v3, and v4
  launched through real Triton-Ascend backend paths with low numerical error.
- Ran the official AKG Bench Lite benchmark for `t1/softmax`:
  - `softmax_v1`: pass, speedup `1.0006x`, score `60.01`
  - `softmax_v2`: pass, speedup `0.7315x`, score `43.89`
  - `softmax_v3`: pass, speedup `0.8871x`, score `53.22`
  - `softmax_v4`: pass, speedup `0.9225x`, score `55.35`
- Promoted the softmax rowwise-correct but performance-negative lesson into
  the Skill Library.
- Added deterministic reference `ModelNew` candidates for the 9 remaining AKG
  Bench Lite cases:
  - `t1/matmul_basic`
  - `t1/matmul_biasadd`
  - `t2/rope`
  - `t2/add_rmsnorm_cast`
  - `t2/add_rmsnorm_quant`
  - `t2/moe_topk_softmax`
  - `t3/causal_conv1d`
  - `t3/decode_mla`
  - `t3/layernorm_gated`
- Added `scripts/create_remaining_reference_submissions.sh`,
  `experiments/runs/2026-07-08-remaining-reference-preeval.yaml`, and
  `experiments/reports/2026-07-08-remaining-reference-preeval.yaml`.
- Ran the official AKG Bench Lite benchmark for the remaining reference set:
  - `t1/matmul_basic`: pass, speedup `1.0012x`, score `60.01`
  - `t1/matmul_biasadd`: pass, speedup `0.8901x`, score `53.41`
  - `t2/add_rmsnorm_cast`: pass, speedup `0.9999x`, score `89.99`
  - `t2/add_rmsnorm_quant`: pass, speedup `0.9998x`, score `89.98`
  - `t2/moe_topk_softmax`: pass, speedup `1.003x`, score `90.05`
  - `t2/rope`: pass, speedup `1.0006x`, score `90.01`
  - `t3/causal_conv1d`: environment/runtime failure in the official
    reference path because CANN/TBE initialization could not import `tbe`
  - `t3/decode_mla`: pass, speedup `1.0004x`, score `120.01`
  - `t3/layernorm_gated`: pass, speedup `0.9999x`, score `119.99`
- Promoted the `causal_conv1d` reference-path environment failure lesson into
  the benchmark evaluation Skill Library.
- Re-ran the remaining-reference pre-evaluation after updating AKG to commit
  `47aa428fcdc8c68f78d331dc578bc6c74fb9d91d`; the active baseline is now
  `experiments/reports/2026-07-09-remaining-reference-preeval-updated-akg.yaml`.
- Diagnosed the `t3/causal_conv1d` CANN/TBE failure: `set_env.sh` made `tbe`,
  `te`, and `auto_tune` importable, but the initial benchmark command
  overwrote CANN's `PYTHONPATH` with only `/data/KernelForge-Agent`.
- Re-ran `t3/causal_conv1d` with
  `export PYTHONPATH=/data/KernelForge-Agent:${PYTHONPATH:-}` after sourcing
  CANN and activating the venv; the official reference path passed.
- Corrected remaining-reference baseline under the current runner:
  - `t1/matmul_basic`: pass, speedup `1.0036x`, score `60.04`
  - `t1/matmul_biasadd`: pass, speedup `0.891x`, score `53.46`
  - `t2/add_rmsnorm_cast`: pass, speedup `0.9984x`, score `89.86`
  - `t2/add_rmsnorm_quant`: pass, speedup `0.9996x`, score `89.96`
  - `t2/moe_topk_softmax`: pass, speedup `1.0002x`, score `90.0`
  - `t2/rope`: pass, speedup `1.01x`, score `90.15`
  - `t3/causal_conv1d`: pass, speedup `0.9971x`, score `119.65`
  - `t3/decode_mla`: pass, speedup `0.9983x`, score `119.79`
  - `t3/layernorm_gated`: pass, speedup `0.9893x`, score `118.71`
- Added the base project-book source and PDF under `pdfs/`.
- Drafted the Chinese full project book at `docs/project_book_full_zh.md`,
  using the base project book plus current code architecture, provider
  workflow, benchmark evidence, risks, roadmap, and Step 3 submission plan.
- Added the Step 3 technical design document at `docs/technical_design.md`.
- Added the GitLink package README, PR text, email text, and checklist at
  `docs/submission_package_readme.md`.
- Added standalone PR and email drafts under `docs/submission/`.
- Added the Chinese GitLink package root README source at
  `docs/submission/package_readme_zh.md`; the package helper installs it as
  `README.md` and preserves the project development README as
  `PROJECT_README.md`.
- Added `scripts/export_project_book.py` to create a standalone Markdown
  project-book attachment from the full project book, technical design, and
  package notes.
- Added `scripts/prepare_gitlink_package.py` to assemble a clean Step 3
  GitLink package under `outputs/gitlink_package/`.
- Added `docs/submission/step3_completion_audit.md` to record local
  deliverable status, verification commands, and manual PR/email actions.

## In Progress

- No automatic external submission action is in progress. Local Step 3
  materials are ready for manual review, PR creation, and email sending.

## Blockers

- `gelu_triton_v13` remains much slower than the PyTorch baseline, so GELU is
  not a good next single-operator tuning target without a new backend strategy.
- T2/T3 cases with symbolic shape setup are currently discovered by the
  registry but marked `parse_failed` until the extractor supports local shape
  variables and more complex input construction.
- The current cloud SSH endpoint is served by `SSHPiper` and only advertises
  password authentication. Local no-password SSH works only while a
  `ControlMaster` session remains alive; durable automation needs either
  provider-level key configuration or a manually opened persistent master
  connection.

## Next Actions

1. Manually review `docs/project_book_full_zh.md` for Chinese wording and
   team-specific details.
2. Manually copy the generated GitLink package into the GitLink fork and open
   the PR.
3. Rerun `scripts/export_project_book.py --pr-link <GitLink PR URL>` after the
   PR exists.
4. Manually email the updated project book to `contact@public.mindspore.cn`.
5. Record the PR link, email date, and submission status in `docs/status.md`.
6. Run a first live `provider=openai` Pass@4 generation cycle for
   `t1/sigmoid_scale_sum` once credentials and model selection are available.
7. Keep `replay` as the deterministic CI/regression provider.
8. Import live generated benchmark results after Ascend verification.
9. Use the completed manual Pass@4 cycles as retrieval examples:
   `sigmoid_scale_sum_v2` for a positive reduction trajectory and
   `fused_silu_and_mul` for a correctness-positive but performance-negative
   fused-elementwise trajectory, and `softmax_v4` for a correctness-positive
   but still-slower rowwise softmax trajectory.
10. Use
    `experiments/reports/2026-07-09-remaining-reference-preeval-updated-akg.yaml`
    as the baseline for all remaining AKG Bench Lite operators before live
    provider generation; all nine remaining reference cases now pass when
    CANN's `PYTHONPATH` entries are preserved.
11. Rerun key Pass@4 reports under updated AKG commit
    `47aa428fcdc8c68f78d331dc578bc6c74fb9d91d` before final result claims,
    because the standalone correctness protocol changed.
12. Add backend-probe fields to future generated experiment records by default.
13. Keep model/provider information explicit in every generated experiment
   record.
14. Compare the standalone `tools/run_bench.py` path with the AKG Agents
   `run_torch_bench_lite.py` path on one completed Pass@4 case.

## Latest Handoff

Date: 2026-07-09
Agent: Codex
Branch: main
Summary:
- Diagnosed the `t3/causal_conv1d` CANN/TBE import failure on the Ascend
  worker. The worker already has `tbe`; the failed benchmark command had
  overwritten CANN's `PYTHONPATH`.
- Re-ran `reference_t3_causal_conv1d` with the repository path prepended to the
  existing CANN `PYTHONPATH`; it passed correctness and benchmarked at
  `0.9971x` speedup with weighted score `119.65`.
- Updated
  `experiments/reports/2026-07-09-remaining-reference-preeval-updated-akg.yaml`
  to be a 9/9 passing active baseline for remaining AKG Bench Lite operators
  under the current runner.

Changed Files:
- `docs/status.md`
- `experiments/reports/2026-07-09-remaining-reference-preeval-updated-akg.yaml`
- `experiments/runs/2026-07-09-causal-conv1d-tbe-pythonpath-fix.yaml`
- `experiments/runs/2026-07-09-remaining-reference-preeval-updated-akg.yaml`
- `skills/ascend_debug/SKILL.md`
- `skills/benchmark_evaluation/SKILL.md`
- `tasks/active.md`

Verification:
- `ssh -o BatchMode=yes ascend-kf 'bash -lc '\''cd /data/KernelForge-Agent; source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1; source /data/venvs/kf-triton-ascend/bin/activate; python -c "import importlib.util; print(importlib.util.find_spec(\"tbe\")); print(importlib.util.find_spec(\"te\")); print(importlib.util.find_spec(\"auto_tune\"))"'\'''`
- `ssh -o BatchMode=yes ascend-kf 'bash -lc '\''cd /data/KernelForge-Agent && source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 && source /data/venvs/kf-triton-ascend/bin/activate && export PYTHONPATH=/data/KernelForge-Agent:${PYTHONPATH:-} && python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py outputs/submissions/remaining_reference_2026_07_09 --team reference_t3_causal_conv1d --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite --output outputs/results/causal_conv1d_envfix_2026_07_09 --warmup 10 --iterations 100 --num-trials 3'\'''`
- Corrected single-case result: `t3/causal_conv1d` pass,
  `max_abs_diff=0.0`, `max_rel_diff=0.0`, speedup `0.9971x`, weighted score
  `119.65`.

Open Issues:
- GitLink PR is not opened yet; this is a manual user action.
- The project-book email has not been sent yet; this is a manual user action.
- After the PR is opened, rerun the exporter with `--pr-link <GitLink PR URL>`
  and attach that final output to the email.
- Need to extend parsing for symbolic shape construction in T2/T3 cases.
- Need to rerun important Pass@4 benchmark reports under the updated AKG runner
  before treating previous speed/correctness numbers as final.
- Need to run the first real live-provider generation and compare it with
  replay/manual Pass@4 results when credentials/model selection are available.

Next Suggested Step:
- Rerun the `t1/sigmoid_scale_sum` Pass@4 report under AKG
  `47aa428fcdc8c68f78d331dc578bc6c74fb9d91d` with CANN's `PYTHONPATH`
  preserved, because it is the strongest positive Triton-Ascend result so far.
