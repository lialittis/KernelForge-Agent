# Pre-Submission No-Key Development Plan

Date: 2026-07-09

## Context

Before the first submission, we do not yet have live AI model credentials.
Development should therefore focus on deterministic, reviewable project
evidence that strengthens the system design and does not depend on an external
provider.

## Goals

1. Complete OpSpec coverage for all 13 AKG Bench Lite cases.
2. Add Sketch templates for any newly covered operator categories.
3. Add tests that enforce full benchmark coverage.
4. Keep deterministic reference and replay evaluation reproducible.
5. Improve pre-AI infrastructure that will be reused by live agents later.
6. Update submission-facing documentation so the first submission can claim
   complete Lite benchmark understanding even before live generation.

## Task 1: Complete OpSpec Coverage

Status: complete.

Parsed coverage before this task: `10/13`.

Parsed coverage after this task: `13/13`.

Missing cases:

- `t1/matmul_basic`
- `t1/matmul_biasadd`
- `t2/moe_topk_softmax`

Added files:

- `benchmarks/parsed/t1_matmul_basic.yaml`
- `benchmarks/parsed/t1_matmul_biasadd.yaml`
- `benchmarks/parsed/t2_moe_topk_softmax.yaml`

Updated support summary:

```text
total_cases: 13
by_support.opspec_supported: 13
parse_failed: 0
unsupported: 0
```

Deliverables:

- Add these cases to deterministic OpSpec extraction support.
- Infer input and output tensor specs.
- Record semantics, validation, performance metadata, and submission metadata.
- Generate parsed YAML files under `benchmarks/parsed/`.
- Update the raw benchmark registry so `by_support.opspec_supported == 13`.

Why this matters:

- It proves the project understands the full Lite benchmark problem surface.
- It makes prompt/RAG context available for every official case.
- It improves the first submission even without live AI generation.

## Task 2: Add Missing Sketch Templates

Status: complete and hardened for the new OpSpec cases.

Required templates:

- Matmul:
  - `t1/matmul_basic`
  - `t1/matmul_biasadd`
  - tile axes: M/N/K
  - accumulation dtype and optional bias broadcast
  - backend risks: BF16/FP16 accumulation, UB/L1 tiling, matmul backend choice
- MoE top-k softmax:
  - `t2/moe_topk_softmax`
  - rowwise softmax over expert logits
  - top-k selection and renormalization
  - tuple outputs: probabilities and indices
  - backend risks: tie behavior, stable top-k ordering, probability
    renormalization, index dtype

Implemented hardening:

- Matmul sketches now include explicit `axis_map` entries for M/N/K, blocked
  tile candidates over M/N/K, dtype plans for lhs/rhs/bias/accumulator/output,
  bias broadcast contracts, UB/L1 accumulator tile risk, and lowering
  preference notes for backend matmul/dot intrinsics.
- MoE top-k softmax sketch now includes explicit row/expert/top-k axis mapping,
  tile candidates, tuple `output_contract` entries for probabilities and
  indices, numerical plan for max-subtracted softmax and selected-probability
  renormalization, and risk tags for top-k tie ordering, stable ordering,
  renormalization, and int64 indices.

## Task 3: Enforce Full Coverage In Tests

Status: complete.

Update tests so they assert:

- scanner finds all 13 official cases,
- all 13 are `opspec_supported`,
- there are no `parse_failed` cases,
- all 13 parsed OpSpec YAML files exist,
- every OpSpec has required fields and a non-generic Sketch,
- tuple-output cases describe all outputs explicitly.

Implemented in `tests/test_benchmark_registry_and_opspec.py`.

## Task 4: Deterministic Reference And Replay Evaluation

Status: complete for the current no-key baseline.

After OpSpec coverage is complete:

- rerun or refresh reference pre-evaluation records if benchmark runner inputs
  or output schemas changed,
- ensure result import handles multi-output cases,
- keep `scripts/run_replay_regression.py` as the deterministic replay
  regression path for provider-generated candidates,
- keep all benchmark claims tied to the updated AKG commit
  `47aa428fcdc8c68f78d331dc578bc6c74fb9d91d`.

Implemented state:

- The active deterministic reference baseline is
  `experiments/reports/2026-07-09-remaining-reference-preeval-updated-akg.yaml`,
  which supersedes the July 8 baseline and is tied to AKG commit
  `47aa428fcdc8c68f78d331dc578bc6c74fb9d91d`.
- Result import now preserves per-output correctness details when the runner
  JSON includes them, while remaining compatible with the current aggregate
  AKG Bench Lite schema.
- Pass@N summaries and generated experiment updates now carry per-output
  details into candidate rows, completed-case records, and benchmark-result
  metadata.
- `scripts/run_replay_regression.py` remains the deterministic replay
  regression path and now guards against accidentally claiming results from an
  unexpected AKG submodule commit. Use `--allow-akg-commit-mismatch` only for
  explicitly labeled exploratory runs.
- `scripts/audit_pre_key_readiness.py --json` now gates on full Lite coverage:
  `total_cases == 13`, `opspec_supported == 13`, `unsupported == 0`, and
  `parse_failed == 0`.

## Task 5: Pre-AI Infrastructure Improvements

Status: complete for the first provider-independent pass.

Useful provider-independent work:

- OpSpec validator.
- Sketch validator.
- prompt context snapshot tests.
- package hygiene tests.
- replay regression tests.
- result comparison tools.
- stronger Skill Library entries for matmul, top-k softmax, normalization,
  reduction, and layout/transpose patterns.

Implemented state:

- Added reusable OpSpec and Sketch validation helpers in
  `kernel_forge/benchmark/validation.py`.
- Added `scripts/validate_opspecs.py` as a deterministic CLI gate for
  `benchmarks/parsed`; the current parsed Lite set validates as `13/13`.
- Added prompt context snapshot coverage for matmul bias-add and MoE top-k
  softmax so retrieved skills, OpSpec YAML, Sketch YAML, and tuple-output
  contracts remain visible to future providers.
- Added package hygiene tests around `scripts/prepare_gitlink_package.py` so
  submission packages keep source/docs/tests/skills/prompts and exclude
  runtime outputs and caches.
- Kept replay regression deterministic by testing the updated-AKG commit guard
  exposed by `scripts/run_replay_regression.py`.
- Extended result comparison to preserve per-output candidate details when
  standalone or AKG Agents JSON includes them.
- Strengthened Skill Library entries for matmul-like, reduction/top-k softmax,
  normalization, and transpose/layout patterns.
- Added `pre_ai_infrastructure` to `scripts/audit_pre_key_readiness.py` so the
  deterministic readiness audit tracks these files.

## Task 6: Submission-Facing Updates

Status: complete for local PR/email materials.

Update submission docs after full OpSpec coverage:

- `docs/status.md`
- `docs/competition_alignment.md`
- `docs/project_book_full_zh.md`
- `docs/technical_design.md`
- package README / PR text if needed

Suggested claim after completion:

```text
AKG Bench Lite 的 OpSpec 覆盖已达到 13/13；当前可执行候选与 Pass@4
实验证据已覆盖优先验证子集；完整的实时 AI 生成对比仅受模型/API 配置限制。
```

Implemented state:

- Updated `docs/competition_alignment.md`, `docs/technical_design.md`, and
  `docs/project_book_full_zh.md` to remove stale T1-only coverage wording.
- Updated package/PR/email entry points under `docs/submission/` and
  `docs/submission_package_readme.md` with the same 13/13 coverage and
  priority-subset Pass@4 claim.
- Left external actions explicit: GitLink PR creation and project-book email
  sending remain manual steps after the user fills in the PR link.

## Deferred Until Credentials Exist

- Live `provider=openai` Pass@4 generation.
- AKG Agents `run_torch_bench_lite.py --mode full` comparison with generated
  attempts.
- Provider/model ablation.
