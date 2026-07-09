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

Status: complete for the new OpSpec cases.

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

## Task 3: Enforce Full Coverage In Tests

Update tests so they assert:

- scanner finds all 13 official cases,
- all 13 are `opspec_supported`,
- there are no `parse_failed` cases,
- all 13 parsed OpSpec YAML files exist,
- every OpSpec has required fields and a non-generic Sketch,
- tuple-output cases describe all outputs explicitly.

## Task 4: Deterministic Reference And Replay Evaluation

After OpSpec coverage is complete:

- rerun or refresh reference pre-evaluation records if benchmark runner inputs
  or output schemas changed,
- ensure result import handles multi-output cases,
- keep `scripts/run_replay_regression.py` as the deterministic replay
  regression path for provider-generated candidates,
- keep all benchmark claims tied to the updated AKG commit
  `47aa428fcdc8c68f78d331dc578bc6c74fb9d91d`.

## Task 5: Pre-AI Infrastructure Improvements

Useful provider-independent work:

- OpSpec validator.
- Sketch validator.
- prompt context snapshot tests.
- package hygiene tests.
- replay regression tests.
- result comparison tools.
- stronger Skill Library entries for matmul, top-k softmax, normalization,
  reduction, and layout/transpose patterns.

## Task 6: Submission-Facing Updates

Update submission docs after full OpSpec coverage:

- `docs/status.md`
- `docs/competition_alignment.md`
- `docs/project_book_full_zh.md`
- `docs/technical_design.md`
- package README / PR text if needed

Suggested claim after completion:

```text
Lite benchmark OpSpec coverage: 13/13. Current executable candidates and
Pass@4 evidence cover a priority subset. Full live AI generation remains gated
only by model/API configuration.
```

## Deferred Until Credentials Exist

- Live `provider=openai` Pass@4 generation.
- AKG Agents `run_torch_bench_lite.py --mode full` comparison with generated
  attempts.
- Provider/model ablation.
