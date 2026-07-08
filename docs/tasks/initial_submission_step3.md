# Initial Submission Step 3 Task Plan

Date: 2026-07-08

## Context

The first two initial-round actions are already done:

1. A basic project book was prepared.
2. The basic project book was emailed to request compute resources.

The urgent remaining initial-round task is Step 3: start project development,
improve the project book, submit project code through a PR to the GitLink
competition repository, and email the updated project book.

Live LLM generation is not blocking this stage because no OpenAI API key is
currently available. The Step 3 package should emphasize the implemented
deterministic pipeline, benchmark evidence, skill library, prompts, and the
replaceable provider interface.

## Submission Goal

Produce a PR-ready project package for:

- GitLink repository: `https://www.gitlink.org.cn/mindspore-ai/ccf-akg`
- Email target: `contact@public.mindspore.cn`

The package should show that SketchSkill-AKG has moved from proposal to an
implemented prototype with reproducible benchmark evidence.

## Required Outputs

### Code PR Package

Prepare a team-scoped project package for the GitLink PR. If the upstream
repository does not define a stricter layout, use a self-contained layout such
as:

```text
projects/<team-name>/SketchSkill-AKG/
  README.md
  docs/
  kernel_forge/
  prompts/
  skills/
  scripts/
  experiments/
```

Do not include large runtime output directories, generated caches, raw logs,
compiled artifacts, or local virtual environments.

Include these minimum files:

- project/package `README.md`
- `docs/technical_design.md`
- `docs/competition_alignment.md`
- `docs/architecture.md`
- `docs/project_workflow.md`
- `docs/benchmark_spec.md`
- `docs/dev_guide.md`
- `kernel_forge/`
- `scripts/`
- `prompts/`
- `skills/`
- concise experiment records and reports under `experiments/`

### Updated Project Book

Update the project book from the basic version to a submission-ready improved
version. It should include:

- team introduction
- technical solution overview
- AI technology selection
- operator generation flow
- expected innovation points
- expected results and performance goals
- references and prior work
- current implementation progress
- current benchmark evidence
- code repository and PR link after the PR is opened

### PR Text

Suggested PR title:

```text
Initial submission: SketchSkill-AKG Ascend NPU skill-driven operator generation system
```

The PR body should include:

- team name
- project summary
- current implementation status
- reproduction commands
- benchmark results
- project-book/email status
- known limitations
- next development plan

### Email Text

Suggested email subject:

```text
Initial Round + <team-name>: SketchSkill-AKG updated project book
```

Attach the updated project book and include the GitLink PR link once available.

## Evidence To Highlight

Use the current repository evidence:

- AKG Bench Lite benchmark is pinned as a submodule.
- Ascend worker environment has been reconstructed and documented.
- OpSpec extraction exists for the first T1 non-matmul subset.
- NPU-aware Sketch templates exist for elementwise, fused elementwise,
  reduction, softmax, and unsupported placeholders.
- Skill Library and prompt templates exist.
- GELU tuning produced a correctness-positive Triton-Ascend case study, with
  `gelu_triton_v13` as the best tracked real Triton candidate.
- `t1/sigmoid_scale_sum` Pass@4 passed on Ascend; best manual candidate reached
  about `2.0279x` speedup, and replay-generated Pass@4 reproduced the pipeline
  with best speedup about `1.998x`.
- `t1/fused_silu_and_mul` Pass@4 passed correctness, but Triton variants were
  slower; this is useful as a negative performance lesson in the Skill Library.
- A live `openai` provider adapter exists but is optional and not required for
  the initial submission package.

## Immediate Implementation Order

1. Create `docs/technical_design.md`.
2. Create a PR/package README that describes the prototype, directory layout,
   reproducibility commands, and current results.
3. Add a packaging helper or checklist so files can be copied into a GitLink
   fork without runtime outputs.
4. Update the project book with current code architecture and benchmark
   evidence.
5. Open the GitLink PR.
6. Send the updated project book to the competition email address.
7. Record the PR link and email status in `docs/status.md`.

## Acceptance Criteria

The Step 3 task is complete when:

- the GitLink PR is opened,
- the updated project book is emailed,
- the repository records the PR link and email date,
- the PR package contains enough code and docs for reviewers to reproduce the
  current deterministic benchmark flow,
- the package clearly explains that live LLM generation is provider-pluggable
  and not required for the current prototype evidence.

## Deferred Until After Step 3

- Run live `provider=openai` Pass@4 experiments.
- Extend symbolic parsing for T2/T3 cases.
- Add a full Repair Agent loop.
- Add deeper profiler/search automation.
- Run multi-backend TileLang-Ascend or Ascend C experiments.
