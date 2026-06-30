# AGENTS.md

This file is the shared operating guide for humans and AI agents working on
SketchSkill-AKG. Keep it current whenever the project workflow, architecture,
or experiment process changes.

## Project Purpose

SketchSkill-AKG is a feedback-driven operator generation and optimization
system for Ascend 910 NPU benchmarks. The active design is based on the
SketchSkill-AKG proposal:

- main path: AKG Agents + Triton-Ascend
- intermediate plan: NPU-aware Operator Sketch
- reusable memory: operator-pattern Skill Library
- validation: correctness loop plus Pass@1/Pass@4
- optimization: real hardware profiling plus lightweight search
- assets: bad-to-good trajectories, prompt templates, repair rules, tuning
  rules, and benchmark scripts

The project should not rely on one-shot low-level kernel generation. Agents
should move through OpSpec extraction, Sketch planning, retrieval, code
generation, compile/run verification, repair, profiling, search, and Skill
Library write-back.

## Source Of Truth

- Architecture: `docs/architecture.md`
- Development loop: `docs/dev_guide.md`
- Project workflow: `docs/project_workflow.md`
- Benchmark requirements: `docs/benchmark_spec.md`
- Current status: `docs/status.md`
- Roadmap: `docs/roadmap.md`
- Decisions: `docs/decisions/`
- Active work: `tasks/active.md`
- Experiment schema and rules: `experiments/README.md`
- Skill Library: `skills/`

If chat history and repository files disagree, trust the repository files and
update them as needed.

## Startup Checklist

Before starting work:

1. Pull the latest branch state.
2. Read `docs/status.md`.
3. Read `tasks/active.md`.
4. Read `docs/architecture.md` for the current system design.
5. Read `docs/dev_guide.md` if work will be run on a separate Ascend machine.
6. Check recent decision records in `docs/decisions/`.
7. Confirm whether your task touches benchmark research, harness code, agent
   code, experiments, skills, or documentation.

Recommended Git flow:

```bash
git pull --rebase
git checkout -b task/short-name
```

Use branch-per-task for concurrent work. Avoid sharing important progress only
through uncommitted local changes.

## Work Session Rules

- Keep changes scoped to the current task.
- Preserve the SketchSkill-AKG architecture unless a new decision record
  explicitly changes it.
- Treat `docs/benchmark_spec.md` as the benchmark contract once official
  details are confirmed.
- Record generated candidates and experiments under `experiments/runs/`.
- Promote reusable generation, repair, debugging, or tuning lessons into
  `skills/`.
- Do not commit build outputs, raw logs, generated binaries, caches, or large
  candidate output directories unless a maintainer explicitly decides
  otherwise.

Before ending a work session:

1. Run relevant validation commands if implementation files changed.
2. Update `docs/status.md` or `tasks/active.md`.
3. Add experiment records if benchmark or generation runs were performed.
4. Update relevant `skills/*/SKILL.md` files if a reusable lesson was found.
5. Commit a coherent unit of work.
6. Push the branch if the work should be shared across machines.

## Handoff Format

Use this format in `docs/status.md` when handing work to another agent or
machine:

```md
## Latest Handoff

Date: YYYY-MM-DD
Agent: name-or-tool
Branch: branch-name
Summary:
- What changed
- Why it changed

Changed Files:
- path/to/file

Verification:
- Command and result, or "Not run" with reason

Open Issues:
- Any blocker or unresolved design point

Next Suggested Step:
- One concrete next action
```

## Implementation Priorities

The implementation order is:

1. Confirm official benchmark format and Ascend 910 environment constraints.
2. Reproduce AKG benchmark baseline locally or on the provided cloud resource.
3. Implement OpSpec parsing for `akg_kernels_bench_lite` tasks.
4. Define and validate NPU-aware Operator Sketch templates.
5. Generate Triton-Ascend candidates through AKG Agents.
6. Build compile/run/correctness verification and Pass@1/Pass@4 reporting.
7. Add targeted Repair Agent routing from error categories.
8. Add Skill Retriever and Skill Writer.
9. Add Profiler/Search Agent for correct kernels.
10. Try TileLang-Ascend or Ascend C only for selected representative kernels.

## Experiment Discipline

Every generated candidate should be traceable to:

- benchmark task and operator category
- OpSpec version
- Sketch version
- backend target
- prompt version
- model or agent
- retrieved skills or examples
- generated code path
- compile result
- correctness result
- Pass@N status
- profiling/performance result
- logs or artifact paths
- repair iteration count
- final status

Change one major variable per experiment cycle whenever possible. This keeps
results interpretable.
