# AGENTS.md

This file is the shared operating guide for humans and AI agents working on
KernelForge-Agent. Keep it current whenever the project workflow, architecture,
or experiment process changes.

## Project Purpose

KernelForge-Agent is intended to become a feedback-driven AI operator generation
and optimization system. The system should read official benchmark operator
tasks, generate candidate implementations, compile and run them, verify
correctness, benchmark performance, repair failures, optimize successful
candidates, and persist reusable knowledge in a Skill Library.

## Source Of Truth

- Project workflow: `docs/project_workflow.md`
- Current status: `docs/status.md`
- Roadmap: `docs/roadmap.md`
- Decisions: `docs/decisions/`
- Active work: `tasks/active.md`
- Experiment schema and rules: `experiments/README.md`

If chat history and repository files disagree, trust the repository files and
update them as needed.

## Startup Checklist

Before starting work:

1. Pull the latest branch state.
2. Read `docs/status.md`.
3. Read `tasks/active.md`.
4. Check recent decision records in `docs/decisions/`.
5. Confirm whether your task touches implementation, experiments, docs, or all
   three.

Recommended Git flow:

```bash
git pull --rebase
git checkout -b task/short-name
```

Use branch-per-task for concurrent work. Avoid sharing important progress only
through uncommitted local changes.

## Work Session Rules

- Keep changes scoped to the current task.
- Prefer existing project conventions once implementation code exists.
- Update docs/status files when the current project state changes.
- Record important design decisions as decision records.
- Record experiments as structured metadata under `experiments/runs/`.
- Do not commit build outputs, raw logs, generated binaries, caches, or large
  candidate output directories unless a project maintainer explicitly decides
  otherwise.

Before ending a work session:

1. Run relevant validation commands if implementation files changed.
2. Update `docs/status.md` or `tasks/active.md`.
3. Add experiment records if benchmark or generation runs were performed.
4. Commit a coherent unit of work.
5. Push the branch if the work should be shared across machines.

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

Build the reliable benchmark harness before optimizing prompts or agent
autonomy. The recommended order is:

1. Benchmark specification.
2. Manual runnable baseline for one operator.
3. Automated compile/run/validate harness.
4. Single-shot code generation.
5. Feedback repair loop.
6. Multi-candidate generation and Pass@N.
7. Retrieval from Skill Library.
8. Performance optimization loop.
9. Final evaluation and ablation report.

## Experiment Discipline

Every generated candidate should be traceable to:

- benchmark task
- prompt version
- model or agent
- retrieved skills or examples
- generated code path
- compile result
- correctness result
- performance result
- logs or artifact paths
- repair iteration count
- final status

Change one major variable per experiment cycle whenever possible. This keeps
results interpretable.

