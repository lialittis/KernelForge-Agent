# Status

Last updated: 2026-06-30

## Current Phase

Project setup and workflow definition.

## Completed

- Initial project idea drafted in `idea-draft.md`.
- Repository-level multi-agent guidance added in `AGENTS.md`.
- Research and implementation workflow documented in
  `docs/project_workflow.md`.
- Initial roadmap, task tracking, and experiment metadata rules added.

## In Progress

- Researching official benchmark requirements and competition constraints.
- Preparing the project structure needed for implementation and experiments.

## Blockers

- Exact official benchmark input/output format is not yet captured in the repo.
- Supported operator implementation runtime and submission constraints still
  need to be confirmed from official competition materials.

## Next Actions

1. Create `docs/benchmark_spec.md` from official competition requirements.
2. Collect at least one official sample operator task.
3. Define the first manual runnable baseline.
4. Add an initial experiment record once the first baseline or generated
   candidate is run.

## Latest Handoff

Date: 2026-06-30
Agent: Codex
Branch: main
Summary:
- Added persistent cross-machine and cross-agent project memory files.
- Captured workflow, roadmap, task tracking, experiment rules, and the first
  decision record.
- Added a README index for the shared project-memory files.

Changed Files:
- `AGENTS.md`
- `README.md`
- `docs/project_workflow.md`
- `docs/status.md`
- `docs/roadmap.md`
- `docs/decisions/0001-persist-project-memory-in-repo.md`
- `experiments/README.md`
- `experiments/runs/.gitkeep`
- `tasks/active.md`
- `.gitignore`

Verification:
- Documentation-only change. No code tests were run.

Open Issues:
- Need official benchmark details before implementation can be specified.

Next Suggested Step:
- Write `docs/benchmark_spec.md` using official benchmark and submission
  requirements.
