# 0001: Persist Project Memory In The Repository

Date: 2026-06-30

## Status

Accepted

## Context

SketchSkill-AKG will be developed across multiple machines and by multiple
humans or AI agents. Chat history and local notes are not reliable shared state.
The project needs durable guidance, status, decisions, skills, and experiment
metadata that travel with the repository.

## Decision

Use Git-tracked repository files as the source of truth for project memory:

- `AGENTS.md` for agent operating instructions
- `docs/project_workflow.md` for the research and implementation workflow
- `docs/status.md` for current state and handoffs
- `docs/roadmap.md` for milestones
- `docs/decisions/` for architecture and workflow decisions
- `experiments/` for experiment metadata
- `skills/` for reusable operator-generation and optimization knowledge
- `tasks/` for active work tracking

Large generated artifacts, build outputs, caches, and raw logs should not be
committed by default. Commit summaries, manifests, schemas, and curated examples
instead.

## Consequences

Positive:

- Agents can resume work from any machine by reading repository files.
- Important decisions are reviewable and versioned.
- Experiment results can be compared over time.
- Handoffs do not depend on one chat session.

Negative:

- Contributors must keep status and task files current.
- Merge conflicts may occur if many agents edit the same status file.

## Follow-Up

If merge conflicts become frequent, move handoffs into append-only files under a
dedicated `handoffs/` directory.
