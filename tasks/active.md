# Active Tasks

Keep this file short. It tracks the next concrete work items across machines
and agents.

## Current Priorities

1. Write `docs/benchmark_spec.md` from official competition materials.
2. Collect official sample benchmark tasks under a future `benchmarks/raw/`
   directory.
3. Define the first manual baseline operator and required build/run commands.
4. Create the initial harness design in `docs/architecture.md`.
5. Add the first experiment record under `experiments/runs/` after a baseline
   or generated candidate is executed.

## Research Questions

- What exact operator task format does the official benchmark provide?
- Which implementation language/runtime is required?
- What are the correctness tolerances?
- What performance metric determines ranking?
- What are the submission packaging constraints?
- Are official sample operators available for local validation?

## Coordination Rules

- Use branch-per-task for concurrent work.
- Update `docs/status.md` before handing off work.
- Add a decision record for non-trivial architecture or workflow changes.
- Do not overwrite another agent's unmerged work.
- Prefer append-only experiment records over editing old records, unless
  correcting an obvious metadata mistake.

