# Active Tasks

Keep this file short. It tracks the next concrete work items across machines
and agents.

## Current Priorities

1. Start the next deterministic Pass@4 cycle on `t1/fused_silu_and_mul`.
2. Reuse `sigmoid_scale_sum_v2` as the first positive non-GELU retrieval
   example for rowwise reductions.
3. Add backend-probe fields to future generated experiment records by default.
4. Record model/provider, prompt version, retrieved skills, and candidate index
   for every generated experiment.
5. Promote reusable generation, repair, or tuning lessons into `skills/`.
6. Draft `docs/technical_design.md` from the architecture, workflow, roadmap,
   and competition alignment docs.
7. Decide whether to commit `SketchSkill_AKG_项目书基础版.pdf`.

## Research Questions

- Which AKG Agents runner path should be treated as authoritative for the final
  submission: standalone `tools/run_bench.py`, `run_torch_bench_lite.py`, or
  both?
- Which Triton-Ascend APIs can be relied on across rented workers?
- Which CANN and Ascend 910 worker variants should be included in final
  reproducibility notes?
- Should project experiments use Pass@4, or match the AKG Agents runner default
  `--pass-n 3` unless overridden?
- What is the minimum symbolic-shape parser needed for T2/T3 case metadata?
- Which LLM provider adapter should be implemented first after the deterministic
  T1 loop is stable?
- Can the cloud SSH gateway be configured for provider-level key auth, or do
  agents need an explicitly maintained `ControlMaster` session?

## Coordination Rules

- Use branch-per-task for concurrent work.
- Update `docs/status.md` before handing off work.
- Add a decision record for non-trivial architecture or workflow changes.
- Do not overwrite another agent's unmerged work.
- Prefer append-only experiment records over editing old records, unless
  correcting an obvious metadata mistake.
- Promote reusable experiment lessons into `skills/`.
