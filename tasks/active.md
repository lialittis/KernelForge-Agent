# Active Tasks

Keep this file short. It tracks the next concrete work items across machines
and agents.

## Current Priorities

1. Commit and push the completed `t1/fused_silu_and_mul` Pass@4 result record.
2. Sync the final result commit to the Ascend worker.
3. Implement the first pluggable LLM/provider adapter for Sketch/Code/Repair
   experiments.
4. Use completed manual Pass@4 cycles as retrieval examples:
   `sigmoid_scale_sum_v2` for a positive reduction trajectory and
   `fused_silu_and_mul` for a correctness-positive but performance-negative
   fused-elementwise trajectory.
5. Reuse `sigmoid_scale_sum_v2` as the first positive non-GELU retrieval
   example for rowwise reductions.
6. Add backend-probe fields to future generated experiment records by default.
7. Record model/provider, prompt version, retrieved skills, and candidate index
   for every generated experiment.
8. Promote reusable generation, repair, or tuning lessons into `skills/`.
9. Draft `docs/technical_design.md` from the architecture, workflow, roadmap,
   and competition alignment docs.
10. Decide whether to commit `SketchSkill_AKG_项目书基础版.pdf`.

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
