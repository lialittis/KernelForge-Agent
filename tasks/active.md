# Active Tasks

Keep this file short. It tracks the next concrete work items across machines
and agents.

## Current Priorities

1. Sync the latest commits to the Ascend worker.
2. Create the `t1/sigmoid_scale_sum` Pass@4 submissions:
   `bash scripts/create_sigmoid_scale_sum_pass4_submissions.sh`.
3. Probe at least one Triton candidate with
   `scripts/probe_sigmoid_scale_sum_backend.py`.
4. Run the official benchmark on the Ascend worker and import result JSON with
   `scripts/import_benchmark_result.py`.
5. Summarize Pass@4 with `scripts/summarize_passn.py`.
6. Record model/provider, prompt version, retrieved skills, and candidate index
   for every generated experiment.
7. Promote reusable generation, repair, or tuning lessons into `skills/`.
8. Draft `docs/technical_design.md` from the architecture, workflow, roadmap,
   and competition alignment docs.
9. Decide whether to commit `SketchSkill_AKG_项目书基础版.pdf`.

## Research Questions

- Which AKG Agents runner path should be treated as authoritative for the final
  submission: standalone `tools/run_bench.py`, `run_torch_bench_lite.py`, or
  both?
- What Triton-Ascend version and APIs are available?
- What CANN and Ascend 910 environment are provided?
- Should project experiments use Pass@4, or match the AKG Agents runner default
  `--pass-n 3` unless overridden?
- Which official cases are runnable without `torch_npu` dependencies on the
  available machine?
- What is the minimum symbolic-shape parser needed for T2/T3 case metadata?
- Which LLM provider adapter should be implemented first after the deterministic
  T1 loop is stable?

## Coordination Rules

- Use branch-per-task for concurrent work.
- Update `docs/status.md` before handing off work.
- Add a decision record for non-trivial architecture or workflow changes.
- Do not overwrite another agent's unmerged work.
- Prefer append-only experiment records over editing old records, unless
  correcting an obvious metadata mistake.
- Promote reusable experiment lessons into `skills/`.
