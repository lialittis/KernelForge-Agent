# Active Tasks

Keep this file short. It tracks the next concrete work items across machines
and agents.

## Current Priorities

1. Review the committed T1 non-matmul OpSpecs:
   `benchmarks/parsed/t1_fused_silu_and_mul.yaml`,
   `benchmarks/parsed/t1_sigmoid_scale_sum.yaml`, and
   `benchmarks/parsed/t1_softmax.yaml`.
2. Start the first non-GELU Pass@4 cycle for `t1/sigmoid_scale_sum`.
3. Use `scripts/create_submission.py` for new candidate layouts instead of
   adding another one-off shell generator.
4. Run the official benchmark on the Ascend worker and import result JSON with
   `scripts/import_benchmark_result.py`.
5. Record model/provider, prompt version, retrieved skills, and candidate index
   for every generated experiment.
6. Promote reusable generation, repair, or tuning lessons into `skills/`.
7. Draft `docs/technical_design.md` from the architecture, workflow, roadmap,
   and competition alignment docs.
8. Decide whether to commit `SketchSkill_AKG_项目书基础版.pdf`.

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
