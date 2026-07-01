# Active Tasks

Keep this file short. It tracks the next concrete work items across machines
and agents.

## Current Priorities

1. Pull the latest project changes on the Ascend worker.
2. Analyze `gelu_triton_v5` failure with
   `python scripts/analyze_gelu_candidate_error.py --candidate outputs/submissions/gelu_triton_v5/gelu_triton_v5/t1/gelu.py --shape 32 512 1024`.
3. Use the worst-error input to decide whether the next repair should target
   negative tail, transition region, or positive path.
4. If no pure Triton math path can satisfy official relative error, switch GELU
   to a benchmark-safe `torch_npu`/framework implementation and move custom
   Triton work to a less numerically fragile case.
6. Record the custom candidate experiment under `experiments/runs/`.
7. Expand OpSpec extraction beyond GELU after the custom GELU loop is complete.
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

## Coordination Rules

- Use branch-per-task for concurrent work.
- Update `docs/status.md` before handing off work.
- Add a decision record for non-trivial architecture or workflow changes.
- Do not overwrite another agent's unmerged work.
- Prefer append-only experiment records over editing old records, unless
  correcting an obvious metadata mistake.
- Promote reusable experiment lessons into `skills/`.
