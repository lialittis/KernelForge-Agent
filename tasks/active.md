# Active Tasks

Keep this file short. It tracks the next concrete work items across machines
and agents.

## Current Priorities

1. Install/verify `triton-ascend` on the Ascend worker.
2. Generate a first custom Triton-Ascend GELU candidate.
3. Run the custom candidate through `tools/run_bench.py`.
4. Compare the custom candidate against the manual PyTorch baseline.
5. Record the custom candidate experiment under `experiments/runs/`.
6. Expand OpSpec extraction beyond GELU after the custom GELU loop is complete.
7. Decide whether to commit `SketchSkill_AKG_项目书基础版.pdf`.

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
