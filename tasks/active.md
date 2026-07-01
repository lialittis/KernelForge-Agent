# Active Tasks

Keep this file short. It tracks the next concrete work items across machines
and agents.

## Current Priorities

1. Pull the latest project changes on the Ascend worker.
2. Generate `gelu_triton_v5` with
   `bash scripts/create_gelu_triton_v5_submission.sh`.
3. Probe v5 with `python scripts/probe_gelu_triton_backend.py --candidate outputs/submissions/gelu_triton_v5/gelu_triton_v5/t1/gelu.py --shape 32 512 1024`.
4. Run the official benchmark for `gelu_triton_v5`.
5. If v5 passes, compare latency against the manual baseline and tune block
   size or tail threshold.
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
