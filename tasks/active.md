# Active Tasks

Keep this file short. It tracks the next concrete work items across machines
and agents.

## Current Priorities

1. Pull the latest project changes on the Ascend worker.
2. Run `python scripts/probe_gelu_triton_backend.py` and record
   `last_backend`.
3. If `last_backend` is `triton`, compare the custom candidate against the
   manual PyTorch baseline and tune `gelu_triton_v1`.
4. If `last_backend` is `torch_fallback_*`, debug Triton-Ascend availability or
   `tl.erf` lowering.
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
