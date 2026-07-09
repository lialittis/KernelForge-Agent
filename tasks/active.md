# Active Tasks

Keep this file short. It tracks the next concrete work items across machines
and agents.

## Current Priorities

1. Keep the local initial-round Step 3 materials ready for manual PR/email
   submission, as described in `docs/tasks/initial_submission_step3.md`.
2. Review `docs/project_book_full_zh.md` only for final human wording or
   team-specific edits before submission.
3. Use `scripts/export_project_book.py` to regenerate the project book for
   email after the manual PR link exists.
4. Use `scripts/prepare_gitlink_package.py`,
   `docs/submission/package_readme_zh.md`, and
   `docs/submission_package_readme.md` to regenerate the GitLink PR package.
5. Manual external action: open the GitLink PR with
   `docs/submission/gitlink_pr_title.txt` and
   `docs/submission/gitlink_pr_body.md`.
6. Manual external action: send the updated project book with
   `docs/submission/project_book_email_zh.md`, then record the email
   date/status.
7. Keep `replay` as the deterministic CI/regression provider; use
   `scripts/run_replay_regression.py` for the current updated-AKG replay
   `t1/sigmoid_scale_sum` regression baseline.
8. Use completed manual Pass@4 cycles as retrieval examples:
   updated-AKG `sigmoid_scale_sum_v2` for a positive reduction trajectory and
   updated-AKG `fused_silu_and_mul_v3` for a correctness-positive but
   performance-negative fused-elementwise trajectory, and updated-AKG
   `softmax_v4` for a correctness-positive but still-slower rowwise softmax
   trajectory, and `add_rmsnorm_cast_v2` for a positive T2 normalization
   trajectory, and `rope_v4`/`rope_v1` for RoPE intrinsic-vs-Triton parity.
9. Use
   `experiments/reports/2026-07-09-remaining-reference-preeval-updated-akg.yaml`
   as the deterministic reference baseline for all remaining AKG Bench Lite
   cases under the current runner.
10. Preserve CANN's `PYTHONPATH` entries in all Ascend benchmark commands;
   prepend the repository path with
   `export PYTHONPATH=/data/KernelForge-Agent:${PYTHONPATH:-}` instead of
   replacing `PYTHONPATH`.
11. Continue rerunning key Pass@4 reports under AKG commit
   `47aa428fcdc8c68f78d331dc578bc6c74fb9d91d` before final result claims;
   manual `t1/sigmoid_scale_sum`, `t1/softmax`, `t1/fused_silu_and_mul`, and
   replay `t1/sigmoid_scale_sum` are done.
12. Choose between `t2/add_rmsnorm_quant` and `t3/layernorm_gated` for the next
   deterministic manual seed before live provider generation.
13. Configure an AKG Agents `standard` model level before rerunning
   `run_torch_bench_lite.py --mode full` for full runner-path comparison.
14. Record model/provider, prompt version, retrieved skills, and candidate index
   for every generated experiment.
15. Promote reusable generation, repair, or tuning lessons into `skills/`.

## Research Questions

- After configuring an AKG Agents `standard` model level, do standalone
  `tools/run_bench.py` and AKG Agents `run_torch_bench_lite.py --mode full`
  agree closely enough to cite both for final evidence?
- Which Triton-Ascend APIs can be relied on across rented workers?
- Which CANN and Ascend 910 worker variants should be included in final
  reproducibility notes?
- Should project experiments use Pass@4, or match the AKG Agents runner default
  `--pass-n 3` unless overridden?
- Which live model should be the first measured `openai` provider backend for
  generated candidates after the initial submission package is done?
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
