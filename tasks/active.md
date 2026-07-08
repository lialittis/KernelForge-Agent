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
7. Keep `replay` as the deterministic CI/regression provider.
8. Use completed manual Pass@4 cycles as retrieval examples:
   `sigmoid_scale_sum_v2` for a positive reduction trajectory and
   `fused_silu_and_mul` for a correctness-positive but performance-negative
   fused-elementwise trajectory, and `softmax_v4` for a correctness-positive
   but still-slower rowwise softmax trajectory.
9. Use `experiments/reports/2026-07-08-remaining-reference-preeval.yaml` as
   the deterministic reference baseline for all remaining AKG Bench Lite cases.
10. Repair or document the current `t3/causal_conv1d` environment failure:
   official reference `F.conv1d` cannot initialize CANN/TBE because Python
   module `tbe` is missing.
11. Rerun key reports under AKG commit
   `47aa428fcdc8c68f78d331dc578bc6c74fb9d91d` before final result claims,
   because standalone `run_bench.py` now uses three seeded correctness trials
   and independent reference/solution inputs.
12. Add backend-probe fields to future generated experiment records by default.
13. Record model/provider, prompt version, retrieved skills, and candidate index
   for every generated experiment.
14. Promote reusable generation, repair, or tuning lessons into `skills/`.

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
- Does the Ascend worker need an additional CANN/TBE Python path or package to
  run the official `t3/causal_conv1d` reference?
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
