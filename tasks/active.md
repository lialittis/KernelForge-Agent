# Active Tasks

Keep this file short. It tracks the next concrete work items across machines
and agents.

## Current Priorities

1. Treat full Lite OpSpec coverage as complete: all 13 official cases now have
   parsed OpSpecs and non-generic sketches, including `t1/matmul_basic`,
   `t1/matmul_biasadd`, and `t2/moe_topk_softmax`.
2. Keep the local initial-round Step 3 materials ready for manual PR/email
   submission, as described in `docs/tasks/initial_submission_step3.md`.
3. Review `docs/project_book_full_zh.md` only for final human wording or
   team-specific edits before submission.
4. Use `scripts/export_project_book.py` to regenerate the project book for
   email after the manual PR link exists.
5. Use `scripts/prepare_gitlink_package.py`,
   `docs/submission/package_readme_zh.md`, and
   `docs/submission_package_readme.md` to regenerate the GitLink PR package.
6. Submission-facing docs now consistently claim `13/13` Lite OpSpec coverage,
   priority-subset executable Pass@4 evidence, and live AI generation gated
   only by model/API configuration.
7. Manual external action: open the GitLink PR with
   `docs/submission/gitlink_pr_title.txt` and
   `docs/submission/gitlink_pr_body.md`.
8. Manual external action: send the updated project book with
   `docs/submission/project_book_email_zh.md`, then record the email
   date/status.
9. Deterministic reference/replay evaluation is complete for the current
   no-key baseline: result import preserves multi-output details when present,
   `scripts/run_replay_regression.py` guards the expected updated AKG commit,
   and `scripts/audit_pre_key_readiness.py --json` gates on exact `13/13`
   Lite OpSpec support.
10. Provider-independent infrastructure is complete for the first pass:
   OpSpec/Sketch validators, prompt context snapshot tests, package hygiene
   tests, replay guard tests, multi-output result-comparison preservation, and
   strengthened Skill Library entries are in place.
11. Keep `replay` as the deterministic CI/regression provider; use
   `scripts/run_replay_regression.py` for the current updated-AKG replay
   `t1/sigmoid_scale_sum` regression baseline.
12. Use `docs/tasks/pre_key_objective_audit.md` as the current pre-key
   objective audit: deterministic replay/import, T2/T3 OpSpecs, priority
   sketches, and priority manual seeds are complete; full AKG Agents runner
   parity remains blocked on `standard` model configuration.
13. Use `scripts/audit_pre_key_readiness.py --json` as the machine-checkable
   pre-key readiness gate. Current expected status without a key is
   `pre_key_deterministic_complete_provider_config_missing`; after credentials
   exist, run it with `--require-standard-config`. Add `--check-ascend-ssh`
   when diagnosing whether the local machine currently has BatchMode SSH access
   to the Ascend worker.
14. Use completed manual Pass@4 cycles as retrieval examples:
   updated-AKG `sigmoid_scale_sum_v2` for a positive reduction trajectory and
   updated-AKG `fused_silu_and_mul_v3` for a correctness-positive but
   performance-negative fused-elementwise trajectory, and updated-AKG
   `softmax_v4` for a correctness-positive but still-slower rowwise softmax
   trajectory, and `add_rmsnorm_cast_v2` for a positive T2 normalization
   trajectory, and `rope_v4`/`rope_v1` for RoPE intrinsic-vs-Triton parity,
   and `add_rmsnorm_quant_v2`-`v4` as quantized-normalization boundary
   failures under the exact int8 gate, and `layernorm_gated_v4` for a positive
   T3 fp16 gated-RMSNorm row-grouping trajectory.
15. Use
   `experiments/reports/2026-07-09-remaining-reference-preeval-updated-akg.yaml`
   as the deterministic reference baseline for all remaining AKG Bench Lite
   cases under the current runner.
16. Preserve CANN's `PYTHONPATH` entries in all Ascend benchmark commands;
   prepend the repository path with
   `export PYTHONPATH=/data/KernelForge-Agent:${PYTHONPATH:-}` instead of
   replacing `PYTHONPATH`.
17. Continue rerunning key Pass@4 reports under AKG commit
   `47aa428fcdc8c68f78d331dc578bc6c74fb9d91d` before final result claims;
   manual `t1/sigmoid_scale_sum`, `t1/softmax`, `t1/fused_silu_and_mul`, and
   replay `t1/sigmoid_scale_sum` are done.
18. The priority T2/T3 pre-key manual seeds are complete; next pre-key
   development should focus on opening the Step 3 GitLink PR/email, runner
   comparison/provider setup, or a non-priority operator such as
   `t2/moe_topk_softmax` if more manual evidence is needed.
19. Configure an AKG Agents `standard` model level, verify it with
   `scripts/check_akg_agents_model_config.py --level standard`, then rerun
   `scripts/run_akg_agents_full_comparison.sh` for full runner-path comparison
   and use `scripts/compare_runner_results.py` to compare the full-mode JSON
   with the standalone replay Pass@4 report.
20. Use `scripts/run_akg_agents_verifier_probe.py` only as a no-key
   verifier-only smoke path for existing candidates; from the local machine use
   `scripts/run_ascend_verifier_probe.sh` after reopening the Ascend
   `ControlMaster` session. Do not treat this as final runner parity because
   it does not produce AKG Agents full-mode Pass@4 or leaderboard scores.
21. Record model/provider, prompt version, retrieved skills, and candidate index
   for every generated experiment.
22. Use `scripts/validate_opspecs.py --json` as the deterministic OpSpec/Sketch
   validation gate before editing parsed benchmark specs, prompt templates, or
   generated experiment context.
23. Promote reusable generation, repair, or tuning lessons into `skills/`.

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
