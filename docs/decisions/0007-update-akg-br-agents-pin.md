# 0007: Update AKG br_agents Benchmark Pin

Date: 2026-07-09

## Status

Accepted

## Context

We were told the previous AKG `br_agents` branch revision might have benchmark
runner problems. The project had pinned `third_party/akg` to:

```text
bea77cb38db5713056a7e06e5e8a0cbe9d26954b
```

The latest `br_agents` head checked from AtomGit on 2026-07-09 is:

```text
47aa428fcdc8c68f78d331dc578bc6c74fb9d91d
```

## Decision

Update the AKG submodule pin and setup script to
`47aa428fcdc8c68f78d331dc578bc6c74fb9d91d`.

The benchmark case files under
`akg_agents/benchmark/akg_kernels_bench_lite` did not change between the old
and new pins. The changed files are:

```text
RUNNER.md
tools/run_bench.py
```

The standalone runner now:

- regenerates independent seeded inputs for reference and solution paths
- runs three correctness trials
- rejects NaN/Inf in reference output, solution output, or computed diffs
- keeps the strict `max_abs_diff <= atol and max_rel_diff <= rtol` correctness
  gate

## Consequences

- Existing historical experiment records still correctly identify the old
  source commit used for those runs.
- Future benchmark results should use the new source commit.
- Previous results may need rerunning when comparing final numbers because the
  correctness protocol changed even though operator files did not.
- Candidate code with in-place input mutation is less likely to contaminate
  reference comparison under the updated runner.

## Follow-Up

- Rerun key Pass@4 and reference pre-evaluation reports on the Ascend worker
  with the updated runner before final result claims.
- Keep `benchmarks/raw/akg_kernels_bench_lite_registry.yaml` unchanged unless
  future branch updates change case files.
