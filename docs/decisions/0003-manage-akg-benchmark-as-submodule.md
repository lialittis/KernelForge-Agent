# 0003: Manage AKG Benchmark As A Pinned Sparse Submodule

Date: 2026-06-30

## Status

Accepted

## Context

The official benchmark lives inside the external AKG repository:

```text
https://atomgit.com/mindspore/akg.git
branch: br_agents
path: akg_agents/benchmark/akg_kernels_bench_lite
```

The project needs reproducible access to that benchmark across machines, but
vendoring third-party files directly into this repository would blur provenance
and make future benchmark updates harder.

## Decision

Manage the AKG repository as a Git submodule at:

```text
third_party/akg
```

Pin the submodule to:

```text
bea77cb38db5713056a7e06e5e8a0cbe9d26954b
```

Use sparse checkout so local working trees materialize only:

```text
third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite
```

Provide `scripts/setup_benchmark_submodule.sh` as the standard setup command
for new machines.

## Consequences

Positive:

- The repository stores only a submodule pointer, not vendored benchmark files.
- The benchmark source and commit are explicit.
- New machines can reproduce the same benchmark checkout.
- The local working tree stays small through sparse checkout.

Negative:

- Users must initialize submodules after cloning.
- Sparse-checkout configuration is local Git state, so the setup script must be
  run on each new clone.
- The submodule points to the whole AKG repository even though only one
  benchmark subdirectory is checked out locally.

## Follow-Up

- Use the submodule benchmark path for the first `t1/gelu.py` reproduction.
- Update the pinned commit only through a separate decision or explicit status
  note.

