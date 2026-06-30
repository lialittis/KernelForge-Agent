# Third Party Dependencies

This directory contains Git submodules and external source checkouts.

## AKG Benchmark

The official benchmark is managed as a Git submodule:

```text
third_party/akg
```

The submodule points to:

```text
repo:   https://atomgit.com/mindspore/akg.git
branch: br_agents
commit: bea77cb38db5713056a7e06e5e8a0cbe9d26954b
```

Only the benchmark path is needed locally:

```text
third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite
```

Run this from the repository root after cloning:

```bash
bash scripts/setup_benchmark_submodule.sh
```

