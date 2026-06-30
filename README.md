# SketchSkill-AKG

SketchSkill-AKG is a skill-driven operator generation and hardware-feedback
optimization system for Ascend 910 NPU benchmarks. This repository was
originally initialized as KernelForge-Agent; the active project design now
follows the SketchSkill-AKG proposal in `SketchSkill_AKG_项目书基础版.pdf`.

Competition track:

https://www.gitlink.org.cn/competitions/track2_2026MindSpore

## Purpose

The project targets AI/Agent-based NPU operator generation. Instead of asking a
large model to write a low-level kernel in one shot, the system decomposes
operator development into:

```text
Benchmark task
-> OpSpec extraction
-> NPU-aware Operator Sketch planning
-> Skill retrieval
-> backend code generation
-> compile and correctness verification
-> repair loop
-> Ascend 910 profiling and performance search
-> best-kernel selection
-> Skill Library write-back
```

The initial main path is **AKG Agents + Triton-Ascend** for fast end-to-end
benchmark coverage. Enhancement paths include TileLang-Ascend, Ascend C, legacy
CCE/TBE knowledge, MLIR/AscendNPU IR alignment, and CUDA/Triton-to-Ascend
knowledge migration.

## Core Methods

- NPU-aware Operator Sketch as an intermediate representation between operator
  semantics and backend code.
- Operator-pattern Skill Library organized by elementwise, broadcast,
  reduction, transpose/layout, normalization, matmul-like, debugging,
  performance, benchmark evaluation, and CUDA-to-Ascend migration.
- Correctness loop using compiler logs, runtime errors, numerical comparison,
  shape/dtype failures, and Pass@N metrics.
- Performance loop using real Ascend 910 profiling, latency/throughput
  measurement, tile/core/vector/unroll/double-buffer search, and baseline
  comparison.
- Bad-to-good trajectory capture so successful fixes and optimizations become
  reusable skills.

## Project Memory

This repository is the shared source of truth for multi-machine and multi-agent
work.

- Agent guide: `AGENTS.md`
- Architecture: `docs/architecture.md`
- Workflow: `docs/project_workflow.md`
- Dev guide: `docs/dev_guide.md`
- Benchmark spec: `docs/benchmark_spec.md`
- Current status: `docs/status.md`
- Roadmap: `docs/roadmap.md`
- Active tasks: `tasks/active.md`
- Experiment rules: `experiments/README.md`
- Skill Library: `skills/`
- Decisions: `docs/decisions/`

If chat history and repository files disagree, trust the repository files.

## Benchmark Setup

The official AKG benchmark is managed as a Git submodule, pinned to the
inspected `br_agents` commit:

```text
repo:   https://atomgit.com/mindspore/akg.git
branch: br_agents
commit: bea77cb38db5713056a7e06e5e8a0cbe9d26954b
path:   akg_agents/benchmark/akg_kernels_bench_lite
```

After cloning this repository, initialize the benchmark with:

```bash
bash scripts/setup_benchmark_submodule.sh
```

The benchmark will be available at:

```text
third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite
```

The submodule keeps our Git history small: this repository stores only a
pointer to the AKG commit. The setup script enables sparse checkout so the local
working tree only materializes the benchmark path we need.

For a normal recursive clone, users can also run:

```bash
git submodule update --init --depth 1 --filter=blob:none third_party/akg
bash scripts/setup_benchmark_submodule.sh
```
