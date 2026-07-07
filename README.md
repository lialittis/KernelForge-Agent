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

The final product is the reusable **SketchSkill-AKG agent system** plus
benchmark evidence. Individual optimized kernels are generated outputs of that
system, not the whole project.

The initial main path is **AKG Agents + Triton-Ascend** for fast end-to-end
benchmark coverage. Enhancement paths include TileLang-Ascend, Ascend C, legacy
CCE/TBE knowledge, MLIR/AscendNPU IR alignment, and CUDA/Triton-to-Ascend
knowledge migration.

The default generation and repair backend should be a strong coding/reasoning
LLM, such as a Codex/GPT-5-class agent, but the provider must stay replaceable
so local/open code models can be compared later under the same OpSpec, Sketch,
skill, benchmark, and result-import pipeline.

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
- Competition alignment: `docs/competition_alignment.md`
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

On a fresh Ascend worker, use the bootstrap script to recreate the benchmark,
venv, Triton-Ascend backend, and diagnostics setup used by current
experiments:

```bash
bash scripts/bootstrap_ascend_env.sh
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

## Benchmark Metadata Workflow

The first supported automation subset is T1 non-matmul:

```text
t1/gelu
t1/fused_silu_and_mul
t1/sigmoid_scale_sum
t1/softmax
```

Scan the official benchmark into a case registry:

```bash
python scripts/scan_benchmark_cases.py \
  --output benchmarks/raw/akg_kernels_bench_lite_registry.yaml \
  --repo-root .
```

Generate parsed OpSpecs and Sketches for supported cases:

```bash
python scripts/extract_opspec_batch.py \
  --output-dir benchmarks/parsed \
  --repo-root .
```

Create an official submission layout from any tracked candidate source:

```bash
python scripts/create_submission.py \
  --team gelu_triton_v13 \
  --candidate gelu_triton_v13 \
  --case t1/gelu=kernel_forge/candidates/gelu_triton_v13.py
```

After an Ascend worker run, import the official benchmark JSON into either a
summary YAML or an existing experiment record:

```bash
python scripts/import_benchmark_result.py \
  --result-json outputs/results/gelu_triton_v13_triton_backend/gelu_triton_v13.json \
  --experiment experiments/runs/2026-07-03-gelu-triton-v13-planned.yaml \
  --in-place
```
