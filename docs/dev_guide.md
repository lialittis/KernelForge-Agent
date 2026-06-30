# Development Guide

This guide describes the recommended workflow when development happens on one
machine and Ascend NPU experiments run on another machine.

## Roles

Use two roles:

```text
local dev machine
  -> edit code, docs, skills, schemas, prompts, experiment metadata
  -> commit project changes

Ascend worker machine
  -> pull latest code
  -> initialize benchmark submodule
  -> run correctness/performance experiments
  -> produce logs, result JSON, profiles, and summaries
```

The local machine is the source of truth for design and implementation. The
Ascend machine is the source of truth for hardware execution results.

## What Goes In Git

Commit these:

- source code
- documentation
- scripts
- OpSpec files under `benchmarks/parsed/`
- concise experiment records under `experiments/runs/*.yaml`
- reusable lessons in `skills/*/SKILL.md`
- decision records under `docs/decisions/`

Do not commit these by default:

- generated candidate directories
- raw benchmark output folders
- large logs
- profiling dumps
- temporary submissions
- virtual environments
- downloaded package caches

`outputs/` is ignored by Git for this reason.

## Local Dev Workflow

On the local development machine:

```bash
git pull --rebase
bash scripts/setup_benchmark_submodule.sh
```

Make implementation or documentation changes, then commit them:

```bash
git status
git diff
git add <changed-files>
git commit -m "Describe the change"
git push
```

If the work changes experiment expectations, update:

- `docs/status.md`
- `tasks/active.md`
- relevant `skills/*/SKILL.md`
- `experiments/runs/*.yaml` if an experiment was completed

## Ascend Worker Setup

On the Ascend machine:

```bash
git clone <repo-url>
cd KernelForge-Agent
git submodule update --init --depth 1 --filter=blob:none third_party/akg
bash scripts/setup_benchmark_submodule.sh
```

Source the Ascend runtime environment:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

If the path differs, locate it:

```bash
find /usr/local/Ascend -maxdepth 3 -name set_env.sh
```

Verify the NPU and Python stack:

```bash
npu-smi info

python - <<'PY'
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available())
print("device count:", torch.npu.device_count())
PY
```

## First Manual GELU Run

The first target is `t1/gelu.py`.

Expected submission layout:

```text
outputs/submissions/manual_baseline/
  manual_baseline/
    meta.json
    t1/
      gelu.py
```

Create it from tracked project files:

```bash
bash scripts/create_manual_gelu_submission.sh
```

Run the official benchmark:

```bash
python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \
  outputs/submissions/manual_baseline \
  --team manual_baseline \
  --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite \
  --output outputs/results/manual_baseline_gelu \
  --warmup 10 \
  --iterations 100 \
  --num-trials 3
```

The script writes into `outputs/`, which is ignored by Git. That is intentional:
the submission is a runtime artifact, while the script is the reproducible
source.

## Extracting GELU OpSpec

The current parser milestone supports `t1/gelu.py`.

Generate an OpSpec YAML from the official case file:

```bash
python scripts/extract_opspec.py \
  --case third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/t1/gelu.py \
  --experiment experiments/runs/2026-06-30-gelu-manual-baseline.yaml \
  --repo-root . \
  --output /tmp/t1_gelu.generated.yaml
```

The committed canonical OpSpec is:

```text
benchmarks/parsed/t1_gelu.yaml
```

Do not overwrite committed OpSpecs automatically. Generate to a temporary path,
compare the result, then intentionally update `benchmarks/parsed/` if the
schema or extracted facts change.

## Copying Results Back

Copy back only the useful result files, not the whole `outputs/` tree.

Example:

```bash
rsync -avz <ascend-host>:/path/to/KernelForge-Agent/outputs/results/manual_baseline_gelu/ \
  /tmp/manual_baseline_gelu/
```

Then summarize the run locally in:

```text
experiments/runs/YYYY-MM-DD-gelu-manual-baseline.yaml
```

The experiment record should include:

- benchmark commit
- device name
- CANN version
- Python version
- torch and torch_npu versions
- command used
- correctness status
- baseline latency
- solution latency
- speedup
- failure category if any
- paths to any retained artifacts

## Experiment Handoff

When stopping work on the Ascend machine, write a short handoff and copy it
back to the local machine:

```text
date:
machine:
branch:
benchmark commit:
command:
status:
important output:
artifact paths:
next action:
```

Then update `docs/status.md` from the local machine and commit the project
metadata.

## Branching Rule

Use branch-per-task if multiple machines or agents are active:

```bash
git checkout -b task/gelu-baseline
```

Before running experiments on the worker, make sure it is on the intended
commit:

```bash
git rev-parse HEAD
git status --short
git submodule status
```

The experiment record should include that commit hash.
