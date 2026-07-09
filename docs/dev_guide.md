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

## SSH Automation Caveat

Some rented Ascend containers are reached through a provider SSH gateway rather
than a normal direct `sshd`. In the current worker, the server banner is
`SSHPiper` and fresh connections advertise only password authentication:

```text
Authentications that can continue: password
```

In that setup, adding a public key to `~/.ssh/authorized_keys` inside the
container is not enough to enable key auth from the local machine. A second
command may appear passwordless only because it is reusing an existing
`ControlMaster` connection.

Use this key-only diagnostic from the local machine:

```bash
ssh -S none \
  -o BatchMode=yes \
  -o PasswordAuthentication=no \
  -o KbdInteractiveAuthentication=no \
  -o PreferredAuthentications=publickey \
  -vvv ascend-kf true
```

If that reports password-only authentication, use one of these workflows:

- Configure the public key at the provider/gateway layer if the platform
  supports it.
- Start a persistent master connection manually after one password login:

```bash
ssh -MNf ascend-kf
ssh -O check ascend-kf
```

Keep `ControlPersist` long enough for the experiment session, for example:

```sshconfig
Host ascend-kf
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
```

Automation should use `BatchMode=yes` so it fails instead of hanging on a
password prompt:

```bash
ssh -o BatchMode=yes ascend-kf 'cd /data/KernelForge-Agent && git rev-parse HEAD'
```

Avoid scripting password entry into a pseudo-terminal. If input is sent before
the actual password prompt disables echo, the password can appear in terminal
logs. Prefer opening the master connection manually, then let agents use only
`BatchMode=yes` commands over that existing connection.

Fast path for reconstructing a known-good worker environment:

```bash
bash scripts/bootstrap_ascend_env.sh
```

The bootstrap script:

- initializes the pinned AKG benchmark submodule
- creates `/data/venvs/kf-triton-ascend` with `--system-site-packages`
- sources `/usr/local/Ascend/ascend-toolkit/set_env.sh`
- installs `triton-ascend==3.2.0` plus the Python dependencies needed by the
  Triton-Ascend backend
- runs `npu-smi info`, a `torch_npu` GELU smoke test, and
  `scripts/diagnose_triton_ascend.py`

The standalone benchmark runner is enough for deterministic replay/manual
submissions. To probe the AKG Agents orchestration runner
`run_torch_bench_lite.py`, install the additional LangChain/LangGraph runtime
stack after the main Ascend bootstrap:

```bash
bash scripts/setup_akg_agents_runner_deps.sh
```

That script expands the AKG sparse checkout to include `akg_agents/`, installs
the runner imports into `/data/venvs/kf-triton-ascend`, and prints the `--help`
probe command. The runner can import and emit correctness-mode JSON without an
API key, but successful Agent attempts require an AKG Agents `standard` model
level configured via `AKG_AGENTS_STANDARD_*`, `~/.akg/settings.json`, or
`.akg/settings*.json`.

Before launching the full AKG Agents runner, check that the `standard` level is
complete. This command does not call the provider and masks any key it prints:

```bash
python scripts/check_akg_agents_model_config.py --level standard
```

Environment-variable setup:

```bash
export AKG_AGENTS_STANDARD_BASE_URL="https://api.openai.com/v1"
export AKG_AGENTS_STANDARD_API_KEY="<redacted>"
export AKG_AGENTS_STANDARD_MODEL_NAME="<model-name>"
```

Local file setup, if environment variables are inconvenient:

```json
{
  "models": {
    "standard": {
      "base_url": "https://api.openai.com/v1",
      "api_key": "<redacted>",
      "model_name": "<model-name>",
      "provider_type": "openai"
    }
  }
}
```

Save that as `.akg/settings.local.json` in the project or
`~/.akg/settings.json` on the worker. The project-local
`.akg/settings.local.json` path is ignored by Git.

After a `standard` model level is configured, use the project wrapper for the
full runner-path comparison:

```bash
CASE=sigmoid_scale_sum \
OUTPUT=outputs/results/akg_agents_full_sigmoid_scale_sum_$(date +%Y_%m_%d).json \
bash scripts/run_akg_agents_full_comparison.sh
```

The wrapper sources CANN and the known venv when present, preserves CANN's
Python paths, adds the AKG Agents Python path, and fails early with a clear
message if no `standard` model configuration is visible.

After the full-mode runner JSON exists, compare it with standalone evidence:

```bash
python scripts/compare_runner_results.py \
  --standalone-report experiments/reports/2026-07-09-replay-sigmoid-scale-sum-pass4-updated-akg.yaml \
  --akg-agents-json outputs/results/akg_agents_full_sigmoid_scale_sum_$(date +%Y_%m_%d).json \
  --case t1/sigmoid_scale_sum \
  --akg-log-dir outputs/akg_agents_logs \
  --output experiments/reports/runner-comparison-after-standard-config.yaml
```

If no model key is available yet, a narrower AKG Agents verifier probe can
exercise the LangGraph verifier/project-generation/logging path on an existing
candidate. This is useful as a pre-key smoke test, but it is not a substitute
for the full `run_torch_bench_lite.py --mode full` comparison because it does
not generate Pass@4 attempts or leaderboard scores:

```bash
python scripts/run_akg_agents_verifier_probe.py \
  --case t1/sigmoid_scale_sum \
  --candidate kernel_forge/candidates/sigmoid_scale_sum_v2.py \
  --task-type profile \
  --output outputs/results/akg_agents_verifier_probe_sigmoid_v2.json
```

The output JSON can be passed to `scripts/compare_runner_results.py`; the
comparator will classify it as `verifier_only_probe` and keep standalone
`tools/run_bench.py` authoritative until full-mode AKG Agents results exist.

From the local machine, after the Ascend SSH `ControlMaster` session is open,
the same probe plus comparison can be launched through the BatchMode wrapper:

```bash
bash scripts/run_ascend_verifier_probe.sh
```

Use `CHECK_ONLY=1 bash scripts/run_ascend_verifier_probe.sh` to inspect the
remote paths and commands without connecting.

Use environment variables if the rented machine differs:

```bash
ASCEND_VENV=/data/venvs/kf-triton-ascend \
ASCEND_SET_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh \
TRITON_ASCEND_VERSION=3.2.0 \
bash scripts/bootstrap_ascend_env.sh
```

If packages are already installed and you only want the benchmark and
diagnostics:

```bash
SKIP_PIP_INSTALL=1 bash scripts/bootstrap_ascend_env.sh
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

## Extracting Benchmark Metadata

The current parser milestone supports the first T1 non-matmul subset:

```text
t1/gelu
t1/fused_silu_and_mul
t1/sigmoid_scale_sum
t1/softmax
```

Generate the full benchmark case registry:

```bash
python scripts/scan_benchmark_cases.py \
  --output benchmarks/raw/akg_kernels_bench_lite_registry.yaml \
  --repo-root .
```

Generate OpSpec and Sketch YAML for all supported cases:

```bash
python scripts/extract_opspec_batch.py \
  --output-dir benchmarks/parsed \
  --repo-root .
```

Generate a single OpSpec YAML from an official case file:

```bash
python scripts/extract_opspec.py \
  --case third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/t1/gelu.py \
  --experiment experiments/runs/2026-06-30-gelu-manual-baseline.yaml \
  --repo-root . \
  --output /tmp/t1_gelu.generated.yaml
```

The committed canonical OpSpecs are:

```text
benchmarks/parsed/t1_gelu.yaml
benchmarks/parsed/t1_fused_silu_and_mul.yaml
benchmarks/parsed/t1_sigmoid_scale_sum.yaml
benchmarks/parsed/t1_softmax.yaml
```

Do not overwrite committed OpSpecs automatically. Generate to a temporary path,
compare the result, then intentionally update `benchmarks/parsed/` if the
schema or extracted facts change.

## First Triton-Ascend GELU Candidate

The first custom candidate source is tracked at:

```text
kernel_forge/candidates/gelu_triton_v1.py
```

It implements a Triton-style 1D GELU kernel for contiguous NPU tensors. If
Triton, `tl.erf`, or the NPU backend launch path is unavailable, it falls back
to `torch.nn.functional.gelu` so correctness experiments can still run and
report the environment gap.

Generate the official submission layout:

```bash
bash scripts/create_gelu_triton_submission.sh
```

For new non-GELU candidates, prefer the generic materializer:

```bash
python scripts/create_submission.py \
  --team <team_name> \
  --candidate <candidate_label> \
  --case t1/softmax=kernel_forge/candidates/<candidate_file>.py
```

This writes:

```text
outputs/submissions/gelu_triton_v1/
  gelu_triton_v1/
    meta.json
    t1/
      gelu.py
```

Probe the backend path on the Ascend worker:

```bash
python scripts/probe_gelu_triton_backend.py
```

The probe prints JSON with `last_backend`, `last_error`, and correctness deltas.
Use `--shape 32 512 1024` to probe the exact benchmark shape.

If the probe reports `torch_fallback_after_error` with
`RuntimeError: 0 active drivers ([]). There should only be one.`, Triton
imported but no usable backend driver is registered. Diagnose the environment:

```bash
python scripts/diagnose_triton_ascend.py
```

If the diagnosis shows `triton-ascend: null` and
`ModuleNotFoundError: No module named 'triton_ascend'`, install the Ascend
backend package and recheck:

```bash
python -m pip install triton-ascend==3.2.0
python scripts/diagnose_triton_ascend.py
python scripts/probe_gelu_triton_backend.py
```

For the reconstructed worker environment, prefer the pinned bootstrap command
above. It installs the extra packages that the Triton-Ascend backend needed in
the previous Ascend worker:

```text
wheel pybind11 ninja cmake attrs==24.2.0 numpy==1.26.4 scipy==1.13.1
decorator==5.1.1 psutil==6.0.0 pyyaml
```

Do not treat benchmark speedup as custom-kernel speedup until the probe reports:

```text
last_backend: triton
```

If `gelu_triton_v1` launches Triton but fails the official benchmark with a
small absolute error and large relative error, test the erfc-form v2 candidate:

```bash
bash scripts/create_gelu_triton_v2_submission.sh
python scripts/probe_gelu_triton_backend.py \
  --candidate outputs/submissions/gelu_triton_v2/gelu_triton_v2/t1/gelu.py \
  --shape 32 512 1024
python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \
  outputs/submissions/gelu_triton_v2 \
  --team gelu_triton_v2 \
  --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite \
  --output outputs/results/gelu_triton_v2_triton_backend \
  --warmup 10 \
  --iterations 100 \
  --num-trials 3
```

If v2 falls back because `tl.erfc` is unavailable, test the hybrid v3 candidate:

```bash
bash scripts/create_gelu_triton_v3_submission.sh
python scripts/probe_gelu_triton_backend.py \
  --candidate outputs/submissions/gelu_triton_v3/gelu_triton_v3/t1/gelu.py \
  --shape 32 512 1024
python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \
  outputs/submissions/gelu_triton_v3 \
  --team gelu_triton_v3 \
  --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite \
  --output outputs/results/gelu_triton_v3_triton_backend \
  --warmup 10 \
  --iterations 100 \
  --num-trials 3
```

If v3 still fails relative error, test v4. It widens the exact tail repair to
`x < -2.1` and repairs only indexed tail values:

```bash
bash scripts/create_gelu_triton_v4_submission.sh
python scripts/probe_gelu_triton_backend.py \
  --candidate outputs/submissions/gelu_triton_v4/gelu_triton_v4/t1/gelu.py \
  --shape 32 512 1024
python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \
  outputs/submissions/gelu_triton_v4 \
  --team gelu_triton_v4 \
  --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite \
  --output outputs/results/gelu_triton_v4_triton_backend \
  --warmup 10 \
  --iterations 100 \
  --num-trials 3
```

If v4 passes but is too slow, test v5. It keeps the whole computation inside one
Triton kernel by using an erfc tail approximation for `x < -2.1`:

```bash
bash scripts/create_gelu_triton_v5_submission.sh
python scripts/probe_gelu_triton_backend.py \
  --candidate outputs/submissions/gelu_triton_v5/gelu_triton_v5/t1/gelu.py \
  --shape 32 512 1024
python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \
  outputs/submissions/gelu_triton_v5 \
  --team gelu_triton_v5 \
  --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite \
  --output outputs/results/gelu_triton_v5_triton_backend \
  --warmup 10 \
  --iterations 100 \
  --num-trials 3
```

For any failed GELU candidate, locate the worst input value:

```bash
python scripts/analyze_gelu_candidate_error.py \
  --candidate outputs/submissions/gelu_triton_v5/gelu_triton_v5/t1/gelu.py \
  --shape 32 512 1024
```

If the analyzer shows the NPU reference follows tanh-approximate GELU rather
than erf-exact GELU, test v6:

```bash
bash scripts/create_gelu_triton_v6_submission.sh
python scripts/probe_gelu_triton_backend.py \
  --candidate outputs/submissions/gelu_triton_v6/gelu_triton_v6/t1/gelu.py \
  --shape 32 512 1024
python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \
  outputs/submissions/gelu_triton_v6 \
  --team gelu_triton_v6 \
  --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite \
  --output outputs/results/gelu_triton_v6_triton_backend \
  --warmup 10 \
  --iterations 100 \
  --num-trials 3
```

If v6 still fails relative error in the far negative tail, test v7. It uses the
stable sigmoid form of tanh-approximate GELU:

```bash
bash scripts/create_gelu_triton_v7_submission.sh
python scripts/probe_gelu_triton_backend.py \
  --candidate outputs/submissions/gelu_triton_v7/gelu_triton_v7/t1/gelu.py \
  --shape 32 512 1024
python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \
  outputs/submissions/gelu_triton_v7 \
  --team gelu_triton_v7 \
  --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite \
  --output outputs/results/gelu_triton_v7_triton_backend \
  --warmup 10 \
  --iterations 100 \
  --num-trials 3
```

If v7 passes correctness but is slower than the baseline, test v8. It keeps the
same math and raises the Triton block size from 1024 to 4096:

```bash
bash scripts/create_gelu_triton_v8_submission.sh
python scripts/probe_gelu_triton_backend.py \
  --candidate outputs/submissions/gelu_triton_v8/gelu_triton_v8/t1/gelu.py \
  --shape 32 512 1024
python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \
  outputs/submissions/gelu_triton_v8 \
  --team gelu_triton_v8 \
  --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite \
  --output outputs/results/gelu_triton_v8_triton_backend \
  --warmup 10 \
  --iterations 100 \
  --num-trials 3
```

Equivalent inline smoke check:

```bash
python - <<'PY'
import importlib.util
import torch
import torch_npu

path = "outputs/submissions/gelu_triton_v1/gelu_triton_v1/t1/gelu.py"
spec = importlib.util.spec_from_file_location("gelu_candidate", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.ModelNew().npu().eval()
x = torch.randn(1024, 1024).npu()
with torch.no_grad():
    y = model(x)
torch.npu.synchronize()

print("output:", y.device, y.dtype, tuple(y.shape))
print("last_backend:", getattr(model, "_last_backend", None))
print("last_error:", getattr(model, "_last_error", None))
PY
```

Then run the official benchmark:

```bash
python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \
  outputs/submissions/gelu_triton_v1 \
  --team gelu_triton_v1 \
  --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite \
  --output outputs/results/gelu_triton_v1 \
  --warmup 10 \
  --iterations 100 \
  --num-trials 3
```

If `last_backend` is `triton`, compare speedup against the manual baseline. If
it is a `torch_fallback_*` value, record the fallback reason and treat the run
as an environment or backend-support finding, not a kernel-performance result.

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

Prefer importing the official JSON mechanically:

```bash
python scripts/import_benchmark_result.py \
  --result-json /tmp/manual_baseline_gelu/manual_baseline.json \
  --experiment experiments/runs/YYYY-MM-DD-gelu-manual-baseline.yaml \
  --in-place
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
