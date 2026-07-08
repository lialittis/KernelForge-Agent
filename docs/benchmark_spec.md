# Benchmark Specification

This file is the project contract for official benchmark behavior. It should be
updated whenever the official benchmark branch or runner behavior changes.

For the requirement-by-requirement competition mapping, see
`docs/competition_alignment.md`.

## Official Source

- Repository: <https://atomgit.com/mindspore/akg.git>
- Web path:
  <https://atomgit.com/mindspore/akg/tree/br_agents/akg_agents/benchmark/akg_kernels_bench_lite>
- Branch: `br_agents`
- Inspected commit: `47aa428fcdc8c68f78d331dc578bc6c74fb9d91d`
- Benchmark path: `akg_agents/benchmark/akg_kernels_bench_lite`
- Local submodule path:
  `third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite`

Change note:

- The previous project pin was
  `bea77cb38db5713056a7e06e5e8a0cbe9d26954b`.
- The update to `47aa428fcdc8c68f78d331dc578bc6c74fb9d91d` changed
  `RUNNER.md` and `tools/run_bench.py`; the benchmark case files and scanned
  case registry are unchanged.
- The standalone runner now regenerates independent seeded inputs for
  reference and solution paths, runs three correctness trials, and fails on
  NaN/Inf in reference, solution, or diff tensors.

Recommended inspection command:

```bash
bash scripts/setup_benchmark_submodule.sh
```

## Target

- Target hardware: Ascend 910 NPU.
- Main implementation path: AKG Agents + Triton-Ascend.
- Benchmark dataset: `akg_kernels_bench_lite`.
- Required participant implementation: a Python module defining `ModelNew`.
- Reference implementation: each task file defines `Model`.
- Performance should be measured only after correctness passes.

## Directory Layout

The benchmark currently contains three tiers:

```text
akg_kernels_bench_lite/
  README.md
  RUNNER.md
  t1/
  t2/
  t3/
  tools/
    run_bench.py
    scoring.py
    validate_submission.py
    gen_leaderboard.py
```

Case list at inspected commit:

| Tier | File | Operator |
| --- | --- | --- |
| `t1` | `gelu.py` | GELU activation |
| `t1` | `fused_silu_and_mul.py` | SwiGLU via `torch_npu.npu_swiglu` |
| `t1` | `matmul_basic.py` | BF16 matmul, 32 x 8192 x 8192 |
| `t1` | `matmul_biasadd.py` | FP16 matmul plus bias add, 4096 x 4096 x 4096 |
| `t1` | `softmax.py` | Softmax |
| `t1` | `sigmoid_scale_sum.py` | Sigmoid plus scale plus sum fusion |
| `t2` | `rope.py` | RoPE via `torch_npu.npu_rotary_mul` |
| `t2` | `add_rmsnorm_cast.py` | Add plus RMSNorm plus cast |
| `t2` | `add_rmsnorm_quant.py` | Add plus RMSNorm plus int8 quantization |
| `t2` | `moe_topk_softmax.py` | MoE TopK softmax |
| `t3` | `causal_conv1d.py` | Causal Conv1D |
| `t3` | `decode_mla.py` | Paged MLA decode |
| `t3` | `layernorm_gated.py` | Gated LayerNorm |

## Case File Contract

Each official case file defines:

```python
class Model(nn.Module):
    """Golden/reference implementation."""

    def __init__(self, ...):
        ...

    def forward(self, ...):
        ...

def get_inputs():
    return [input1, input2, ...]

def get_init_inputs():
    return [param1, param2, ...]
```

Participant solutions define matching `ModelNew`:

```python
class ModelNew(nn.Module):
    def __init__(self, ...):
        ...

    def forward(self, ...):
        ...
```

`ModelNew.__init__` must accept the same arguments returned by
`get_init_inputs()`. `ModelNew.forward` must accept the same arguments returned
by `get_inputs()` and return outputs matching `Model.forward()`.

## Submission Layout

The standalone `tools/run_bench.py` and `tools/validate_submission.py` expect a
submission root containing one or more team directories:

```text
submissions/
  team_name/
    meta.json
    t1/
      gelu.py
      softmax.py
    t2/
      rope.py
```

`meta.json` must include:

```json
{
  "team_name": "team_name"
}
```

Each submitted case file must exist in the official benchmark registry and must
define `ModelNew`.

## Correctness Validation

The benchmark validates:

- output type: tensor or tensor list/tuple
- output count
- exact shape match
- numerical closeness
- absence of NaN/Inf in reference output, solution output, and computed diffs
- three seeded correctness trials per case in the standalone runner

Default tolerance:

```text
rtol = 1e-2
atol = 1e-2
```

The checker computes:

```text
max_abs_diff = max(abs(ref - sol))
max_rel_diff = max(abs(ref - sol) / (abs(ref) + 1e-8))
```

A case passes correctness only if:

```text
max_abs_diff <= atol and max_rel_diff <= rtol
```

The updated standalone runner uses strict AND semantics for those two
thresholds on every trial. It also regenerates reference and solution inputs
with the same seed instead of sharing one input object, which avoids in-place or
stateful behavior in one path contaminating the other path.

## Performance Measurement

`tools/run_bench.py` measures reference and solution latency after correctness
passes.

Defaults:

```text
warmup = 10
iterations = 100
num_trials = 3
timeout = 300 seconds
```

The reported solution latency uses median trial latency in milliseconds.

Speedup:

```text
speedup = baseline_time / solution_time
```

## Scoring

Correctness is a hard gate. Incorrect cases receive zero score.

Raw score:

- `speedup <= 0`: 0
- `speedup < 1.0`: `60 * speedup`
- `speedup == 1.0`: 60
- `speedup > 1.0`: increases from 60 toward 100
- `speedup >= 5.0`: capped at 100

Tier weights:

| Tier | Weight |
| --- | --- |
| `t1` | 1.0 |
| `t2` | 1.5 |
| `t3` | 2.0 |
| `t4` | 2.5 |
| `t5` | 3.0 |

Weighted score:

```text
weighted_score = raw_score * tier_weight
```

## Runner Entrypoints

Standalone submission runner:

```bash
python akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \
  <submissions_dir> \
  --team <team_name> \
  --bench-dir akg_agents/benchmark/akg_kernels_bench_lite \
  --output results/ \
  --rtol 1e-2 \
  --atol 1e-2 \
  --warmup 10 \
  --iterations 100 \
  --num-trials 3
```

Submission validator:

```bash
python akg_agents/benchmark/akg_kernels_bench_lite/tools/validate_submission.py \
  submission.tar.gz \
  --bench-dir akg_agents/benchmark/akg_kernels_bench_lite
```

AKG Agents runner described in `RUNNER.md`:

```text
akg_agents/examples/kernel_related/run_torch_bench_lite.py
```

It supports modes `correctness`, `performance`, and `full`, with backend
selection among `cpu`, `gpu`, `npu`, and `all`. Its CLI supports `--pass-n`; the
documented default is 3, while this project will use `--pass-n 4` for Pass@4
experiments unless the official scoring instruction says otherwise.

## Initial Task Classification

The generated registry is tracked at:

```text
benchmarks/raw/akg_kernels_bench_lite_registry.yaml
```

| Case | Primary category | Notes |
| --- | --- | --- |
| `t1/gelu.py` | `elementwise` | Best first OpSpec/Sketch target |
| `t1/fused_silu_and_mul.py` | `elementwise` | Uses `torch_npu`; likely NPU-only |
| `t1/softmax.py` | `reduction` | Reduction-like normalization |
| `t1/sigmoid_scale_sum.py` | `reduction` | Fusion plus sum |
| `t1/matmul_basic.py` | `matmul_like` | Large BF16 matmul |
| `t1/matmul_biasadd.py` | `matmul_like` | Large FP16 matmul plus bias |
| `t2/rope.py` | `transpose_layout` | Rotary position embedding, NPU intrinsic reference |
| `t2/add_rmsnorm_cast.py` | `normalization` | RMSNorm plus cast |
| `t2/add_rmsnorm_quant.py` | `normalization` | RMSNorm plus quantization |
| `t2/moe_topk_softmax.py` | `reduction` | TopK plus softmax |
| `t3/causal_conv1d.py` | `unknown` | Complex sequence kernel |
| `t3/decode_mla.py` | `matmul_like` | Complex decode attention path |
| `t3/layernorm_gated.py` | `normalization` | LayerNorm plus gating |

## First Reproduction Checklist

1. Clone or sparse-checkout the official benchmark source.
2. Record exact source location and commit in experiment metadata.
3. Set up Ascend 910 environment.
4. Run `t1/gelu.py` reference baseline.
5. Create a trivial `ModelNew` implementation that wraps the reference behavior
   or a safe PyTorch equivalent.
6. Run `tools/run_bench.py` on the single-case submission.
7. Save command output summary in `experiments/runs/`.
8. Create `benchmarks/parsed/t1_gelu.yaml` as the first OpSpec example.

## Open Items

- Confirm exact CANN, MindSpore, AKG, Triton-Ascend, Python, and compiler
  versions on the provided Ascend 910 environment.
- Confirm whether competition evaluation uses the standalone `tools/run_bench.py`
  submission runner, the AKG Agents runner, or both.
- Confirm whether official Pass@N should be Pass@3 from runner defaults or
  Pass@4 from the project proposal/competition narrative.
- Confirm whether it is acceptable to vendor small official sample files under
  `benchmarks/raw/`, or whether the repo should store only source URLs and
  commits.
