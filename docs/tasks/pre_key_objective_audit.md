# Pre-Key Development Objective Audit

Date: 2026-07-09
Agent: Codex
Branch: main

## Scope

This audits the active pre-key development objective:

- compare standalone `tools/run_bench.py` and AKG Agents
  `run_torch_bench_lite.py`
- automate generated-result import
- extend T2/T3 OpSpec parsing
- prepare sketches for priority remaining operators
- build deterministic replay regression
- optionally add manual Pass@4 seeds before live AI provider use

## Audit Result

The pre-key deterministic development work is substantially complete. The one
remaining objective item that is not fully completable before model/API
configuration is the full AKG Agents runner comparison, because
`run_torch_bench_lite.py --mode full` requires an AKG Agents `standard` model
level.

Current pre-key evidence should trust standalone
`akg_kernels_bench_lite/tools/run_bench.py` as the authoritative scorer for
manual and replay submissions. AKG Agents runner evidence is limited to import,
help, environment-check, case-discovery, and JSON-schema probes until model
configuration exists.

## Requirement Status

### Runner Path Comparison

Status: partially complete, blocked for full parity comparison.

Evidence:

- Report:
  `experiments/reports/2026-07-09-runner-path-comparison-sigmoid-scale-sum.yaml`
- Run record:
  `experiments/runs/2026-07-09-runner-path-comparison-sigmoid-scale-sum.yaml`
- Standalone runner completed updated-AKG replay `t1/sigmoid_scale_sum`
  Pass@4:
  - Pass@1 true
  - Pass@4 true
  - best replay candidate `sigmoid_scale_sum_replay_v2`
  - speedup `1.9794x`
  - weighted score `69.79`
- AKG Agents runner dependency setup completed and `--help` works.
- AKG Agents correctness probe reached worker registration, environment check,
  case discovery, and JSON output.
- AKG Agents runner failed at model creation:
  `模型级别 'standard' 未配置，无法创建 LLM 客户端`.

Why not complete:

- The AKG Agents runner treats submission directories as output from generated
  attempts, not as input for existing replay/manual candidates.
- A comparable Pass@4/performance run requires live model configuration through
  `AKG_AGENTS_STANDARD_*`, `~/.akg/settings.json`, or `.akg/settings*.json`.

Next unblock:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /data/venvs/kf-triton-ascend/bin/activate
export PYTHONPATH=/data/KernelForge-Agent:/data/KernelForge-Agent/third_party/akg/akg_agents/python:${PYTHONPATH:-}
python third_party/akg/akg_agents/examples/kernel_related/run_torch_bench_lite.py \
  --backend npu \
  --mode full \
  --cases sigmoid_scale_sum \
  --pass-n 4 \
  --max-concurrent 1 \
  --output outputs/results/akg_agents_full_sigmoid_YYYY_MM_DD.json
```

### Generated-Result Import Automation

Status: complete for replay/provider-generated candidates.

Evidence:

- `scripts/run_replay_regression.py` performs generation, official benchmark,
  backend probes, Pass@N summary, report writing, and generated-experiment
  update.
- `kernel_forge/experiments/passn.py` provides:
  - `summarize_passn`
  - `enrich_passn_summary`
  - `apply_passn_to_generated_experiment`
- `tests/test_experiment_result_import.py` validates backend-probe enrichment,
  benchmark-result import, and generated experiment updates.
- Runner comparison report records an Ascend smoke test:
  `2026-07-09-replay-sigmoid-regression-smoke`, Pass@1 true, Pass@4 true,
  4/4 pass, best candidate `sigmoid_scale_sum_replay_v2`.

### T2/T3 OpSpec Parsing

Status: complete for current supported scope.

Evidence:

- Parsed T2/T3 OpSpecs exist:
  - `benchmarks/parsed/t2_add_rmsnorm_cast.yaml`
  - `benchmarks/parsed/t2_add_rmsnorm_quant.yaml`
  - `benchmarks/parsed/t2_rope.yaml`
  - `benchmarks/parsed/t3_layernorm_gated.yaml`
  - `benchmarks/parsed/t3_causal_conv1d.yaml`
  - `benchmarks/parsed/t3_decode_mla.yaml`
- `tests/test_benchmark_registry_and_opspec.py` asserts:
  - 13 total official cases
  - 10 `opspec_supported`
  - no `parse_failed` entries
  - T3 symbolic-shape cases extract useful sketches.

Remaining unsupported cases are intentional, not parse failures:

- `t1/matmul_basic`
- `t1/matmul_biasadd`
- `t2/moe_topk_softmax`

### Priority Remaining-Operator Sketches

Status: complete for the named priority operators.

Evidence:

- `t2/add_rmsnorm_cast`: parsed OpSpec and sketch, Pass@4 report
  `experiments/reports/2026-07-09-add-rmsnorm-cast-pass4.yaml`.
- `t2/add_rmsnorm_quant`: parsed OpSpec and sketch, negative Pass@4 report
  `experiments/reports/2026-07-09-add-rmsnorm-quant-pass4.yaml`.
- `t2/rope`: parsed OpSpec and sketch, Pass@4 report
  `experiments/reports/2026-07-09-rope-pass4.yaml`.
- `t3/layernorm_gated`: parsed OpSpec and sketch, Pass@4 report
  `experiments/reports/2026-07-09-layernorm-gated-pass4.yaml`.

### Deterministic Replay Regression Command

Status: complete.

Evidence:

- Script: `scripts/run_replay_regression.py`
- Smoke result recorded in
  `experiments/reports/2026-07-09-runner-path-comparison-sigmoid-scale-sum.yaml`.
- The script updates generated experiment records with backend probes and
  benchmark result fields.

### Optional Manual Seed Candidates

Status: complete for priority pre-key targets.

Evidence:

- Positive T1 reduction:
  `experiments/reports/2026-07-09-sigmoid-scale-sum-pass4-updated-akg.yaml`
- Positive T2 normalization:
  `experiments/reports/2026-07-09-add-rmsnorm-cast-pass4.yaml`
- RoPE parity/intrinsic lesson:
  `experiments/reports/2026-07-09-rope-pass4.yaml`
- Negative int8 quantization lesson:
  `experiments/reports/2026-07-09-add-rmsnorm-quant-pass4.yaml`
- Positive T3 normalization:
  `experiments/reports/2026-07-09-layernorm-gated-pass4.yaml`

## Current Best Retrieval Examples

- `sigmoid_scale_sum_v2`: positive row-reduction speedup.
- `add_rmsnorm_cast_v2`: positive T2 rowwise RMSNorm/cast speedup.
- `layernorm_gated_v4`: positive T3 fp16 gated-RMSNorm row-grouping speedup.
- `softmax_v4`: correctness-positive, performance-negative rowwise softmax.
- `fused_silu_and_mul_v3`: correctness-positive, strongly performance-negative
  fused elementwise trajectory.
- `rope_v1` and `rope_v4`: intrinsic-vs-Triton parity.
- `add_rmsnorm_quant_v2`-`v4`: exact-int8 quantization boundary failures.

## Remaining Work Before Full Objective Completion

1. Configure an AKG Agents `standard` model level.
2. Run `run_torch_bench_lite.py --mode full --backend npu --cases
   sigmoid_scale_sum --pass-n 4`.
3. Compare AKG Agents full-mode generated/extracted outputs against the
   standalone runner schema, Pass@4, speedup, score, and logs.
4. After API credentials exist, run the first live provider Pass@4 cycle and
   compare it with replay/manual evidence.
