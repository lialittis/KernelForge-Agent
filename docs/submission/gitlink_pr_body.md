# Team

算子炼金术师

# Summary

This PR submits SketchSkill-AKG, a skill-driven AI/Agent prototype for Ascend NPU operator generation and hardware-feedback optimization.

SketchSkill-AKG is not a single hand-written kernel submission. It is a reusable pipeline that converts AKG Bench Lite tasks into OpSpec records, builds NPU-aware Operator Sketches, retrieves operator-pattern skills, generates `ModelNew` candidates through a provider boundary, materializes official submission layouts, runs correctness/performance validation on Ascend hardware, imports benchmark results, summarizes Pass@N, and writes reusable lessons back to the Skill Library.

# Current Implementation

- Pinned AKG Bench Lite benchmark submodule.
- Benchmark registry for all 13 official cases.
- Lite benchmark OpSpec coverage: 13/13.
- NPU-aware Operator Sketch coverage for all 13 parsed Lite cases, including elementwise, fused elementwise, reduction, softmax, normalization, matmul-like, MoE top-k softmax, layout/transpose, convolution, and decode/attention patterns.
- Deterministic OpSpec/Sketch validation gate through `scripts/validate_opspecs.py`.
- Skill Library organized by operator pattern and engineering task.
- Prompt templates for Code Agent, Repair Agent, and Skill Writer.
- Deterministic `replay` provider and OpenAI Responses API provider boundary.
- Candidate generation, official submission materialization, result import, and Pass@N reporting.
- Ascend benchmark evidence for a priority executable subset.

# Reproduction

Initialize the benchmark:

```bash
bash scripts/setup_benchmark_submodule.sh
```

Generate the benchmark registry:

```bash
python scripts/scan_benchmark_cases.py \
  --output benchmarks/raw/akg_kernels_bench_lite_registry.yaml \
  --repo-root .
```

Generate OpSpec and Sketch records:

```bash
python scripts/extract_opspec_batch.py \
  --output-dir benchmarks/parsed \
  --repo-root .
```

Run the provider-generation smoke test:

```bash
python scripts/generate_candidate.py \
  --opspec benchmarks/parsed/t1_sigmoid_scale_sum.yaml \
  --provider replay \
  --backend triton_ascend \
  --pass-n 1 \
  --run-id replay-provider-smoke \
  --output-root /tmp/kf-generated-smoke
```

Run focused local tests:

```bash
python -m pytest -q tests/test_agent_generation_workflow.py tests/test_fused_silu_and_mul_pass4.py
```

# Benchmark Evidence

| Task | Batch | Pass@1 | Pass@4 | Best speedup | Notes |
| --- | --- | --- | --- | ---: | --- |
| `t1/sigmoid_scale_sum` | manual | true | true, 4/4 | `2.0279x` | Positive rowwise-reduction case |
| `t1/sigmoid_scale_sum` | replay provider | true | true, 4/4 | `1.9980x` | Provider workflow reproduced the known trajectory |
| `t1/fused_silu_and_mul` | manual | true | true, 4/4 | `1.0027x` | Triton variants pass correctness but are performance-negative lessons |
| `t2/add_rmsnorm_cast` | manual | true | true, 4/4 | `2.0135x` | Positive T2 normalization case |
| `t3/layernorm_gated` | manual | true | true, 4/4 | `1.5137x` | Positive T3 fp16 gated RMSNorm case |
| `t1/gelu` | tuning case study | true | not formal Pass@4 | `0.6059x` | Correctness-positive Triton-Ascend case, slower than baseline |

Summary claim:

```text
Lite benchmark OpSpec coverage: 13/13. Current executable candidates and
Pass@4 evidence cover a priority subset. Full live AI generation remains gated
only by model/API configuration.
```

# Included Documents

- `docs/project_book_full_zh.md`
- `docs/technical_design.md`
- `docs/submission_package_readme.md`
- `docs/competition_alignment.md`
- `docs/architecture.md`
- `docs/project_workflow.md`
- `docs/benchmark_spec.md`
- `docs/dev_guide.md`

# Known Limitations

- Lite benchmark OpSpec/Sketch coverage is complete; larger Benchmark suites,
  dynamic shapes, and AKG Agents full-mode comparison remain future work.
- Full Repair Agent and Profiler/Search Agent automation is still under development.
- Live OpenAI/provider benchmark comparison is planned after model/API configuration exists.
- GELU Triton-Ascend is correctness-positive but still slower than the framework baseline.

# Project Book And Email Status

- Updated Chinese project book: `docs/project_book_full_zh.md`
- Technical design: `docs/technical_design.md`
- Email target: `contact@public.mindspore.cn`
- Email status: pending until the PR link is available.
