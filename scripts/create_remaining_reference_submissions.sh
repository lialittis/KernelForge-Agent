#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/submissions/remaining_reference}"

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team reference_t1_matmul_basic \
  --candidate reference_t1_matmul_basic \
  --case t1/matmul_basic=kernel_forge/candidates/reference_matmul_basic.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team reference_t1_matmul_biasadd \
  --candidate reference_t1_matmul_biasadd \
  --case t1/matmul_biasadd=kernel_forge/candidates/reference_matmul_biasadd.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team reference_t2_rope \
  --candidate reference_t2_rope \
  --case t2/rope=kernel_forge/candidates/reference_rope.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team reference_t2_add_rmsnorm_cast \
  --candidate reference_t2_add_rmsnorm_cast \
  --case t2/add_rmsnorm_cast=kernel_forge/candidates/reference_add_rmsnorm_cast.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team reference_t2_add_rmsnorm_quant \
  --candidate reference_t2_add_rmsnorm_quant \
  --case t2/add_rmsnorm_quant=kernel_forge/candidates/reference_add_rmsnorm_quant.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team reference_t2_moe_topk_softmax \
  --candidate reference_t2_moe_topk_softmax \
  --case t2/moe_topk_softmax=kernel_forge/candidates/reference_moe_topk_softmax.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team reference_t3_causal_conv1d \
  --candidate reference_t3_causal_conv1d \
  --case t3/causal_conv1d=kernel_forge/candidates/reference_causal_conv1d.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team reference_t3_decode_mla \
  --candidate reference_t3_decode_mla \
  --case t3/decode_mla=kernel_forge/candidates/reference_decode_mla.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team reference_t3_layernorm_gated \
  --candidate reference_t3_layernorm_gated \
  --case t3/layernorm_gated=kernel_forge/candidates/reference_layernorm_gated.py

cat <<EOF
Remaining reference submissions created:

$ROOT_DIR/$OUTPUT_ROOT

Run all candidates with:
python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \\
  $OUTPUT_ROOT \\
  --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite \\
  --output outputs/results/remaining_reference \\
  --warmup 10 \\
  --iterations 100 \\
  --num-trials 3
EOF
