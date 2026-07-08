#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/submissions/fused_silu_and_mul_pass4}"

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team fused_silu_and_mul_v1 \
  --candidate fused_silu_and_mul_v1 \
  --case t1/fused_silu_and_mul=kernel_forge/candidates/fused_silu_and_mul_v1.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team fused_silu_and_mul_v2 \
  --candidate fused_silu_and_mul_v2 \
  --case t1/fused_silu_and_mul=kernel_forge/candidates/fused_silu_and_mul_v2.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team fused_silu_and_mul_v3 \
  --candidate fused_silu_and_mul_v3 \
  --case t1/fused_silu_and_mul=kernel_forge/candidates/fused_silu_and_mul_v3.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team fused_silu_and_mul_v4 \
  --candidate fused_silu_and_mul_v4 \
  --case t1/fused_silu_and_mul=kernel_forge/candidates/fused_silu_and_mul_v4.py

cat <<EOF
Fused SiLU and mul Pass@4 submissions created:

$ROOT_DIR/$OUTPUT_ROOT

Run all candidates with:
python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \\
  $OUTPUT_ROOT \\
  --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite \\
  --output outputs/results/fused_silu_and_mul_pass4 \\
  --warmup 10 \\
  --iterations 100 \\
  --num-trials 3
EOF
