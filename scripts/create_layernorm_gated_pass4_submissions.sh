#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/submissions/layernorm_gated_pass4}"

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team layernorm_gated_v1 \
  --candidate layernorm_gated_v1 \
  --case t3/layernorm_gated=kernel_forge/candidates/layernorm_gated_v1.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team layernorm_gated_v2 \
  --candidate layernorm_gated_v2 \
  --case t3/layernorm_gated=kernel_forge/candidates/layernorm_gated_v2.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team layernorm_gated_v3 \
  --candidate layernorm_gated_v3 \
  --case t3/layernorm_gated=kernel_forge/candidates/layernorm_gated_v3.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team layernorm_gated_v4 \
  --candidate layernorm_gated_v4 \
  --case t3/layernorm_gated=kernel_forge/candidates/layernorm_gated_v4.py

cat <<EOF
LayerNorm gated Pass@4 submissions created:

$OUTPUT_ROOT

Run all candidates with:
python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \\
  $OUTPUT_ROOT \\
  --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite \\
  --output outputs/results/layernorm_gated_pass4 \\
  --warmup 10 \\
  --iterations 100 \\
  --num-trials 3
EOF
