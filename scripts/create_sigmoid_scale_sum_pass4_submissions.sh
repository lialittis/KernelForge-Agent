#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/submissions/sigmoid_scale_sum_pass4}"

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team sigmoid_scale_sum_v1 \
  --candidate sigmoid_scale_sum_v1 \
  --case t1/sigmoid_scale_sum=kernel_forge/candidates/sigmoid_scale_sum_v1.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team sigmoid_scale_sum_v2 \
  --candidate sigmoid_scale_sum_v2 \
  --case t1/sigmoid_scale_sum=kernel_forge/candidates/sigmoid_scale_sum_v2.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team sigmoid_scale_sum_v3 \
  --candidate sigmoid_scale_sum_v3 \
  --case t1/sigmoid_scale_sum=kernel_forge/candidates/sigmoid_scale_sum_v3.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team sigmoid_scale_sum_v4 \
  --candidate sigmoid_scale_sum_v4 \
  --case t1/sigmoid_scale_sum=kernel_forge/candidates/sigmoid_scale_sum_v4.py

cat <<EOF
Sigmoid scale sum Pass@4 submissions created:

$ROOT_DIR/$OUTPUT_ROOT

Run all candidates with:
python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \\
  $OUTPUT_ROOT \\
  --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite \\
  --output outputs/results/sigmoid_scale_sum_pass4 \\
  --warmup 10 \\
  --iterations 100 \\
  --num-trials 3
EOF
