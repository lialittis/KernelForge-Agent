#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/submissions/softmax_pass4}"

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team softmax_v1 \
  --candidate softmax_v1 \
  --case t1/softmax=kernel_forge/candidates/softmax_v1.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team softmax_v2 \
  --candidate softmax_v2 \
  --case t1/softmax=kernel_forge/candidates/softmax_v2.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team softmax_v3 \
  --candidate softmax_v3 \
  --case t1/softmax=kernel_forge/candidates/softmax_v3.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team softmax_v4 \
  --candidate softmax_v4 \
  --case t1/softmax=kernel_forge/candidates/softmax_v4.py

cat <<EOF
Softmax Pass@4 submissions created:

$ROOT_DIR/$OUTPUT_ROOT

Run all candidates with:
python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \\
  $OUTPUT_ROOT \\
  --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite \\
  --output outputs/results/softmax_pass4 \\
  --warmup 10 \\
  --iterations 100 \\
  --num-trials 3
EOF
