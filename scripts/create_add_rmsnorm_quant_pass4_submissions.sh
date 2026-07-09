#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/submissions/add_rmsnorm_quant_pass4}"

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team add_rmsnorm_quant_v1 \
  --candidate add_rmsnorm_quant_v1 \
  --case t2/add_rmsnorm_quant=kernel_forge/candidates/add_rmsnorm_quant_v1.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team add_rmsnorm_quant_v2 \
  --candidate add_rmsnorm_quant_v2 \
  --case t2/add_rmsnorm_quant=kernel_forge/candidates/add_rmsnorm_quant_v2.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team add_rmsnorm_quant_v3 \
  --candidate add_rmsnorm_quant_v3 \
  --case t2/add_rmsnorm_quant=kernel_forge/candidates/add_rmsnorm_quant_v3.py

python scripts/create_submission.py \
  --layout flat \
  --output-root "$OUTPUT_ROOT" \
  --team add_rmsnorm_quant_v4 \
  --candidate add_rmsnorm_quant_v4 \
  --case t2/add_rmsnorm_quant=kernel_forge/candidates/add_rmsnorm_quant_v4.py

cat <<EOF
Add RMSNorm quant Pass@4 submissions created:

$OUTPUT_ROOT

Run all candidates with:
python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \\
  $OUTPUT_ROOT \\
  --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite \\
  --output outputs/results/add_rmsnorm_quant_pass4 \\
  --warmup 10 \\
  --iterations 100 \\
  --num-trials 3
EOF
