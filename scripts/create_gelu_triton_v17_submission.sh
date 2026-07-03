#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
TEAM_NAME="${TEAM_NAME:-gelu_triton_v17}"
SUBMISSION_ROOT="$ROOT_DIR/outputs/submissions/$TEAM_NAME/$TEAM_NAME"
CANDIDATE_SOURCE="$ROOT_DIR/kernel_forge/candidates/gelu_triton_v17.py"

case "$TEAM_NAME" in
  ""|*[!A-Za-z0-9_-]*)
    echo "TEAM_NAME must contain only letters, numbers, underscores, and hyphens." >&2
    exit 1
    ;;
esac

mkdir -p "$SUBMISSION_ROOT/t1"

cat > "$SUBMISSION_ROOT/meta.json" <<JSON
{
  "team_name": "$TEAM_NAME",
  "candidate": "gelu_triton_v17",
  "source": "kernel_forge/candidates/gelu_triton_v17.py"
}
JSON

cp "$CANDIDATE_SOURCE" "$SUBMISSION_ROOT/t1/gelu.py"

cat <<EOF
Triton GELU v17 submission created:

$SUBMISSION_ROOT

Run with:
python third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite/tools/run_bench.py \\
  outputs/submissions/$TEAM_NAME \\
  --team $TEAM_NAME \\
  --bench-dir third_party/akg/akg_agents/benchmark/akg_kernels_bench_lite \\
  --output outputs/results/$TEAM_NAME \\
  --warmup 10 \\
  --iterations 100 \\
  --num-trials 3
EOF

