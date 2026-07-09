#!/usr/bin/env bash
set -euo pipefail

ASCEND_HOST="${ASCEND_HOST:-ascend-kf}"
REMOTE_DIR="${REMOTE_DIR:-/data/KernelForge-Agent}"
ASCEND_SET_ENV="${ASCEND_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
ASCEND_VENV="${ASCEND_VENV:-/data/venvs/kf-triton-ascend}"
CASE_ID="${CASE_ID:-t1/sigmoid_scale_sum}"
CANDIDATE="${CANDIDATE:-kernel_forge/candidates/sigmoid_scale_sum_v2.py}"
TASK_TYPE="${TASK_TYPE:-profile}"
REMOTE_OUTPUT="${REMOTE_OUTPUT:-outputs/results/akg_agents_verifier_probe_sigmoid_scale_sum_v2_$(date +%Y_%m_%d).json}"
REMOTE_COMPARE_OUTPUT="${REMOTE_COMPARE_OUTPUT:-outputs/results/akg_agents_verifier_probe_sigmoid_scale_sum_v2_compare_$(date +%Y_%m_%d).yaml}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-outputs/akg_agents_verifier_logs}"
STANDALONE_REPORT="${STANDALONE_REPORT:-experiments/reports/2026-07-09-replay-sigmoid-scale-sum-pass4-updated-akg.yaml}"
CHECK_ONLY="${CHECK_ONLY:-0}"

usage() {
  cat <<'EOF'
Fast-forward the Ascend worker, run the no-key AKG Agents verifier-only probe,
and compare the probe JSON with the standalone replay Pass@4 report.

This script uses ssh BatchMode only. It will fail rather than prompting for a
password; reopen the SSH ControlMaster session first if the gateway requires it.

Environment overrides:
  ASCEND_HOST=ascend-kf
  REMOTE_DIR=/data/KernelForge-Agent
  ASCEND_SET_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
  ASCEND_VENV=/data/venvs/kf-triton-ascend
  CASE_ID=t1/sigmoid_scale_sum
  CANDIDATE=kernel_forge/candidates/sigmoid_scale_sum_v2.py
  TASK_TYPE=profile
  REMOTE_OUTPUT=outputs/results/akg_agents_verifier_probe_sigmoid_scale_sum_v2_YYYY_MM_DD.json
  REMOTE_COMPARE_OUTPUT=outputs/results/akg_agents_verifier_probe_sigmoid_scale_sum_v2_compare_YYYY_MM_DD.yaml
  REMOTE_LOG_DIR=outputs/akg_agents_verifier_logs
  STANDALONE_REPORT=experiments/reports/2026-07-09-replay-sigmoid-scale-sum-pass4-updated-akg.yaml
  CHECK_ONLY=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$CHECK_ONLY" == "1" ]]; then
  cat <<EOF
Resolved Ascend verifier probe:
  ASCEND_HOST=$ASCEND_HOST
  REMOTE_DIR=$REMOTE_DIR
  CASE_ID=$CASE_ID
  CANDIDATE=$CANDIDATE
  TASK_TYPE=$TASK_TYPE
  REMOTE_OUTPUT=$REMOTE_OUTPUT
  REMOTE_COMPARE_OUTPUT=$REMOTE_COMPARE_OUTPUT
  REMOTE_LOG_DIR=$REMOTE_LOG_DIR
  STANDALONE_REPORT=$STANDALONE_REPORT

Remote command will:
  1. git pull --ff-only
  2. source CANN and venv if present
  3. export PYTHONPATH=$REMOTE_DIR:\${PYTHONPATH:-}
  4. run scripts/run_akg_agents_verifier_probe.py
  5. run scripts/compare_runner_results.py
EOF
  exit 0
fi

ssh -o BatchMode=yes "$ASCEND_HOST" \
  "REMOTE_DIR='$REMOTE_DIR' ASCEND_SET_ENV='$ASCEND_SET_ENV' ASCEND_VENV='$ASCEND_VENV' CASE_ID='$CASE_ID' CANDIDATE='$CANDIDATE' TASK_TYPE='$TASK_TYPE' REMOTE_OUTPUT='$REMOTE_OUTPUT' REMOTE_COMPARE_OUTPUT='$REMOTE_COMPARE_OUTPUT' REMOTE_LOG_DIR='$REMOTE_LOG_DIR' STANDALONE_REPORT='$STANDALONE_REPORT' bash -s" <<'REMOTE'
set -euo pipefail

cd "$REMOTE_DIR"
git pull --ff-only

if [[ -f "$ASCEND_SET_ENV" ]]; then
  # shellcheck source=/dev/null
  source "$ASCEND_SET_ENV"
fi

if [[ -d "$ASCEND_VENV" ]]; then
  # shellcheck source=/dev/null
  source "$ASCEND_VENV/bin/activate"
fi

export PYTHONPATH="$REMOTE_DIR:${PYTHONPATH:-}"

python scripts/run_akg_agents_verifier_probe.py \
  --case "$CASE_ID" \
  --candidate "$CANDIDATE" \
  --task-type "$TASK_TYPE" \
  --output "$REMOTE_OUTPUT" \
  --log-dir "$REMOTE_LOG_DIR"

python scripts/compare_runner_results.py \
  --standalone-report "$STANDALONE_REPORT" \
  --akg-agents-json "$REMOTE_OUTPUT" \
  --case "$CASE_ID" \
  --akg-log-dir "$REMOTE_LOG_DIR" \
  --output "$REMOTE_COMPARE_OUTPUT"

printf 'Probe JSON: %s\n' "$REMOTE_OUTPUT"
printf 'Comparison YAML: %s\n' "$REMOTE_COMPARE_OUTPUT"
REMOTE
